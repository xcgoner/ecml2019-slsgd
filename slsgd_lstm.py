import argparse, time, logging, os, math, random, glob
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"


import numpy as np
from scipy import stats

import mxnet as mx
from mxnet import nd, gluon, autograd
import gluonnlp as nlp

from os import listdir
import os.path
import argparse

import pickle

from mpi4py import MPI

mpi_comm = MPI.COMM_WORLD
mpi_size = mpi_comm.Get_size()
mpi_rank = mpi_comm.Get_rank()

# print('rank: %d' % (mpi_rank), flush=True)

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, help="dir of the data", required=True)
parser.add_argument("--valdir", type=str, help="dir of the val data", required=True)
parser.add_argument("--epochs", type=int, help="epochs", default=100)
parser.add_argument("--interval", type=int, help="log interval", default=10)
parser.add_argument("--nsplit", type=int, help="number of split", default=40)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--alpha", type=float, help="moving average", default=1.0)
parser.add_argument("--alpha-decay", type=float, help="decay factor of alpha", default=0.5)
parser.add_argument("--alpha-decay-epoch", type=str, help="epoch of alpha decay", default='800')
parser.add_argument("--log", type=str, help="dir of the log file", default='train_cifar100.log')
# parser.add_argument("--classes", type=int, help="number of classes", default=20)
parser.add_argument("--iterations", type=int, help="number of local epochs", default=50)
parser.add_argument("--aggregation", type=str, help="aggregation method", default='mean')
parser.add_argument("--nbyz", type=int, help="number of Byzantine workers", default=0)
parser.add_argument("--trim", type=int, help="number of trimmed workers on one side", default=0)
parser.add_argument("--lr-decay", type=float, help="lr decay rate", default=0.5)
parser.add_argument("--lr-decay-epoch", type=str, help="lr decay epoch", default='2000')
parser.add_argument("--iid", type=int, help="IID setting", default=0)
parser.add_argument("--model", type=str, help="model", default='mobilenetv2_1.0')
parser.add_argument("--save", type=int, help="save", default=0)
parser.add_argument("--seed", type=int, help="random seed", default=733)
 
args = parser.parse_args()

# print(args, flush=True)

filehandler = logging.FileHandler(args.log)
streamhandler = logging.StreamHandler()

if mpi_rank == 0:
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)

mx.random.seed(args.seed + mpi_rank)
random.seed(args.seed + mpi_rank)
np.random.seed(args.seed + mpi_rank)

data_dir = os.path.join(args.dir, 'dataset_split_{}'.format(args.nsplit))
train_dir = os.path.join(data_dir, 'train')
# val_dir = os.path.join(data_dir, 'val')
val_train_dir = os.path.join(args.valdir, 'train')
val_val_dir = os.path.join(args.valdir, 'val')

training_files = []
for filename in sorted(listdir(train_dir)):
    absolute_filename = os.path.join(train_dir, filename)
    if filename != 'vocab.pkl':
        training_files.append(absolute_filename)

context = mx.cpu()

# classes = args.classes

def get_train_batch(train_filename):
    with open(train_filename, "rb") as f:
        data = pickle.load(f)
        data = pickle.loads(data)
    return data[0], data[1]

def get_train_batch_byz(train_filename):
    with open(train_filename, "rb") as f:
        data = pickle.load(f)
        data = pickle.loads(data)
    for target in data[1]:
        target[:] = nd.relu(target - 1000)
    return data[0], data[1]

def get_val_train_batch(data_dir):
    test_filename = os.path.join(data_dir, 'train_data_%03d.pkl' % mpi_rank)
    with open(test_filename, "rb") as f:
        data = pickle.load(f)
        data = pickle.loads(data)
    return data[0], data[1]

def get_val_val_batch(data_dir):
    test_filename = os.path.join(data_dir, 'val_data_%03d.pkl' % mpi_rank)
    with open(test_filename, "rb") as f:
        data = pickle.load(f)
        data = pickle.loads(data)
    return data[0], data[1]

train_data_list = []
for training_file in training_files:
    [train_X, train_Y] = get_train_batch(training_file)
    train_data_list.append([train_X, train_Y])

[val_train_X, val_train_Y] = get_val_train_batch(val_train_dir)

[val_val_X, val_val_Y] = get_val_val_batch(val_val_dir)

vocab_dir = os.path.join(train_dir, 'vocab.pkl')
with open(vocab_dir, "rb") as f:
        data = pickle.load(f)
        vocab = pickle.loads(data)

model_name = args.model

if model_name == 'default':
    model_kwargs = {'ctx': context, 'vocab': vocab, 'dataset_name': None, 'pretrained': False}
    net, vocab = nlp.model.get_model('standard_lstm_lm_200', **model_kwargs)
else:
    model_kwargs = {'ctx': context, 'vocab': vocab, 'dataset_name': None, 'pretrained': False}
    net, vocab = nlp.model.get_model(model_name, **model_kwargs)

net.initialize(mx.init.Xavier(), ctx=context)


# # no weight decay
# for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#     v.wd_mult = 0.0
    
optimizer = 'sgd'
lr = args.lr
# optimizer_params = {'momentum': 0.9, 'learning_rate': lr, 'wd': 0.0001}
optimizer_params = {'momentum': 0.0, 'learning_rate': lr, 'wd': 0.0}
grad_clip = 0.25
batch_size = 20

lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
alpha_decay_epoch = [int(i) for i in args.alpha_decay_epoch.split(',')]

trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

train_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
test_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()

def detach(hidden):
    if isinstance(hidden, (tuple, list)):
        hidden = [detach(i) for i in hidden]
    else:
        hidden = hidden.detach()
    return hidden

def evaluate(model, data_source, batch_size, ctx):
    total_L = 0.0
    ntotal = 0
    hidden = model.begin_state(batch_size=batch_size, func=mx.nd.zeros, ctx=ctx)
    for i, (data, target) in enumerate(zip(*data_source)):
        # data = data.as_in_context(ctx)
        # target = target.as_in_context(ctx)
        output, hidden = model(data, hidden)
        hidden = detach(hidden)
        L = test_cross_entropy(output.reshape(-3, -1), target.reshape(-1))
        total_L += mx.nd.sum(L).asscalar()
        ntotal += L.size
    return [total_L, ntotal]

# warmup
# print('warm up', flush=True)
trainer.set_learning_rate(0.01)
# train_data = random.choice(train_data_list)
train_data = train_data_list[90]
for local_epoch in range(5):
    hiddens = net.begin_state(batch_size, func=mx.nd.zeros, ctx=context)
    for i, (data, target) in enumerate(zip(*train_data)):
        hiddens = detach(hiddens)
        with autograd.record():
            output, h = net(data, hiddens)
            batch_L = train_cross_entropy(output.reshape(-3, -1), target.reshape(-1,))
            L = batch_L / data.size
            hiddens = h
        L.backward()
        grads = [p.grad() for p in net.collect_params().values()]
        gluon.utils.clip_global_norm(grads, grad_clip)
        trainer.step(1)

if mpi_rank == 0:
    params_prev = [param.data().copy() for param in net.collect_params().values()]
else:
    params_prev = None

nd.waitall()

# broadcast
params_prev = mpi_comm.bcast(params_prev, root=0)
for param, param_prev in zip(net.collect_params().values(), params_prev):
    param.set_data(param_prev)

if mpi_rank == 0:
    worker_list = list(range(mpi_size))

training_file_index_list = [i for i in range(len(training_files))]

alpha = args.alpha

randperm_choice_list = []
randperm_list = [i for i in range(args.nsplit)]
for i in range(int(math.ceil(args.epochs * mpi_size / args.nsplit) * 2)):
    random.shuffle(randperm_list)
    randperm_choice_list = randperm_choice_list + randperm_list

nd.waitall()

time_0 = time.time()

for epoch in range(1, args.epochs+1):
        # train_metric.reset()

        if epoch in lr_decay_epoch:
            lr = lr * args.lr_decay

        if epoch in alpha_decay_epoch:
            alpha = alpha * args.alpha_decay

        tic = time.time()

        if args.iid == 0:
            if mpi_rank == 0:
                training_file_index_sublist = randperm_choice_list[(mpi_size * epoch):(mpi_size * epoch + mpi_size)]
                # logger.info(training_file_index_sublist)
            else:
                training_file_index_sublist = None
            training_file_index = mpi_comm.scatter(training_file_index_sublist, root=0)
            train_data = train_data_list[training_file_index]

        trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)
        trainer.set_learning_rate(lr)

        if alpha < 1:
            for param, param_prev in zip(net.collect_params().values(), params_prev):
                if param.grad_req != 'null':
                    param_prev[:] = param.data() * (1-alpha)

        # select byz workers
        if args.nbyz > 0:
            if mpi_rank == 0:
                random.shuffle(worker_list)
                byz_worker_list = worker_list[0:args.nbyz]
            else:
                byz_worker_list = None
            byz_worker_list = mpi_comm.bcast(byz_worker_list, root=0)
        else:
            byz_worker_list = []

        if mpi_rank in byz_worker_list:
            # byz worker
            [byz_train_X, byz_train_Y] = get_train_batch_byz(random.choice(training_files))
            net.initialize(mx.init.MSRAPrelu(), ctx=context, force_reinit=True)
            for local_epoch in range(args.iterations):
                hiddens = net.begin_state(batch_size, func=mx.nd.zeros, ctx=context)
                for i, (data, target) in enumerate(zip(byz_train_X, byz_train_Y)):
                    hiddens = detach(hiddens)
                    with autograd.record():
                        output, h = net(data, hiddens)
                        batch_L = train_cross_entropy(output.reshape(-3, -1), target.reshape(-1,))
                        L = batch_L / data.size
                        hiddens = h
                    L.backward()
                    grads = [p.grad() for p in net.collect_params().values()]
                    gluon.utils.clip_global_norm(grads, grad_clip)
                    trainer.step(1)
        else:
            # train
            # local epoch
            for local_epoch in range(args.iterations):
                if args.iid == 1:
                    train_data = random.choice(train_data_list)
                hiddens = net.begin_state(batch_size, func=mx.nd.zeros, ctx=context)
                for i, (data, target) in enumerate(zip(*train_data)):
                    hiddens = detach(hiddens)
                    with autograd.record():
                        output, h = net(data, hiddens)
                        batch_L = train_cross_entropy(output.reshape(-3, -1), target.reshape(-1,))
                        L = batch_L / data.size
                        hiddens = h
                    L.backward()
                    grads = [p.grad() for p in net.collect_params().values()]
                    gluon.utils.clip_global_norm(grads, grad_clip)
                    trainer.step(1)
        
        # aggregation
        nd.waitall()
        params_np = [param.data().copy().asnumpy() for param in net.collect_params().values()]
        params_np_list = mpi_comm.gather(params_np, root=0)
        if mpi_rank == 0:
            n_params = len(params_np)
            if args.aggregation == "trim" or args.trim > 0:
                params_np = [ ( stats.trim_mean( np.stack( [params[j] for params in params_np_list], axis=0), args.trim/mpi_size, axis=0 ) ) for j in range(n_params) ]
            else:
                params_np = [ ( np.mean( np.stack( [params[j] for params in params_np_list], axis=0), axis=0 ) ) for j in range(n_params) ]
        else:
            params_np = None
        params_np = mpi_comm.bcast(params_np, root=0)
        params_nd = [ nd.array(param_np) for param_np in params_np ]
        for param, param_nd in zip(net.collect_params().values(), params_nd):
            param.set_data(param_nd)
        if alpha < 1:
            # moving average
            for param, param_prev in zip(net.collect_params().values(), params_prev):
                if param.grad_req != 'null':
                    weight = param.data()
                    weight[:] = weight * alpha + param_prev

        # test
        nd.waitall()

        toc = time.time()

        if epoch % args.interval == 0 :
            train_result = evaluate(net, [val_train_X, val_train_Y], batch_size, context)
            test_result = evaluate(net, [val_val_X, val_val_Y], batch_size, context)

            train_result_list = mpi_comm.gather(train_result, root=0)
            test_result_list = mpi_comm.gather(test_result, root=0)

            if mpi_rank == 0:
                train_result = np.array(train_result_list)
                test_result = np.array(test_result_list)
                train_loss = np.sum(train_result[:,0])/np.sum(train_result[:,1])
                test_loss = np.sum(test_result[:,0])/np.sum(test_result[:,1])
                logger.info('[Epoch %d] validation: train loss=%f, train ppl=%.2f, val loss=%f, test ppl=%.2f, lr=%f, alpha=%f, time=%f, elapsed=%f'%(epoch, train_loss, math.exp(train_loss), test_loss, math.exp(test_loss), trainer.learning_rate, alpha, toc-tic, time.time()-time_0))
                # logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f'%(epoch, top1, top5))

                if args.save == 1:
                    [dirname, postfix] = os.path.splitext(args.log)
                    filename = dirname + ("_%04d.params" % (epoch))
                    net.save_parameters(filename)
        
        nd.waitall()






