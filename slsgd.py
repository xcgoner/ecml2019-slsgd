import argparse, time, logging, os, math, random
os.environ["MXNET_USE_OPERATOR_TUNING"] = "0"


import numpy as np
from scipy import stats
import mxnet as mx
from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, LRScheduler

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
parser.add_argument("--batchsize", type=int, help="batchsize", default=8)
parser.add_argument("--epochs", type=int, help="epochs", default=100)
parser.add_argument("--interval", type=int, help="log interval", default=10)
parser.add_argument("--nsplit", type=int, help="number of split", default=40)
parser.add_argument("--lr", type=float, help="learning rate", default=0.001)
parser.add_argument("--alpha", type=float, help="moving average", default=1.0)
parser.add_argument("--alpha-decay", type=float, help="decay factor of alpha", default=0.5)
parser.add_argument("--alpha-decay-epoch", type=str, help="epoch of alpha decay", default='800')
parser.add_argument("--log", type=str, help="dir of the log file", default='train_cifar100.log')
parser.add_argument("--classes", type=int, help="number of classes", default=20)
parser.add_argument("--iterations", type=int, help="number of local epochs", default=50)
parser.add_argument("--aggregation", type=str, help="aggregation method", default='mean')
parser.add_argument("--nbyz", type=int, help="number of Byzantine workers", default=0)
parser.add_argument("--trim", type=int, help="number of trimmed workers on one side", default=0)
# parser.add_argument("--lr-decay", type=float, help="lr decay rate", default=0.1)
# parser.add_argument("--lr-decay-epoch", type=str, help="lr decay epoch", default='400')
parser.add_argument("--iid", type=int, help="IID setting", default=0)
parser.add_argument("--model", type=str, help="model", default='mobilenetv2_1.0')
parser.add_argument("--save", type=int, help="save", default=0)
parser.add_argument("--start-epoch", type=int, help="epoch start from", default=-1)
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
    training_files.append(absolute_filename)

context = mx.cpu()

classes = args.classes

def get_train_batch(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)

    # return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(L)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

def get_train_batch_byz(train_filename):
    with open(train_filename, "rb") as f:
        B, L = pickle.load(f)

    # return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(classes - 1 - L)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(classes - 1 - L)

def get_val_train_batch(data_dir):
    test_filename = os.path.join(data_dir, 'train_data_%03d.pkl' % mpi_rank)
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)
    
    # return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(L)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

def get_val_val_batch(data_dir):
    test_filename = os.path.join(data_dir, 'val_data_%03d.pkl' % mpi_rank)
    with open(test_filename, "rb") as f:
        B, L = pickle.load(f)
    
    # return nd.transpose(nd.array(B.astype('float32') / 255.0), (0, 3, 1, 2)), nd.array(L)
    return nd.transpose(nd.array(B), (0, 3, 1, 2)), nd.array(L)

train_data_list = []
for training_file in training_files:
    [train_X, train_Y] = get_train_batch(training_file)
    train_dataset = mx.gluon.data.dataset.ArrayDataset(train_X, train_Y)
    train_data = gluon.data.DataLoader(train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
    train_data_list.append(train_data)

[val_train_X, val_train_Y] = get_val_train_batch(val_train_dir)
val_train_dataset = mx.gluon.data.dataset.ArrayDataset(val_train_X, val_train_Y)
val_train_data = gluon.data.DataLoader(val_train_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=1)

[val_val_X, val_val_Y] = get_val_val_batch(val_val_dir)
val_val_dataset = mx.gluon.data.dataset.ArrayDataset(val_val_X, val_val_Y)
val_val_data = gluon.data.DataLoader(val_val_dataset, batch_size=1000, shuffle=False, last_batch='keep', num_workers=1)

model_name = args.model

if model_name == 'default':
    net = gluon.nn.Sequential()
    with net.name_scope():
        #  First convolutional layer
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(rate=0.25))
        #  Second convolutional layer
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Third convolutional layer
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.Conv2D(channels=128, kernel_size=3, padding=(1,1), activation='relu'))
        net.add(gluon.nn.BatchNorm())
        net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        net.add(gluon.nn.Dropout(rate=0.25))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.Conv2D(channels=64, kernel_size=3, padding=(1,1), activation='relu'))
        # net.add(gluon.nn.MaxPool2D(pool_size=2, strides=2))
        # Flatten and apply fullly connected layers
        net.add(gluon.nn.Flatten())
        # net.add(gluon.nn.Dense(512, activation="relu"))
        # net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dense(512, activation="relu"))
        net.add(gluon.nn.Dropout(rate=0.25))
        net.add(gluon.nn.Dense(classes))
else:
    model_kwargs = {'ctx': context, 'pretrained': False, 'classes': classes}
    net = get_model(model_name, **model_kwargs)

if model_name.startswith('cifar') or model_name == 'default':
    net.initialize(mx.init.Xavier(), ctx=context)
else:
    net.initialize(mx.init.MSRAPrelu(), ctx=context)

# # no weight decay
# for k, v in net.collect_params('.*beta|.*gamma|.*bias').items():
#     v.wd_mult = 0.0
    
optimizer = 'sgd'
lr = args.lr
# optimizer_params = {'momentum': 0.9, 'learning_rate': lr, 'wd': 0.0001}
optimizer_params = {'momentum': 0.0, 'learning_rate': lr, 'wd': 0.0}

# lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
alpha_decay_epoch = [int(i) for i in args.alpha_decay_epoch.split(',')]

trainer = gluon.Trainer(net.collect_params(), optimizer, optimizer_params)

loss_func = gluon.loss.SoftmaxCrossEntropyLoss()

train_metric = mx.metric.Accuracy()

acc_top1 = mx.metric.Accuracy()
acc_top5 = mx.metric.TopKAccuracy(5)
train_cross_entropy = mx.metric.CrossEntropy()

# warmup
# print('warm up', flush=True)
trainer.set_learning_rate(0.01)
# train_data = random.choice(train_data_list)
train_data = train_data_list[90]
for local_epoch in range(5):
    for i, (data, label) in enumerate(train_data):
        with ag.record():
            outputs = net(data)
            loss = loss_func(outputs, label)
        loss.backward()
        trainer.step(args.batchsize)
        if args.start_epoch > 0:
            break
    if args.start_epoch > 0:
        break

# # force initialization
# train_data = random.choice(train_data_list)
# for i, (data, label) in enumerate(train_data):
#     outputs = net(data)

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
for i in range(int(math.ceil(args.epochs * mpi_size / args.nsplit))):
    random.shuffle(randperm_list)
    randperm_choice_list = randperm_choice_list + randperm_list

if args.start_epoch > 0:
    [dirname, postfix] = os.path.splitext(args.log)
    filename = dirname + ("_%04d.params" % (args.start_epoch))
    net.load_parameters(filename, ctx=context)

    acc_top1.reset()
    acc_top5.reset()
    train_cross_entropy.reset()
    for i, (data, label) in enumerate(val_val_data):
        outputs = net(data)
        acc_top1.update(label, outputs)
        acc_top5.update(label, outputs)

    for i, (data, label) in enumerate(val_train_data):
        outputs = net(data)
        train_cross_entropy.update(label, nd.softmax(outputs))

    _, top1 = acc_top1.get()
    _, top5 = acc_top5.get()
    _, crossentropy = train_cross_entropy.get()

    top1_list = mpi_comm.gather(top1, root=0)
    top5_list = mpi_comm.gather(top5, root=0)
    crossentropy_list = mpi_comm.gather(crossentropy, root=0)

    if mpi_rank == 0:
        top1_list = np.array(top1_list)
        top5_list = np.array(top5_list)
        crossentropy_list = np.array(crossentropy_list)
        logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f, loss=%f, lr=%f, alpha=%f'%(args.start_epoch, top1_list.mean(), top5_list.mean(), crossentropy_list.mean(), trainer.learning_rate, alpha))

nd.waitall()

time_0 = time.time()

for epoch in range(args.start_epoch+1, args.epochs):
        # train_metric.reset()

        # if epoch in lr_decay_epoch:
        #     lr = lr * args.lr_decay

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
            byz_train_dataset = mx.gluon.data.dataset.ArrayDataset(byz_train_X, byz_train_Y)
            byz_train_data = gluon.data.DataLoader(byz_train_dataset, batch_size=args.batchsize, shuffle=True, last_batch='rollover', num_workers=1)
            net.initialize(mx.init.MSRAPrelu(), ctx=context, force_reinit=True)
            for local_epoch in range(args.iterations):
                for i, (data, label) in enumerate(byz_train_data):
                    with ag.record():
                        outputs = net(data)
                        loss = loss_func(outputs, label)
                    loss.backward()
                    trainer.step(args.batchsize)
        else:
            # train
            # local epoch
            for local_epoch in range(args.iterations):
                if args.iid == 1:
                    train_data = random.choice(train_data_list)
                for i, (data, label) in enumerate(train_data):
                    with ag.record():
                        outputs = net(data)
                        loss = loss_func(outputs, label)
                    loss.backward()
                    trainer.step(args.batchsize)
        
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

        if  ( epoch % args.interval == 0 or epoch == args.epochs-1 ) :
            acc_top1.reset()
            acc_top5.reset()
            train_cross_entropy.reset()
            for i, (data, label) in enumerate(val_val_data):
                outputs = net(data)
                acc_top1.update(label, outputs)
                acc_top5.update(label, outputs)

            for i, (data, label) in enumerate(val_train_data):
                outputs = net(data)
                train_cross_entropy.update(label, nd.softmax(outputs))

            _, top1 = acc_top1.get()
            _, top5 = acc_top5.get()
            _, crossentropy = train_cross_entropy.get()

            top1_list = mpi_comm.gather(top1, root=0)
            top5_list = mpi_comm.gather(top5, root=0)
            crossentropy_list = mpi_comm.gather(crossentropy, root=0)

            if mpi_rank == 0:
                top1_list = np.array(top1_list)
                top5_list = np.array(top5_list)
                crossentropy_list = np.array(crossentropy_list)

                logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f, loss=%f, lr=%f, alpha=%f, time=%f, elapsed=%f'%(epoch, top1_list.mean(), top5_list.mean(), crossentropy_list.mean(), trainer.learning_rate, alpha, toc-tic, time.time()-time_0))
                # logger.info('[Epoch %d] validation: acc-top1=%f acc-top5=%f'%(epoch, top1, top5))

                if args.save == 1:
                    [dirname, postfix] = os.path.splitext(args.log)
                    filename = dirname + ("_%04d.params" % (epoch))
                    net.save_parameters(filename)
        
        nd.waitall()






