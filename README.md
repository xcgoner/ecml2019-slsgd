# Robust Federated Learning

### This is the python implementation of the paper "SLSGD: Secure and Efficient Distributed On-device Machine Learning"

### Requirements

The following python packages needs to be installed by pip:

1. MXNET (we use Intel CPU cluster, thus mxnet-mkl is preferred)
2. Gluon-CV
3. Numpy
4. MPI4py
5. Keras (with Tensorflow backend, we use this only for dataset preparation, not for model training)
6. PIL (also for dataset preparation)
7. Gluon-NLP (only for the experiment of LSTM on Wikitext-2)

The users can simply run the following commond in their own virtualenv:

```bash
pip install --no-cache-dir numpy mxnet-mkl gluoncv mpi4py keras pillow gluonnlp
```

### Prepare the dataset

#### Options:

| Option     | Desctiption | 
| ---------- | ----------- | 
|--output DATASET_DIR| the directory where the dataset will be placed|
|--nsplit 100| partition to 100 devices|
|--normalize 1| Normalize the data|
|--step 8| Increment of the partition size|

* CIFAR-10 with balanced partition:
```bash
python convert_cifar10_to_np_normalized.py --nsplit 100 --normalize 1 --output DATASET_DIR
```

* CIFAR-10 with unbalanced partition:
```bash
python convert_cifar10_to_np_normalized_unbalanced.py --nsplit 100 --normalize 1 --step 8 --output DATASET_DIR
```

#### Note that balanced partition is always needed for validation.

### Run the demo

#### Options:

| Option     | Desctiption | 
| ---------- | ----------- | 
|--dir DATASET_DIR| the directory where the training dataset is placed|
|--valdir VAL_DATASET_DIR| the directory where the validation dataset is placed|
|--batchsize 50| batch size of the workers|
|--epochs 800| total number of epochs|
|--interval 10| log interval|
|--nsplit 100| training data is partitioned to 100 devices|
|--lr 0.1| learning rate|
|--lr-decay | lr decay rate|
|--lr-decay-epoch | epochs where lr decays|
|--alpha 1| weight of moving average|
|--alpha-decay | alpha decay rate|
|--alpha-decay-epoch | epochs where alpha decays|
|--log | path to the log file|
|--classes 10| number of different classes/labels|
|--iterations 1| number of local iterations in each epoch|
|--aggregation mean| aggregation method, mean or trim|
|--nbyz 2| number of malicious workers|
|--trim 4| hyperparameter $b$ of the trimmed mean|
|--model default | name of the model, "default" means the CNN used in the paper experiments|
|--seed 337 | random seed|

* Train with no malicious users, mean as aggregation, $k=10$ workers are randomly selected in each epoch:
```bash
mpirun -np 10 -machinefile hostfile python ecml_federated/slsgd.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.1 --alpha 1 --alpha-decay 0.9 --alpha-decay-epoch 400 --epochs 800 --iterations 1 --seed 733 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee $watchfile
```

* Train with 2 nefarious users, trimmed mean as aggregation, $k=10$ workers are randomly selected in each epoch:
```bash
mpirun -np 10 -machinefile hostfile python icml_federated/slsgd.py --classes 10 --model default --nsplit 100 --batchsize 50 --lr 0.1 --alpha 1 --alpha-decay 0.9 --alpha-decay-epoch 400 --epochs 800 --iterations 1 --seed 733 --nbyz 2 --trim 4 --dir $inputdir --valdir $valdir -o $logfile 2>&1 | tee $watchfile
```

There is a demo script *experiment_script_1.sh*

### Notes:

When executed on the Intel vLab Academic Cluster, the following instructions might be needed:

1. manually install libfabric 1.1.0
2. set I_MPI_FABRICS=ofi