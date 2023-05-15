# ISC 23 - Tutorial on Distributed Training of Neural Networks

Code material for the tutorial can be found in this repository. 

**Contents** 
* [Setup](#setup)
* [Basics of Model Training](#basics-of-model-training)
* [Data Parallelism](#data-parallelism)
* [Tensor Parallelism](#tensor-parallelism)
* [Pipeline Parallelism](#pipeline-parallelism)

## Setup 

We have built the dependencies required for this tutorial in a shared python virtual enviroment, which can be activated as follows:

```bash
. /scratch/zt1/project/isc2023/shared/tutorial-venv/bin/activate

```

We have also put the training dataset i.e. [MNIST](http://yann.lecun.com/exdb/mnist/)  used in this tutorial in `. /scratch/zt1/project/bhatele-lab/shared/MNIST`



## Basics of Model Training

### Mixed Precision

### Activation Checkpointing


## Data Parallelism

```bash
cd session_2_data_parallelism
```

### Pytorch Distributed Data Parallel (DDP)

```bash
sbatch run_ddp.sh
```

### Zero Redundancy Optimizer (ZeRO)


```bash
sbatch run_zero.sh
```

## Tensor Parallelism

## Pipeline Parallelism


