# ISC 23 - Tutorial on Distributed Training of Deep Neural Networks

[![Join slack](https://img.shields.io/badge/slack-axonn--users-blue)](https://join.slack.com/t/axonn-users/shared_invite/zt-1vw4fm25c-XAH9n9d_3hg5TuHMw_7Ggw)

All the code for the hands-on exercies can be found in this repository. 

**Table of Contents**

* [Setup](#setup)
* [Basics of Model Training](#basics-of-model-training)
* [Data Parallelism](#data-parallelism)
* [Tensor Parallelism](#tensor-parallelism)
* [Pipeline Parallelism](#pipeline-parallelism)

## Setup 

To request an account on Zaratan, please fill [this form](https://docs.google.com/forms/d/e/1FAIpQLSeHoELzzWfOlo3YnCDxLyfY581hWuSidjWgzIvUq2gGFOinWw/viewform?usp=sf_link).

We have pre-built the dependencies required for this tutorial on Zaratan. This
will be activated automatically when you run the bash scripts.

The training dataset i.e. [MNIST](http://yann.lecun.com/exdb/mnist/) has also
been downloaded in `/scratch/zt1/project/isc2023/shared/MNIST`.

## Basics of Model Training

### Using PyTorch

```bash
cd session_1_basics/
sbatch --reservation=isc2023 run.sh
```

### Mixed Precision

```bash
MIXED_PRECISION=true sbatch --reservation=isc2023 run.sh
```

### Activation Checkpointing

```bash
CHECKPOINT_ACTIVATIONS=true sbatch --reservation=isc2023 run.sh
```

## Data Parallelism

### Pytorch Distributed Data Parallel (DDP)

```bash
cd session_2_data_parallelism
sbatch --reservation=isc2023 run_ddp.sh
```

### Zero Redundancy Optimizer (ZeRO)


```bash
sbatch --reservation=isc2023 run_deepspeed.sh
```

## Tensor Parallelism

## Pipeline Parallelism


