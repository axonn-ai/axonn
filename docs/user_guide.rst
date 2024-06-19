**********
User Guide
**********

Initializing AxoNN
==================

Tensor with Easy API
====================

Overview
--------

The EasyAPI offers a streamlined approach to leverage the computational power of multiple GPUs for accelerating matrix multiplications in neural network layers. Specifically, the `nn.Linear` layer is parallelized, distributing its computation across GPUs to expedite training and inference processes.

Parallel Implementation
------------------------

Sequential Approach (`nn.Linear`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the sequential approach, the nn.Linear module is traditionally used for fully connected layers within neural networks. This module performs matrix multiplications sequentially, often limiting performance when dealing with large datasets or complex models.

Parallel Approach (`axonn.intra_layer.Linear`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The EasyAPI for Tensor simplifies the parallelization of matrix multiplications, particularly in fully connected layers (`nn.Linear`), across multiple GPUs using Argarwal's 3-D matrix multiplication algorithm. Please see https://arxiv.org/abs/2305.13525 for more details.

Usage
-----

Creating a Fully Connected Network (FC_Net) using `nn.Linear`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    :linenos: 
    :emphasize-lines: 18, 20

    import torch.nn as nn
    import torch

    class FC_Net(nn.Module):
        def __init__(self, num_layers, input_size, hidden_size, output_size):
            super(FC_Net, self).__init__()
            self.embed = nn.Linear(input_size, hidden_size)
            self.layers = nn.ModuleList([FC_Net_Layer(hidden_size) for _ in range(num_layers)])
            self.clf = nn.Linear(hidden_size, output_size)

        def forward(self, x, checkpoint_activations=False):
            ...

    class FC_Net_Layer(nn.Module):
        def __init__(self, hidden_size):
            super(FC_Net_Layer, self).__init__()
            self.norm = nn.LayerNorm(hidden_size)
            self.linear_1 = nn.Linear(in_features=hidden_size, out_features=4 * hidden_size)
            self.relu = nn.ReLU()
            self.linear_2 = nn.Linear(in_features = 4 * hidden_size, out_features = hidden_size)

        def forward(self, x):
            ...

Creating FC_Net with Distributed Tensor Layers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following code snippet demonstrates how to utilize the parallelization of the `nn.Linear` layer using the EasyAPI for Tensor:

.. code-block:: python
    :linenos: 
    :emphasize-lines: 19, 21

    import torch.nn as nn
    import torch
    from axonn.intra_layer import Linear

    class FC_Net(nn.Module):
        def __init__(self, num_layers, input_size, hidden_size, output_size):
            super(FC_Net, self).__init__()
            self.embed = nn.Linear(input_size, hidden_size)
            self.layers = nn.ModuleList([FC_Net_Layer(hidden_size) for _ in range(num_layers)])
            self.clf = nn.Linear(hidden_size, output_size)

        def forward(self, x, checkpoint_activations=False):
            ...

    class FC_Net_Layer(nn.Module):
        def __init__(self, hidden_size):
            super(FC_Net_Layer, self).__init__()
            self.norm = nn.LayerNorm(hidden_size)
            self.linear_1 = Linear(in_features=hidden_size, out_features=4 * hidden_size)
            self.relu = nn.ReLU()
            self.linear_2 = Linear(in_features = 4 * hidden_size, out_features = hidden_size)

        def forward(self, x):
            ...

Training with Parallel FC Net
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The provided code block demonstrates the training process using the parallel FC Net within the AxoNN framework. This example showcases the usage of AxoNN for distributed training. It is important to note that this snippet includes only the relevant portions related to AxoNN. The entire training code, along with additional details, can be found `here <https://github.com/axonn-ai/distrib-dl-tutorial/blob/develop/session_5_easy_intra_layer_parallelism/train.py>`_. Additionally the serial code is available `here <https://github.com/axonn-ai/distrib-dl-tutorial/blob/develop/session_1_basics/train.py>`_.

.. code-block:: python

    import torch
    import torchvision
    import sys
    import os
    from torchvision import transforms
    import numpy as np
    from axonn import axonn as ax

    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

    from model.fc_net_easy_tensor_parallel import FC_Net
    from utils import print_memory_stats, num_params, log_dist
    from args import create_parser

    NUM_EPOCHS=2
    PRINT_EVERY=200

    ## Set the torch seed 
    torch.manual_seed(0)

    if __name__ == "__main__":
        parser = create_parser()
        args = parser.parse_args()

        ## Step 1 - Initialize AxoNN
        ax.init(
                    G_data=args.G_data,
                    G_inter=1,
                    G_intra_r=args.G_intra_r,
                    G_intra_c=args.G_intra_c,
                    mixed_precision=True,
                    fp16_allreduce=True,
                )

        ...

        ## Step 2 - Create dataset with augmentations
        ...

        ## Step 3 - Create dataloader using AxoNN
        train_loader = ax.create_dataloader(
            train_dataset,
            args.batch_size,
            args.micro_batch_size,
            num_workers=1,
        )

        ## Step 4 - Create Neural Network 
        ...

        ## Step 5 - Create Optimizer 
        ...

        ## Step 6 - register model and optimizer with AxoNN
        ## This creates the required data structures for
        ## mixed precision
        net, optimizer = ax.register_model_and_optimizer(net, optimizer)

        ## Step 7 - Create Loss Function and register it
        ...
        ax.register_loss_fn(loss_fn)

        ## Step 8 - Train
        ...

Monkey Patching
~~~~~~~~~~~~~~~



Tensor using Advanced API
=====================================

Combining Tensor in AxoNN with PyTorch DDP
==========================================

Integration with other Parallel APIs
====================================

Huggingface
-----------

AxoNN seamlessly integrates with the Hugging Face Accelerate API, providing a uniform interface for leveraging parallel computing frameworks. This integration enables efficient training of deep learning models across multiple GPUs.

AxoNN Plugin
~~~~~~~~~~~~

We define a plugin that makes accelerate compatible with AxoNN as a backend. Our implementation can be found in this accelerate `fork <https://github.com/axonn-ai/accelerate>`_. To use accerate + AxoNN, one can simply: 

.. code-block:: python 

    pip install git@github.com:axonn-ai/accelerate.git

A concrete fine-tuning `example <https://github.com/axonn-ai/axonn-examples/blob/develop/llm_finetuning/run_clm_no_trainer.py>`_ demonstrates how the AxoNN plugin can be used. More information can be found in the Fine-Tuning section under Examples. 




Pipelining in AxoNN 
===================

Combining Pipelining in AxoNN with Data Parallelism 
===================================================

