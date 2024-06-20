****************
AxoNN User Guide
****************


AxoNN is a highly scalable framework for training, finetuning, and running inference of deep learning models on multiple 
GPUs. It is designed to maximize performance and ease of use, making it simple to scale your models across several GPUs.

There are three steps for using AxoNN:
1. Initializing AxoNN.
2. Mapping your dataloaders to GPUs.
3. Parallelizing your model.

Initializing AxoNN
==================

Before initializing AxoNN, you need to initialize ``torch.distributed`` via ``torch.distributed.init_process_group()``. This 
step creates the global communicator with all processes. For more information, refer to the 
`torch.distributed documentation <https://pytorch.org/docs/stable/distributed.html>`_.

.. code-block:: python
    
    import torch.distributed as dist 
    import axonn as ax 

    dist.init_process_group(backend='nccl')
    
Next, call ``ax.init()`` to initialize the subcommunicators/sub-process groups for AxoNN's 3D tensor, pipeline, and data parallelism. The arguments ``G_intra_r``, ``G_intra_c``, ``G_intra_d`` correspond to our tensor parallel algorithm. ``G_inter`` is for pipeline parallelism, and ``G_data`` is for data parallelism. The product of these five dimensions should equal the number of GPUs.

.. code-block:: python

    ax.init(G_data: int,       # Size of each data parallel group
            G_intra_r: int ,  # Size of each row tensor parallel group
            G_intra_c: int ,  # Size of each column tensor parallel group
            G_intra_d: int ,  # Size of each depth tensor parallel group
            G_inter: int)        # Size of each pipeline parallel group


Mapping your dataloaders to GPUs.
==================================

In any parallel deep learning algorithm, it is not just the neural network but also the data that is sharded across the GPUs. 
In AxoNN, an input batch is divided equally across the data-parallel (``G_data``) and depth tensor parallel (``G_intra_d``) 
dimensions. In other words, all GPUs that have the same data-parallel and depth tensor-parallel ranks get to see the same 
shard of the input batch.

To make this data division easy for the end-user, AxoNN provides a ``create_dataloader`` function, which takes as input a 
PyTorch dataset object (``torch.utils.data.Dataset``) and returns a parallelized dataloader.

.. code-block:: python

    ax.create_dataloader(
        dataset: torch.utils.data.Dataset,  # The dataset to be loaded
        global_batch_size: int,             # The total batch size across all GPUs
        num_workers: int = 0,               # Number of subprocesses to use for data loading
        *args,                              # Additional arguments
        **kwargs                            # Additional keyword arguments
    ) -> torch.utils.data.DataLoader


.. note::

   The ``global batch size`` argument here denotes the total batch size across ALL GPUs, and not the per GPU batch size. 


Parallelizing Your Model with AxoNN's Tensor Parallelism
========================================================

Modern neural networks, especially large language models (LLMs), often exceed the memory capacity of a single GPU. To address this, AxoNN 
offers two key parallelization techniques: tensor parallelism and pipeline parallelism. This user guide will cover tensor parallelism, which 
is designed to balance ease-of-use with high performance, and is our recommended approach for parallel training and inference. If you want to 
learn about pipeline parallelism, please refer to the section on :ref:`Advanced Usage`.

Tensor Parallelism
------------------

In neural networks, most parameters and computations reside in layers such as Fully Connected (``torch.nn.Linear``) or Convolutional 
(``torch.nn.Conv2d``) layers. Tensor parallelism involves parallelizing these operations across multiple GPUs. AxoNN implements a 3D tensor 
parallel algorithm inspired by Agarwal's 3D matrix multiplication approach. For detailed insights, refer to our paper on  
`A 4D Hybrid Algorithm to Scale Parallel Training to Thousands of GPUs <https://arxiv.org/abs/2305.13525>`_.

AxoNN's 3D tensor parallel algorithm is designed to balance ease-of-use with high performance, making it our recommended approach for efficient 
parallel training and inference.

Automatic Tensor Parallelism with ``auto_parallelize``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

AxoNN simplifies tensor parallelism with its ``auto_parallelize`` API, which automates the parallelization of operations within the specified context. Specifically, this API intercepts all declarations of ``torch.nn.Linear`` layers and replaces them with tensor parallelized equivalents from ``axonn.intra_layer.Linear``. This allows you to seamlessly integrate tensor parallelism into your workflow, without any changes to your model definitions! 

.. code-block:: python

    from axonn.intra_layer import auto_parallelize
    
    with auto_parallelize():
        net = # declare your sequential model here. AxoNN will automatically parallelize all FC layers


.. note::

   Autoparallelize currently supports parallelizing ``torch.nn.Linear`` layers only. If you intend to use tensor parallel convolution layers, please refer to the next section on manual parallelization.


Manual Tensor Parallelization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For scenarios where the user wants more control over which parts of their neural network they want parallelized, AxoNN provides an alternative approach to parallelize tensor operations in your neural network definitions. This approach requires explicit modifications to your model definitions to incorporate tensor parallelism. Let us understand this with a simple example using an MLP (Multi-layer Perceptron) block.

In the following example, we'll demonstrate the transformation from a sequential MLP implementation to a tensor parallelized version. In the original sequential implementation, most of the compute and parameters reside in the linear layers (highlighted lines), which we will subsequently replace with AxoNN's tensor parallel linear layers.

.. code-block:: python
    :emphasize-lines: 7, 9

    import torch.nn as nn

    class SequentialMLP(nn.Module):
        def __init__(self, hidden_size):
            super(SequentialMLP, self).__init__()
            self.norm = nn.LayerNorm(hidden_size)
            self.linear_1 = nn.Linear(in_features=hidden_size, out_features=4 * hidden_size)
            self.relu = nn.GELU()
            self.linear_2 = nn.Linear(in_features=4 * hidden_size, out_features=hidden_size)

Now, let's transform the sequential MLP into a tensor parallelized version using AxoNN. All you need to do is replace instances of ``nn.Linear`` with ``axonn.intra_layer.Linear``:

.. code-block:: python
    :emphasize-lines: 2, 8, 10

    import torch.nn as nn
    import axonn

    class ParallelMLP(nn.Module):
        def __init__(self, hidden_size):
            super(ParallelMLP, self).__init__()
            self.norm = nn.LayerNorm(hidden_size)
            self.linear_1 = axonn.intra_layer.Linear(in_features=hidden_size, out_features=4 * hidden_size)
            self.relu = nn.GELU()
            self.linear_2 = axonn.intra_layer.Linear(in_features=4 * hidden_size, out_features=hidden_size)



That's it! You do not need to make any changes to other layers and the forward pass of your module! 


Putting it all together
=======================

The coolest part of using our tensor parallelism is that apart from the aforementioned steps, everything else is identical to single GPU 
training with PyTorch. The training or inference loop can be written as if the user is training on a single GPU. Further, our tensor 
parallelism is inter-operable with most of PyTorch's features like automatic mixed precision (``torch.autocast``) and activation checkpointing.

A complete training example with our tensor parallelism can be found in our tutorial on distributed deep learning
`here <https://github.com/axonn-ai/distrib-dl-tutorial/tree/develop/session_3_intra_layer_parallelism>`_. 



Integration with other Parallel APIs
====================================

Huggingface Transformers
------------------------

PyTorch Lightning
-----------------


Huggingface Accelerate
----------------------








.. Pipeline Parallelism
.. --------------------


.. .. Training with Parallel FC Net
.. .. ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. .. The provided code block demonstrates the training process using the parallel FC Net within the AxoNN framework. This example showcases the usage of AxoNN for distributed training. It is important to note that this snippet includes only the relevant portions related to AxoNN. The entire training code, along with additional details, can be found `here <https://github.com/axonn-ai/distrib-dl-tutorial/blob/develop/session_5_easy_intra_layer_parallelism/train.py>`_. Additionally the serial code is available `here <https://github.com/axonn-ai/distrib-dl-tutorial/blob/develop/session_1_basics/train.py>`_.

.. .. .. code-block:: python

.. ..     import torch
.. ..     import torchvision
.. ..     import sys
.. ..     import os
.. ..     from torchvision import transforms
.. ..     import numpy as np
.. ..     from axonn import axonn as ax

.. ..     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

.. ..     from model.fc_net_easy_tensor_parallel import FC_Net
.. ..     from utils import print_memory_stats, num_params, log_dist
.. ..     from args import create_parser

.. ..     NUM_EPOCHS=2
.. ..     PRINT_EVERY=200

.. ..     ## Set the torch seed 
.. ..     torch.manual_seed(0)

.. ..     if __name__ == "__main__":
.. ..         parser = create_parser()
.. ..         args = parser.parse_args()

.. ..         ## Step 1 - Initialize AxoNN
.. ..         ax.init(
.. ..                     G_data=args.G_data,
.. ..                     G_inter=1,
.. ..                     G_intra_r=args.G_intra_r,
.. ..                     G_intra_c=args.G_intra_c,
.. ..                     mixed_precision=True,
.. ..                     fp16_allreduce=True,
.. ..                 )

.. ..         ...

.. ..         ## Step 2 - Create dataset with augmentations
.. ..         ...

.. ..         ## Step 3 - Create dataloader using AxoNN
.. ..         train_loader = ax.create_dataloader(
.. ..             train_dataset,
.. ..             args.batch_size,
.. ..             args.micro_batch_size,
.. ..             num_workers=1,
.. ..         )

.. ..         ## Step 4 - Create Neural Network 
.. ..         ...

.. ..         ## Step 5 - Create Optimizer 
.. ..         ...

.. ..         ## Step 6 - register model and optimizer with AxoNN
.. ..         ## This creates the required data structures for
.. ..         ## mixed precision
.. ..         net, optimizer = ax.register_model_and_optimizer(net, optimizer)

.. ..         ## Step 7 - Create Loss Function and register it
.. ..         ...
.. ..         ax.register_loss_fn(loss_fn)

.. ..         ## Step 8 - Train
.. ..         ...

.. .. Monkey Patching
.. .. ~~~~~~~~~~~~~~~



.. .. Tensor using Advanced API
.. .. =====================================

.. .. Combining Tensor in AxoNN with PyTorch DDP
.. .. ==========================================

.. .. Integration with other Parallel APIs
.. .. ====================================

.. .. Huggingface
.. .. -----------

.. .. AxoNN seamlessly integrates with the Hugging Face Accelerate API, providing a uniform interface for leveraging parallel computing frameworks. This integration enables efficient training of deep learning models across multiple GPUs.

.. .. AxoNN Plugin
.. .. ~~~~~~~~~~~~

.. .. We define a plugin that makes accelerate compatible with AxoNN as a backend. Our implementation can be found in this accelerate `fork <https://github.com/axonn-ai/accelerate>`_. To use accerate + AxoNN, one can simply: 

.. .. .. code-block:: python 

.. ..     pip install git@github.com:axonn-ai/accelerate.git

.. .. A concrete fine-tuning `example <https://github.com/axonn-ai/axonn-examples/blob/develop/llm_finetuning/run_clm_no_trainer.py>`_ demonstrates how the AxoNN plugin can be used. More information can be found in the Fine-Tuning section under Examples. 




.. .. Pipelining in AxoNN 
.. .. ===================

.. .. Combining Pipelining in AxoNN with Data Parallelism 
.. .. ===================================================

