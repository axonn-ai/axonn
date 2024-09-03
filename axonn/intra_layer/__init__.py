# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
from .fully_connected import Linear  # noqa: F401
from .conv import Conv2d  # noqa: F401
from .embedding import Embedding  # noqa: F401

from .communication import Drop, Gather
from .gradient_normalization import clip_grad_norm_  # noqa: F401

from axonn import axonn as ax
import torch
import torch.distributed as dist
from .automatic_parallelism import auto_parallelize  # noqa: F401
from .overlap_communication import optimize_communication, overlap_all_gathers_for_checkpointed_forward # noqa: F401 


NO_GRADIENT_SYNC = False

@contextmanager
def no_grad_sync():
    global NO_GRADIENT_SYNC
    old_val = NO_GRADIENT_SYNC
    try:
        NO_GRADIENT_SYNC = True
        yield None
    finally:
        NO_GRADIENT_SYNC = old_val


@torch.no_grad()
def sync_gradients(
    model, gradient_attr_name="grad", mean=False, vectorize=False
):
    if NO_GRADIENT_SYNC:
        return
    grads_to_sync = {
            "tensor_parallel_weights": [], 
            "tensor_parallel_biases": [],         
            "others": []}
    for param in model.parameters():
        if param.requires_grad:
            grad = getattr(param, gradient_attr_name)
            if grad is not None:
                if hasattr(param, "is_tensor_parallel") and param.is_tensor_parallel:
                    if hasattr(param, "needs_depth_parallel_gradient_sync"):
                        if param.needs_depth_parallel_gradient_sync:
                            grads_to_sync["tensor_parallel_biases"].append(grad)
                        else:
                            grads_to_sync["tensor_parallel_weights"].append(grad)
                    else:
                        raise ValueError
                else:
                    grads_to_sync["others"].append(grad)

    data_parallel_group = ax.comm_handle.data_parallel_group
    depth_parallel_group = ax.comm_handle.depth_intra_layer_parallel_group
    G_data = ax.config.G_data
    G_intra_d = ax.config.G_intra_d 

    if vectorize:
        raise NotImplementedError
    else:
        for grad in grads_to_sync["tensor_parallel_weights"]:
            # weights are already reduced over the depth parallel groups
            # so we only need the reduction over the data parallel group
            dist.all_reduce(grad, group=data_parallel_group)
            if mean:
                grad.div_(G_data * G_intra_d)

        for grad in grads_to_sync["tensor_parallel_biases"]:
            # biases need to be reduced over both the data parallel
            # and depth parallel groups
            dist.all_reduce(grad, group=data_parallel_group)
            dist.all_reduce(grad, group=depth_parallel_group)
            if mean:
                grad.div_(G_data * G_intra_d)

        for grad in grads_to_sync["others"]:
            # all other weights are purely data parallel
            dist.all_reduce(grad)
            if mean:
                grad.div_(torch.distributed.get_world_size())


