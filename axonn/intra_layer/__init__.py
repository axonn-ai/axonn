# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
from .fully_connected import Linear  # noqa: F401
from .conv import Conv2d  # noqa: F401
from .embedding import Embedding  # noqa: F401

from .communication import Drop, Gather  # noqa: F401
from .gradient_normalization import clip_grad_norm_  # noqa: F401

from axonn import axonn as ax
import torch
import torch.distributed as dist
from .automatic_parallelism import auto_parallelize  # noqa: F401
from .overlap_communication import optimize_communication  # noqa: F401
from .overlap_communication import (  # noqa: F401
    overlap_all_gathers_for_checkpointed_forward,  # noqa: F401
)  # noqa: F401


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
def sync_gradients_expert_mode_depth_parallel(
    model, gradient_attr_name="grad", mean=False, vectorize=False
):
    if NO_GRADIENT_SYNC:
        return
    grads_to_sync = []
    world_size = dist.get_world_size(ax.comm_handle.depth_intra_layer_parallel_group)
    for param in model.parameters():
        if param.requires_grad:
            grad = getattr(param, gradient_attr_name)
            if grad is not None:
                if mean:
                    grad.div_(world_size)
                if hasattr(param, "is_tensor_parallel") and param.is_tensor_parallel:
                    if (
                        hasattr(param, "needs_gradient_sync")
                        and param.needs_gradient_sync
                    ):
                        grads_to_sync.append(grad)
                else:
                    grads_to_sync.append(grad)

    if not grads_to_sync:
        return

    if vectorize:
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        global_grad = _flatten_dense_tensors(grads_to_sync)
        dist.all_reduce(
            global_grad, group=ax.comm_handle.depth_intra_layer_parallel_group
        )

        for old_tensor, new_tensor in zip(
            grads_to_sync, _unflatten_dense_tensors(global_grad, grads_to_sync)
        ):
            old_tensor.data = new_tensor
    else:
        for grad in grads_to_sync:
            dist.all_reduce(grad, group=ax.comm_handle.depth_intra_layer_parallel_group)


@torch.no_grad()
def sync_gradients_expert_mode_data_parallel(
    model, gradient_attr_name="grad", mean=False, vectorize=False
):
    if NO_GRADIENT_SYNC:
        return
    grads_to_sync = []
    world_size = dist.get_world_size(ax.comm_handle.data_parallel_group)
    for param in model.parameters():
        if param.requires_grad:
            grad = getattr(param, gradient_attr_name)
            if grad is not None:
                if mean:
                    grad.div_(world_size)
                grads_to_sync.append(grad)

    if not grads_to_sync:
        return

    if vectorize:
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        global_grad = _flatten_dense_tensors(grads_to_sync)
        dist.all_reduce(global_grad, group=ax.comm_handle.data_parallel_group)
        for old_tensor, new_tensor in zip(
            grads_to_sync, _unflatten_dense_tensors(global_grad, grads_to_sync)
        ):
            old_tensor.data = new_tensor
    else:
        for grad in grads_to_sync:
            dist.all_reduce(grad, group=ax.comm_handle.data_parallel_group)


@torch.no_grad()
def sync_gradients(
    model, gradient_attr_name="grad", mean=False, vectorize=False, expert_mode=False
):
    if NO_GRADIENT_SYNC:
        return
    if expert_mode:
        sync_gradients_expert_mode_depth_parallel(
            model, gradient_attr_name, mean, vectorize
        )
        sync_gradients_expert_mode_data_parallel(
            model, gradient_attr_name, mean, vectorize
        )
        return
    ax.get_timers().start("sync-gradients-non-expert")
    grads_to_sync = {
        "tensor_parallel_weights": [],
        "tensor_parallel_biases": [],
        "others": [],
    }
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

    if vectorize:
        raise NotImplementedError
    else:
        for grad in grads_to_sync["tensor_parallel_weights"]:
            # weights are already reduced over the depth parallel groups
            # so we only need the reduction over the data parallel group
            dist.all_reduce(grad, group=data_parallel_group)
            if mean:
                grad.div_(torch.distributed.get_world_size())

        for grad in grads_to_sync["tensor_parallel_biases"]:
            # biases need to be reduced over both the data parallel
            # and depth parallel groups
            dist.all_reduce(grad, group=data_parallel_group)
            dist.all_reduce(grad, group=depth_parallel_group)
            if mean:
                grad.div_(torch.distributed.get_world_size())

        ax.get_timers().start("AR-others-world")
        for grad in grads_to_sync["others"]:
            # all other weights are purely data parallel
            dist.all_reduce(grad)
            if mean:
                grad.div_(torch.distributed.get_world_size())
        ax.get_timers().stop("AR-others-world")
    ax.get_timers().stop("sync-gradients-non-expert")
