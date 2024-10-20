# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
import torch
import axonn
import torch.distributed as dist

OVERLAP_REDUCE_SCATTER = False
OVERLAP_ALL_REDUCE = False
ALL_GATHER_ITERATOR = None

handles = []
pending_grad_accumulations = []
weights_cache = {}


def register_handle(handle):
    # ToDo: This might be unnecesary since
    # we are calling synchronize in clear_handles
    global handles
    handles.append(handle)


def clear_handles():
    global handles
    torch.cuda.synchronize()
    handles = []


def accumulate_later(param, grad):
    global pending_grad_accumulations
    pending_grad_accumulations.append([param, grad])


@torch.no_grad()
def accumulate():
    global pending_grad_accumulations
    for param, grad in pending_grad_accumulations:
        if param.grad is None:
            param.grad = grad.to(param.dtype)
        else:
            param.grad.add_(grad.to(param.dtype))

    pending_grad_accumulations = []


def clear_weights_cache():
    global weights_cache
    weights_cache = {}


def trigger_async_all_gathers(model):
    global weights_cache
    for module in model.modules():
        if isinstance(module, axonn.intra_layer.Linear) or isinstance(
            module, axonn.intra_layer.Conv2d
        ):
            weight = module.weight
            if weight not in weights_cache:
                # only trigger all gathers if not in cache
                process_group = module.depth_group
                world_size = dist.get_world_size(process_group)
                if world_size == 1:
                    all_gathered_weight = weight
                    handle = None
                else:
                    assert weight.ndim == 1
                    output_shape = weight.shape[0] * world_size
                    all_gathered_weight = torch.empty(
                        output_shape, dtype=weight.dtype, device=weight.device
                    )
                    handle = dist.all_gather_into_tensor(
                        all_gathered_weight, weight, group=process_group, async_op=True
                    )
                weights_cache[weight] = [all_gathered_weight, handle]
            yield


def enqueue_next_all_gather():
    global ALL_GATHER_ITERATOR
    assert ALL_GATHER_ITERATOR is not None
    try:
        next(ALL_GATHER_ITERATOR)
    except StopIteration:
        pass


def retrieve_all_gathered_weight(weight, delete):
    global ALL_GATHER_ITERATOR
    assert weight in weights_cache
    all_gathered_weight, handle = weights_cache[weight]
    if ALL_GATHER_ITERATOR is not None:
        enqueue_next_all_gather()
    if delete:
        del weights_cache[weight]
    return all_gathered_weight, handle


@contextmanager
def overlap_all_gathers_for_checkpointed_forward(
    model_object_for_overlapping_allgathers,
):
    global ALL_GATHER_ITERATOR
    if ALL_GATHER_ITERATOR is None:  # this is a false call
        try:
            yield None
        finally:
            pass
    else:
        old_iterator = ALL_GATHER_ITERATOR
        ALL_GATHER_ITERATOR = trigger_async_all_gathers(
            model_object_for_overlapping_allgathers
        )
        enqueue_next_all_gather()
        try:
            yield None
        finally:
            ALL_GATHER_ITERATOR = old_iterator


@contextmanager
def optimize_communication(
    overlap_all_reduce=True,
    overlap_reduce_scatter=False,
    overlap_all_gather=False,
    model_object_for_overlapping_allgathers=None,
    *args,
    **kwargs,
):
    global OVERLAP_ALL_REDUCE, OVERLAP_REDUCE_SCATTER
    global ALL_GATHER_ITERATOR
    OVERLAP_ALL_REDUCE = overlap_all_reduce
    OVERLAP_REDUCE_SCATTER = overlap_reduce_scatter

    if overlap_all_gather:
        if model_object_for_overlapping_allgathers is None:
            raise ValueError(
                "You need to pass your model as an argument - "
                "optimize_communication(...,model_object_"
                "for_overlapping_allgathers=model, ...)"
                "if overlap_all_gather is True"
            )
        ALL_GATHER_ITERATOR = trigger_async_all_gathers(
            model_object_for_overlapping_allgathers
        )
        enqueue_next_all_gather()

    try:
        yield None
    finally:
        clear_handles()
        accumulate()
        clear_weights_cache()
        OVERLAP_ALL_REDUCE = False
        OVERLAP_REDUCE_SCATTER = False
        ALL_GATHER_ITERATOR = None
