from contextlib import contextmanager
from .fully_connected import Linear  # noqa: F401
from .conv import Conv2d  # noqa: F401

from .communication import Drop, Gather
from .gradient_normalization import clip_grad_norm_  # noqa: F401

from axonn import axonn as ax
import torch
import torch.distributed as dist
from .automatic_parallelism import auto_parallelize  # noqa: F401


def drop(
    x, transpose=False, dim=-1, batch_dim=0, skip_channels=False, skip_batch=False
):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group

    if not skip_channels:
        x = Drop.apply(x, group, dim)
    if not skip_batch:
        x = Drop.apply(x, ax.comm_handle.depth_intra_layer_parallel_group, batch_dim)
    return x


def gather(
    x, transpose=False, dim=-1, batch_dim=0, skip_channels=False, skip_batch=False
):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group

    if not skip_channels:
        x = Gather.apply(x, group, dim)
    if not skip_batch:
        x = Gather.apply(x, ax.comm_handle.depth_intra_layer_parallel_group, batch_dim)
    return x


OVERLAP_REDUCE_SCATTER = False
OVERLAP_ALL_REDUCE = False
ALL_GATHER_ITERATOR = None
NO_GRADIENT_SYNC = False
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
        if isinstance(module, Linear) or isinstance(module, Conv2d):
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
    **kwargs
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
        OVERLAP_ALL_REDUCE = False
        OVERLAP_REDUCE_SCATTER = False
        ALL_GATHER_ITERATOR = None


@contextmanager
def no_grad_sync():
    global NO_GRADIENT_SYNC
    old_val = NO_GRADIENT_SYNC
    try:
        NO_GRADIENT_SYNC = True
    finally:
        NO_GRADIENT_SYNC = old_val


@torch.no_grad()
def sync_gradients_depth_parallel(
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
def sync_gradients_data_parallel(
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
        dist.all_reduce(
            global_grad, group=ax.comm_handle.data_parallel_group
        )
        if mean:
            global_grad.div_(world_size)

        for old_tensor, new_tensor in zip(
            grads_to_sync, _unflatten_dense_tensors(global_grad, grads_to_sync)
        ):
            old_tensor.data = new_tensor
    else:
        for grad in grads_to_sync:
            dist.all_reduce(grad, group=ax.comm_handle.data_parallel_group)
