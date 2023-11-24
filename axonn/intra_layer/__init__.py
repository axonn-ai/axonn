from contextlib import contextmanager
from .fully_connected import Linear  # noqa: F401
from .conv import Conv2d as Tensor_Parallel_Conv2d  # noqa: F401

from .communication import Drop, Gather
from .gradient_normalization import clip_grad_norm_  # noqa: F401

from axonn import axonn as ax
import torch


def drop(x, transpose=False, dim=-1, batch_dim=0):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group

    x = Drop.apply(x, group, dim)
    x = Drop.apply(x, ax.comm_handle.depth_intra_layer_parallel_group, batch_dim)
    return x


def gather(x, transpose=False, dim=-1, batch_dim=0):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group

    x = Gather.apply(x, group, dim)
    x = Gather.apply(x, ax.comm_handle.depth_intra_layer_parallel_group, batch_dim)
    return x


OVERLAP_COMM = False
handles = []
pending_grad_accumulations = []


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
            param.grad = grad
        else:
            param.grad.add_(grad)

    pending_grad_accumulations = []


@contextmanager
def optimize_communication(*args, **kwargs):
    global OVERLAP_COMM
    OVERLAP_COMM = True
    try:
        yield None
    finally:
        clear_handles()
        accumulate()
        OVERLAP_COMM = False
