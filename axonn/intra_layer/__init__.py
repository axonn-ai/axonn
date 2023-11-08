from contextlib import contextmanager
from .fully_connected import Linear  # noqa: F401
from .conv import Conv2d as Tensor_Parallel_Conv2d  # noqa: F401

from .communication import Drop, Gather
from .gradient_normalization import clip_grad_norm_  # noqa: F401

from axonn import axonn as ax


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


OVERLAP_COMM=False
handles = []
pending_grad_accumulations = []

def register_handle(handle):
    global handles
    handles.append(handle)

def clear_handles():
    global handles
    for handle in handles:
        handle.wait()
    handles = []

def accumulate_later(param, grad, main_grad):
    global pending_grad_accumulations

    pending_grad_accumulations.append([param, grad, main_grad])

def accumulate():
    global pending_grad_accumulations
    for param, grad, main_grad in pending_grad_accumulations:
        if main_grad is None:
            param.grad = grad
        else:
            param.grad = main_grad.data.add_(grad)

    pending_grad_accumulations = []

@contextmanager
def optimize_communication(enabled : bool = True):
    ## no sync -> True, cache weights, prevent extra all-gathers
    ## else -> clear weights
    global OVERLAP_COMM
    if not enabled:
        try:
            yield None
        finally:
            return 

    OVERLAP_COMM=True
    try:
        yield None
    finally:
        clear_handles()
        accumulate()
        OVERLAP_COMM=False


