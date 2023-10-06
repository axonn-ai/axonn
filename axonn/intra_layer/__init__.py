from .fully_connected import Linear as Tensor_Parallel_Linear  # noqa: F401
from .conv import Conv2d as Tensor_Parallel_Conv2d # noqa: F401
from .communication import Drop, Gather
from axonn import axonn as ax


def drop(x, transpose=False, dim=-1):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group

    return Drop.apply(x, group, dim)


def gather(x, transpose=False, dim=-1):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group
    return Gather.apply(x, group, dim)
