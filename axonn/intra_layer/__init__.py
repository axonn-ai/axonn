from .fully_connected import Linear  # noqa: F401
from .conv import Conv2d as Tensor_Parallel_Conv2d  # noqa: F401

from .communication import Drop, Gather
from .gradient_normalization import clip_grad_norm_  # noqa: F401

from axonn import axonn as ax


def drop(x, transpose=False, dim=-1):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group

    x = Drop.apply(x, group, dim)
    x = Drop.apply(x, ax.comm_handle.depth_intra_layer_parallel_group, 0)
    return x


def gather(x, transpose=False, dim=-1):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group

    x = Gather.apply(x, group, dim)
    x = Gather.apply(x, ax.comm_handle.depth_intra_layer_parallel_group, 0)
    return x
