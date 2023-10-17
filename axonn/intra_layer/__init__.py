from .fully_connected import Linear  # noqa: F401
from .communication import Drop, Gather
from axonn import axonn as ax


def drop(x, transpose=False):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group

    return Drop.apply(x, group)


def gather(x, transpose=False):
    if not transpose:
        group = ax.comm_handle.inner_intra_layer_parallel_group
    else:
        group = ax.comm_handle.outer_intra_layer_parallel_group
    return Gather.apply(x, group)
