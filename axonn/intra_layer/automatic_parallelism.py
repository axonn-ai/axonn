import torch.nn as nn
from axonn import axonn as ax
from axonn.intra_layer import Linear
from contextlib import contextmanager

reference_to_original_linear_class = nn.Linear


def is_parallelizable(in_features, out_features):
    G_row = ax.config.G_intra_r
    G_col = ax.config.G_intra_c
    G_depth = ax.config.G_intra_d
    row_col_condition = out_features % G_row == 0 and in_features % G_col == 0
    depth_condition = (out_features * in_features // (G_row * G_col)) % G_depth == 0
    return row_col_condition and depth_condition


class patched_linear:
    def __new__(cls, in_features, out_features, bias=True, device=None, dtype=None):
        if is_parallelizable(in_features, out_features):
            parallel_layer = Linear(in_features, out_features, bias=bias)
            if device is not None:
                parallel_layer = parallel_layer.to(device)
            if dtype is not None:
                parallel_layer = parallel_layer.to(dtype)
            return parallel_layer
        else:
            sequential_layer = reference_to_original_linear_class(
                in_features, out_features, bias=bias
            )
            if device is not None:
                sequential_layer = sequential_layer.to(device)
            if dtype is not None:
                sequential_layer = sequential_layer.to(dtype)
            return sequential_layer


@contextmanager
def auto_parallelize():
    nn.Linear = patched_linear
    try:
        yield None
    finally:
        nn.Linear = reference_to_original_linear_class
