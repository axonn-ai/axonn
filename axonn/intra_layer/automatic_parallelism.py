# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.nn as nn
from axonn import axonn as ax
from axonn.intra_layer import Linear, Embedding
from contextlib import contextmanager

reference_to_original_linear_class = nn.Linear
reference_to_original_embedding_class = nn.Embedding


def is_parallelizable_linear(in_features, out_features):
    G_row = ax.config.G_intra_r
    G_col = ax.config.G_intra_c
    G_depth = ax.config.G_intra_d
    row_col_condition = out_features % G_row == 0 and in_features % G_col == 0
    depth_condition = (out_features * in_features // (G_row * G_col)) % G_depth == 0
    return row_col_condition and depth_condition


def is_parallelizable_embedding(num_embeddings, embedding_dim):
    G_row = ax.config.G_intra_r
    G_col = ax.config.G_intra_c
    G_depth = ax.config.G_intra_d
    row_col_condition = embedding_dim % G_row == 0 and G_col == 1
    depth_condition = (num_embeddings * embedding_dim // (G_row * G_col)) % G_depth == 0
    return row_col_condition and depth_condition


class patched_linear:
    def __new__(cls, in_features, out_features, bias=True, device=None, dtype=None):
        if is_parallelizable_linear(in_features, out_features):
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


class patched_embedding:
    def __new__(
        cls, num_embeddings, embedding_dim, *args, device=None, dtype=None, **kwargs
    ):
        if is_parallelizable_embedding(num_embeddings, embedding_dim):
            parallel_layer = Embedding(num_embeddings, embedding_dim, *args, **kwargs)
            if device is not None:
                parallel_layer = parallel_layer.to(device)
            if dtype is not None:
                parallel_layer = parallel_layer.to(dtype)
            return parallel_layer
        else:
            sequential_layer = reference_to_original_embedding_class(
                num_embeddings, embedding_dim, *args, **kwargs
            )
            if device is not None:
                sequential_layer = sequential_layer.to(device)
            if dtype is not None:
                sequential_layer = sequential_layer.to(dtype)
            return sequential_layer


@contextmanager
def auto_parallelize():
    nn.Linear = patched_linear
    #    nn.Embedding = patched_embedding
    try:
        yield None
    finally:
        nn.Linear = reference_to_original_linear_class


#        nn.Embedding = reference_to_original_embedding_class
