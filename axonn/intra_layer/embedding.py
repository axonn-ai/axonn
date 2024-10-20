# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.distributed as dist
import torch
import torch.nn.functional as F

from axonn import axonn as ax
from .communication import (
    Drop,
    Gather,
    ForwardGather_BackwardReduceScatter,
)


def divide(a, b):
    assert a % b == 0
    return a // b


@torch.no_grad()
def extract_local_params_from_full_params(
    params, out_features_group, in_features_group, depth_group
):
    params = Drop.apply(params, out_features_group)
    params = Drop.apply(torch.t(params).contiguous(), in_features_group)
    params = torch.t(params).contiguous()
    params = Drop.apply(params.reshape(-1), depth_group)  # create 1D view
    return params


@torch.no_grad()
def initialize_params(
    out_features,
    in_features,
    out_features_group,
    in_features_group,
    depth_group,
    init_method,
    init_device="cuda",
):
    params = torch.empty((in_features, out_features), device=init_device)
    init_method(params)
    params = extract_local_params_from_full_params(
        params, out_features_group, in_features_group, depth_group
    ).cpu()
    return params


@torch.no_grad()
def default_init_method(weight, padding_idx=None):
    return torch.nn.init.normal_(weight)


class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        _weight=None,
        _freeze=False,
        *args,
        transpose=False,
        init_method=None,
        expert_mode=False,
        **kwargs,
    ):
        assert not _weight, "_weight argument is not supported."
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert (
                    padding_idx < self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
            elif padding_idx < 0:
                assert (
                    padding_idx >= -self.num_embeddings
                ), "Padding_idx must be within num_embeddings"
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse

        self.inner_group = ax.comm_handle.inner_intra_layer_parallel_group
        self.outer_group = ax.comm_handle.outer_intra_layer_parallel_group
        self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group

        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        self.depth_group_size = dist.get_world_size(self.depth_group)

        self.out_features = self.embedding_dim
        self.in_features = self.num_embeddings

        if init_method is None:
            init_method = default_init_method

        if not transpose:
            assert self.inner_group_size == 1
            assert self.embedding_dim % self.outer_group_size == 0
            self.local_in_features = self.num_embeddings
            self.local_out_features = divide(self.embedding_dim, self.outer_group_size)
            initial_params = initialize_params(
                self.out_features,
                self.in_features,
                self.outer_group,
                self.inner_group,
                self.depth_group,
                init_method,
            )
        else:
            assert self.outer_group_size == 1
            assert embedding_dim % self.inner_group_size == 0
            self.local_in_features = self.num_embeddings
            self.local_out_features = divide(self.embedding_dim, self.inner_group_size)
            initial_params = initialize_params(
                self.out_features,
                self.in_features,
                self.inner_group,
                self.outer_group,
                self.depth_group,
                init_method,
            )

        if self.padding_idx is not None:
            initial_params[padding_idx].fill_(0)

        self.weight = torch.nn.Parameter(initial_params, requires_grad=not _freeze)

        setattr(self.weight, "is_tensor_parallel", True)
        setattr(self.weight, "needs_depth_parallel_gradient_sync", False)
        setattr(
            self.weight,
            "process_group_for_norm_reduction",
            ax.comm_handle.intra_layer_group,
        )

        self.expert_mode = expert_mode
        self.transpose = transpose
        self._old_load_from_state_dict = self._load_from_state_dict
        self._load_from_state_dict = self._modified_load_from_state_dict

    def get_output_feature_size(self):
        return self.local_out_features

    def forward(self, x):
        # gather weights from depth parallel group
        # reduce scatter in the backward pass
        weight = self.weight
        weight = ForwardGather_BackwardReduceScatter.apply(
            weight, self.depth_group
        ).reshape(self.local_in_features, self.local_out_features)
        x = F.embedding(
            x,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )
        if not self.expert_mode:
            x = Gather.apply(
                x, self.outer_group if not self.transpose else self.inner_group
            )

        return x

    def _is_full_weight_matrix(self, weight):
        return (
            weight.ndim == 2
            and weight.size(0) == self.in_features
            and weight.size(1) == self.out_features
        )

    def _is_sharded_weight_matrix(self, weight):
        return weight.ndim == 1 and weight.size(0) == divide(
            self.local_out_features * self.local_in_features, self.depth_group_size
        )

    @torch.no_grad()
    def _modified_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight = (
            state_dict[prefix + "weight"] if prefix + "weight" in state_dict else None
        )

        if weight is not None:
            is_full_weight_matrix = self._is_full_weight_matrix(weight)
            is_sharded_weight_matrix = self._is_sharded_weight_matrix(weight)

            assert (
                is_full_weight_matrix or is_sharded_weight_matrix
            ), "This is neither a full checkpoint nor a sharded checkpoint"

            if is_full_weight_matrix:
                out_features_group, in_features_group = (
                    self.outer_group,
                    self.inner_group,
                )
                if self.transpose:
                    out_features_group, in_features_group = (
                        self.inner_group,
                        self.outer_group,
                    )
                weight = extract_local_params_from_full_params(
                    weight, out_features_group, in_features_group, self.depth_group
                )

            state_dict[prefix + "weight"] = weight

        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)
