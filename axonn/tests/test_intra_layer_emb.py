# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import pytest
from axonn import axonn as ax
from axonn.intra_layer.communication import _drop, _gather
from axonn.intra_layer import (
    Embedding,
    clip_grad_norm_,
    sync_gradients_depth_parallel,
)


@pytest.mark.parametrize("B", [2, 4, 8])
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("H", [1024, 2048])
@pytest.mark.parametrize("V", [50304, 32000, 128256])
@pytest.mark.parametrize("G_intra_r,  G_intra_d", [(2, 1), (1, 2)])
@pytest.mark.parametrize("expert_mode", [True, False])
def test_fw_pass(G_intra_r, G_intra_d, B, S, H, V, expert_mode):
    # These tests are in fp-32
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    torch.manual_seed(42)
    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=1,
        G_intra_d=G_intra_d,
    )

    X = torch.randint(0, V, (B, S)).cuda()

    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    X_local = _drop(X, 0, depth_group)  # divide rows of X along the depth tensor group

    layer = Embedding(V, H, expert_mode=expert_mode).cuda()

    layer_sequential = torch.nn.Embedding(V, H).cuda()

    # test if load state dict works with a sequential checkpoint
    layer.load_state_dict(layer_sequential.state_dict())
    # test if load state dict works with a sharded checkpoint
    layer.load_state_dict(layer.state_dict())

    with torch.no_grad():
        # parallel FW pass
        Y_local = layer(X_local)
        Y_parallel = _gather(Y_local.clone(), 0, depth_group)
        if expert_mode:  # gather output manually
            Y_parallel = _gather(Y_parallel.clone(), -1, outer_group)
        Y_sequential = layer_sequential(X)

    assert torch.allclose(Y_sequential, Y_parallel), "FW Pass - output does not match"


@pytest.mark.mpi
@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("H", [1024, 2048])
@pytest.mark.parametrize("V", [50304, 32000])
@pytest.mark.parametrize("G_intra_r,  G_intra_d", [(2, 1), (1, 2)])
@pytest.mark.parametrize("clip_grad_norm", [1e-3, -1])
@pytest.mark.parametrize("expert_mode", [True, False])
# comm opt is pending
def test_bw_pass(G_intra_r, G_intra_d, B, S, H, V, expert_mode, clip_grad_norm):
    # These tests are in fp-32
    torch.manual_seed(42)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=1,
        G_intra_d=G_intra_d,
    )
    X = torch.randint(0, V, (B, S)).cuda()
    Y_grad = torch.randn(B, S, H).cuda() * 0.01

    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    layer = Embedding(V, H, expert_mode=expert_mode).cuda()
    layer_sequential = torch.nn.Embedding(V, H).cuda()
    # test if load state dict works with a sequential checkpoint
    layer.load_state_dict(layer_sequential.state_dict())
    # test if load state dict works with a sharded checkpoint
    layer.load_state_dict(layer.state_dict())

    X_local = (
        _drop(X, 0, depth_group).detach().clone()
    )  # divide colunns of X along the inner tensor group

    Y_local_grad = _drop(Y_grad, 0, depth_group).detach().clone()
    if expert_mode:
        Y_local_grad = _drop(Y_local_grad, -1, outer_group).detach().clone()

    Y_local = layer(X_local)
    Y_local.backward(Y_local_grad)

    sync_gradients_depth_parallel(layer)

    # sequential backward pass
    Y_sequential = layer_sequential(X)
    Y_sequential.backward(Y_grad)

    if clip_grad_norm > 0:
        clip_grad_norm_(layer.parameters(), max_norm=clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(
            layer_sequential.parameters(), max_norm=clip_grad_norm
        )

    weight_grad_parallel = _gather(layer.weight.grad, 0, depth_group).reshape(
        layer.local_in_features, layer.local_out_features
    )

    weight_grad_parallel = _gather(weight_grad_parallel, 1, outer_group)

    assert torch.allclose(
        weight_grad_parallel, layer_sequential.weight.grad
    ), "BW Pass - gradients of weight do not match"
