# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import pytest
from axonn import axonn as ax
from axonn.intra_layer.communication import _drop, _gather
from axonn.intra_layer import (
    Linear,
    clip_grad_norm_,
    sync_gradients,
    optimize_communication,
)


@pytest.mark.parametrize("B, H", [(32, 64), (16, 128), (2, 256)])
@pytest.mark.parametrize(
    "G_intra_r, G_intra_c, G_intra_d", [(2, 1, 1), (1, 2, 1), (1, 1, 2)]
)
@pytest.mark.parametrize("expert_mode", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_fw_pass(G_intra_r, G_intra_c, G_intra_d, B, H, expert_mode, bias):
    # These tests are in fp-32
    torch.manual_seed(42)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
        G_intra_d=G_intra_d,
    )

    X = torch.randn(B, H).cuda() * 0.01

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    if expert_mode:
        # manually divide input
        X_local = _drop(
            X, 0, depth_group
        )  # divide rows of X along the depth tensor group
        X_local = _drop(
            X_local, 1, inner_group
        )  # divide colunns of X along the inner tensor group
        # manually divide input
    else:
        X_local = _drop(X, 0)  # simply divide the batch equally among all GPUs

    layer = Linear(
        in_features=H, out_features=H, bias=bias, expert_mode=expert_mode
    ).cuda()
    layer_sequential = torch.nn.Linear(in_features=H, out_features=H, bias=bias).cuda()

    # test if load state dict works with a sequential checkpoint
    layer.load_state_dict(layer_sequential.state_dict())
    # test if load state dict works with a sharded checkpoint
    layer.load_state_dict(layer.state_dict())

    with torch.no_grad():
        # parallel FW pass
        Y_local = layer(X_local)
        if expert_mode:  # gather output manually
            Y_parallel = _gather(Y_local.clone(), 0, depth_group)
            Y_parallel = _gather(Y_parallel.clone(), 1, outer_group)
        else:
            # simply gather the output along the batch dimension
            Y_parallel = _gather(Y_local.clone(), 0)

        Y_sequential = layer_sequential(X)

    assert torch.allclose(Y_sequential, Y_parallel), "FW Pass - output does not match"

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("B, H", [(32, 64), (16, 128), (2, 256)])
@pytest.mark.parametrize(
    "G_intra_r, G_intra_c, G_intra_d", [(2, 1, 1), (1, 2, 1), (1, 1, 2)]
)
@pytest.mark.parametrize("comm_opt_level", [0, 4])
@pytest.mark.parametrize("expert_mode", [False, True])
@pytest.mark.parametrize("clip_grad_norm", [-1, 1e-3])
@pytest.mark.parametrize("bias", [False, True])
def test_bw_pass(
    G_intra_r,
    G_intra_c,
    G_intra_d,
    B,
    H,
    comm_opt_level,
    expert_mode,
    clip_grad_norm,
    bias,
):
    if bias:
        pytest.skip()  # ToDO: Fix this convergence bug
    # These tests are in fp-32
    torch.manual_seed(42)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl")
    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
        G_intra_d=G_intra_d,
    )
    X = torch.randn(B, H).cuda() * 0.01
    Y_grad = torch.randn(B, H).cuda() * 0.01

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    # parallel backward pass
    layer = Linear(
        in_features=H, out_features=H, bias=bias, expert_mode=expert_mode
    ).cuda()
    layer_sequential = torch.nn.Linear(in_features=H, out_features=H, bias=bias).cuda()

    # test if load state dict works with a sequential checkpoint
    layer.load_state_dict(layer_sequential.state_dict())
    # test if load state dict works with a sharded checkpoint
    layer.load_state_dict(layer.state_dict())

    if expert_mode:
        X_local = (
            _drop(X, 0, depth_group).detach().clone()
        )  # divide colunns of X along the inner tensor group
        X_local = (
            _drop(X_local, 1, inner_group).detach().clone()
        )  # divide colunns of X along the inner tensor group
    else:
        X_local = (
            _drop(X, 0).detach().clone()
        )  # simply divide the batch dimension of X among all GPUs

    X_local.requires_grad = True

    if expert_mode:
        Y_local_grad = _drop(Y_grad, 0, depth_group).detach().clone()
        Y_local_grad = _drop(Y_local_grad, 1, outer_group).detach().clone()
    else:
        Y_local_grad = _drop(Y_grad, 0).detach().clone()

    with optimize_communication(
        overlap_all_reduce=comm_opt_level >= 1,
        overlap_reduce_scatter=comm_opt_level >= 2,
        cache_weights=comm_opt_level >= 3,
        overlap_all_gather=comm_opt_level == 4,
        model_object_for_overlapping_allgathers=layer,
    ):
        Y_local = layer(X_local)
        Y_local.backward(Y_local_grad)

    sync_gradients(layer, expert_mode=expert_mode)

    # sequential backward pass
    X.requires_grad = True
    Y_sequential = layer_sequential(X)
    Y_sequential.backward(Y_grad)

    if clip_grad_norm > 0:
        clip_grad_norm_(layer.parameters(), max_norm=clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(
            layer_sequential.parameters(), max_norm=clip_grad_norm
        )

    if expert_mode:
        X_grad_parallel = _gather(X_local.grad, 0, depth_group)
        X_grad_parallel = _gather(X_grad_parallel, 1, inner_group)
    else:
        X_grad_parallel = _gather(X_local.grad, 0)

    assert torch.allclose(
        X_grad_parallel, X.grad
    ), "BW Pass - gradients of input do not match"

    weight_grad_parallel = _gather(layer.weight.grad, 0, depth_group).reshape(
        layer.local_out_features, layer.local_in_features
    )

    weight_grad_parallel = _gather(
        _gather(weight_grad_parallel, 1, inner_group), 0, outer_group
    )

    assert torch.allclose(
        weight_grad_parallel, layer_sequential.weight.grad
    ), "BW Pass - gradients of weight do not match"

    if bias:
        bias_grad_parallel = _gather(layer.bias.grad, 0, outer_group)
        assert torch.allclose(
            bias_grad_parallel, layer_sequential.bias.grad
        ), "BW Pass - gradients of bias do not match"

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    test_bw_pass(
        G_intra_r=1,
        G_intra_c=2,
        G_intra_d=1,
        B=2,
        H=256,
        comm_opt_level=4,
        expert_mode=False,
        clip_grad_norm=1e-3,
        bias=True,
    )
