import torch
import pytest
from axonn import axonn as ax
from axonn.intra_layer.communication import _drop, _gather
from axonn.intra_layer import Tensor_Parallel_Linear


@pytest.mark.mpi
@pytest.mark.parametrize("B, H", [(32, 64), (16, 128), (2, 256)])
@pytest.mark.parametrize("G_intra_r, G_intra_c", [(1, 2), (2, 1)])
def test_fw_pass(G_intra_r, G_intra_c, B, H):
    # These tests are in fp-32
    torch.manual_seed(42)
    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
    )

    X = torch.randn(B, H).cuda() * 0.01

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group

    X_local = _drop(
        X, 1, inner_group
    )  # divide colunns of X along the inner tensor group
    layer = Tensor_Parallel_Linear(
        in_features=H, out_features=H, skip_bias_add=True
    ).cuda()

    with torch.no_grad():
        # parallel FW pass
        Y_local, _ = layer(X_local)
        Y_parallel = _gather(Y_local.clone(), 1, outer_group)

        # sequential FW pass
        layer_sequential = torch.nn.Linear(
            in_features=H, out_features=H, bias=False
        ).cuda()
        weight_sequential = _gather(
            _gather(layer.linear.weight, 1, inner_group), 0, outer_group
        )
        layer_sequential.weight.copy_(weight_sequential)
        Y_sequential = layer_sequential(X)

    assert torch.allclose(Y_sequential, Y_parallel), "FW Pass - output does not match"


@pytest.mark.mpi
@pytest.mark.parametrize("B, H", [(32, 64), (16, 128), (2, 256)])
@pytest.mark.parametrize("G_intra_r, G_intra_c", [(1, 2), (2, 1)])
@pytest.mark.parametrize("async_comm_in_backward_pass", [True, False])
def test_bw_pass(G_intra_r, G_intra_c, B, H):
    # These tests are in fp-32
    torch.manual_seed(42)
    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
    )
    X = torch.randn(B, H).cuda() * 0.01
    Y_grad = torch.randn(B, H).cuda() * 0.01

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group

    # parallel backward pass
    layer = Tensor_Parallel_Linear(
        in_features=H, out_features=H, skip_bias_add=True, async_comm_in_backward_pass=async_comm_in_backward_pass
    ).cuda()
    X_local = (
        _drop(X, 1, inner_group).detach().clone()
    )  # divide colunns of X along the inner tensor group
    X_local.requires_grad = True
    Y_local, _ = layer(X_local)
    Y_local_grad = _drop(Y_grad, 1, outer_group)
    Y_local.backward(Y_local_grad)

    # sequential backward pass
    layer_sequential = torch.nn.Linear(in_features=H, out_features=H, bias=False).cuda()
    with torch.no_grad():
        weight_sequential = _gather(
            _gather(layer.linear.weight, 1, inner_group), 0, outer_group
        )
        layer_sequential.weight.copy_(weight_sequential)
    X.requires_grad = True
    Y_sequential = layer_sequential(X)
    Y_sequential.backward(Y_grad)

    X_grad_parallel = _gather(X_local.grad, 1, inner_group)
    assert torch.allclose(
        X_grad_parallel, X.grad
    ), "BW Pass - gradients of input do not match"

    weight_grad_parallel = _gather(
        _gather(layer.linear.weight.grad, 1, inner_group), 0, outer_group
    )
    assert torch.allclose(
        weight_grad_parallel, layer_sequential.weight.grad
    ), "BW Pass - gradients of weight do not match"
