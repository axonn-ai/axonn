import torch
import pytest
from mpi4py import MPI  # noqa: F401
from axonn import axonn as ax
from axonn.intra_layer.communication import _drop, _gather
from axonn.intra_layer import (
    Conv2d,
    optimize_communication,
    clear_weights_cache,
    sync_gradients,
)
import math
import torch.distributed as dist


def log_dist(msg, ranks=[]):
    assert dist.is_initialized()
    if dist.get_rank() in ranks:
        print(f"Rank {dist.get_rank()} : {msg}")


def norm_allclose(X, Y):
    epsilon = 1e-6
    squared_diff = torch.square(X - Y)
    mse = torch.mean(squared_diff).item()
    rmse = math.sqrt(mse)

    log_dist(f"RMSE:{rmse}", [0])
    log_dist(f"L2Norm:{torch.norm(X - Y, 2)}", [0])

    if rmse < epsilon:
        return True
    else:
        return False


@pytest.mark.mpi
@pytest.mark.parametrize("H, W, C", [(64, 64, 4), (64, 64, 8), (64, 32, 8)])
@pytest.mark.parametrize("B", [2, 4, 16])
@pytest.mark.parametrize(
    "G_intra_r, G_intra_c, G_intra_d", [(1, 2, 1), (2, 1, 1), (1, 1, 2)]
)
@pytest.mark.parametrize("easy_tp", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.skip(reason="torch.all_close does not work with conv")
def test_fw_pass(G_intra_r, G_intra_c, G_intra_d, B, H, W, C, easy_tp, bias):
    # These tests are in fp-32
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    # Need to remove all non-determinism from convolutions
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # This is required because TF32 cores only look at the first 10 bits of mantissa
    torch.backends.cudnn.allow_tf32 = False

    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
        G_intra_d=G_intra_d,
    )

    X = torch.randn(B, C, H, W).cuda() * 0.01

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    if not easy_tp:
        X_local = _drop(
            X, 1, inner_group
        )  # divide channels of X along the inner tensor group
        X_local = _drop(
            X_local, 0, depth_group
        )  # divide input channels of X along the depth tensor group
    else:
        X_local = X

    layer = Conv2d(in_channels=C, out_channels=2 * C, kernel_size=5, bias=bias).cuda()

    with torch.no_grad():
        # parallel FW pass
        Y_local = layer(X_local, scatter_input=easy_tp, gather_output=easy_tp)
        if not easy_tp:
            Y_parallel = _gather(Y_local.clone(), 1, outer_group)
            Y_parallel = _gather(Y_parallel.clone(), 0, depth_group)
        else:
            Y_parallel = Y_local

        # sequential FW pass
        layer_sequential = torch.nn.Conv2d(
            in_channels=C,
            out_channels=C * 2,
            kernel_size=5,
            bias=bias,
        ).cuda()
        weight_sequential = _gather(
            _gather(
                _gather(layer.weight, 0, depth_group).reshape(
                    layer.local_out_channels,
                    layer.local_in_channels,
                    layer.kernel_size,
                    layer.kernel_size,
                ),
                1,
                inner_group,
            ),
            0,
            outer_group,
        )
        layer_sequential.weight.copy_(weight_sequential)
        if bias:
            layer_sequential.bias.zero_()
        Y_sequential = layer_sequential(X)

    assert torch.allclose(Y_sequential, Y_parallel), "FW Pass - output does not match"


@pytest.mark.mpi
@pytest.mark.parametrize("H, W, C", [(64, 64, 4), (64, 64, 8), (64, 32, 8)])
@pytest.mark.parametrize("B", [2, 4, 16])
@pytest.mark.parametrize(
    "G_intra_r, G_intra_c, G_intra_d", [(1, 2, 1), (2, 1, 1), (1, 1, 2)]
)
@pytest.mark.parametrize("easy_tp", [True, False])
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("comm_opt_level", [0, 3])
@pytest.mark.skip(reason="torch.all_close does not work with conv")
def test_bw_pass(
    G_intra_r, G_intra_c, G_intra_d, B, H, W, C, easy_tp, bias, comm_opt_level
):
    # These tests are in fp-32
    # Need to remove all non-determinism from convolutions
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # This is required because TF32 cores only look at the first 10 bits of mantissa
    torch.backends.cudnn.allow_tf32 = False

    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
        G_intra_d=G_intra_d,
    )
    X = torch.randn(B, C, H, W).cuda() * 0.01
    Y_grad = torch.randn(B, 2 * C, H - 4, W - 4).cuda() * 0.01

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    # parallel backward pass
    layer = Conv2d(in_channels=C, out_channels=2 * C, kernel_size=5, bias=bias).cuda()

    if not easy_tp:
        X_local = (
            _drop(X, 1, inner_group).detach().clone()
        )  # divide input channels of X along the inner tensor group
        X_local = (
            _drop(X_local, 0, depth_group).detach().clone()
        )  # divide input channels of X along the depth tensor group
    else:
        X_local = X

    X_local.requires_grad = True
    if not easy_tp:
        Y_local_grad = _drop(Y_grad, 1, outer_group).detach().clone()
        Y_local_grad = _drop(Y_local_grad, 0, depth_group).detach().clone()
    else:
        Y_local_grad = Y_grad

    with optimize_communication(
        overlap_reduce_scatter=comm_opt_level >= 1,
        cache_weights=comm_opt_level >= 2,
        overlap_all_gather=comm_opt_level == 3,
        model_object_for_overlapping_allgathers=layer,
    ):
        Y_local = layer(X_local, scatter_input=easy_tp, gather_output=easy_tp)
        Y_local.backward(Y_local_grad)

    if not easy_tp:
        sync_gradients(layer)
    if comm_opt_level >= 3:
        clear_weights_cache()

    # sequential backward pass
    layer_sequential = torch.nn.Conv2d(
        in_channels=C,
        out_channels=C * 2,
        kernel_size=5,
        bias=bias,
    ).cuda()
    with torch.no_grad():
        weight_sequential = _gather(
            _gather(
                _gather(layer.weight, 0, depth_group).reshape(
                    layer.local_out_channels,
                    layer.local_in_channels,
                    layer.kernel_size,
                    layer.kernel_size,
                ),
                1,
                inner_group,
            ),
            0,
            outer_group,
        )
        layer_sequential.weight.copy_(weight_sequential)
        if bias:
            layer_sequential.bias.zero_()
    X.requires_grad = True
    Y_sequential = layer_sequential(X)
    Y_sequential.backward(Y_grad)

    if not easy_tp:
        X_grad_parallel = _gather(X_local.grad, 0, depth_group)
        X_grad_parallel = _gather(X_grad_parallel, 1, inner_group)
    else:
        X_grad_parallel = X_local.grad

    assert norm_allclose(
        X_grad_parallel, X.grad
    ), "BW Pass - gradients of input do not match"

    weight_grad_parallel = _gather(
        _gather(
            _gather(layer.weight.grad, 0, depth_group).reshape(
                layer.local_out_channels,
                layer.local_in_channels,
                layer.kernel_size,
                layer.kernel_size,
            ),
            1,
            inner_group,
        ),
        0,
        outer_group,
    )

    assert norm_allclose(
        weight_grad_parallel, layer_sequential.weight.grad
    ), "BW Pass - gradients of weight do not match"

    if bias:
        bias_grad_parallel = _gather(layer.bias.grad, 0, outer_group)
        assert norm_allclose(
            bias_grad_parallel, layer_sequential.bias.grad
        ), "BW Pass - gradients of bias do not match"
