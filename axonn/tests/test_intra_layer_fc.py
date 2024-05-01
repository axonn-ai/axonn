import torch
import pytest
from axonn import axonn as ax
from axonn.intra_layer.communication import _drop, _gather
from axonn.intra_layer import (
    Linear,
    clip_grad_norm_,
    optimize_communication,
    clear_weights_cache,
    sync_gradients,
)


@pytest.mark.mpi
@pytest.mark.parametrize("B, H", [(32, 64), (16, 128), (2, 256)])
@pytest.mark.parametrize(
    "G_intra_r, G_intra_c, G_intra_d", [(2, 1, 1), (1, 2, 1), (1, 1, 2)]
)
@pytest.mark.parametrize("easy_tp", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_fw_pass(G_intra_r, G_intra_c, G_intra_d, B, H, easy_tp, bias, device):
    # These tests are in fp-32
    torch.manual_seed(42)

    # GPU runs on axonn-cpu currently do not work with mixed_precision or fp16_allreduce
    # if set_device == "cpu":
    #     bool set_mixed_precision = False
    #     bool set_fp16_allreduce = False

    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
        G_intra_d=G_intra_d,
        mixed_precision=False,
        fp16_allreduce=False,
        device=device,
    )

    X = torch.randn(B, H).to(device) * 0.01

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    X_local = _drop(X, 0, depth_group)  # divide rows of X along the depth tensor group

    if not easy_tp:
        # manually divide input
        X_local = _drop(
            X, 1, inner_group
        )  # divide colunns of X along the inner tensor group
        # manually divide input

    layer = Linear(in_features=H, out_features=H, bias=bias).to(device)
    layer_sequential = torch.nn.Linear(in_features=H, out_features=H, bias=bias).to(
        device
    )

    # test if load state dict works with a sequential checkpoint
    layer.load_state_dict(layer_sequential.state_dict())
    # test if load state dict works with a sharded checkpoint
    layer.load_state_dict(layer.state_dict())

    with torch.no_grad():
        # parallel FW pass
        Y_local = layer(X_local, scatter_input=easy_tp, gather_output=easy_tp)
        Y_parallel = _gather(Y_local.clone(), 0, depth_group)
        if not easy_tp:  # gather output manually
            Y_parallel = _gather(Y_local.clone(), 1, outer_group)
        Y_sequential = layer_sequential(X)

    assert torch.allclose(Y_sequential, Y_parallel), "FW Pass - output does not match"


@pytest.mark.mpi
@pytest.mark.parametrize("B, H", [(32, 64), (16, 128), (2, 256)])
@pytest.mark.parametrize(
    "G_intra_r, G_intra_c, G_intra_d", [(2, 1, 1), (1, 2, 1), (1, 1, 2)]
)
@pytest.mark.parametrize("comm_opt_level", [0, 4])
@pytest.mark.parametrize("easy_tp", [False, True])
@pytest.mark.parametrize("clip_grad_norm", [-1, 1e-3])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_bw_pass(
    G_intra_r,
    G_intra_c,
    G_intra_d,
    B,
    H,
    comm_opt_level,
    easy_tp,
    clip_grad_norm,
    bias,
    device,
):
    # These tests are in fp-32
    if device == "cpu" and G_intra_d > 1:
        return  # Gloo doesnt support reduce scatter

    torch.manual_seed(42)
    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=G_intra_c,
        G_intra_d=G_intra_d,
        mixed_precision=False,
        fp16_allreduce=False,
        device=device,
    )
    X = torch.randn(B, H).to(device) * 0.01
    Y_grad = torch.randn(B, H).to(device) * 0.01

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    # parallel backward pass
    layer = Linear(
        in_features=H,
        out_features=H,
        bias=bias,
    ).to(device)
    layer_sequential = torch.nn.Linear(in_features=H, out_features=H, bias=bias).to(
        device
    )

    # test if load state dict works with a sequential checkpoint
    layer.load_state_dict(layer_sequential.state_dict())
    # test if load state dict works with a sharded checkpoint
    layer.load_state_dict(layer.state_dict())

    X_local = (
        _drop(X, 0, depth_group).detach().clone()
    )  # divide colunns of X along the inner tensor group
    if not easy_tp:
        X_local = (
            _drop(X_local, 1, inner_group).detach().clone()
        )  # divide colunns of X along the inner tensor group

    X_local.requires_grad = True

    Y_local_grad = _drop(Y_grad, 0, depth_group).detach().clone()
    if not easy_tp:
        Y_local_grad = _drop(Y_local_grad, 1, outer_group).detach().clone()

    with optimize_communication(
        overlap_all_reduce=comm_opt_level >= 1,
        overlap_reduce_scatter=comm_opt_level >= 2 and device != "cpu",
        cache_weights=comm_opt_level >= 3,
        overlap_all_gather=comm_opt_level == 4 and device != "cpu",
        model_object_for_overlapping_allgathers=layer,
    ):
        Y_local = layer(X_local, scatter_input=easy_tp, gather_output=easy_tp)
        Y_local.backward(Y_local_grad)

    sync_gradients(layer)
    if comm_opt_level >= 3:
        clear_weights_cache()
    # sequential backward pass
    X.requires_grad = True
    Y_sequential = layer_sequential(X)
    Y_sequential.backward(Y_grad)

    if clip_grad_norm > 0:
        clip_grad_norm_(layer.parameters(), max_norm=clip_grad_norm)
        torch.nn.utils.clip_grad_norm_(
            layer_sequential.parameters(), max_norm=clip_grad_norm
        )

    X_grad_parallel = _gather(X_local.grad, 0, depth_group)
    if not easy_tp:
        X_grad_parallel = _gather(X_grad_parallel, 1, inner_group)

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
