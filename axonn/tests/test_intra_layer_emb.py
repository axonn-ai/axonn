import torch
import pytest
from axonn import axonn as ax
from axonn.intra_layer.communication import _drop, _gather
from axonn.intra_layer import (
    Embedding,
    clip_grad_norm_,
    optimize_communication,
    clear_weights_cache,
    sync_gradients_depth_parallel,
)

@pytest.mark.mpi
@pytest.mark.parametrize("B", [2, 4, 8])
@pytest.mark.parametrize("S", [1024, 2048])
@pytest.mark.parametrize("H", [1024, 2048])
@pytest.mark.parametrize("V", [50304, 32000, 128256])
@pytest.mark.parametrize(
    "G_intra_r,  G_intra_d", [(2, 1), (1, 2)]
)
@pytest.mark.parametrize("expert_mode", [True])
def test_fw_pass(G_intra_r, G_intra_d, B, S, H, V, expert_mode):
    # These tests are in fp-32
    torch.manual_seed(42)
    ax.init(
        G_data=1,
        G_inter=1,
        G_intra_r=G_intra_r,
        G_intra_c=1,
        G_intra_d=G_intra_d,
    )

    X = torch.randint(0, V, (B,S)).cuda()

    inner_group = ax.comm_handle.inner_intra_layer_parallel_group
    outer_group = ax.comm_handle.outer_intra_layer_parallel_group
    depth_group = ax.comm_handle.depth_intra_layer_parallel_group

    X_local = _drop(X, 0, depth_group)  # divide rows of X along the depth tensor group

    layer = Embedding(
            V,H, expert_mode=expert_mode
    ).cuda()

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


if __name__ == "__main__":
    test_fw_pass(2, 1, 1, 1024, 1024, 50304, True)
