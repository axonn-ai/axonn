# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import torchvision
from external.models.vit import DistributedViT
from torchvision.transforms import ToTensor
import torch
from tqdm import tqdm
import pytest
import os


@pytest.mark.mpi
def test_vit_mnist():
    from axonn import axonn as ax
    from axonn import optim

    G_inter = int(os.environ.get("G_inter"))
    assert 6 % G_inter == 0
    G_data = int(os.environ.get("G_data"))
    bs = int(os.environ.get("batch_size", 64))
    mbs = int(os.environ.get("micro_batch_size", 16))
    epochs = int(os.environ.get("epochs", 10))
    cpu_offload = bool(os.environ.get("memopt"))
    N, D, H = 6, 128, 8

    ax.init(
        G_data=G_data,
        G_inter=G_inter,
        mixed_precision=True,
        fp16_allreduce=True,
        cpu_offload=cpu_offload,
    )

    ilp_rank = ax.config.inter_layer_parallel_rank
    G_inter = ax.config.G_inter

    model = DistributedViT(
        image_size=28,
        channels=1,
        patch_size=4,
        num_classes=10,
        dim=D,
        depth=N,
        heads=H,
        dim_head=D // H,
        mlp_dim=D * 4,
        dropout=0.1,
        emb_dropout=0.1,
        inter_layer_parallel_rank=ilp_rank,
        G_inter=G_inter,
    ).cuda()

    if cpu_offload:
        optimizer = optim.CPUAdam(model.parameters(), lr=0.001)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    ax.register_model_and_optimizer(model, optimizer)

    ax.register_loss_fn(torch.nn.CrossEntropyLoss())

    train_dataset = torchvision.datasets.MNIST(
        root="./", train=True, transform=ToTensor()
    )
    train_loader = ax.create_dataloader(train_dataset, bs, mbs, 0)
    previous_model_state_memory = None
    for epoch_number in range(epochs):
        epoch_loss = 0
        for x, y in tqdm(train_loader, disable=True):
            optimizer.zero_grad()
            if ilp_rank == 0:
                x, y = x.cuda(), y.cuda()
            if G_inter > 1:
                if ilp_rank == 0:
                    ax.comm_handle.send(y, G_inter - 1, tag=0, async_op=False)
                elif ilp_rank == G_inter - 1:
                    y = y.long().cuda()
                    ax.comm_handle.recv(y, 0, tag=0, async_op=False)
            batch_loss = ax.run_batch(x, y, eval_mode=False)
            optimizer.step()
            epoch_loss += batch_loss
            current_model_state_memory = torch.cuda.memory_allocated()
            assert (not previous_model_state_memory) or (
                current_model_state_memory == previous_model_state_memory
            ), "model state memory should stay the same throughout training"
        if ilp_rank == G_inter - 1:
            ax.print_status(
                f"Epoch {epoch_number+1} : epoch loss {epoch_loss/len(train_loader)}"
                f": model state memory = {torch.cuda.memory_allocated()/2**30} GB"
            )

    assert epoch_loss / len(train_loader) < 0.1, "model did not converge"


if __name__ == "__main__":
    test_vit_mnist()
