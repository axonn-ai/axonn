# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from axonn import axonn as ax
import torchvision
from models.vit import DistributedViT
from torchvision.transforms import ToTensor
import torch


def test_vit_mnist():
    bs = 32 * 3 * 8
    mbs = 32
    epochs = 10
    N, D, H = 12, 768, 12

    ax.init(G_data=2, G_inter=3)

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

    ax.register_model(model)
    ax.register_loss_fn(torch.nn.CrossEntropyLoss())

    train_dataset = torchvision.datasets.MNIST(
        root="./tests/datasets/", train=True, transform=ToTensor()
    )
    train_loader = ax.create_dataloader(train_dataset, bs, mbs, 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    for epoch_number in range(epochs):
        epoch_loss = 0
        bn = 0
        for x, y in train_loader:
            bn += 1
            optimizer.zero_grad()
            if ilp_rank == 0:
                x, y = x.cuda(), y.cuda()
            if G_inter > 1:
                if ilp_rank == 0:
                    ax.comm_handle.send(y, G_inter - 1, tag=0, async_op=False)
                elif ilp_rank == G_inter - 1:
                    y = y.long().cuda()
                    ax.comm_handle.recv(y, 0, tag=0, async_op=False)
            batch_loss = ax.run_batch(x, y)
            optimizer.step()
            epoch_loss += batch_loss
        if ilp_rank == G_inter - 1:
            ax.print_status(
                f"Epoch {epoch_number+1} : epoch loss {epoch_loss/len(train_loader)}"
            )


test_vit_mnist()
