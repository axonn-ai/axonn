# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from axonn import axonn as ax
from axonn import optim
import torchvision
from external.models.vit import DistributedViT
from torchvision.transforms import ToTensor
import torch
from tqdm import tqdm


def test_vit_mnist():
    bs_per_gpu = 64
    num_gpus = 6
    bs = num_gpus * bs_per_gpu
    mbs = bs_per_gpu
    epochs = 10
    cpu_offload = True
    N, D, H = 12, 768, 12

    ax.init(
        G_data=2,
        G_inter=3,
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
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ax.register_model_and_optimizer(model, optimizer)

    ax.register_loss_fn(torch.nn.CrossEntropyLoss())

    train_dataset = torchvision.datasets.MNIST(
        root="./examples/dataset/", train=True, transform=ToTensor()
    )
    train_loader = ax.create_dataloader(train_dataset, bs, mbs, 0)

    for epoch_number in range(epochs):
        epoch_loss = 0
        for x, y in tqdm(
            train_loader,
            disable=not (ilp_rank == 0 and ax.config.data_parallel_rank == 0),
        ):
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
