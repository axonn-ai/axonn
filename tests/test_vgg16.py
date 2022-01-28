# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


from axonn import axonn as ax
import torchvision
from models.vit import DistributedViT
from torchvision.transforms import ToTensor
import torch
from tqdm import tqdm 
import torchvision.models as models
import os
import time

def test_vgg_imagenet():
    bs_per_gpu = 64
    num_gpus = int(os.environ['WORLD_SIZE'])
    bs = num_gpus * bs_per_gpu
    mbs = bs_per_gpu
    epochs = 10

    ax.init(G_data=num_gpus, G_inter=1, mixed_precision=True)
    ax.print_status(f"Running on {num_gpus} gpus")

    ilp_rank = ax.config.inter_layer_parallel_rank
    G_inter = ax.config.G_inter

    model = models.vgg16().cuda()

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    model, optimizer = ax.register_model_and_optimizer(model, optimizer)

    ax.register_loss_fn(torch.nn.CrossEntropyLoss())

    train_dataset = torchvision.datasets.FakeData(
        size = 64*12*12,#1281167,
        image_size = (3,224,224),
        num_classes = 1000,
        transform=ToTensor()
    )
    train_loader = ax.create_dataloader(train_dataset, bs, mbs, 0)

    for epoch_number in range(epochs):
        epoch_loss = 0
        start_time = time.time()
        for x, y in tqdm(train_loader, disable= not (ilp_rank == 0 and ax.config.data_parallel_rank == 0)):
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
        if ilp_rank == G_inter - 1 and ax.config.data_parallel_rank == 0:
            ax.print_status(
                f"Epoch {epoch_number+1} : epoch loss {epoch_loss/len(train_loader)}, Epoch time = {time.time()-start_time} s"
            )
            
test_vgg_imagenet()
