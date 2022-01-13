from axonn import axonn as ax
import torchvision
from models.vit import DistributedViT
from torchvision.transforms import ToTensor
import torch


def test_vit_mnist():
    bs = 64
    mbs = 1
    epochs=10
    N, D, H = 2, 128, 16


    ax.init(2,1,mbs, bs)

    train_dataset = torchvision.datasets.MNIST(root='./tests/datasets/', train=True, transform=ToTensor())
    train_loader = ax.create_dataloader(train_dataset, bs, 0)

    model = DistributedViT(image_size=28, channels=1, patch_size=4, num_classes=10, dim=D, depth=N, heads=H, dim_head=D//H, mlp_dim=D*4, dropout=0.1, emb_dropout=0.1, inter_layer_parallel_rank=ax.comm_handle.inter_layer_parallel_rank, G_inter=ax.config.G_inter).cuda()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    for epoch_number in range(epochs):
        train_iter = iter(train_loader)
        epoch_loss = 0
        acc = 0
        if ax.comm_handle.inter_layer_parallel_rank == 0:
            x,y = next(train_iter, (None, None))
            if x is None:
                break
            x,y = x.cuda(), y.cuda()
            ax.comm_handle.send(y, ax.config.G_inter-1, tag=0, async_op=False)
        elif ax.comm_handle.inter_layer_parallel_rank == ax.config.G_inter-1:
            y = torch.cuda.LongTensor(bs)
            ax.comm_handle.recv(y, 0, tag=0, async_op=False)
            x = None
        else:
            x = None

        optimizer.zero_grad()
        x = x.cuda()
        y = y.cuda()
        logits  = model(x)
        loss = loss_fn(logits,y)
        acc += torch.sum((torch.max(logits, 1)[1]) == (y)).item()
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()

    assert acc*100/len(train_loader)/bs > 95

