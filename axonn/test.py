import axonn as ax
import torchvision

bs = 8
mbs = 1

ax.init(2,2,mbs, bs)

train_dataset = torchvision.datasets.MNIST(root='/gpfs/alpine/csc452/scratch/ssingh37/MNIST', train=True, download=True)
train_loader = ax.create_dataloader(train_dataset, bs, 0)


