import torch
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np
from torch.cuda.amp import GradScaler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from mpi4py import MPI

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from model.fc_net_sequential import FC_Net
from utils import print_memory_stats, num_params, log_dist, set_seed
from args import create_parser

NUM_EPOCHS=2
PRINT_EVERY=200

def set_device_and_init_torch_dist():
    # initialize torch distributed
    torch.distributed.init_process_group(backend='nccl')
    # assign a unique GPU to each MPI process on a node    
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    set_seed(args.seed)
    
    ## Step 1 - Initialize Pytorch Distributed
    set_device_and_init_torch_dist()
    log_dist('initialized pytorch dist', ranks=[0])

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    ## Step 2 - Create Dataloaders with sampler
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    # create a sampler to assign different data to each GPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=args.batch_size // dist.get_world_size(), drop_last=True, num_workers=1, sampler=train_sampler)


    ## Step 3 - Create Neural Network 
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10).cuda()
    params = num_params(net) / 1e6 
    
    ## Step 4 - Pass model through DDP constructor
    net = DDP(net, device_ids=[torch.cuda.current_device()])

    ## Step 5 - Create Optimizer and LR scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    ## Step 6 - Create Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    ## Step 7 - Train
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
   
    log_dist(f"Model Size = {params} M", ranks=[0])
    log_dist("Start training with DDP...\n", [0])
    scaler = GradScaler()

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        iter_ = 0
        iter_times = []
        for img, label in train_loader:
            start_event.record()
            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = net(img, checkpoint_activations=args.checkpoint_activations)
                iter_loss = loss_fn(output, label)

            # DDP does all reduce in the backward pass
            scaler.scale(iter_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += iter_loss
            stop_event.record()
            torch.cuda.synchronize()
            iter_time = start_event.elapsed_time(stop_event)
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0:
                log_dist(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time/1000:.6f} s", [0])
            iter_ += 1
        log_dist(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times)/1000:.6f} s", [0])
        print_memory_stats()
    
    log_dist("\nEnd training ..", [0])
