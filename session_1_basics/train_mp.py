import torch
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.cuda.amp import GradScaler

from model.fc_net_sequential import FC_Net
from utils import print_memory_stats, num_params
from args import create_parser

NUM_EPOCHS=2
PRINT_EVERY=200


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    ## Step 1 - Create Dataloaders
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=args.batch_size, drop_last=True, num_workers=1)
    
    ## Step 2 - Create Neural Network 
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10).cuda()
    params = num_params(net) / 1e9 
    
    ## Step 3 - Create Optimizer and LR scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    ## Step 4 - Create Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    ## Step 5 - Train
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
   
    print("Start training in mixed-precision ...\n")
    print(f"Model Size = {params} B")

    ## Scales the loss prior to the backward pass to prevent underflow of gradients
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

            ## autocast selectively runs certain ops in fp16 and others in fp32
            ## usually ops that do accumulation (like batch norm) or exponentiation 
            ## (like softmax) need to be run in fp32 for numerical stability
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = net(img, checkpoint_activations=args.checkpoint_activations)
                iter_loss = loss_fn(output, label)
           
            ## scaling loss before doing the backward pass
            scaler.scale(iter_loss).backward()
            ## unscale gradients and run optimizer
            scaler.step(optimizer)
            ## update loss scale 
            scaler.update()
            
            epoch_loss += iter_loss
            stop_event.record()
            torch.cuda.synchronize()
            iter_time = start_event.elapsed_time(stop_event)
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0:
                print(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time/1000:.6f} s")
                # print_memory_stats()
            iter_ += 1
        print(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times)/1000:.6f} s")
        
    print("\nEnd of training.")

