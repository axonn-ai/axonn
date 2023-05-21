import torch
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.fc_net_sequential import FC_Net 
from utils import print_memory_stats, num_params
from args import create_parser

NUM_EPOCHS=2
PRINT_EVERY=200

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    ## Step 0 - Create a dataset object with transformations
    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    ## Step 1 - Create Dataloader for MNIST
    train_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=args.batch_size, drop_last=True, num_workers=1)

   
    ## Step 2 - Create Neural Network 
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10).cuda()

    params = num_params(net) / 1e9 
    
    ## Step 3 - Create Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    ## Step 4 - Create Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    ## Create CUDA events for profiling
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
   
    ## Step 5 - Train
    print("Start training ...\n")
    print(f"Model Size = {params} B")

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        iter_ = 0
        iter_times = []
        for img, label in train_loader:
            start_event.record()
            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()

            # Forward pass
            output = net(img)
            iter_loss = loss_fn(output, label)
            
            # Backawrd Pass
            iter_loss.backward()
            
            # Optimizer
            optimizer.step()
            
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

