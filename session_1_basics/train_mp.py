import torch
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from torch.cuda.amp import GradScaler


from model.vit import ViT
from utils import print_memory_stats, num_params
from args import create_parser

NUM_EPOCHS=10
PRINT_EVERY=1


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


    test_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, transform=augmentations
    )

    test_loader = torch.utils.data.DataLoader(train_dataset, 
            batch_size=args.batch_size, drop_last=True)
    
    ## Step 2 - Create Neural Network 
    net = ViT(
                image_size=args.image_size,
                channels=1,
                patch_size=4,
                num_classes=10,
                dim=args.hidden_size,
                depth=args.num_layers,
                heads=16,
                dim_head=args.hidden_size // 16,
                mlp_dim=args.hidden_size * 4,
                dropout=0.1,
                emb_dropout=0.1,
            ).cuda()

    params = num_params(net) / 1e9 
    
    ## Step 3 - Create Optimizer and LR scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    
    ## Step 4 - Create Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    ## Step 5 - Train
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
   
    print(f"Model Size = {params} B")

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
           
            scaler.scale(iter_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += iter_loss
            stop_event.record()
            torch.cuda.synchronize()
            iter_time = start_event.elapsed_time(stop_event)
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0:
                print(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time/1000:.6f} s")
                print_memory_stats()
            iter_ += 1
        print(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times)/1000:.6f} s")
        
