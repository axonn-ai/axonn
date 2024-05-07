import torch
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np
import deepspeed


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from model.fc_net_sequential import FC_Net
from utils import print_memory_stats, num_params, log_dist, set_seed
from args import create_parser

NUM_EPOCHS=2
PRINT_EVERY=200



if __name__ == "__main__":
    parser = create_parser()
    ## deepspeed requires us to add this arg
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    set_seed(args.seed)

    ## Step 1 - Initialize DeepSpeed Distributed
    deepspeed.init_distributed()
    log_dist('initialized deepspeed', ranks=[0])

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    ## Step 2 - Create dataset
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    ## Step 3 - Create Neural Network and optimizer
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10).cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    params = num_params(net) / 1e6
   

    ## Step 4 - Create model, optimizer, trainloader
    
    model_engine, optimizer, train_loader, __ = deepspeed.initialize(
        args=args, model=net, optimizer=optimizer, training_data=train_dataset)
   

    ## Step 6 - Create Loss Function
    loss_fn = torch.nn.CrossEntropyLoss()

    ## Step 7 - Train
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)
   
    log_dist(f"Model Size = {params} M", ranks=[0])
    log_dist(f"Start training with DeepSpeed \n", [0])

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        iter_ = 0
        iter_times = []
        for img, label in train_loader:
            start_event.record()
            optimizer.zero_grad()
            img = img.cuda().half() ## manually typecast input to fp16
            label = label.cuda()
            
            output = model_engine(img, checkpoint_activations=args.checkpoint_activations)
            iter_loss = loss_fn(output, label)
            model_engine.backward(iter_loss)
            model_engine.step()
            
            epoch_loss += iter_loss
            stop_event.record()
            torch.cuda.synchronize()
            iter_time = start_event.elapsed_time(stop_event)
            iter_times.append(iter_time)
            if iter_ % PRINT_EVERY == 0:
                log_dist(f"Epoch {epoch} | Iter {iter_}/{len(train_loader)} | Iter Train Loss = {iter_loss:.3f} | Iter Time = {iter_time/1000:.6f} s", [0])
            iter_ += 1
        print_memory_stats()
        log_dist(f"Epoch {epoch} : Epoch Train Loss= {epoch_loss/len(train_loader):.3f} | Average Iter Time = {np.mean(iter_times)/1000:.6f} s", [0])
        
    log_dist("\n End training ...", [0])
