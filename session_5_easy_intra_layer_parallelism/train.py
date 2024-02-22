import torch
import torch.nn as nn
import torchvision
import sys
import os
from torchvision import transforms
import numpy as np
from axonn import axonn as ax
from axonn.intra_layer import Linear
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
from model.fc_net_sequential import FC_Net
from utils import print_memory_stats, num_params, log_dist
from args import create_parser
import torch.fx
from torch.fx import symbolic_trace


NUM_EPOCHS=2
PRINT_EVERY=200

torch.manual_seed(0)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    torch.distributed.init_process_group() 

    ## Step 1 - Initialize AxoNN
    ax.init(
                G_data=args.G_data,
                G_inter=1,
                G_intra_r=args.G_intra_r,
                G_intra_c=args.G_intra_c,
                mixed_precision=True,
                fp16_allreduce=True,
            )

    log_dist('initialized AxoNN', ranks=[0])

    augmentations = transforms.Compose(
        [
            transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

    ## Step 2 - Create dataset with augmentations
    train_dataset = torchvision.datasets.MNIST(
        root=args.data_dir, train=True, transform=augmentations
    )

    ## Step 3 - Create dataloader using AxoNN
    train_loader = ax.create_dataloader(
        train_dataset,
        args.batch_size,
        args.micro_batch_size,
        num_workers=1,
    )

    model = FC_Net(num_layers=4, input_size=256, hidden_size=2048, output_size=10)
    symbolic_traced:torch.fx.GraphModule = symbolic_trace(model)
    print(symbolic_traced.graph)
    modules_to_replace = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            modules_to_replace.append((name, module))

    for name, _ in modules_to_replace:
        setattr(model, name, Linear(module.in_features, module.out_features))

#    symbolic_traced = torch.fx.symbolic_trace(model)

#    for name, module in model.named_modules():
#        if isinstance(module, Linear):
#            print("Found custom layer", name)

    print(model)

#    exit(1) 





    ## Step 4 - Create Neural Network 
    net = FC_Net(args.num_layers, args.image_size**2, args.hidden_size, 10).cuda()
    params = num_params(net) / 1e9

    ## Step 5 - Create Optimizer 
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    ## Step 6 - register model and optimizer with AxoNN
    ## This creates the required data structures for
    ## mixed precision
  #  net, optimizer = ax.register_model_and_optimizer(net, optimizer)

    ## Step 7 - Create Loss Function and register it
    loss_fn = torch.nn.CrossEntropyLoss()
   # ax.register_loss_fn(loss_fn)

    ## Step 8 - Train
    start_event = torch.cuda.Event(enable_timing=True)
    stop_event = torch.cuda.Event(enable_timing=True)

    log_dist(f"Model Params = {num_params(net)*ax.config.G_intra/1e9} B", [0])
    log_dist(f"Start Training with AxoNN's Intra-Layer Parallelism", [0])

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        iter_ = 0
        iter_times = []
        for img, label in train_loader:
            start_event.record()
            optimizer.zero_grad()
            img = img.cuda()
            label = label.cuda()
          #  iter_loss = ax.run_batch(img, label)
            output = net(img)
            iter_loss = loss_fn(output, label)

            # Backawrd Pass
            iter_loss.backward()

 
            optimizer.step()

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
        log_dist(f"End Training ...", [0])
