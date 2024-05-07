import torch
import torch.distributed as dist
import random
import numpy as np

def print_memory_stats():
    curr_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Current Memory Usage = {curr_memory:.2f} GB | Peak Memory Usage = {peak_memory:.2f} GB")
    else:
        print(f"Current Memory Usage = {curr_memory:.2f} GB | Peak Memory Usage = {peak_memory:.2f} GB")

def num_params(model):
    params = 0
    for param in model.parameters():
        params += param.numel()
    return params

def log_dist(msg, ranks=[]):
    assert dist.is_initialized()
    if dist.get_rank() in ranks:
        print(f"Rank {dist.get_rank()} : {msg}")

def report_local_and_global_params(net):
    assert dist.is_initialized()
    local_params = num_params(net)/1e6
    log_dist(f"Local Model Params  = {local_params:.3f} M", list(range(dist.get_world_size())))
    dist.barrier()
    total_params = torch.tensor([local_params], device='cuda')
    dist.all_reduce(total_params)
    log_dist(f"Total Model Params = {total_params.item():.3f} M", [0])


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
