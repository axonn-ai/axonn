import torch

def print_memory_stats():
    curr_memory = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
    peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

    print(f"Current Memory Usage = {curr_memory:.2f} GB | Peak Memory Usage = {peak_memory:.2f} GB")

def num_params(model):
    params = 0
    for param in model.parameters():
        params += param.numel()
    return params
