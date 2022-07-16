import torch
from . import config
import os
from .axonn import model_params_fp16, model_params_fp32, model


def save_model_and_optimizer(model, optimizer, checkpoint_folder):
    inter_rank = config.inter_layer_parallel_rank
    intra_rank = config.intra_layer_parallel_rank
    data_rank = config.data_parallel_rank

    if intra_rank == 0 and data_rank == 0:
        model_path = os.path.join(checkpoint_folder, f"model_{inter_rank}.pt")
        optim_path = os.path.join(checkpoint_folder, f"optim_{inter_rank}.pt")
        torch.save(model.state_dict(), model_path)
        torch.save(optimizer.state_dict(), optim_path)


def load_model(model, checkpoint_folder):
    inter_rank = config.inter_layer_parallel_rank

    model_path = os.path.join(checkpoint_folder, f"model_{inter_rank}.pt")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    return model


def load_optimizer(optimizer, checkpoint_folder):
    inter_rank = config.inter_layer_parallel_rank
    optim_path = os.path.join(checkpoint_folder, f"optim_{inter_rank}.pt")
    optimizer.load_state_dict(torch.load(optim_path, map_location="cpu"))
    if model is not None:
        model_params_fp16.copy_(model_params_fp32)
    return optimizer
