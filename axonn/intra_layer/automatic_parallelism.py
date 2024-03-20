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


def auto_parallellize(model):
    G_row = ax.config.G_intra_r
    G_col = ax.config.G_intra_c
    G_depth = ax.config.G_intra_r
    for name, module in model.named_modules():
        if isinstance(module, nn.Module):
            for attr_name, attr_module in module.named_children():
                if isinstance(attr_module, nn.Linear):
                    if (attr_module.out_features % G_row == 0) and \
                       (attr_module.in_features % G_col == 0) and \
                       ((attr_module.out_features * attr_module.in_features) / (G_row * G_col) % G_depth == 0):
                        setattr(module, attr_name, Linear(attr_module.in_features, attr_module.out_features))
    return model
