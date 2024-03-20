import torch.nn as nn
from axonn import axonn as ax
from axonn.intra_layer import Linear


def auto_parallelize(model):
    G_row = ax.config.G_intra_r
    G_col = ax.config.G_intra_c
    G_depth = ax.config.G_intra_d
    # Iterate through all modules in the model
    for name, module in model.named_modules():
        if isinstance(module, nn.Module):
            # Iterate through all child modules of each module
            for attr_name, attr_module in module.named_children():
                # Check if the module is a linear layer
                if isinstance(attr_module, nn.Linear):
                    # Check if layer is "parallelizable"
                    if (
                        (attr_module.out_features % G_row == 0)
                        and (attr_module.in_features % G_col == 0)
                        and (
                            (attr_module.out_features * attr_module.in_features)
                            / (G_row * G_col)
                            % G_depth
                            == 0
                        )
                    ):
                        # Replace the linear layer with Axonn's linear layer
                        setattr(
                            module,
                            attr_name,
                            Linear(attr_module.in_features, attr_module.out_features),
                        )
    return model
