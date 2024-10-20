# Copyright 2022-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from . import config
import os


def get_prefix_for_checkpoint():
    row_tp_rank = config.intra_layer_row_parallel_rank
    column_tp_rank = config.intra_layer_column_parallel_rank
    depth_tp_rank = config.intra_layer_depth_parallel_rank
    return f"tp_row_{row_tp_rank}_col_{column_tp_rank}_depth_{depth_tp_rank}"


def save(state, checkpoint_folder, checkpoint_name, overwrite=True):
    if config.data_parallel_rank == 0:
        checkpoint_folder = os.path.join(checkpoint_folder, get_prefix_for_checkpoint())
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_file = os.path.join(checkpoint_folder, f"{checkpoint_name}.pt")
        if os.path.exists(checkpoint_file) and not overwrite:
            raise ValueError(f"Checkpoint {checkpoint_file} already exists")
        torch.save(state, checkpoint_file)


def load(state, checkpoint_folder, checkpoint_name):
    assert os.path.isdir(
        checkpoint_folder
    ), f"folder {checkpoint_folder} does not exist"
    checkpoint_file = os.path.join(
        checkpoint_folder, f"{get_prefix_for_checkpoint()}_{checkpoint_name}.pt"
    )
    torch.load(checkpoint_file)
    return state
