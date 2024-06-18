# Copyright 2021 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from . import config
import os
from typing import Any


def _get_prefix_for_checkpoint() -> str:
    """Create a unique folder-name for each tensor parallel rank.

    Returns:
        prefix (str): The unique folder name for each tensor parallel rank
    """
    row_tp_rank = config.intra_layer_row_parallel_rank
    column_tp_rank = config.intra_layer_column_parallel_rank
    depth_tp_rank = config.intra_layer_depth_parallel_rank
    return f"tp_row_{row_tp_rank}_col_{column_tp_rank}_depth_{depth_tp_rank}"


def save(
    state: object, checkpoint_folder: str, checkpoint_name: str, overwrite: bool = True
) -> None:
    """Save a tensor parallel model checkpoint. The model is saved in
    a sharded fashion inside the checkpoint_folder as
    checkpoint_folder/<shard_id>/checkpoint_name.pt, where the shard_id is
    generated uniquely for each rank using the _get_prefix_for_checkpoint()
    function.

    Arguments:
        state (object): saved object
        checkpoint_folder (str): the folder in which the checkpoint will be created
        checkpoint_name (str): filename for the checkpoint. This is suffixed
        with .pt automatically
        overwrite (bool): whether to overwrite an existing checkpoint

    """
    assert (
        config.G_inter == 1
    ), "axonn.checkpoint.save does not work with inter-layer parallelism"

    if config.data_parallel_rank == 0:
        checkpoint_folder = os.path.join(
            checkpoint_folder, _get_prefix_for_checkpoint()
        )
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        checkpoint_file = os.path.join(checkpoint_folder, f"{checkpoint_name}.pt")
        if os.path.exists(checkpoint_file) and not overwrite:
            raise ValueError(f"Checkpoint {checkpoint_file} already exists")
        torch.save(state, checkpoint_file)


def load(checkpoint_folder: str, checkpoint_name: str) -> Any:
    """Load a tensor parallel model checkpoint. It is assumed that the user 
    is loading a model saved with
    axonn.checkpoint.save. The model is loaded in a sharded fashion
    from the checkpoint_folder as 
    checkpoint_folder/<shard_id>/checkpoint_name.pt, where the shard_id is generated uniquely 
    for each rank using the _get_prefix_for_checkpoint() function.

    Arguments:
        checkpoint_folder (str): the folder in which the checkpoint will be created
        checkpoint_name (str): filename for the checkpoint. This is 
        suffixed with .pt automatically

    Returns:
        checkpoint (Any): the loaded checkpoint object

    """
    assert (
        config.G_inter == 1
    ), "axonn.checkpoint.load does not work with inter-layer parallelism"
    assert os.path.isdir(
        checkpoint_folder
    ), f"folder {checkpoint_folder} does not exist"
    checkpoint_file = os.path.join(
        checkpoint_folder, f"{_get_prefix_for_checkpoint()}_{checkpoint_name}.pt"
    )
    return torch.load(checkpoint_file)
