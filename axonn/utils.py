# Copyright 2021-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from typing import List


def _coalesce_and_reassign(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Coalesce tensors into a flattened 1D tensor and reassign them to
    subtensors in this 1D tensor.

    TODO:- By creating a flat tensor first this doubles the gpu memory.
          Make this less memory consuming

    Arguments:
        tensors (List[torch.Tensor]): list of tensors to be coalesced

    Returns:
        flatenned_tensors (torch.tensor): the flattened tensor.

    """
    flattened_tensor = _flatten_dense_tensors(tensors)
    for old_tensor, new_tensor in zip(
        tensors, _unflatten_dense_tensors(flattened_tensor, tensors)
    ):
        old_tensor.data = new_tensor
    return flattened_tensor
