# write an all gather function that can use either nccl or mpi.
#  
# all_gather.py

import torch
import torch.distributed as dist
from mpi4py import MPI
from typing import List, Optional, Union
from .request import Request
from .process_groups import ProcessGroups
import numpy as np
from .utils import _torch_to_mpi

def _all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False
) -> Optional[Request]:

    # Case 1: torch.distributed.ProcessGroup
    if group is None or isinstance(group, dist.ProcessGroup):
        # Delegate to torch.distributed.all_gather
        request = dist.all_gather_into_tensor(output_tensor, input_tensor, group, async_op)
    # Case 2: mpi4py.MPI.Comm
    elif isinstance(group, MPI.Comm):
        # make sure that the cpu is synchronized with the current stream
        torch.cuda.current_stream().synchronize()
        if async_op:
            request = group.Iallgather(input_tensor.detach(), output_tensor)
        else:
            request = group.Allgather(input_tensor.detach(), output_tensor)
    else:
        raise TypeError(
            f"Unsupported group type: {type(group)}. "
            "Expected torch.distributed.ProcessGroup or mpi4py.MPI.Comm."
        )
    return request 

def all_gather_2D(output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[ProcessGroups] = None,
    async_op: bool = False):

    assert not async_op, "Non blocking version not implemented"

    assert input_tensor.dim() == 1 and output_tensor.dim() == 1, "all_gather_2D only admits 1D tensors"
    intra_node_group_size, inter_node_group_size = group.get_world_size()


    # Step-1 inter-node all-gather 
    output_intermediate = torch.empty(input_tensor.size(0) * inter_node_group_size, 
                                      device=input_tensor.device, 
                                      dtype=input_tensor.dtype)
    _all_gather(output_intermediate, input_tensor, group.get_outer_group(), async_op=False)

    # Step-2 intra-node all-gather
    _all_gather(output_tensor, output_intermediate, group.get_inner_group(), async_op=False)

    # Step-3 on device permutation
    output_splits = torch.split(output_tensor, split_size_or_sections=input_tensor.size(0)) 
    ordered_tensors = []
    for i in range(inter_node_group_size):
        idxes = list(np.arange(i, intra_node_group_size*inter_node_group_size, inter_node_group_size ))
        ordered_tensors.extend([output_splits[idx] for idx in idxes])
    output_tensor.copy_(torch.cat(ordered_tensors))


