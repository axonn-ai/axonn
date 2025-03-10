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

def recursive_doubling_allgather_mpi(output_tensor: torch.Tensor,
                                       input_tensor: torch.Tensor,
                                       group: Optional[MPI.Comm] = None,
                                       async_op: bool = False):
    """
    Performs a recursive doubling based all-gather on CUDA tensors using MPI point-to-point 
    Sendrecv operations.
    
    Each process starts with a 1D input_tensor (its local block of data) and the final output_tensor 
    is a 1D tensor of size (P * block_size) where the data from process i is stored in 
    output_tensor[i*block_size:(i+1)*block_size].
    
    This implementation assumes that the number of processes (P) is a power of 2.
    
    Parameters:
      output_tensor : torch.Tensor
          Pre-allocated tensor of shape (P * block_size,) on a CUDA device.
      input_tensor : torch.Tensor
          1D tensor of shape (block_size,) representing local data.
      group : Optional[MPI.Comm]
          MPI communicator; defaults to MPI.COMM_WORLD.
      async_op : bool
          Non-blocking operations are not supported in this implementation.
    """
    assert not async_op, "non-blocking primitives not supported"
    comm = MPI.COMM_WORLD if group is None else group
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Determine block size and ensure output_tensor is large enough.
    block_size = input_tensor.numel()
    assert output_tensor.numel() == size * block_size, "Output tensor has incorrect size"

    # Copy local data into the proper slot.
    output_tensor[rank * block_size : (rank + 1) * block_size].copy_(input_tensor)

    # Make sure any CUDA work is complete before we start communication.
    torch.cuda.current_stream().synchronize()

    # Recursive doubling: at each step, the segment size doubles.
    seg_size = 1  # number of blocks currently gathered (each process starts with 1 block)
    while seg_size < size:
        # The partner for this step is computed using a bitwise XOR.
        partner = rank ^ seg_size

        # Compute the start of the group of blocks (of size 2*seg_size)
        # that contains the current process’s rank.
        group_start = (rank // (2 * seg_size)) * (2 * seg_size)
        # Depending on whether this process is in the lower or upper half of the group,
        # decide which contiguous segment to send and where to place the received data.
        if rank < partner:
            # In the lower half: send the first seg_size blocks,
            # receive the next seg_size blocks.
            send_offset = group_start * block_size
            recv_offset = (group_start + seg_size) * block_size
        else:
            # In the upper half: send the upper seg_size blocks,
            # receive them into the lower seg_size slots.
            send_offset = (group_start + seg_size) * block_size
            recv_offset = group_start * block_size

        # Number of elements to send/receive.
        count = seg_size * block_size

        # Prepare a temporary tensor for the incoming data.
        tmp = torch.empty(count, dtype=output_tensor.dtype, device=output_tensor.device)

        # Synchronize CUDA streams before communication.
        torch.cuda.current_stream().synchronize()

        # Exchange the contiguous segment with the partner.
        comm.Sendrecv(sendbuf=output_tensor[send_offset:send_offset + count],
                      dest=partner, sendtag=0,
                      recvbuf=tmp, source=partner, recvtag=0)

        # Place the received data into the proper position in output_tensor.
        output_tensor[recv_offset:recv_offset + count].copy_(tmp)

        # Double the segment size for the next iteration.
        seg_size *= 2

    # At the end, output_tensor holds blocks from rank 0, 1, ..., size-1 in order.


def _all_gather(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False,
    use_recursive_doubling_in_mpi: bool = False
) -> Optional[Request]:

    # Case 1: torch.distributed.ProcessGroup
    if group is None or isinstance(group, dist.ProcessGroup):
        # Delegate to torch.distributed.all_gather
        request = dist.all_gather_into_tensor(output_tensor, input_tensor, group, async_op)
    # Case 2: mpi4py.MPI.Comm
    elif isinstance(group, MPI.Comm):
        # make sure that the cpu is synchronized with the current stream
        torch.cuda.current_stream().synchronize()
        if use_recursive_doubling_in_mpi:
            request = recursive_doubling_allgather_mpi(output_tensor, 
                                                       input_tensor, 
                                                       group, 
                                                       async_op)
        else:
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
    async_op: bool = False,
    use_recursive_doubling_in_mpi: bool = False):


    assert not async_op, "Non blocking version not implemented"

    assert input_tensor.dim() == 1 and output_tensor.dim() == 1, "all_gather_2D only admits 1D tensors"
    intra_node_group_size, inter_node_group_size = group.get_world_size()


    # Step-1 inter-node all-gather 
    output_intermediate = torch.empty(input_tensor.size(0) * inter_node_group_size, 
                                      device=input_tensor.device, 
                                      dtype=input_tensor.dtype)
    _all_gather(output_intermediate, 
                input_tensor, 
                group.get_outer_group(), 
                async_op=False,
                use_recursive_doubling_in_mpi=use_recursive_doubling_in_mpi)

    # Step-2 intra-node all-gather
    _all_gather(output_tensor, 
                output_intermediate, 
                group.get_inner_group(), 
                async_op=False,
                use_recursive_doubling_in_mpi=use_recursive_doubling_in_mpi)

    # Step-3 on device permutation
    output_splits = torch.split(output_tensor, split_size_or_sections=input_tensor.size(0)) 
    ordered_tensors = []
    for i in range(inter_node_group_size):
        idxes = list(np.arange(i, intra_node_group_size*inter_node_group_size, inter_node_group_size ))
        ordered_tensors.extend([output_splits[idx] for idx in idxes])
    output_tensor.copy_(torch.cat(ordered_tensors))



