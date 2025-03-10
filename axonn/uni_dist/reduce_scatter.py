import torch
import torch.distributed as dist
from mpi4py import MPI
from typing import List, Optional, Union
from .request import Request
from .process_groups import ProcessGroups
import numpy as np
from .utils import _torch_to_mpi
import math

def recursive_halving_reduce_scatter_mpi(output_tensor: torch.Tensor,
                                    input_tensor: torch.Tensor,
                                    group: Optional[MPI.Comm] = None,
                                    async_op: bool = False,
                                    op=torch.add,):
    """
    Performs a reduce-scatter using arithmetic offsets.
    
    Each process starts with a 1D input_tensor of size (P * block_size),
    which is viewed as P contiguous blocks in natural order. The goal is to reduce
    (using op, e.g. torch.add) the j-th block (over all processes) so that at the end,
    process j ends with the fully reduced block j.
    
    Instead of using bitwise XOR to pick partners (which naturally produces a bit-reversed
    ordering), this algorithm uses arithmetic offsets:
      - In round 1, each process exchanges with the partner at rank ± (p/2)
      - In round 2, the offset is (p/4)
      - etc.
    
    At each round the process splits its current buffer (which initially has p blocks)
    into two equal halves. Then:
      - If the process’s position within its current group is in the lower half, it keeps
        the lower half (which should contain the blocks destined for lower-ranked processes)
        and sends the upper half.
      - Otherwise it keeps the upper half and sends the lower half.
    
    After log2(P) rounds, only one block remains—and it is naturally ordered (process i gets block i).
    
    Assumes that P (the number of processes) is a power of 2.
    """
    assert not async_op, "Non-blocking operations not supported"
    comm = MPI.COMM_WORLD if group is None else group
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    total_elems = input_tensor.numel()
    assert total_elems % size == 0, "Input tensor size must be divisible by number of processes"
    block_size = total_elems // size

    # Reshape the input tensor into a 2D tensor of shape (P, block_size)
    buf = input_tensor.view(size, block_size).clone()

    # The algorithm runs for log2(size) rounds.
    rounds = int(math.log2(size))
    for r in range(rounds):
        # At round r, we imagine that the current "group" size is: group_size = size / (2^r)
        # and we split our current buffer (which has exactly group_size blocks) into two equal halves.
        group_size = buf.size(0)  # This is the number of blocks we still hold.
        half = group_size // 2

        # To decide which half to keep, we use the process's global rank.
        # In round 1 (group_size == size), a natural choice is:
        #   - if rank < size/2: keep lower half; else keep upper half.
        # In subsequent rounds, we use the remainder of rank modulo the current group size.
        if (rank % (size // (2**r))) < (size // (2**(r+1))):
            # Lower half: keep lower half, send upper half.
            partner = (rank + half) % size
            send_data = buf[half:].clone()
            recv_buf = torch.empty((half, block_size), dtype=buf.dtype, device=buf.device)
            torch.cuda.current_stream().synchronize()
            comm.Sendrecv(sendbuf=send_data, dest=partner, sendtag=r,
                          recvbuf=recv_buf, source=partner, recvtag=r)
            # Combine the received upper half into our lower half.
            buf[:half] = op(buf[:half], recv_buf)
            # Discard the upper half.
            buf = buf[:half]
        else:
            # Upper half: keep upper half, send lower half.
            partner = (rank - half) % size
            send_data = buf[:half].clone()
            recv_buf = torch.empty((half, block_size), dtype=buf.dtype, device=buf.device)
            torch.cuda.current_stream().synchronize()
            comm.Sendrecv(sendbuf=send_data, dest=partner, sendtag=r,
                          recvbuf=recv_buf, source=partner, recvtag=r)
            # Combine the received lower half into our upper half.
            buf[half:] = op(buf[half:], recv_buf)
            buf = buf[half:]
    # At the end, buf has exactly one block—the reduced result for the block this process is responsible for.
    output_tensor.copy_(buf.view(-1))

def ring_reduce_scatter_mpi(output_tensor: torch.Tensor,
                            input_tensor: torch.Tensor,
                            group: Optional[MPI.Comm] = None,
                            async_op: bool = False, 
                            op=torch.add):
    """
    Performs a ring-based reduce-scatter using torch tensors.
    
    Each process holds a 2D tensor 'sendbuf' of shape (P, N), where P is the number 
    of processes and N is the block size. Each row of the tensor corresponds to a block.
    The goal is to reduce the i-th block (row) from all processes (using the provided op)
    so that at the end, each process gets the fully reduced block corresponding to its 
    assigned segment.
    
    The algorithm performs P-1 steps. At each step, each process:
      - Sends a designated block to its right neighbor.
      - Receives a block from its left neighbor.
      - Immediately reduces the received block into its local copy.
      
    After the loop, the fully reduced block is at index equal to the process’s rank.
    """
    assert not async_op, "non-blocking primitives not supported"
    comm = MPI.COMM_WORLD if group is None else group
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    assert input_tensor.dim() == 1, "input tensor should be 1D"
    assert output_tensor.dim() == 1, "output tensor should be 1D"

    sendbuf = input_tensor.view(size, -1)
    assert sendbuf.size(0) == size, "sendbuf must have one block per process"
    block_size = sendbuf.size(1)
    
    # Work on a local copy of the buffer.
    buf = sendbuf.clone()
    tmp = torch.empty(block_size, dtype=sendbuf.dtype, device=sendbuf.device)
    
    # Perform size-1 steps of the ring reduce-scatter.
    for step in range(size - 1):
        # Adjusted indices: subtract one extra so that the final index is the natural rank.
        send_idx = (rank - step - 1) % size
        recv_idx = (rank - step - 2) % size
        
        # Identify neighbors in the ring.
        dest = (rank + 1) % size
        source = (rank - 1 + size) % size
        
        # Copy the block to be sent.
        send_data = buf[send_idx].clone()
        
        # Send the designated block to the right and receive a block from the left.
        torch.cuda.current_stream().synchronize()
        comm.Sendrecv(sendbuf=send_data, dest=dest, sendtag=0,
                      recvbuf=tmp, source=source, recvtag=0)
        
        # Reduce the received block into our local block.
        buf[recv_idx] = op(buf[recv_idx], tmp)
    
    # Now, the fully reduced block is at index equal to rank.
    output_tensor.copy_(buf[rank])


def _reduce_scatter(
    output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[Union[dist.ProcessGroup, MPI.Comm]] = None,
    async_op: bool = False,
    use_recursive_halving_in_mpi: bool = False
) -> Optional[Request]:

    # Case 1: torch.distributed.ProcessGroup
    if group is None or isinstance(group, dist.ProcessGroup):
        # Delegate to torch.distributed.all_gather
        request = dist.reduce_scatter_tensor(output_tensor, 
                                             input_tensor, 
                                             group=group, 
                                             async_op=async_op)
    # Case 2: mpi4py.MPI.Comm
    elif isinstance(group, MPI.Comm):
        # make sure that the cpu is synchronized with the current stream
        if use_recursive_halving_in_mpi:
            request = recursive_halving_reduce_scatter_mpi(output_tensor, input_tensor, group, async_op)
        else:
            request = ring_reduce_scatter_mpi(output_tensor, input_tensor, group, async_op)
    else:
        raise TypeError(
            f"Unsupported group type: {type(group)}. "
            "Expected torch.distributed.ProcessGroup or mpi4py.MPI.Comm."
        )
    return request 

def reduce_scatter_2D(output_tensor: torch.Tensor,
    input_tensor: torch.Tensor,
    group: Optional[ProcessGroups] = None,
    async_op: bool = False,
    use_recursive_halving_in_mpi=False):

    assert not async_op, "Non blocking version not implemented"

    assert input_tensor.dim() == 1 and output_tensor.dim() == 1, "all_gather_2D only admits 1D tensors"
    intra_node_group_size, inter_node_group_size = group.get_world_size()

    # step 1: on-device permutation 
    world_size = inter_node_group_size * intra_node_group_size
    output_msg_size = input_tensor.size(0) // world_size
    input_splits = torch.split(input_tensor, split_size_or_sections=output_msg_size)
    permuted_tensors = []
    for i in range(intra_node_group_size):
        idxes = list(np.arange(i, world_size, intra_node_group_size))
        permuted_tensors.extend([input_splits[idx] for idx in idxes])
    input_permuted = torch.cat(permuted_tensors)

    # Step-2 intra-node reduce-scatter
    output_intermediate = torch.empty(input_permuted.size(0) // intra_node_group_size, 
                                      device=input_tensor.device, 
                                      dtype=input_tensor.dtype)
    _reduce_scatter(output_intermediate, 
                    input_permuted, 
                    group.get_inner_group(), 
                    async_op=False, 
                    use_recursive_halving_in_mpi=use_recursive_halving_in_mpi)

    # Step-3 inter-node 
    _reduce_scatter(output_tensor, 
                    output_intermediate, 
                    group.get_outer_group(), 
                    async_op=False,
                    use_recursive_halving_in_mpi=use_recursive_halving_in_mpi)


