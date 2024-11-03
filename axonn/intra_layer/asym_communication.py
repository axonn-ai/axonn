# Copyright 2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.distributed as dist
from axonn import axonn as ax


def print_rank(msg):
    if dist.get_rank() == 0:
        print(f"{dist.get_rank()} | {msg}")


@torch.no_grad()
def gather_batch_sizes(local_batch_size, process_group=None):
    ax.get_timers().start("gather-batch-sizes")
    world_size = dist.get_world_size(process_group)
    local_batch_tensor = torch.tensor(local_batch_size, device="cuda")
    global_batch_tensor = torch.empty(
        (world_size), device="cuda", dtype=local_batch_tensor.dtype
    )
    dist.all_gather_into_tensor(
        global_batch_tensor, local_batch_tensor, group=process_group
    )

    global_batch_tensor = global_batch_tensor.cpu()
    ax.get_timers().stop("gather-batch-sizes")
    return global_batch_tensor


@torch.no_grad()
def _allgatherv(tensor, rank_local_batch_sizes, process_group=None):
    ax.get_timers().start("allgatherv")
    output_tensor_list = []
    for batch_size in rank_local_batch_sizes:
        shape = list(tensor.shape)
        shape[0] = batch_size.item()
        output_tensor_list.append(
            torch.empty(tuple(shape), device=tensor.device, dtype=tensor.dtype)
        )
    input_tensor_list = [tensor.contiguous() for _ in rank_local_batch_sizes]
    dist.all_to_all(output_tensor_list, input_tensor_list, group=process_group)
    ax.get_timers().stop("allgatherv")
    return torch.cat(output_tensor_list)


class Gatherv(torch.autograd.Function):
    """
    All gather activations with different batch sizes on each rank.
    For example if rank-0 has a tensor of shape [3,4], and rank-1 has a tensor
    of shape [8,4], then this function will return a tensor of [11,4] on each
    rank.
    """

    @staticmethod
    def symbolic(graph, input_, rank_local_batch_sizes, process_group=None):
        output = _allgatherv(input_, rank_local_batch_sizes, process_group)
        graph.rank_local_batch_sizes = rank_local_batch_sizes
        graph.process_group = process_group
        return output

    @staticmethod
    def forward(ctx, input_, rank_local_batch_sizes, process_group=None):
        ax.get_timers().start("extra-non-expert-communication")
        output = _allgatherv(input_, rank_local_batch_sizes, process_group)
        ctx.save_for_backward(rank_local_batch_sizes)
        # print_rank(f"Gatherv forward - {rank_local_batch_sizes}")
        ctx.process_group = process_group
        ax.get_timers().stop("extra-non-expert-communication")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print_rank("Start - Gatherv Back")
        rank = dist.get_rank(ctx.process_group)
        # print_rank(f"GatherVBack - rank = {rank}")
        (rank_local_batch_sizes,) = ctx.saved_tensors
        # print_rank("Gatherv back - retrieve from cache")
        # print(rank_local_batch_sizes)
        end = torch.sum(rank_local_batch_sizes[: rank + 1])
        start = end - rank_local_batch_sizes[rank]
        # print_rank(f"start={start} end={end}")
        grad_input = grad_output[start:end]
        # print_rank("End - GatherVBack")
        return grad_input, None, None


class Dropv(torch.autograd.Function):
    """
    Opposite of Gatherv operation.
    """

    @staticmethod
    def symbolic(graph, input_, rank_local_batch_sizes, process_group=None):
        rank = dist.get_rank(process_group)
        end = torch.sum(rank_local_batch_sizes[: rank + 1])
        start = end - rank_local_batch_sizes[rank]
        output = input_[start:end]
        graph.process_group = process_group
        return output

    @staticmethod
    def forward(ctx, input_, rank_local_batch_sizes, process_group=None):
        rank = dist.get_rank(process_group)
        end = torch.sum(rank_local_batch_sizes[: rank + 1])
        start = end - rank_local_batch_sizes[rank]
        output = input_[start:end]
        ctx.process_group = process_group
        ctx.save_for_backward(rank_local_batch_sizes)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ax.get_timers().start("extra-non-expert-communication")
        (rank_local_batch_sizes,) = ctx.saved_tensors
        # print_rank("Start - DropVBack")
        grad_input = _allgatherv(grad_output, rank_local_batch_sizes, ctx.process_group)
        ax.get_timers().stop("extra-non-expert-communication")
        # print_rank("End - DropVBack")
        return grad_input, None, None


@torch.no_grad()
def _gather_batch_scatter_channels(input_, rank_local_batch_sizes, process_group=None):
    # if input in GPU i is of shape [m_{i},...,k], and process group size is G
    # then this returns a tensor of [sum_{i}(m_{i}),....,k/G].
    ax.get_timers().start("gather-batch-scatter-channels")
    ax.get_timers().start("alltoallv")
    input_ = input_.contiguous()
    world_size = torch.distributed.get_world_size(process_group)
    send_tensors = list(torch.chunk(input_, world_size, dim=-1))
    send_tensors = [s.contiguous() for s in send_tensors]
    recv_tensors = []
    for i in range(world_size):
        shape = list(input_.shape)
        assert shape[-1] % world_size == 0
        shape[-1] = shape[-1] // world_size
        shape[0] = rank_local_batch_sizes[i].item()
        recv_tensors.append(
            torch.empty(tuple(shape), device="cuda", dtype=input_.dtype)
        )
    torch.distributed.all_to_all(recv_tensors, send_tensors, group=process_group)
    ax.get_timers().stop("alltoallv")
    ax.get_timers().stop("gather-batch-scatter-channels")
    return torch.cat(recv_tensors, dim=0)


@torch.no_grad()
def _gather_channels_scatter_batch(input_, rank_local_batch_sizes, process_group=None):
    # if input in GPU i is of shape [m,...,k/G], and process group size is G
    # then this returns a tensor of [m_{i},....,k],
    # where m_{i} = rank_local_batch_sizes[i]
    ax.get_timers().start("gather-channels-scatter-batch")
    ax.get_timers().start("alltoallv")
    input_ = input_.contiguous()
    world_size = torch.distributed.get_world_size(process_group)
    send_tensors = list(torch.split(input_, list(rank_local_batch_sizes), dim=0))
    send_tensors = [s.contiguous() for s in send_tensors]
    recv_tensors = []
    for i in range(world_size):
        shape = list(input_.shape)
        shape[-1] = shape[-1]
        shape[0] = rank_local_batch_sizes[dist.get_rank(process_group)].item()
        recv_tensors.append(
            torch.empty(tuple(shape), device="cuda", dtype=input_.dtype)
        )

    torch.distributed.all_to_all(recv_tensors, send_tensors, group=process_group)
    ax.get_timers().stop("alltoallv")
    ax.get_timers().stop("gather-channels-scatter-batch")
    return torch.cat(recv_tensors, dim=-1)


class GatherBatchScatterChannels(torch.autograd.Function):
    """
    if input in GPU i is of shape [m_{i},...,k], and process group size is G
    then this returns a tensor of [sum_{i}(m_{i}),....,k/G].
    """

    @staticmethod
    def symbolic(graph, input_, rank_local_batch_sizes, process_group=None):
        output = _gather_batch_scatter_channels(
            input_, rank_local_batch_sizes, process_group
        )
        graph.process_group = process_group
        graph.rank_local_batch_sizes = rank_local_batch_sizes
        return output

    @staticmethod
    def forward(ctx, input_, rank_local_batch_sizes, process_group=None):
        ax.get_timers().start("extra-non-expert-communication")
        output = _gather_batch_scatter_channels(
            input_, rank_local_batch_sizes, process_group
        )
        ctx.process_group = process_group
        ctx.save_for_backward(rank_local_batch_sizes)
        ax.get_timers().stop("extra-non-expert-communication")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ax.get_timers().start("extra-non-expert-communication")
        (rank_local_batch_sizes,) = ctx.saved_tensors
        # print_rank("Start - GBSC back")
        grad_input = _gather_channels_scatter_batch(
            grad_output, rank_local_batch_sizes, ctx.process_group
        )
        ax.get_timers().stop("extra-non-expert-communication")
        # print_rank("End - GBSC back")
        return grad_input, None, None


class GatherChannelsScatterBatch(torch.autograd.Function):
    """
    if input in GPU i is of shape [m,...,k/G], and process group size is G
    then this returns a tensor of [m_{i},....,k]
    where m_{i} = rank_local_batch_sizes[i]
    """

    @staticmethod
    def symbolic(graph, input_, rank_local_batch_sizes, process_group=None):
        output = _gather_channels_scatter_batch(
            input_, rank_local_batch_sizes, process_group
        )
        graph.process_group = process_group
        graph.rank_local_batch_sizes = rank_local_batch_sizes
        return output

    @staticmethod
    def forward(ctx, input_, rank_local_batch_sizes, process_group=None):
        ax.get_timers().start("extra-non-expert-communication")
        output = _gather_channels_scatter_batch(
            input_, rank_local_batch_sizes, process_group
        )
        ctx.process_group = process_group
        ctx.save_for_backward(rank_local_batch_sizes)
        ax.get_timers().stop("extra-non-expert-communication")
        return output

    @staticmethod
    def backward(ctx, grad_output):
        ax.get_timers().start("extra-non-expert-communication")
        (rank_local_batch_sizes,) = ctx.saved_tensors
        # print_rank("Start - GCSB  back")
        grad_input = _gather_batch_scatter_channels(
            grad_output, rank_local_batch_sizes, ctx.process_group
        )
        # print_rank("End - GCSB  back")
        ax.get_timers().stop("extra-non-expert-communication")
        return grad_input, None, None


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    ax.init(G_intra_r=dist.get_world_size())

    tensor = torch.randn(
        (dist.get_rank() + 5, 8),
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    # output, _ = GatherBatchScatterChannels.apply(tensor)
    # output.backward(output)
    # print(tensor - tensor.grad)

    output, rank_local_batch_sizes = Gatherv.apply(tensor)
    output = Dropv.apply(output, rank_local_batch_sizes)
    output.backward(output)
