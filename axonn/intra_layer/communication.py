# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.distributed as dist
import torch
import axonn.intra_layer.overlap_communication as overlap_communication
from axonn import axonn as ax


def _all_reduce(input_, process_group=None, overlap_comm=False):
    ax.get_timers().start("all-reduce")
    input_ = input_.contiguous()
    if dist.get_world_size(process_group) > 1:
        handle = dist.all_reduce(
            input_.contiguous(), group=process_group, async_op=overlap_comm
        )
        if overlap_comm:
            overlap_communication.register_handle(handle)
    ax.get_timers().stop("all-reduce")
    return input_


def _drop(input_, dim, process_group=None):
    """Divide a tensor among the tensor parallel ranks"""
    if dist.get_world_size(process_group) == 1:
        return input_
    ax.get_timers().start("drop")
    total_chunks = dist.get_world_size(process_group)
    this_chunk = dist.get_rank(process_group)
    assert input_.shape[dim] % total_chunks == 0
    chunk_size = input_.shape[dim] // total_chunks
    ax.get_timers().stop("drop")
    return torch.narrow(input_, dim, this_chunk * chunk_size, chunk_size)


def _gather(input_, dim, process_group=None, cache=False):
    """Gather tensors and concatenate them along a dimension"""
    if dist.get_world_size(process_group) == 1:
        return input_
    ax.get_timers().start("gather")
    if input_ in overlap_communication.weights_cache:
        output, handle = overlap_communication.retrieve_all_gathered_weight(
            input_, delete=not cache
        )
        if handle is not None:
            handle.wait()
            if cache:
                overlap_communication.weights_cache[input_][1] = None
    else:
        input_ = input_.contiguous()
        # Size and dimension.
        rank = dist.get_rank(process_group)

        tensor_list = [
            torch.empty_like(input_) for _ in range(dist.get_world_size(process_group))
        ]
        tensor_list[rank] = input_
        dist.all_gather(tensor_list, input_, group=process_group)

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=dim).contiguous()

        if cache:
            overlap_communication.weights_cache[input_] = output, None
    ax.get_timers().stop("gather")
    return output


def _reduce_scatter(input_, dim, process_group=None, overlap_comm=False):
    assert dim == 0, "reduce scatter only implemented for dim=0"

    if dist.get_world_size(process_group) == 1:
        return input_
    ax.get_timers().start("reduce-scatter")
    total_chunks = dist.get_world_size(process_group)
    assert input_.shape[dim] % total_chunks == 0
    tensor_shape = list(input_.shape)
    tensor_shape[dim] //= total_chunks
    output = torch.empty(
        tensor_shape, dtype=input_.dtype, device=torch.cuda.current_device()
    )

    ax.get_timers().start("reduce-scatter-dist")
    if hasattr(torch.distributed, "reduce_scatter_tensor"):
        handle = torch.distributed.reduce_scatter_tensor(
            output, input_, group=process_group, async_op=overlap_comm
        )
    else:
        handle = torch.distributed._reduce_scatter_base(
            output, input_, group=process_group, async_op=overlap_comm
        )

    ax.get_timers().stop("reduce-scatter-dist")
    if overlap_comm:
        overlap_communication.register_handle(handle)
    ax.get_timers().stop("reduce-scatter")
    return output


class ForwardAllReduce(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, process_group=None):
        return _all_reduce(input_, process_group)

    @staticmethod
    def forward(ctx, input_, process_group=None):
        return _all_reduce(input_, process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class BackwardAllReduce(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, process_group=None, overlap_comm=False):
        return input_

    @staticmethod
    def forward(ctx, input_, process_group=None, overlap_comm=False):
        ctx.process_group = process_group
        ctx.overlap_comm = overlap_comm
        ctx.input = input_
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _all_reduce(grad_output, ctx.process_group, ctx.overlap_comm)
        if not ctx.overlap_comm:
            return grad_input, None, None
        else:
            overlap_communication.accumulate_later(ctx.input, grad_input)
            return None, None, None


class Drop(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, process_group=None, dim=-1):
        return _drop(input_, dim=dim, process_group=process_group)

    @staticmethod
    def forward(ctx, input_, process_group=None, dim=-1):
        ctx.process_group = process_group
        ctx.dim = dim
        return _drop(input_, dim=dim, process_group=process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _gather(grad_output, dim=ctx.dim, process_group=ctx.process_group),
            None,
            None,
        )


class Gather(torch.autograd.Function):
    @staticmethod
    def symbolic(graph, input_, process_group=None, dim=-1):
        return _gather(input_, dim=dim, process_group=process_group)

    @staticmethod
    def forward(ctx, input_, process_group=None, dim=-1):
        ctx.process_group = process_group
        ctx.dim = dim
        return _gather(input_, dim=dim, process_group=process_group)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _drop(grad_output, dim=ctx.dim, process_group=ctx.process_group),
            None,
            None,
        )


class ForwardGather_BackwardReduceScatter(torch.autograd.Function):
    @staticmethod
    def symbolic(
        graph,
        input_,
        process_group=None,
        dim=0,
        overlap_comm=False,
        cache_all_gather=False,
    ):
        return _gather(input_, dim=dim, process_group=process_group)

    @staticmethod
    def forward(
        ctx,
        input_,
        process_group=None,
        dim=0,
        overlap_comm=False,
        cache_all_gather=False,
    ):
        assert dim == 0
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.overlap_comm = overlap_comm
        ctx.input = input_
        return _gather(
            input_, dim=dim, process_group=process_group, cache=cache_all_gather
        )

    @staticmethod
    def backward(ctx, grad_output):
        assert ctx.dim == 0
        grad_input = _reduce_scatter(
            grad_output,
            dim=ctx.dim,
            process_group=ctx.process_group,
            overlap_comm=ctx.overlap_comm,
        )
        if not ctx.overlap_comm:
            return (grad_input, None, None, None, None)
        else:
            overlap_communication.accumulate_later(ctx.input, grad_input)
            return None, None, None, None, None
