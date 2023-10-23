import torch.distributed as dist
import torch


def _all_reduce(input_, process_group=None):
    if dist.get_world_size(process_group) > 1:
        dist.all_reduce(input_.contiguous(), group=process_group)
    return input_


def _drop(input_, dim, process_group=None):
    """Divide a tensor among the tensor parallel ranks"""
    if dist.get_world_size(process_group) == 1:
        return input_

    total_chunks = dist.get_world_size(process_group)
    this_chunk = dist.get_rank(process_group)
    assert input_.shape[dim] % total_chunks == 0
    chunk_size = input_.shape[dim] // total_chunks

    return torch.narrow(input_, dim, this_chunk * chunk_size, chunk_size)


def _gather(input_, dim, process_group=None):
    """Gather tensors and concatenate them along a dimension"""
    if dist.get_world_size(process_group) == 1:
        return input_

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
    def symbolic(graph, input_, process_group=None):
        return input_

    @staticmethod
    def forward(ctx, input_, process_group=None):
        ctx.process_group = process_group
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _all_reduce(grad_output, ctx.process_group), None


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