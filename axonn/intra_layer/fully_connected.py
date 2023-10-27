from axonn import axonn as ax
import torch.distributed as dist
import torch
from .communication import Drop, Gather
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import math

def divide(a, b):
    assert a % b == 0
    return a // b


@torch.no_grad()
def initialize_params(
    out_features, in_features, out_features_group, in_features_group, init_method
):
    params = torch.empty((out_features, in_features))
    init_method(params)
    params = Drop.apply(torch.t(params).contiguous(), out_features_group)
    params = torch.t(params).contiguous()
    params = Drop.apply(params, in_features_group)
    return params

class AsyncLinear(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input_,
        weight,
        forward_all_reduce_group,
        backward_all_reduce_group,
        backward_comm_async,
    ):
        ctx.save_for_backward(input_, weight)
        ctx.backward_all_reduce_group = backward_all_reduce_group
        ctx.backward_comm_async = backward_comm_async
        output = input_.matmul(weight.t())
        dist.all_reduce(output, group=forward_all_reduce_group, async_op=False)
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        handle = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
            handle = dist.all_reduce(
                grad_input,
                group=ctx.backward_all_reduce_group,
                async_op=ctx.backward_comm_async,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = (
                grad_output.view(-1, grad_output.shape[-1])
                .t()
                .mm(input_.view(-1, input_.shape[-1]))
            )
        if handle and ctx.backward_comm_async:
            handle.wait()
        return grad_input, grad_weight, None, None, None


def default_init_method(weight):
    return torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *args,
        transpose=False,
        skip_bias_add=False,
        init_method=None,
        async_comm_in_backward_pass=True,
        **kwargs
    ):
        super(Linear, self).__init__()
        self.inner_group = ax.comm_handle.inner_intra_layer_parallel_group
        self.outer_group = ax.comm_handle.outer_intra_layer_parallel_group

        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)

        self.async_comm_in_backward_pass = async_comm_in_backward_pass

        if init_method is None:
            init_method = default_init_method

        if not transpose:
            assert in_features % self.inner_group_size == 0
            assert out_features % self.outer_group_size == 0
            self.local_in_features = divide(in_features, self.inner_group_size)
            self.local_out_features = divide(out_features, self.outer_group_size)
            initial_params = initialize_params(
                out_features,
                in_features,
                self.outer_group,
                self.inner_group,
                init_method,
            )
        else:
            assert out_features % self.inner_group_size == 0
            assert in_features % self.outer_group_size == 0
            self.local_in_features = divide(in_features, self.outer_group_size)
            self.local_out_features = divide(out_features, self.inner_group_size)
            initial_params = initialize_params(
                out_features,
                in_features,
                self.inner_group,
                self.outer_group,
                init_method,
            )


        self.weight = torch.nn.Parameter(initial_params, requires_grad=True)

        setattr(self.weight, "is_tensor_parallel", True)

        self.bias = torch.nn.Parameter(
            torch.zeros(
                self.local_out_features,
            )
        )
        self.transpose = transpose
        self.skip_bias_add = skip_bias_add

    def get_output_feature_size(self):
        return self.local_out_features

    def forward(self, x, scatter_input=True, gather_output=True):
        if not self.transpose:
            if scatter_input:
                x = Drop.apply(x, self.inner_group)
            x = AsyncLinear.apply(
                x,
                self.weight,
                self.inner_group,
                self.outer_group,
                self.async_comm_in_backward_pass,
            )
            if gather_output:
                x = Gather.apply(x, self.outer_group)
        else:
            if scatter_input:
                x = Drop.apply(x, self.outer_group)
            x = AsyncLinear.apply(
                x,
                self.weight,
                self.outer_group,
                self.inner_group,
                self.async_comm_in_backward_pass,
            )
            if gather_output:
                x = Gather.apply(x, self.inner_group)
            
        bias = self.bias
        if gather_output:
            bias = Gather.apply(
                self.bias, self.outer_group if not self.transpose else self.inner_group
            )
        if self.skip_bias_add:
            return x, bias
        else:
            return x + bias
