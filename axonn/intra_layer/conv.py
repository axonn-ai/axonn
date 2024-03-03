from axonn import axonn as ax
import axonn
import torch.distributed as dist
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import torch
import math
from .communication import (
    ForwardAllReduce,
    BackwardAllReduce,
    Drop,
    Gather,
    ForwardGather_BackwardReduceScatter,
)
from .utils import divide


@torch.no_grad()
def initialize_params(
    out_channels,
    in_channels,
    kernel_size,
    outer_group,
    inner_group,
    depth_group,
    init_method,
    init_device="cuda",
):
    params = torch.empty(
        (out_channels, in_channels, kernel_size, kernel_size), device=init_device
    )
    init_method(params)
    params = Drop.apply(params, outer_group, 0)
    params = Drop.apply(params, inner_group, 1)
    params = Drop.apply(params.reshape(-1), depth_group)
    params = params.cpu()
    return params


@torch.no_grad()
def default_init_method(weight):
    return torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


class AsyncConv2d(Function):
    @staticmethod
    @custom_fwd
    def forward(
        ctx, 
        input_,
        weight, 
        stride, 
        padding,
        dilation, 
        groups,
        forward_all_reduce_group,
        backward_all_reduce_group,
        backward_comm_async,
        forward_comm_async,
    ):
        ctx.save_for_backward(input_, weight)
        ctx.backward_all_reduce_group = backward_all_reduce_group
        ctx.backward_comm_async = backward_comm_async
        ctx.conf = {
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
        }
        output = torch.nn.functional.conv2d(input_, weight, bias=None, stride=stride, padding=padding, dilation=dilation, groups=groups)
        dist.all_reduce(output, group=forward_all_reduce_group, async_op=False)
        return output
    
    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        backward_all_reduce_group = ctx.backward_all_reduce_group
        backward_comm_async = ctx.backward_comm_async
        stride = ctx.conf["stride"]
        padding = ctx.conf["padding"]
        dilation = ctx.conf["dilation"]
        groups = ctx.conf["groups"]

        assert groups == 1, "Only groups = 1 supported" 

        grad_input = grad_weight = None
        handle = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input_.shape, weight, grad_output, stride=stride, padding=padding, dilation=dilation, groups=groups)

            handle = dist.all_reduce(
                grad_input, 
                group=backward_all_reduce_group,
                async_op=backward_comm_async,
            )
        if ctx.needs_input_grad[1]:
            grad_weight = F.conv2d(input_, grad_out, padding=padding, dilation=stride, stride=dilation)
        
        if handle and ctx.backward_comm_async:
            handle.wait()
        return grad_input, grad_weight, None, None, None, None, None, None, None, None



class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        *args,
        transpose=False,
        bias=True,
        skip_bias_add=False,
        init_method=None,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    ):
        super(Conv2d, self).__init__()

        # For transpose, inner and outer groups are swapped
        if not transpose:
            self.inner_group = ax.comm_handle.inner_intra_layer_parallel_group
            self.outer_group = ax.comm_handle.outer_intra_layer_parallel_group
            self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group
        else:
            self.outer_group = ax.comm_handle.inner_intra_layer_parallel_group
            self.inner_group = ax.comm_handle.outer_intra_layer_parallel_group
            self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group

        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        self.depth_group_size = dist.get_world_size(self.depth_group)

        if init_method is None:
            init_method = default_init_method

        self.local_in_channels = divide(in_channels, self.inner_group_size)
        self.local_out_channels = divide(out_channels, self.outer_group_size)

        initial_params = initialize_params(
            out_channels,
            in_channels,
            kernel_size,
            self.outer_group,
            self.inner_group,
            self.depth_group,
            init_method,
        )

        self.weight = torch.nn.Parameter(initial_params, requires_grad=True)
        setattr(self.weight, "is_tensor_parallel", True)
        setattr(self.weight, "needs_gradient_sync", False)
        setattr(
            self.weight,
            "process_group_for_norm_reduction",
            ax.comm_handle.intra_layer_group,  # What is intra_layer_group?
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(self.local_out_channels), requires_grad=True
            )
            setattr(self.bias, "is_tensor_parallel", True)
            setattr(self.bias, "needs_gradient_sync", True)
            setattr(
                self.bias,
                "process_group_for_norm_reduction",
                self.outer_group,
            )
        else:
            self.bias = None

        self.kernel_size = kernel_size
        self.skip_bias_add = skip_bias_add
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

    def forward(
        self,
        x,
        scatter_input=True,
        gather_output=True,
        cache_weights_in_all_gather=False,
    ):
        # Gather weights from depth parallel group
        # TODO: We should make the OVERLAP_REDUCE_SCATTER flag part of axonn.axonn
        weight = ForwardGather_BackwardReduceScatter.apply(
            self.weight,
            self.depth_group,
            0,
            axonn.intra_layer.OVERLAP_REDUCE_SCATTER,
            cache_weights_in_all_gather,
        ).reshape(
            self.local_out_channels,
            self.local_in_channels,
            self.kernel_size,
            self.kernel_size,
        )

        if scatter_input:
            # Drop input across the in_channels dimension on the inner_group
            x = Drop.apply(x, self.inner_group, 1)
            # Drop input across the batch dimension on the depth_group
            x = Drop.apply(x, self.depth_group, 0)

        x = BackwardAllReduce.apply(x, self.outer_group)
        h = AsyncConv2d.apply(
            x,
            weight,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.inner_group,
            self.outer_group,
            axonn.intra_layer.OVERLAP_ALL_REDUCE,
            False
        )

        #h = torch.nn.functional.conv2d(
        #    x,
        #    weight,
        #    bias=None,
        #    stride=self.stride,
        #    padding=self.padding,
        #    dilation=self.dilation,
        #    groups=self.groups,
        #)

        if gather_output:
            # Gather input across the in_channels dimension on the inner_group
            h = Gather.apply(h, self.outer_group, 1)
            # Gather input across the batch dimension on the depth_group
            h = Gather.apply(h, self.depth_group, 0)

        if self.bias is None:
            return h
        else:
            bias = self.bias
            if gather_output:
                bias = Gather.apply(bias, self.outer_group)

            if self.skip_bias_add:
                return h, bias
            else:
                return h + bias.view(1, -1, 1, 1)
