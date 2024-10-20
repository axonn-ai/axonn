# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from axonn import axonn as ax
import axonn
import torch.distributed as dist
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
        setattr(self.weight, "needs_depth_parallel_gradient_sync", False)
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
            setattr(self.bias, "needs_depth_parallel_gradient_sync", True)
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
        h = torch.nn.functional.conv2d(
            x,
            weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        h = ForwardAllReduce.apply(h, self.inner_group)

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
