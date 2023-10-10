from axonn import axonn as ax
import torch.distributed as dist
import torch
from .communication import ForwardAllReduce, BackwardAllReduce, Drop
from .utils import divide


@torch.no_grad()
def initialize_params(
    out_channels, in_channels, kernel_size, outer_group, inner_group, init_method
):
    params = torch.empty((out_channels, in_channels, kernel_size, kernel_size))
    init_method(params)
    params = Drop.apply(params, outer_group, 0)
    params = Drop.apply(params, inner_group, 1)
    return params


class Conv2d(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        *args,
        transpose=False,
        skip_bias_add=False,
        init_method=None,
        **kwargs
    ):
        super(Conv2d, self).__init__()

        if not transpose:
            self.inner_group = ax.comm_handle.inner_intra_layer_parallel_group
            self.outer_group = ax.comm_handle.outer_intra_layer_parallel_group
        else:
            self.outer_group = ax.comm_handle.inner_intra_layer_parallel_group
            self.inner_group = ax.comm_handle.outer_intra_layer_parallel_group

        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)

        self.in_channels = divide(in_channels, self.inner_group_size)
        self.out_channels = divide(out_channels, self.outer_group_size)

        self.conv = torch.nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            bias=False,
            **kwargs
        )

        if init_method:
            initial_params = initialize_params(
                out_channels,
                in_channels,
                kernel_size,
                self.outer_group,
                self.inner_group,
                init_method,
            )
            self.conv.weight.data.copy_(initial_params)

        self.skip_bias_add = skip_bias_add

        if not self.skip_bias_add:
            self.bias = torch.nn.Parameter(torch.zeros(self.out_channels))

    def forward(self, x):
        x = BackwardAllReduce.apply(x, self.outer_group)
        h = self.conv(x)
        h = ForwardAllReduce.apply(h, self.inner_group)
        if self.skip_bias_add:
            return h
        else:
            return h + self.bias.view(1, -1, 1, 1)
        return h
