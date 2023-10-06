from axonn import axonn as ax
import torch.distributed as dist
import torch
from .communication import ForwardAllReduce, BackwardAllReduce, Drop

class Conv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, *args, transpose=False, **kwargs):
        super(Conv2d, self).__init__()
        self.inner_group = ax.comm_handle.inner_intra_layer_parallel_group
        self.outer_group = ax.comm_handle.outer_intra_layer_parallel_group

        if transpose:
            ordered_groups = [self.outer_group, self.inner_group]
        else:
            ordered_groups = [self.inner_group, self.outer_group]

        self.group_sizes = [dist.get_world_size(group=group) for group in ordered_groups]
        self.ordered_groups = ordered_groups
        self.in_channels, self.out_channels = in_channels, out_channels
        

        assert in_channels % self.group_sizes[0] == 0
        assert out_channels % self.group_sizes[1] == 0

        self.conv = torch.nn.Conv2d(
            in_channels=in_channels // self.group_sizes[0],
            out_channels=out_channels // self.group_sizes[1],
            kernel_size=kernel_size,
            **kwargs)


    def forward(self, x):
        x = BackwardAllReduce.apply(x, self.ordered_groups[1])
        h = self.conv(x)
        h = ForwardAllReduce.apply(h, self.ordered_groups[0])
        return h
