from axonn import axonn as ax
import torch.distributed as dist
import torch
from .communication import ForwardAllReduce, BackwardAllReduce, Drop


class Linear(torch.nn.Module):
    def __init__(self, in_features, out_features, transpose=False):
        super(Linear, self).__init__()
        self.inner_group = ax.comm_handle.inner_intra_layer_parallel_group
        self.outer_group = ax.comm_handle.outer_intra_layer_parallel_group

        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)

        if not transpose:
            assert in_features % self.inner_group_size == 0
            assert out_features % self.outer_group_size == 0
            self.local_in_features = in_features // self.inner_group_size
            self.linear = torch.nn.Linear(
                in_features=in_features // self.inner_group_size,
                out_features=out_features // self.outer_group_size,
            )
        else:
            assert out_features % self.inner_group_size == 0
            assert in_features % self.outer_group_size == 0
            self.local_in_features = in_features // self.outer_group_size
            self.linear = torch.nn.Linear(
                in_features=in_features // self.outer_group_size,
                out_features=out_features // self.inner_group_size,
            )

        self.transpose = transpose

    def forward(self, x):
        if not self.transpose:
            if x.size(-1) == self.local_in_features * self.inner_group_size:
                x = Drop.apply(x, self.inner_group)
            x = BackwardAllReduce.apply(x, self.outer_group)
            x = self.linear(x)
            x = ForwardAllReduce.apply(x, self.inner_group)
        else:
            if x.size(-1) == self.local_in_features * self.outer_group_size:
                x = Drop.apply(x, self.outer_group)
            x = BackwardAllReduce.apply(x, self.inner_group)
            x = self.linear(x)
            x = ForwardAllReduce.apply(x, self.outer_group)
        return x
