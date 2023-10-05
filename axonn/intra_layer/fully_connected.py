from axonn import axonn as ax
import torch.distributed as dist
import torch
from .communication import ForwardAllReduce, BackwardAllReduce, Drop


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


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *args,
        transpose=False,
        skip_bias_add=False,
        init_method=None,
        **kwargs
    ):
        super(Linear, self).__init__()
        self.inner_group = ax.comm_handle.inner_intra_layer_parallel_group
        self.outer_group = ax.comm_handle.outer_intra_layer_parallel_group

        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)

        if not transpose:
            assert in_features % self.inner_group_size == 0
            assert out_features % self.outer_group_size == 0
            self.local_in_features = divide(in_features, self.inner_group_size)
            self.local_out_features = divide(out_features, self.outer_group_size)
            if init_method:
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
            if init_method:
                initial_params = initialize_params(
                    out_features,
                    in_features,
                    self.inner_group,
                    self.outer_group,
                    init_method,
                )

        self.linear = torch.nn.Linear(
            in_features=self.local_in_features,
            out_features=self.local_out_features,
            *args,
            **kwargs,
            bias=False
        )

        if init_method:
            self.linear.weight.data.copy_(initial_params)

        self.bias = torch.nn.Parameter(
            torch.zeros(
                self.local_out_features,
            )
        )
        self.transpose = transpose
        self.skip_bias_add = skip_bias_add

    def get_output_feature_size(self):
        return self.local_out_features

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

        if self.skip_bias_add:
            return x, self.bias
        else:
            return x + self.bias
