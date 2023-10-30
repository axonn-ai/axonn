from axonn import axonn as ax
import torch.distributed as dist
import torch
from .communication import Drop, Gather
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import math
import itertools


def divide(a, b):
    assert a % b == 0
    return a // b


def extract_local_params_from_full_params(
    full_params, out_features_group, in_features_group
):
    params = Drop.apply(torch.t(full_params).contiguous(), out_features_group)
    params = torch.t(params).contiguous()
    params = Drop.apply(params, in_features_group)
    return params


@torch.no_grad()
def initialize_params(
    out_features, in_features, out_features_group, in_features_group, init_method
):
    params = torch.empty((out_features, in_features))
    init_method(params)
    params = extract_local_params_from_full_params(
        params, out_features_group, in_features_group
    )
    return params


def default_init_method(weight):
    return torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))


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
                grad_output.reshape(-1, grad_output.shape[-1])
                .t()
                .mm(input_.view(-1, input_.shape[-1]))
            )
        if handle and ctx.backward_comm_async:
            handle.wait()
        return grad_input, grad_weight, None, None, None


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

        self.in_features = in_features
        self.out_features = out_features

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
        self._old_load_from_state_dict = self._load_from_state_dict
        self._load_from_state_dict = self._modified_load_from_state_dict

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

    def _get_keys_for_state_dict(self, prefix):
        persistent_buffers = {
            k: v
            for k, v in self._buffers.items()
            if k not in self._non_persistent_buffers_set
        }
        local_name_params = itertools.chain(
            self._parameters.items(), persistent_buffers.items()
        )
        local_state = {k: v for k, v in local_name_params if v is not None}
        return [prefix + name for name, param in local_state.items()]

    def _is_full_weight_matrix(self, weight):
        return (weight.size(0) == self.out_features) and (
            weight.size(1) == self.in_features
        )

    def _is_sharded_weight_matrix(self, weight):
        return (weight.size(0) == self.local_out_features) and (
            weight.size(1) == self.local_in_features
        )

    @torch.no_grad()
    def _modified_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        keys = self._get_keys_for_state_dict(prefix)
        # this is very brittle at the moment
        # if the weight or bias is missing from the state dict this will error
        weight, bias = state_dict[keys[0]], state_dict[keys[1]]
        is_full_weight_matrix = self._is_full_weight_matrix(weight)
        is_sharded_weight_matrix = self._is_sharded_weight_matrix(weight)

        assert (
            is_full_weight_matrix or is_sharded_weight_matrix
        ), "This is neither a full checkpoint or a sharded checkpoint"

        if is_full_weight_matrix:
            out_features_group, in_features_group = self.outer_group, self.inner_group
            if self.transpose:
                out_features_group, in_features_group = (
                    self.inner_group,
                    self.outer_group,
                )
            weight = extract_local_params_from_full_params(
                weight, out_features_group, in_features_group
            )
            bias = Drop.apply(bias, out_features_group)
            state_dict[keys[0]] = weight
            state_dict[keys[1]] = bias

        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)
