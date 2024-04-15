import torch.distributed as dist
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import math

from axonn import axonn as ax
import axonn
from .communication import (
    Drop,
    Gather,
    _gather,
    _reduce_scatter,
)


def divide(a, b):
    assert a % b == 0
    return a // b


@torch.no_grad()
def extract_local_params_from_full_params(
    params, out_features_group, in_features_group, depth_group
):
    params = Drop.apply(params, in_features_group)
    params = Drop.apply(torch.t(params).contiguous(), out_features_group)
    params = torch.t(params).contiguous()
    params = Drop.apply(params.reshape(-1), depth_group)  # create 1D view
    return params


@torch.no_grad()
def initialize_params(
    out_features,
    in_features,
    out_features_group,
    in_features_group,
    depth_group,
    init_method,
    init_device="cuda",
):
    params = torch.empty((out_features, in_features), device=init_device)
    init_method(params)
    params = extract_local_params_from_full_params(
        params, out_features_group, in_features_group, depth_group
    ).cpu()
    return params


@torch.no_grad()
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
        depth_parallel_group,
        local_weight_shape,
        cache_weights,
        backward_comm_async,
        forward_comm_async,
    ):
        original_weight = weight
        weight = _gather(
            weight, dim=0, process_group=depth_parallel_group, cache=cache_weights
        )
        weight = weight.reshape(local_weight_shape)
        ctx.save_for_backward(input_, original_weight)
        ctx.backward_all_reduce_group = backward_all_reduce_group
        ctx.depth_parallel_group = depth_parallel_group
        ctx.backward_comm_async = backward_comm_async
        ctx.local_weight_shape = local_weight_shape
        if not forward_comm_async:
            output = input_.matmul(weight.t())
            dist.all_reduce(output, group=forward_all_reduce_group, async_op=False)
        else:
            assert input_.shape[0] % 2 == 0
            input_chunks = torch.chunk(input_, 2)  # each chunk is a view of the tensor
            output_shape = list(input_.shape)
            output_shape[-1] = weight.shape[0]
            outputs = []
            outputs.append(input_chunks[0].matmul(weight.t()))
            handle = dist.all_reduce(
                outputs[-1], group=forward_all_reduce_group, async_op=True
            )
            outputs.append(input_chunks[1].matmul(weight.t()))
            dist.all_reduce(outputs[-1], group=forward_all_reduce_group, async_op=False)
            handle.wait()  # this call might be unnecessary
            output = torch.cat(outputs)

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        weight = _gather(
            weight, dim=0, process_group=ctx.depth_parallel_group, cache=False
        )
        weight = weight.reshape(ctx.local_weight_shape)

        handle = None
        overlap_reduce_scatter = axonn.intra_layer.OVERLAP_REDUCE_SCATTER
        if dist.get_world_size(ctx.backward_all_reduce_group) > 1 or (
            not overlap_reduce_scatter
        ):
            grad_input, grad_weight = None, None

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

            grad_weight = grad_weight.reshape(-1)
            grad_weight = _reduce_scatter(
                grad_weight,
                dim=0,
                process_group=ctx.depth_parallel_group,
                overlap_comm=overlap_reduce_scatter,
            )

            if handle and ctx.backward_comm_async:
                handle.wait()
            if overlap_reduce_scatter:
                axonn.intra_layer.accumulate_later(original_weight, grad_weight)
                grad_weight = None  # weight gradients are not ready yet
            return grad_input, grad_weight, None, None, None, None, None, None, None
        else:
            grad_input, grad_weight = None, None

            if ctx.needs_input_grad[1]:
                grad_weight = (
                    grad_output.reshape(-1, grad_output.shape[-1])
                    .t()
                    .mm(input_.view(-1, input_.shape[-1]))
                ).reshape(-1)
                grad_weight = _reduce_scatter(
                    grad_weight,
                    dim=0,
                    process_group=ctx.depth_parallel_group,
                    overlap_comm=True,
                )
                axonn.intra_layer.accumulate_later(original_weight, grad_weight)
                grad_weight = None  # weight gradients are not ready yet

            if ctx.needs_input_grad[0]:
                grad_input = grad_output.matmul(weight)
            return grad_input, grad_weight, None, None, None, None, None, None, None


class Linear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        *args,
        transpose=False,
        bias=True,
        skip_bias_add=False,
        init_method=None,
        **kwargs
    ):
        super(Linear, self).__init__()
        self.inner_group = ax.comm_handle.inner_intra_layer_parallel_group
        self.outer_group = ax.comm_handle.outer_intra_layer_parallel_group
        self.depth_group = ax.comm_handle.depth_intra_layer_parallel_group

        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        self.depth_group_size = dist.get_world_size(self.depth_group)

        self.in_features = in_features
        self.out_features = out_features

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
                self.depth_group,
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
                self.depth_group,
                init_method,
            )

        self.weight = torch.nn.Parameter(initial_params, requires_grad=True)

        setattr(self.weight, "is_tensor_parallel", True)
        setattr(self.weight, "needs_gradient_sync", False)
        setattr(
            self.weight,
            "process_group_for_norm_reduction",
            ax.comm_handle.intra_layer_group,
        )

        if bias:
            self.bias = torch.nn.Parameter(
                torch.zeros(
                    self.local_out_features,
                )
            )
            setattr(self.bias, "is_tensor_parallel", True)
            setattr(self.bias, "needs_gradient_sync", True)
            if not transpose:
                setattr(
                    self.bias,
                    "process_group_for_norm_reduction",
                    ax.comm_handle.outer_intra_layer_parallel_group,
                )
            else:
                setattr(
                    self.bias,
                    "process_group_for_norm_reduction",
                    ax.comm_handle.inner_intra_layer_parallel_group,
                )
        else:
            self.bias = None

        self.transpose = transpose
        self.skip_bias_add = skip_bias_add
        self._old_load_from_state_dict = self._load_from_state_dict
        self._load_from_state_dict = self._modified_load_from_state_dict

    def get_output_feature_size(self):
        return self.local_out_features

    def forward(
        self,
        x,
        scatter_input=True,
        gather_output=True,
        cache_weights_in_all_gather=False,
    ):
        # gather weights from depth parallel group
        # reduce scatter in the backward pass

        weight = self.weight
        if not self.transpose:
            if scatter_input:
                x = Drop.apply(x, self.inner_group)
            x = AsyncLinear.apply(
                x,
                weight,
                self.inner_group,
                self.outer_group,
                self.depth_group,
                (self.local_out_features, self.local_in_features),
                cache_weights_in_all_gather,
                axonn.intra_layer.OVERLAP_ALL_REDUCE,
                False,
            )
            if gather_output:
                x = Gather.apply(x, self.outer_group)
        else:
            if scatter_input:
                x = Drop.apply(x, self.outer_group)

            x = AsyncLinear.apply(
                x,
                weight,
                self.outer_group,
                self.inner_group,
                self.depth_group,
                (self.local_out_features, self.local_in_features),
                cache_weights_in_all_gather,
                axonn.intra_layer.OVERLAP_ALL_REDUCE,
                False,
            )
            if gather_output:
                x = Gather.apply(x, self.inner_group)

        if self.bias is None:
            return x
        else:
            bias = self.bias
            if gather_output:
                bias = Gather.apply(
                    bias,
                    self.outer_group if not self.transpose else self.inner_group,
                )
            if self.skip_bias_add:
                return x, bias
            else:
                return x + bias

    def _is_full_weight_matrix(self, weight):
        return (
            weight.ndim == 2
            and weight.size(0) == self.out_features
            and weight.size(1) == self.in_features
        )

    def _is_sharded_weight_matrix(self, weight):
        return weight.ndim == 1 and weight.size(0) == divide(
            self.local_out_features * self.local_in_features, self.depth_group_size
        )

    @torch.no_grad()
    def _modified_load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        weight = (
            state_dict[prefix + "weight"] if prefix + "weight" in state_dict else None
        )

        if weight is not None:
            is_full_weight_matrix = self._is_full_weight_matrix(weight)
            is_sharded_weight_matrix = self._is_sharded_weight_matrix(weight)

            assert (
                is_full_weight_matrix or is_sharded_weight_matrix
            ), "This is neither a full checkpoint nor a sharded checkpoint"

            if is_full_weight_matrix:
                out_features_group, in_features_group = (
                    self.outer_group,
                    self.inner_group,
                )
                if self.transpose:
                    out_features_group, in_features_group = (
                        self.inner_group,
                        self.outer_group,
                    )
                weight = extract_local_params_from_full_params(
                    weight, out_features_group, in_features_group, self.depth_group
                )

            state_dict[prefix + "weight"] = weight

        if self.bias is not None:
            bias = (
                state_dict[prefix + "bias"] if prefix + "bias" in state_dict else None
            )
            if bias is not None:
                if bias.size(0) == self.out_features:
                    bias = Drop.apply(
                        bias,
                        self.outer_group if not self.transpose else self.inner_group,
                    )
                    state_dict[prefix + "bias"] = bias
                else:
                    assert (
                        bias.size(0) == self.local_out_features
                    ), "This is neither a full checkpoint nor a sharded checkpoint"

        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)
