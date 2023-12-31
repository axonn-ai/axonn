import torch.distributed as dist
import torch
from torch.autograd import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import math

from axonn import axonn as ax
import axonn
from .communication import Drop, Gather,  _gather, _reduce_scatter, ForwardAllReduce, BackwardAllReduce


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
        depth_group,
        backward_comm_async,
        weight_shape,
        cache_all_gathered_weight
    ):
        weight = _gather(weight, 0, process_group=depth_group, cache=cache_all_gathered_weight)
        weight = weight.reshape(weight_shape)
        ctx.save_for_backward(input_, weight)
        ctx.backward_all_reduce_group = depth_group
        ctx.backward_comm_async = backward_comm_async
        output = input_.matmul(weight.t())
        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input_, weight = ctx.saved_tensors
        handle = None
        if ctx.needs_input_grad[1]:
            grad_weight = (
                grad_output.reshape(-1, grad_output.shape[-1])
                .t()
                .mm(input_.view(-1, input_.shape[-1]))
            )
            grad_weight, handle = _reduce_scatter(grad_weight.reshape(-1), 
                                                  0, 
                                                  process_group=ctx.backward_all_reduce_group, 
                                                  overlap_comm=ctx.backward_comm_async)
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.matmul(weight)
        if handle and ctx.backward_comm_async:
            handle.wait()
        return grad_input, grad_weight, None, None, None, None


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
        #weight = ForwardGather_BackwardReduceScatter.apply(
        #    self.weight,
        #    self.depth_group,
        #    0,
        #    axonn.intra_layer.OVERLAP_REDUCE_SCATTER,
        #    cache_weights_in_all_gather,
        #).reshape(self.local_out_features, self.local_in_features)

        if not self.transpose:
            if scatter_input:
                x = Drop.apply(x, self.inner_group)
                x = Drop.apply(x, self.depth_group, 0)
            x = BackwardAllReduce.apply(x, self.outer_group, False)
            x = AsyncLinear.apply(
                x,
                self.weight,
                self.depth_group,
                axonn.intra_layer.OVERLAP_ALL_REDUCE,
                (self.local_out_features, self.local_in_features),
                cache_weights_in_all_gather,
            )
            x = ForwardAllReduce.apply(x, self.inner_group)
            if gather_output:
                x = Gather.apply(x, self.outer_group)
                x = Gather.apply(x, self.depth_group, 0)
        else:
            if scatter_input:
                x = Drop.apply(x, self.outer_group)
                x = Drop.apply(x, self.depth_group, 0)
            x = BackwardAllReduce.apply(x, self.inner_group, False)
            x = AsyncLinear.apply(
                x,
                self.weight,
                self.depth_group,
                axonn.intra_layer.OVERLAP_ALL_REDUCE,
                (self.local_out_features, self.local_in_features),
                cache_weights_in_all_gather,
            )
            x = ForwardAllReduce.apply(x, self.outer_group)
            if gather_output:
                x = Gather.apply(x, self.inner_group)
                x = Gather.apply(x, self.depth_group, 0)

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
