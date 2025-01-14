# Copyright 2023-2024 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch.distributed as dist
import torch

from torch.autograd import Function

import math

from axonn import axonn as ax
from .communication import (
    Drop,
    Gather,
    _gather,
    _reduce_scatter,
)
import axonn.intra_layer.overlap_communication as overlap_communication
from .asym_communication import (
    Gatherv,
    Dropv,
    GatherBatchScatterChannels,
    GatherChannelsScatterBatch,
    gather_batch_sizes,
)
from typing import Optional, Sequence
from grouped_gemm import backend


# Wrapper for custom_fwd to handle different versions of PyTorch
def version_aware_custom_fwd(*args, **kwargs):
    version = torch.__version__.split(".")
    major_version = int(version[0])
    minor_version = int(version[1])
    if major_version > 2 or (major_version == 2 and minor_version >= 4):
        # For PyTorch version >= 2.4, pass device_type="cuda"
        return torch.amp.custom_fwd(device_type="cuda")(*args, **kwargs)
    else:
        # For PyTorch version < 2.4, no arguments are required
        return torch.cuda.amp.custom_fwd(*args, **kwargs)


# Wrapper for custom_bwd to handle different versions of PyTorch
def version_aware_custom_bwd(*args, **kwargs):
    version = torch.__version__.split(".")
    major_version = int(version[0])
    minor_version = int(version[1])
    if major_version > 2 or (major_version == 2 and minor_version >= 4):
        # For PyTorch version >= 2.4, pass device_type="cuda"
        return torch.amp.custom_bwd(device_type="cuda")(*args, **kwargs)
    else:
        # For PyTorch version < 2.4, no arguments are required
        return torch.cuda.amp.custom_bwd(*args, **kwargs)


def divide(a, b):
    assert a % b == 0
    return a // b


@torch.no_grad()
def extract_local_params_from_full_params(
    params, out_features_group, in_features_group, depth_group
):
    params = Drop.apply(params, in_features_group)
    params = Drop.apply(params, out_features_group, 1)
    params = Drop.apply(params.reshape(-1), depth_group)  # create 1D view
    return params


@torch.no_grad()
def initialize_params(
    num_experts,
    out_features,
    in_features,
    out_features_group,
    in_features_group,
    depth_group,
    init_method,
    init_device="cuda",
):
    params = torch.empty((num_experts, out_features, in_features), device=init_device)
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
    @version_aware_custom_fwd
    def forward(
        ctx,
        input_,
        weight,
        batch_sizes,
        num_experts,
        forward_all_reduce_group,
        backward_all_reduce_group,
        depth_parallel_group,
        local_weight_shape,
        cache_weights,
    ):
        ax.get_timers().start("forward-async")
        original_weight = weight
        weight = _gather(
            weight, dim=0, process_group=depth_parallel_group, cache=cache_weights
        )
        input_ = _gather(
            input_, dim=0, process_group=backward_all_reduce_group
        )
        # random batches ....
        batch_sizes = torch.tensor([input_.shape[0] // num_experts] * num_experts, dtype=torch.long)
        weight = weight.reshape(local_weight_shape)
        ctx.save_for_backward(input_, original_weight, batch_sizes)
        ctx.forward_all_reduce_group = forward_all_reduce_group
        ctx.backward_all_reduce_group = backward_all_reduce_group
        ctx.depth_parallel_group = depth_parallel_group
        ctx.shape = local_weight_shape
        ax.get_timers().start("compute")
        output = backend.gmm(input_, weight, batch_sizes, trans_b=True)
        ax.get_timers().stop("compute")
        #dist.all_reduce(output, group=forward_all_reduce_group, async_op=False)
        output = _reduce_scatter(output, 0, forward_all_reduce_group)
        ax.get_timers().stop("forward-async")
        return output

    @staticmethod
    @version_aware_custom_bwd
    def backward(ctx, grad_output):
        ax.get_timers().start("backward-async")
        input_, original_weight, batch_sizes = ctx.saved_tensors
        weight = _gather(
            original_weight, dim=0, process_group=ctx.depth_parallel_group, cache=False
        )
        grad_output = _gather( 
            grad_output, dim=0, process_group=ctx.forward_all_reduce_group
        )
        weight = weight.reshape(ctx.shape)
        handle = None
        overlap_reduce_scatter = overlap_communication.OVERLAP_REDUCE_SCATTER
        overlap_all_reduce = overlap_communication.OVERLAP_ALL_REDUCE

        grad_output = grad_output.contiguous()

        if dist.get_world_size(ctx.backward_all_reduce_group) > 1 or (
            not overlap_reduce_scatter
        ):
            grad_input, grad_weight = None, None

            if ctx.needs_input_grad[0]:
                ax.get_timers().start("compute")
                #grad_input = grad_output.matmul(weight)
                grad_input = backend.gmm(grad_output, weight, batch_sizes, trans_b=False)
                ax.get_timers().stop("compute")
                # handle = dist.all_reduce(
                #     grad_input,
                #     group=ctx.backward_all_reduce_group,
                #     async_op=overlap_all_reduce,
                # )
                rs_output = _reduce_scatter(grad_input, 
                                                     dim=0, 
                                                     process_group=ctx.backward_all_reduce_group,
                                                     overlap_comm=overlap_all_reduce,
                                                     register_handle=False)
                
                if overlap_all_reduce:
                    grad_input, handle = rs_output 
                else:
                    grad_input = rs_output

            if ctx.needs_input_grad[1]:
                ax.get_timers().start("compute")
                # grad_weight = (
                #     grad_output.reshape(-1, grad_output.shape[-1])
                #     .t()
                #     .mm(input_.view(-1, input_.shape[-1]))
                # )
                grad_weight = backend.gmm(grad_output.reshape(-1, grad_output.shape[-1]), 
                                          input_.view(-1, input_.shape[-1]), 
                                          batch_sizes, 
                                          trans_a=True, 
                                          trans_b=False)
                ax.get_timers().stop("compute")

                grad_weight = grad_weight.reshape(-1)
                grad_weight = _reduce_scatter(
                    grad_weight,
                    dim=0,
                    process_group=ctx.depth_parallel_group,
                    overlap_comm=overlap_reduce_scatter,
                )

            if handle and overlap_all_reduce:
                handle.wait()
            if overlap_reduce_scatter and ctx.needs_input_grad[1]:
                overlap_communication.accumulate_later(original_weight, grad_weight)
                grad_weight = None  # weight gradients are not ready yet
            ax.get_timers().stop("backward-async")
            return grad_input, grad_weight, None, None, None, None, None, None, None
        else:
            grad_input, grad_weight = None, None

            if ctx.needs_input_grad[1]:
                ax.get_timers().start("compute")
                grad_weight = backend.gmm(grad_output.reshape(-1, grad_output.shape[-1]), 
                                          input_.view(-1, input_.shape[-1]), 
                                          batch_sizes, 
                                          trans_a=True, 
                                          trans_b=False).reshape(-1)
                ax.get_timers().stop("compute")
                grad_weight = _reduce_scatter(
                    grad_weight,
                    dim=0,
                    process_group=ctx.depth_parallel_group,
                    overlap_comm=True,
                )
                overlap_communication.accumulate_later(original_weight, grad_weight)
                grad_weight = None  # weight gradients are not ready yet

            if ctx.needs_input_grad[0]:
                ax.get_timers().start("compute")
                #grad_input = grad_output.matmul(weight)
                grad_input = backend.gmm(grad_output, weight, batch_sizes, trans_b=False)
                ax.get_timers().stop("compute")
            ax.get_timers().stop("backward-async")
            return grad_input, grad_weight, None, None, None, None, None, None, None


class MoELinear(torch.nn.Module):
    def __init__(
        self,
        num_experts,
        in_features,
        out_features,
        *args,
        transpose=False,
        bias=True,
        skip_bias_add=False,
        init_method=None,
        expert_mode=False,
        tensor_parallel_dims: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super(MoELinear, self).__init__()

        # weights are shaped [num_experts, out_features, in_features]
        # in_features are distributed across self.inner_group (X tensor parallel group)
        # out_features are distributed across self.inner_group (Y tensor parallel group)
        # if transpose is true then X and Y are swapped
        if tensor_parallel_dims is not None and torch.distributed.get_rank() == 0:
            print(
                "Manually setting TP dims for a layer with shape",
                f" - {(in_features, out_features)} | tp-dims = {tensor_parallel_dims}",
            )
        self.inner_group, self.outer_group, self.depth_group = (
            ax.comm_handle.get_intra_layer_groups(tensor_parallel_dims)
        )
        if transpose:
            self.inner_group, self.outer_group = self.outer_group, self.inner_group

        # calculating the sizes of each tensor parallel process group
        self.inner_group_size = dist.get_world_size(self.inner_group)
        self.outer_group_size = dist.get_world_size(self.outer_group)
        self.depth_group_size = dist.get_world_size(self.depth_group)

        #print(self.inner_group_size, self.outer_group_size, self.depth_group_size, torch.distributed.get_rank())
        #exit()

        assert self.inner_group_size == 1 or self.outer_group_size == 1

        # these are the in and out features of the full global weight matrix
        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts

        # expert mode = True -> user needs to parallelize non-linear layers manually
        # expert mode = False -> non-linear layers are parallelized using
        #                        data parallelism
        #                        automatically by AxoNN. This does involve some
        #                        extra communication
        #                        at the beginning and end of each linear layer.
        self.expert_mode = expert_mode

        # init_method -> function to initialize the weight matrix
        if init_method is None:
            init_method = default_init_method

        # in_features should be divisible by inner_group_size
        assert in_features % self.inner_group_size == 0
        # in_features should be divisible by inner_group_size
        assert out_features % self.outer_group_size == 0
        # local_in_features - this is the number of in_features on each GPU
        self.local_in_features = divide(in_features, self.inner_group_size)
        # local_out_features - this is the number of out_features on each GPU
        self.local_out_features = divide(out_features, self.outer_group_size)
        # initialize the weight matrix and grab the local slice for each GPU
        initial_params = initialize_params(
            num_experts,
            out_features,
            in_features,
            self.outer_group,
            self.inner_group,
            self.depth_group,
            init_method,
        )
        # register the weight matrix as a trainable parameter.
        self.weight = torch.nn.Parameter(initial_params, requires_grad=True)

        # extra book-keeping for the weight tensor.
        # this is needed by AxoNN layer in the sync_gradients and
        # gradient clipping functions.
        setattr(self.weight, "is_tensor_parallel", True)
        setattr(self.weight, "needs_depth_parallel_gradient_sync", False)
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
            setattr(self.bias, "needs_depth_parallel_gradient_sync", True)
            if not transpose:
                setattr(
                    self.bias,
                    "process_group_for_norm_reduction",
                    self.outer_group,
                )
            else:
                setattr(
                    self.bias,
                    "process_group_for_norm_reduction",
                    self.inner_group,
                )
        else:
            self.bias = None

        self.skip_bias_add = skip_bias_add
        self._old_load_from_state_dict = self._load_from_state_dict
        self._load_from_state_dict = self._modified_load_from_state_dict

    def forward(
        self,
        x,
        batch_sizes,
        cache_weights_in_all_gather=False,
    ):
        ax.get_timers().start("forward-linear")
        original_shape_x = x.shape
        x = x.reshape(-1, x.shape[-1])
        #assert batch_sizes.sum() == x.shape[0] 

        weight = self.weight
        if not self.expert_mode and (self.inner_group_size * self.outer_group_size > 1):
            # extra communication to transition from pure data parallelism
            # to 4D hybrid parallelism
            if self.inner_group_size > 1:
                inner_group_batch_sizes = gather_batch_sizes(
                    x.shape[0], self.inner_group
                )
                x = GatherBatchScatterChannels.apply(
                    x, inner_group_batch_sizes, self.inner_group
                )
            if self.outer_group_size > 1:
                outer_group_batch_sizes = gather_batch_sizes(
                    x.shape[0], self.outer_group
                )
                x = Gatherv.apply(x, outer_group_batch_sizes, self.outer_group)

        x = AsyncLinear.apply(
            x,
            weight,
            batch_sizes,
            self.num_experts,
            self.inner_group,
            self.outer_group,
            self.depth_group,
            (self.num_experts, self.local_out_features, self.local_in_features),
            cache_weights_in_all_gather,
        )

        if not self.expert_mode and (self.inner_group_size * self.outer_group_size > 1):
            # extra communication to transition from 4D hybrid parallelism
            # to pure data parallelism
            if self.outer_group_size > 1:
                x = GatherChannelsScatterBatch.apply(
                    x, outer_group_batch_sizes, self.outer_group
                )
            if self.inner_group_size > 1:
                x = Dropv.apply(x, inner_group_batch_sizes, self.inner_group)

        x = x.reshape(-1, *original_shape_x[1:-1], x.shape[-1])

        if self.bias is None:
            ax.get_timers().stop("forward-linear")
            return x
        else:
            bias = self.bias
            if not self.expert_mode:
                bias = Gather.apply(bias, self.outer_group)
            ax.get_timers().stop("forward-linear")
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
                    bias = Drop.apply(bias, self.outer_group)
                    state_dict[prefix + "bias"] = bias
                else:
                    assert (
                        bias.size(0) == self.local_out_features
                    ), "This is neither a full checkpoint nor a sharded checkpoint"

        self._old_load_from_state_dict(state_dict, prefix, *args, **kwargs)
