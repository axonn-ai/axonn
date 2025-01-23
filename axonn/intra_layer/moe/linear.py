# Copyright 2025 Parallel Software and Systems Group, University of Maryland.
# See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import torch
import torch.nn as nn
from axonn import axonn as ax
import math 
try:
    from grouped_gemm import backend
except ImportError:
    from . import naive_gmm as backend
from axonn.intra_layer.communication import _gather, _reduce_scatter
from axonn.intra_layer import Drop
from .communication import TensorParallelUnpermuteAndScatter

@torch.no_grad()
def default_init_method(weight):
    return torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))

def init_params(init_fn, num_experts, local_output_dim, local_input_dim, fsdp_group):
    w = torch.empty(num_experts, local_output_dim, local_input_dim)
    w = init_fn(w)
    return Drop.apply(w.reshape(-1), fsdp_group, 0)

class SimpleFSDPLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, per_expert_token_counts, local_weight_shape, process_group):
        w_gathered = _gather(w, dim=0, process_group=process_group, cache=False).reshape(local_weight_shape)
        y = backend.gmm(x.to(torch.bfloat16), w_gathered.to(torch.bfloat16), per_expert_token_counts, trans_b=True).to(x.dtype)
        # print(x.shape, w_gathered.shape, y.shape)
        # exit()
        ctx.local_weight_shape = local_weight_shape 
        ctx.process_group = process_group 
        ctx.save_for_backward(w, x, per_expert_token_counts)
        return y
        
    @staticmethod
    def backward(ctx, grad_out):
        w, x, per_expert_token_counts = ctx.saved_tensors 
        w_gathered = _gather(w, dim=0, process_group=ctx.process_group, cache=False).reshape(ctx.local_weight_shape)
        
        # dw = dout.t() @ x - this can be overlapped with the previous line 
        # after that do a reduce scatter
        grad_w = backend.gmm(grad_out.to(torch.bfloat16), x.to(torch.bfloat16), per_expert_token_counts, trans_a=True, trans_b=False).reshape(-1).to(x.dtype)
        grad_w = _reduce_scatter(grad_w, dim=0, process_group=ctx.process_group)

        # dx = dout @ w
        grad_x = backend.gmm(grad_out.to(torch.bfloat16), w_gathered.to(torch.bfloat16), per_expert_token_counts, trans_b=False).to(x.dtype)
        return grad_x, grad_w, None, None, None

class ColumnParallelMoE(nn.Module):
    def __init__(self, num_experts, 
                       input_dim, 
                       output_dim, 
                       init_method=None,
                       tensor_parallel_size=1):
        super(ColumnParallelMoE, self).__init__() 
        global_fsdp_dim = ax.config.G_intra_d
        assert global_fsdp_dim % tensor_parallel_size == 0
        local_fsdp_dim = global_fsdp_dim // tensor_parallel_size

        _, self.tensor_parallel_group, self.fsdp_group = (
            ax.comm_handle.get_intra_layer_groups([tensor_parallel_size, 
                                                   1, 
                                                   local_fsdp_dim])
        )
        assert output_dim % tensor_parallel_size == 0
        self.local_input_dim = input_dim 
        self.local_output_dim = output_dim // tensor_parallel_size
        self.num_experts = num_experts
        assert self.num_experts * self.local_input_dim * self.local_output_dim % local_fsdp_dim == 0
        self.local_num_params = self.num_experts * self.local_input_dim * self.local_output_dim // local_fsdp_dim

        if init_method is None:
            init_method = default_init_method 

        params = init_params(init_method, num_experts, self.local_output_dim, self.local_input_dim, self.fsdp_group)
        self.weight = nn.Parameter(params, requires_grad=True)

    def forward(self, x, per_expert_token_counts):
        return SimpleFSDPLinear.apply(x, 
                                      self.weight, 
                                      per_expert_token_counts, 
                                      (self.num_experts, self.local_output_dim, self.local_input_dim), 
                                      self.fsdp_group)

class RowParallelMoE(nn.Module):
    def __init__(self, num_experts, 
                       input_dim, 
                       output_dim, 
                       init_method=None,
                       tensor_parallel_size=1):
        super(RowParallelMoE, self).__init__() 
        global_fsdp_dim = ax.config.G_intra_d
        assert global_fsdp_dim % tensor_parallel_size == 0
        local_fsdp_dim = global_fsdp_dim // tensor_parallel_size
        

        _, self.tensor_parallel_group, self.fsdp_group = (
            ax.comm_handle.get_intra_layer_groups([tensor_parallel_size, 
                                                   1, 
                                                   global_fsdp_dim // tensor_parallel_size])
        )
        
        assert input_dim % tensor_parallel_size == 0
        self.local_input_dim = input_dim // tensor_parallel_size
        self.local_output_dim = output_dim 
        self.num_experts = num_experts
        assert self.num_experts * self.local_input_dim * self.local_output_dim % local_fsdp_dim == 0
        self.local_num_params = self.num_experts * self.local_input_dim * self.local_output_dim // local_fsdp_dim
        
        if init_method is None:
            init_method = default_init_method 

        params = init_params(init_method, num_experts, self.local_output_dim, self.local_input_dim, self.fsdp_group)
        self.weight = nn.Parameter(params, requires_grad=True)

    def forward(self, x, per_expert_token_counts): #sorted_indices_tp, restore_shape, probs=None):
        return SimpleFSDPLinear.apply(x, 
                                      self.weight, 
                                      per_expert_token_counts, 
                                      (self.num_experts, self.local_output_dim, self.local_input_dim), 
                                      self.fsdp_group)
        # if probs is not None:
        #     permuted_partial_output = permuted_partial_output * probs
        # output = TensorParallelUnpermuteAndScatter.apply(
        #     permuted_partial_output, 
        #     sorted_indices_tp, 
        #     restore_shape, 
        #     self.tensor_parallel_group
        # )
       # return output
        
