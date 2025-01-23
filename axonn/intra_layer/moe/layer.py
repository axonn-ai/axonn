import torch.nn as nn 
from .linear import ColumnParallelMoE, RowParallelMoE 
from .routing import DroplessMoERouting 
import torch.nn.functional as F
from .communication import TensorParallelUnpermuteAndScatter
import torch.distributed as dist

class DroplessMoEMLP(nn.Module):
    def __init__(self, num_experts, hdim, idim, tp_size):
        super(DroplessMoEMLP, self).__init__()
        self.fc = ColumnParallelMoE(
                       num_experts, 
                       hdim, 
                       2*idim,
                       tensor_parallel_size=tp_size
        )
        self.proj = RowParallelMoE(
            num_experts, 
            idim, 
            hdim,
            tensor_parallel_size=tp_size
        )
        self.router = DroplessMoERouting(hdim, 
                 num_experts, 
                 top_k=1,
                 tensor_parallel_group=self.fc.tensor_parallel_group)
        self.tp_size = tp_size
        
    def forward(self, x):
        nd_shape = x.shape 
        x = x.reshape(-1, x.shape[-1])
        permuted_logits, sorted_indices, _, per_expert_token_counts = self.router(x)
        y = self.fc(permuted_logits, per_expert_token_counts)
        y1, y2 = y[..., ::2], y[..., 1::2]
        y = F.silu(y1) * y2 
        restore_shape = list(x.shape)
        restore_shape[0] *= self.tp_size
        y = self.proj(y, 
                      per_expert_token_counts) 
        y = TensorParallelUnpermuteAndScatter.apply(
             y, 
             sorted_indices, 
             restore_shape, 
             self.proj.tensor_parallel_group
         )
        y = y.reshape(*nd_shape[:-1], y.shape[-1])
        return y
