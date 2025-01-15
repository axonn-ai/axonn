import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from axonn.intra_layer import Gather, Drop
from .communication import TensorParallelGatherAndPermute


@torch.no_grad()
def create_permutation_indices(routing_map, tensor_parallel_group):
    """
    routing_map - [batch_size, num_experts] - 2D boolean tensor denoting routing decisions 
    tensor_parallel_group - tensor parallel proces group
    """
    num_experts = routing_map.shape[1]
    
    # change shape to [num_experts, batch_size]
    routing_map_t = routing_map.bool().T.contiguous()

    # gather global routing map across all tensor parallel ranks
    routing_map_t_tp = Gather.apply(routing_map_t, tensor_parallel_group, 1)
    
    # Gather the token indices per expert
    expert_token_indices = []
    for e in range(num_experts):
        # if we were to concatenate all tokens on all tp ranks then indices_e 
        # would denote the token indices where are mapped to expert e
        indices_e = torch.where(routing_map_t_tp[e])[0]
        expert_token_indices.append(indices_e)
    
    # Concatenate them in expert order
    sorted_indices = torch.cat(expert_token_indices, dim=0)
    num_tokens_per_expert = routing_map_t_tp.sum(dim=1) 
    return sorted_indices, num_tokens_per_expert
    


class DroplessMoERouting(nn.Module):
    def __init__(self, 
                 input_dim, 
                 num_experts, 
                 top_k,
                 tensor_parallel_group):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.tensor_parallel_group = tensor_parallel_group
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        """
        1) Compute gating probabilities.
        2) Select top_k experts per token.
        3) Build routing_map (token->expert).
        4) permute(...) -> reorder tokens for experts
        """

        batch_size, hidden = x.shape

        # 1) Compute gating
        logits = self.gate(x)  # [batch_size, num_experts]
        probs = F.softmax(logits, dim=-1)

        # 2) Select top_k
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        topk_probs = topk_probs / (topk_probs.sum(dim=-1, keepdim=True) + 1e-8)

        # 3) Build routing_map: [batch_size, num_experts] => True/False
        #    top_k >= 1 => dropless
        routing_map = torch.zeros(batch_size, self.num_experts, dtype=torch.bool, device=x.device)
        for i in range(self.top_k):
            routing_map[torch.arange(batch_size), topk_idx[:, i]] = True

        # 4) permute: group tokens by assigned expert
        permutation_indices, per_expert_token_counts = create_permutation_indices(routing_map, self.tensor_parallel_group)

        # 6) Gather input over tensor parallel groups and permute 
        permuted_x = TensorParallelGatherAndPermute.apply(x, permutation_indices, self.tensor_parallel_group)
        return permuted_x, permutation_indices, topk_probs, per_expert_token_counts.cpu()
                