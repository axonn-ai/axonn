import torch 
from axonn.intra_layer import Gather, Drop
import torch.distributed as dist

def _gather_and_permute(input_, permutation_indices, process_group):
    gathered_input = Gather.apply(input_, process_group, 0)
    permuted_input = gathered_input[permutation_indices]
    return permuted_input, gathered_input.shape

def _unpermute_and_drop(input_, permutation_indices, unpermuted_tensor_shape, process_group):
    # unpermute output gradients
    unpermuted_input = torch.zeros(unpermuted_tensor_shape, 
                                      device=input_.device, 
                                      dtype=input_.dtype)
    
    unpermuted_input.scatter_add_(0, permutation_indices.unsqueeze(1), input_)
    
    # do a reduce scatter
    unpermuted_tensor_shape = list(unpermuted_tensor_shape) 
    world_size = dist.get_world_size(process_group)
    assert unpermuted_tensor_shape[0] % world_size == 0
    unpermuted_tensor_shape[0] //= world_size 
    output = torch.empty(unpermuted_tensor_shape, 
                            dtype=unpermuted_input.dtype, 
                            device=unpermuted_input.device)
    dist.reduce_scatter_tensor(
                                output,
                                unpermuted_input, 
                                group=process_group
                            )
    return output


class TensorParallelGatherAndPermute(torch.autograd.Function):
    @staticmethod 
    def forward(ctx, input_, permutation_indices, process_group):
        output, ctx.unpermuted_tensor_shape = _gather_and_permute(input_, permutation_indices, process_group)
        ctx.process_group = process_group 
        ctx.save_for_backward(permutation_indices)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # unpermute output gradients
        permutation_indices, = ctx.saved_tensors
        grad_input = _unpermute_and_drop(grad_output, 
                                         permutation_indices, 
                                         ctx.unpermuted_tensor_shape, 
                                         ctx.process_group)
        
        return grad_input, None, None

class TensorParallelUnpermuteAndScatter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_, permutation_indices, unpermuted_tensor_shape, process_group):
        # unpermute output
        output = _unpermute_and_drop(input_, 
                            permutation_indices, 
                            unpermuted_tensor_shape, 
                            process_group)
        ctx.process_group = process_group 
        ctx.save_for_backward(permutation_indices)
        return output

    
    @staticmethod 
    def backward(ctx, grad_output):
        permutation_indices, = ctx.saved_tensors
        grad_input, _ = _gather_and_permute(grad_output, 
                                            permutation_indices, 
                                            ctx.process_group)
        return grad_input, None, None, None