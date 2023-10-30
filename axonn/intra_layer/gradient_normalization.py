import torch

try:
    from torch._six import inf
except ImportError:
    from torch import inf
import torch.distributed as dist

from axonn import axonn as ax


def get_total_norm(tensors, norm_type, error_if_nonfinite):
    if len(tensors) == 0:
        return torch.tensor(0.0)
    device = tensors[0].device
    total_norm = torch.norm(
        torch.stack([torch.norm(g.detach(), norm_type).to(device) for g in tensors]),
        norm_type,
    )
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )

    return total_norm


def clip_grad_norm_(parameters, max_norm, norm_type=2.0, error_if_nonfinite=False):
    if norm_type == inf:
        raise NotImplementedError

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    tensor_parallel_params = []
    non_tensor_parallel_params = []
    for p in parameters:
        if hasattr(p, "is_tensor_parallel") and p.is_tensor_parallel:
            tensor_parallel_params.append(p)
        else:
            non_tensor_parallel_params.append(p)

    tensor_parallel_grads = [
        p.grad for p in tensor_parallel_params if p.grad is not None
    ]
    non_tensor_parallel_grads = [
        p.grad for p in non_tensor_parallel_params if p.grad is not None
    ]

    grads = non_tensor_parallel_grads + tensor_parallel_grads

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    non_tensor_parallel_norm = get_total_norm(
        non_tensor_parallel_grads, norm_type, error_if_nonfinite
    )

    local_tensor_parallel_norm = get_total_norm(
        tensor_parallel_grads, norm_type, error_if_nonfinite
    )
    tensor_parallel_norm = local_tensor_parallel_norm**norm_type
    dist.all_reduce(tensor_parallel_norm, group=ax.comm_handle.intra_layer_group)
    tensor_parallel_norm = tensor_parallel_norm ** (1.0 / norm_type)

    total_norm = get_total_norm(
        [tensor_parallel_norm, non_tensor_parallel_norm], norm_type, error_if_nonfinite
    )

    clip_coef = max_norm / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for g in grads:
        g.detach().mul_(clip_coef_clamped.to(g.device))
    return total_norm
