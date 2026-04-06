import torch
from typing import Optional


class GradientPruner:
    """
    Top-K magnitude pruning with error feedback.

    A single instance handles multiple tensors via a per-key error buffer dict.
    For RS pruning use param.data_ptr() as the key (one buffer per layer weight).
    For AR pruning use a fixed key (default 0) for the flat gradient buffer.

    Args:
        sparsity:   fraction of elements to zero out (0.9 = keep top 10%)
        sample_pct: fraction of elements to sample for threshold estimation
                    (< 100 for approximate/faster threshold)
    """

    def __init__(self, sparsity: float, sample_pct: float = 100.0):
        assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)"
        assert 0.0 < sample_pct <= 100.0
        self.sparsity = sparsity
        self.sample_pct = sample_pct
        self._error: dict = {}

    @torch.no_grad()
    def prune(self, tensor: torch.Tensor, key=0, timer=None) -> torch.Tensor:
        """
        Prune tensor in-place with error feedback. Returns tensor.

        Args:
            tensor: gradient tensor to prune (modified in-place)
            key:    identifier for this tensor's error buffer.
                    Use param.data_ptr() for per-layer RS pruning;
                    use a fixed constant for a flat AR buffer.
            timer:  optional _CudaOpTimer to bracket this call.
        """
        if timer is not None:
            timer.start()

        if key in self._error:
            tensor.add_(self._error[key])

        n = tensor.numel()

        if self.sample_pct < 100.0:
            n_sample = max(1, int(n * self.sample_pct / 100.0))
            idx = torch.randint(0, n, (n_sample,), device=tensor.device)
            k = max(1, int(n_sample * self.sparsity))
            threshold = torch.kthvalue(tensor.flatten()[idx].abs(), k)[0]
        else:
            k = max(1, int(n * self.sparsity))
            threshold = torch.kthvalue(tensor.abs().flatten(), k)[0]

        mask = tensor.abs() > threshold

        # save residual, reusing existing buffer to avoid repeated allocation
        if key in self._error:
            self._error[key].copy_(tensor).mul_(~mask)
        else:
            self._error[key] = tensor.clone().mul_(~mask)

        tensor.mul_(mask)

        if timer is not None:
            timer.stop()

        return tensor

    def clear_error(self):
        self._error.clear()
