"""Triton-based GradientPruner implementation."""
import os

import torch
import triton
import triton.language as tl

# =============================================================================
# Triton Kernels (Boilerplate - implement your logic here)
# =============================================================================


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["n_elements"],
)
@triton.jit
def prune_kernel(
    input_ptr,
    output_ptr,
    error_ptr,
    mask_ptr,
    n_elements,
    treshold,
    sparsity,
    keep_error,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for top-k magnitude pruning.

    Args:
        input_ptr: Pointer to input gradient tensor
        output_ptr: Pointer to output pruned tensor
        error_ptr: Pointer to error buffer (for feedback)
        mask_ptr: Pointer to mask tensor (optional, for debugging)
        n_elements: Total number of elements
        sample_pct: Percentage of elements to sample for threshold computation
        sparsity: Fraction of elements to prune (0.0 to 1.0)
        seed: Random seed for sampling
        BLOCK_SIZE: Number of elements per block (tl.constexpr)

    Grid/Block info you can access inside the kernel:
        - tl.program_id(axis=0): Current block index in the 1D grid
        - tl.num_programs(axis=0): Total number of blocks in the grid
        - BLOCK_SIZE: Elements per block (compile-time constant)
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)  # Total number of blocks
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    th = tl.load(treshold + tl.zeros_like(offsets), mask=mask)

    x = tl.load(input_ptr + offsets, mask=mask)
    a = tl.abs(x)
    m = a > th
    
    zeros = tl.zeros_like(x)

    tl.store(input_ptr + offsets, zeros, mask=(mask & (~m)))

    if keep_error:
        tl.store(error_ptr + offsets, tl.where(~m, x, zeros), mask=mask)

    


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 2048}),
        triton.Config({"BLOCK_SIZE": 4096}),
    ],
    key=["n_elements", "n_sample_ements"],
)
@triton.jit
def sample_kernel(
    input_ptr,
    sample_out_ptr,
    n_elements,
    n_sample_ements,
    seed,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for top-k magnitude pruning.

    Args:
        input_ptr: Pointer to input gradient tensor
        output_ptr: Pointer to output pruned tensor
        error_ptr: Pointer to error buffer (for feedback)
        mask_ptr: Pointer to mask tensor (optional, for debugging)
        n_elements: Total number of elements
        sample_pct: Percentage of elements to sample for threshold computation
        sparsity: Fraction of elements to prune (0.0 to 1.0)
        seed: Random seed for sampling
        BLOCK_SIZE: Number of elements per block (tl.constexpr)

    Grid/Block info you can access inside the kernel:
        - tl.program_id(axis=0): Current block index in the 1D grid
        - tl.num_programs(axis=0): Total number of blocks in the grid
        - BLOCK_SIZE: Elements per block (compile-time constant)
    """
    pid = tl.program_id(axis=0)
    grid_size = tl.num_programs(axis=0)  # Total number of blocks
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_sample_ements

    x = tl.load(input_ptr + (tl.randint(seed, offsets).to(tl.int64) % n_elements), mask=mask)
    a = tl.abs(x)
    tl.store(sample_out_ptr + offsets, a, mask=mask)
    

# =============================================================================
# Triton Pruner Class
# =============================================================================


class TritonGradientPruner:
    """
    Triton-based Top-K magnitude pruning with error feedback.

    This is a boilerplate implementation. The actual Triton kernel logic
    needs to be implemented in the kernels above.
    """

    def __init__(self, sparsity: float, sample_pct: float = 100.0):
        assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)"
        assert 0.0 < sample_pct <= 100.0
        self.sparsity = sparsity
        self.sample_pct = sample_pct
        self._error: dict = {}
        self._temp_sample: dict = {}
        self._keep_error: bool = False
        if os.getenv("AXONN_PRUNE_ERROR_ACCUMULATE", "1") == "1":
            self._keep_error = True

    @torch.no_grad()
    def prune(self, tensor: torch.Tensor, key=0) -> torch.Tensor:
        """
        Prune tensor in-place with error feedback using Triton kernel.

        Args:
            tensor: gradient tensor to prune (modified in-place)
            key: identifier for this tensor's error buffer

        Returns:
            Pruned tensor
        """
        # Add error feedback
        if key in self._error and self._keep_error:
            tensor.add_(self._error[key])
        n = tensor.numel()
        n_sample_elems = max(1, int(n * self.sample_pct / 100))
        if key in self._temp_sample:
            sample_out = self._temp_sample[key]
        else:
            sample_out = torch.empty(n_sample_elems, device=tensor.device, dtype=tensor.dtype)
            self._temp_sample[key] = sample_out
            

        if self._keep_error:
            if key in self._error:
                error_buffer = self._error[key]
            else:
                error_buffer = torch.empty_like(tensor)
                self._error[key] = error_buffer
        else:
            error_buffer = tensor

        grid = lambda meta: (triton.cdiv(n_sample_elems, meta["BLOCK_SIZE"]),)
        seed = torch.randint(0, 2**31, (1,)).item()
        sample_kernel[grid](
            tensor,
            sample_out,
            n,
            n_sample_elems,
            seed,
        )
        
        
        k = max(1, int(n_sample_elems * self.sparsity))
        threshold = torch.kthvalue(sample_out, k)[0]
        
        # Launch Triton kernel (computes threshold internally)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        prune_kernel[grid](
            tensor,
            tensor,
            error_buffer,
            None,
            n,
            threshold,
            self.sparsity,
            self._keep_error,
        )

        return tensor

    def clear_error(self):
        """Clear all error buffers."""
        self._error.clear()
