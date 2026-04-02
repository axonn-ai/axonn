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
    INPUT_DTYPE: tl.constexpr,
    ERROR_DTYPE: tl.constexpr,
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
        sparsity: Fraction of elements to prune (0.0 to 1.0)
        keep_error: Whether to write pruned values into error buffer
        INPUT_DTYPE: dtype of input tensor (tl.constexpr), used to cast error on load
        ERROR_DTYPE: dtype to use when storing to error_ptr (tl.constexpr)
        BLOCK_SIZE: Number of elements per block (tl.constexpr)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    th = tl.load(treshold + tl.zeros_like(offsets), mask=mask)

    x = tl.load(input_ptr + offsets, mask=mask)

    if keep_error:
        e = tl.load(error_ptr + offsets, mask=mask).to(INPUT_DTYPE)
        x = x + e

    a = tl.abs(x)
    m = a > th

    zeros = tl.zeros_like(x)

    # store full result (x may have changed due to error add)
    tl.store(input_ptr + offsets, tl.where(m, x, zeros), mask=mask)

    if keep_error:
        tl.store(error_ptr + offsets, tl.where(~m, x, zeros).to(ERROR_DTYPE), mask=mask)


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
    Triton kernel for random sampling of absolute values for threshold estimation.

    Args:
        input_ptr: Pointer to input gradient tensor
        sample_out_ptr: Pointer to output sample buffer
        n_elements: Total number of elements in the input
        n_sample_ements: Number of elements to sample
        seed: Random seed for sampling
        BLOCK_SIZE: Number of elements per block (tl.constexpr)
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_sample_ements

    x = tl.load(input_ptr + (tl.randint(seed, offsets).to(tl.int64) % n_elements), mask=mask)
    a = tl.abs(x)
    tl.store(sample_out_ptr + offsets, a, mask=mask)


# =============================================================================
# Triton Pruner Class
# =============================================================================

# Maps AXONN_PRUNE_ERROR_DTYPE env var value -> (torch.dtype, tl dtype constexpr)
# fp8 types require Triton >= 2.2; on Ampere they are software-emulated.
_ERROR_DTYPE_MAP: dict = {
    "float32":      (torch.float32,        tl.float32),
    "float16":      (torch.float16,        tl.float16),
    "bfloat16":     (torch.bfloat16,       tl.bfloat16),
    "float8_e4m3":  (torch.float8_e4m3fn,  tl.float8e4b15),  # E4M3, bias=15, Ampere-compatible
    "float8_e4m3nv":(torch.float8_e4m3fn,  tl.float8e4nv),   # E4M3, NVIDIA native, Hopper only
    "float8_e5m2":  (torch.float8_e5m2,    tl.float8e5),     # E5M2, Ampere-compatible
}


class TritonGradientPruner:
    """
    Triton-based Top-K magnitude pruning with error feedback.

    Environment variables:
        AXONN_PRUNE_ERROR_ACCUMULATE: Set to "0" to disable error feedback (default "1").
        AXONN_PRUNE_ERROR_DTYPE: dtype for the error accumulator buffer.
            "same"         - match the input tensor dtype (default)
            "float32"      - fp32 accumulation (higher precision)
            "float16"      - fp16
            "bfloat16"     - bf16
            "float8_e4m3"  - E4M3 FP8 (software-emulated on Ampere)
            "float8_e5m2"  - E5M2 FP8 (software-emulated on Ampere)
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

        error_dtype_str = os.getenv("AXONN_PRUNE_ERROR_DTYPE", "same").lower()
        if error_dtype_str == "same":
            self._error_torch_dtype = None  # resolved per-tensor at prune() time
            self._error_tl_dtype = None
        elif error_dtype_str in _ERROR_DTYPE_MAP:
            self._error_torch_dtype, self._error_tl_dtype = _ERROR_DTYPE_MAP[error_dtype_str]
        else:
            raise ValueError(
                f"Unknown AXONN_PRUNE_ERROR_DTYPE={error_dtype_str!r}. "
                f"Valid options: same, {', '.join(_ERROR_DTYPE_MAP)}"
            )

    def _tl_dtype_for(self, tensor: torch.Tensor) -> tl.constexpr:
        """Return the tl dtype to use for the error buffer."""
        if self._error_tl_dtype is not None:
            return self._error_tl_dtype
        _torch_to_tl = {
            torch.float32:        tl.float32,
            torch.float16:        tl.float16,
            torch.bfloat16:       tl.bfloat16,
            torch.float8_e4m3fn:  tl.float8e4b15,  # Ampere-compatible; use float8_e4m3nv for Hopper
            torch.float8_e5m2:    tl.float8e5,
        }
        dtype = _torch_to_tl.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"No tl dtype mapping for tensor dtype {tensor.dtype}")
        return dtype

    def _error_torch_dtype_for(self, tensor: torch.Tensor) -> torch.dtype:
        if self._error_torch_dtype is not None:
            return self._error_torch_dtype
        return tensor.dtype

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
        n = tensor.numel()
        n_sample_elems = max(1, int(n * self.sample_pct / 100))
        if key in self._temp_sample:
            sample_out = self._temp_sample[key]
        else:
            sample_out = torch.empty(n_sample_elems, device=tensor.device, dtype=tensor.dtype)
            self._temp_sample[key] = sample_out

        if self._keep_error:
            err_dtype = self._error_torch_dtype_for(tensor)
            if key in self._error:
                error_buffer = self._error[key]
            else:
                error_buffer = torch.zeros(tensor.shape, device=tensor.device, dtype=err_dtype)
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
            self._tl_dtype_for(tensor),
            self._tl_dtype_for(error_buffer),
        )

        return tensor

    def clear_error(self):
        """Clear all error buffers."""
        self._error.clear()
