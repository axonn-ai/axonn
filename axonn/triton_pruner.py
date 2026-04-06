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
    fp8_scale,
    prev_scale_ptr,
    INPUT_DTYPE: tl.constexpr,
    ERROR_DTYPE: tl.constexpr,
    BITCAST_ERROR: tl.constexpr,
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
        treshold: Pointer to scalar pruning threshold; also the fp8 store scale when fp8_scale=True
        sparsity: Fraction of elements to prune (0.0 to 1.0)
        keep_error: Whether to write pruned values into error buffer
        fp8_scale: Scale error by 1/threshold before storing, unscale by prev_scale on load
        prev_scale_ptr: Pointer to float32 scalar — threshold used when the stored error was written
        INPUT_DTYPE: dtype of input tensor (tl.constexpr)
        ERROR_DTYPE: tl dtype of the error buffer values
        BITCAST_ERROR: True when error is stored as int8 and must be bitcast to ERROR_DTYPE on load/store
        BLOCK_SIZE: Number of elements per block (tl.constexpr)
    """
    ERROR_DECAY: tl.constexpr = tl.constexpr(0.9)
    pid = tl.program_id(axis=0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    th = tl.load(treshold + tl.zeros_like(offsets), mask=mask)

    x = tl.load(input_ptr + offsets, mask=mask)

    if keep_error:
        if BITCAST_ERROR:
            e = tl.load(error_ptr + offsets, mask=mask).to(ERROR_DTYPE, bitcast=True).to(INPUT_DTYPE)
        else:
            e = tl.load(error_ptr + offsets, mask=mask).to(INPUT_DTYPE)
        if fp8_scale:
            prev_scale = tl.load(prev_scale_ptr).to(INPUT_DTYPE)
            e = e * prev_scale
        x = x + e

    a = tl.abs(x)
    m = a > th

    zeros = tl.zeros_like(x)

    # store full result (x may have changed due to error add)
    tl.store(input_ptr + offsets, tl.where(m, x, zeros), mask=mask)

    if keep_error:
        new_e = tl.where(~m, x * ERROR_DECAY, zeros)
        if fp8_scale:
            new_e = new_e / th
        if BITCAST_ERROR:
            tl.store(error_ptr + offsets, new_e.to(ERROR_DTYPE).to(tl.int8, bitcast=True), mask=mask)
        else:
            tl.store(error_ptr + offsets, new_e.to(ERROR_DTYPE), mask=mask)


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
    err_ptr,
    sample_out_ptr,
    n_elements,
    n_sample_ements,
    seed,
    keep_error,
    fp8_scale,
    prev_scale_ptr,
    INPUT_DTYPE: tl.constexpr,
    ERROR_DTYPE: tl.constexpr,
    BITCAST_ERROR: tl.constexpr,
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
        fp8_scale: Unscale error by prev_scale before use (matches prune_kernel store convention)
        prev_scale_ptr: Pointer to float32 scalar — threshold used when the stored error was written
        ERROR_DTYPE: tl dtype of the error buffer values
        BITCAST_ERROR: True when error is stored as int8 and must be bitcast to ERROR_DTYPE on load
        BLOCK_SIZE: Number of elements per block (tl.constexpr)
    """
    pid = tl.program_id(axis=0).to(tl.int64)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_sample_ements

    rnd = (tl.randint(seed, offsets).to(tl.int64) % n_elements)

    x = tl.load(input_ptr + rnd, mask=mask)
    if keep_error:
        if BITCAST_ERROR:
            e = tl.load(err_ptr + rnd, mask=mask).to(ERROR_DTYPE, bitcast=True).to(INPUT_DTYPE)
        else:
            e = tl.load(err_ptr + rnd, mask=mask).to(INPUT_DTYPE)
        if fp8_scale:
            prev_scale = tl.load(prev_scale_ptr).to(INPUT_DTYPE)
            e = e * prev_scale
        a = tl.abs(x + e)
    else:
        a = tl.abs(x)
    tl.store(sample_out_ptr + offsets, a, mask=mask)


# =============================================================================
# Triton Pruner Class
# =============================================================================

# Maps AXONN_PRUNE_ERROR_DTYPE env var value -> (torch.dtype, tl dtype constexpr)
# fp8 types require Triton >= 2.2; on Ampere they are software-emulated.

_ERROR_DTYPE_MAP: dict = {
    "float32":  (torch.float32,  tl.float32,    False),
    "float16":  (torch.float16,  tl.float16,    False),
    "bfloat16": (torch.bfloat16, tl.bfloat16,   False),
    # FP8 types: stored as int8 to avoid Triton typing the pointer as fp8e4nv/fp8e5 on Ampere.
    # The kernel receives *int8 pointers and bitcasts to the correct FP8 type internally.
    "fp8e4":    (torch.int8,     tl.float8e4b15, True),   # E4M3, Ampere-compatible
    "fp8e5":    (torch.int8,     tl.float8e5,    True),   # E5M2, Ampere-compatible
}

_TORCH_TO_TL: dict = {
    torch.float32:       tl.float32,
    torch.float16:       tl.float16,
    torch.bfloat16:      tl.bfloat16,
    torch.float8_e4m3fn: tl.float8e4b15,
    torch.float8_e5m2:   tl.float8e5,
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
            "fp8e4"        - E4M3 FP8, Ampere-compatible
            "fp8e5"        - E5M2 FP8, Ampere-compatible
        AXONN_PRUNE_FP8_SCALE: Set to "1" to enable threshold-based scaling of the FP8
            error buffer (default "0"). Values are divided by the current threshold before
            storing (mapping [-th, th] -> [-1, 1]) and multiplied by the previous threshold
            on load. Only meaningful when AXONN_PRUNE_ERROR_DTYPE is fp8e4 or fp8e5.
    """

    def __init__(self, sparsity: float, sample_pct: float = 100.0):
        assert 0.0 <= sparsity < 1.0, "sparsity must be in [0, 1)"
        assert 0.0 < sample_pct <= 100.0
        self.sparsity = sparsity
        self.sample_pct = sample_pct
        self._error: dict = {}
        self._temp_sample: dict = {}
        self._prev_scale: dict = {}
        self._keep_error: bool = False
        if os.getenv("AXONN_PRUNE_ERROR_ACCUMULATE", "1") == "1":
            self._keep_error = True

        self._fp8_scale: bool = os.getenv("AXONN_PRUNE_FP8_SCALE", "0") == "1"

        error_dtype_str = os.getenv("AXONN_PRUNE_ERROR_DTYPE", "same").lower()
        if error_dtype_str == "same":
            self._error_torch_dtype = None  # resolved per-tensor at prune() time
            self._error_tl_dtype = None
            self._bitcast_error = False
        elif error_dtype_str in _ERROR_DTYPE_MAP:
            self._error_torch_dtype, self._error_tl_dtype, self._bitcast_error = _ERROR_DTYPE_MAP[error_dtype_str]
        else:
            raise ValueError(
                f"Unknown AXONN_PRUNE_ERROR_DTYPE={error_dtype_str!r}. "
                f"Valid options: same, {', '.join(_ERROR_DTYPE_MAP)}"
            )

    def _tl_dtype_for(self, tensor: torch.Tensor) -> tl.constexpr:
        """Return the tl dtype matching tensor.dtype (no error-override)."""
        dtype = _TORCH_TO_TL.get(tensor.dtype)
        if dtype is None:
            raise ValueError(f"No tl dtype mapping for tensor dtype {tensor.dtype}")
        return dtype

    def _error_tl_dtype_for(self, tensor: torch.Tensor) -> tl.constexpr:
        """Return the tl dtype for the error buffer (applies AXONN_PRUNE_ERROR_DTYPE override)."""
        if self._error_tl_dtype is not None:
            return self._error_tl_dtype
        return self._tl_dtype_for(tensor)

    def _error_torch_dtype_for(self, tensor: torch.Tensor) -> torch.dtype:
        if self._error_torch_dtype is not None:
            return self._error_torch_dtype
        return tensor.dtype

    @torch.no_grad()
    def prune(self, tensor: torch.Tensor, key=0, timer=None) -> torch.Tensor:
        """
        Prune tensor in-place with error feedback using Triton kernel.

        Args:
            tensor: gradient tensor to prune (modified in-place)
            key:    identifier for this tensor's error buffer
            timer:  optional _CudaOpTimer to bracket this call

        Returns:
            Pruned tensor
        """
        if timer is not None:
            timer.start()

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

        # prev_scale: threshold from the previous iteration, used to unscale the stored FP8 error.
        # Stored as a 1-element float32 tensor so it can be passed as a device pointer to kernels.
        # Initialized to 1.0 (error buffer is all zeros on first iteration, so scale doesn't matter).
        effective_fp8_scale = self._fp8_scale and self._keep_error
        if key not in self._prev_scale:
            self._prev_scale[key] = torch.ones(1, device=tensor.device, dtype=torch.float32)
        prev_scale_buf = self._prev_scale[key]

        grid = lambda meta: (triton.cdiv(n_sample_elems, meta["BLOCK_SIZE"]),)
        seed = torch.randint(0, 2**31, (1,)).item()
        sample_kernel[grid](
            tensor,
            error_buffer,
            sample_out,
            n,
            n_sample_elems,
            seed,
            self._keep_error,
            effective_fp8_scale,
            prev_scale_buf,
            self._tl_dtype_for(tensor),
            self._error_tl_dtype_for(tensor),
            self._bitcast_error and self._keep_error,
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
            effective_fp8_scale,
            prev_scale_buf,
            self._tl_dtype_for(tensor),
            self._error_tl_dtype_for(tensor),
            self._bitcast_error and self._keep_error,
        )

        # Update prev_scale for the next iteration. This copy_ is queued on the same CUDA stream
        # after both kernels, so it reads the new threshold only after the kernels have finished.
        if effective_fp8_scale:
            prev_scale_buf.copy_(threshold.float().reshape(1))

        if timer is not None:
            timer.stop()

        return tensor

    def clear_error(self):
        """Clear all error buffers and reset FP8 scale history."""
        self._error.clear()
        self._prev_scale.clear()
