"""
Sparse NCCL collectives for AxoNN.

JIT-compiles sparse_comms_ext.cpp on first import (cached in
$SCRATCH/sparse_comms_build or /tmp/sparse_comms_build).

Environment
-----------
    NCCLX_BUILD_DIR        — path to the ncclx build tree containing
                             include/nccl.h and lib/libnccl_static.a
    SPARSE_COMMS_BUILD_DIR — where to cache the compiled extension
                             (default: $SCRATCH/sparse_comms_build)
    USE_SPARSE_RS=1        — use sparse reduce-scatter kernel; else forward
                             directly to torch.distributed (fast path)
    USE_SPARSE_AR=1        — use sparse all-reduce kernel; else forward to
                             torch.distributed (fast path)
    USE_SPARSE_AG=1        — use sparse all-gather kernel; else forward to
                             torch.distributed (fast path)
"""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load as _cpp_load

_HERE = Path(__file__).parent.resolve()

_NCCLX_BUILD_DIR = Path(
    os.environ.get(
        "NCCLX_BUILD_DIR",
        "/pscratch/sd/e/egencer/sparsecomms/torchcomms-sparse/build/ncclx",
    )
)
_NCCLX_INCLUDE = str(_NCCLX_BUILD_DIR / "include")
_NCCLX_LIB_DIR = str(_NCCLX_BUILD_DIR / "lib")

_BUILD_DIR = os.environ.get(
    "SPARSE_COMMS_BUILD_DIR",
    os.path.join(os.environ.get("SCRATCH", "/tmp"), "sparse_comms_build"),
)

_LOG_SPARSITY = os.environ.get("SPARSE_COMMS_LOG_SPARSITY", "0") not in ("0", "", "false", "False")

# Fast-path flags: when False, skip our extension and go straight to torch.dist.
_USE_SPARSE_RS = os.environ.get("USE_SPARSE_RS", "0") == "1"
_USE_SPARSE_AR = os.environ.get("USE_SPARSE_AR", "0") == "1"
_USE_SPARSE_AG = os.environ.get("USE_SPARSE_AG", "0") == "1"


def _log_sparsity(name, tensor):
    rank = dist.get_rank() if dist.is_initialized() else 0
    sparsity = (tensor == 0).sum().item() / tensor.numel()
    print(f"[sparse_comms rank={rank}] {name}: sparsity={sparsity:.4f} ({tensor.numel()} elements)", flush=True)


_ext = _cpp_load(
    name="sparse_comms_ext",
    sources=[str(_HERE / "sparse_comms_ext.cpp")],
    extra_include_paths=[_NCCLX_INCLUDE],
    extra_ldflags=[
        f"-L{_NCCLX_LIB_DIR}",
        "-lnccl",
        f"-Wl,-rpath,{_NCCLX_LIB_DIR}",
        "-lcuda",
    ],
    build_directory=_BUILD_DIR,
    verbose=False,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_backend(group=None):
    pg = group or dist.distributed_c10d._get_default_group()
    return pg, pg._get_backend(torch.device("cuda", torch.cuda.current_device()))


def _get_comm_ptr(group=None):
    pg, backend = _get_backend(group)
    ptr = backend._comm_ptr()
    if ptr == 0:
        dist.barrier(group=pg)
        ptr = backend._comm_ptr()
    if ptr == 0:
        raise RuntimeError(
            "comm_ptr is still 0 after barrier — check that init_process_group "
            "used the 'nccl' backend and that a GPU is visible."
        )
    return ptr


def _get_nccl_stream_ptr(group=None):
    """Return the cudaStream_t (as int64) that torch uses internally for this group/device."""
    pg, backend = _get_backend(group)
    device_index = torch.cuda.current_device()
    ptr = _ext.get_nccl_stream_ptr(backend, device_index)
    if ptr == 0:
        dist.barrier(group=pg)
        ptr = _ext.get_nccl_stream_ptr(backend, device_index)
    if ptr == 0:
        raise RuntimeError(
            "nccl_stream_ptr is still 0 after barrier — communicator not initialized."
        )
    return ptr


# ---------------------------------------------------------------------------
# Work handle
# ---------------------------------------------------------------------------

class _Work:
    """
    Mimics torch.distributed.Work for async sparse collectives.

    Mirrors ProcessGroupNCCL's WorkNCCL pattern:
      - collective is enqueued on torch's internal per-(group,device) NCCL stream
      - ncclEndEvent is recorded on that stream after the op
      - wait() makes current_stream block on ncclEndEvent
    """
    def __init__(self, end_event, device_index):
        self._end_event = end_event
        self._device_index = device_index

    def wait(self):
        current_stream = torch.cuda.current_stream(self._device_index)
        # ncclEndEvent_->block(currentStream) — same as ProcessGroupNCCL::synchronizeStream
        self._end_event.wait(current_stream)


# ---------------------------------------------------------------------------
# Launch logic
# ---------------------------------------------------------------------------

def _launch_sparse(ext_fn, tensors, comm_ptr, group, async_op):
    """
    Enqueue a sparse NCCL collective, matching ProcessGroupNCCL's runCollective:

    async_op=True  (mirrors asyncOp=true in ProcessGroupNCCL):
      1. Use torch's internal per-(group,device) ncclStream.
      2. syncStream: record event on current_stream, make ncclStream wait.
      3. Stash tensors (allocator safety).
      4. Launch NCCL op on ncclStream.
      5. Record ncclEndEvent on ncclStream.
      6. Return _Work; wait() blocks current_stream on ncclEndEvent.

    async_op=False (mirrors asyncOp=false in ProcessGroupNCCL):
      Use current_stream directly — CUDA in-order execution guarantees
      visibility to subsequent ops; no explicit synchronize needed.
    """
    device_index = tensors[0].device.index

    if async_op:
        nccl_stream_ptr = _get_nccl_stream_ptr(group)
        nccl_stream = torch.cuda.ExternalStream(nccl_stream_ptr)
        current_stream = torch.cuda.current_stream(device_index)

        # syncStream: record on current_stream, block ncclStream until done.
        sync_event = torch.cuda.Event()
        sync_event.record(current_stream)
        sync_event.wait(nccl_stream)   # ncclStream waits for sync_event

        # record_stream defers allocator reuse of tensor memory until nccl_stream
        # catches up — sufficient for allocator safety without a Python-level stash.
        for t in tensors:
            t.record_stream(nccl_stream)

        # Launch collective on ncclStream.
        ext_fn(*tensors, comm_ptr, nccl_stream_ptr)

        # Record ncclEndEvent on ncclStream.
        end_event = torch.cuda.Event()
        end_event.record(nccl_stream)

        return _Work(end_event, device_index)
    else:
        # asyncOp=false path: use current_stream directly.
        current_stream_ptr = torch.cuda.current_stream(device_index).cuda_stream
        ext_fn(*tensors, comm_ptr, current_stream_ptr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reduce_scatter_sparse(input, output, group=None, async_op=False):
    """
    Reduce-scatter (sum).

    Fast path (USE_SPARSE_RS=0): forwards directly to torch.distributed.
    Sparse path (USE_SPARSE_RS=1): uses sparse NCCL kernel on torch's internal
    NCCL stream, matching ProcessGroupNCCL's async launch pattern.

    input:  [nranks * recvcount] — full tensor
    output: [recvcount]          — this rank's slice
    """
    if not _USE_SPARSE_RS:
        return dist.reduce_scatter_tensor(output, input, group=group, async_op=async_op)

    if _LOG_SPARSITY:
        _log_sparsity(f"reduce_scatter_sparse/input, dense data size: {input.shape}", input)
    comm_ptr = _get_comm_ptr(group)
    return _launch_sparse(_ext.reduce_scatter_sparse, (input, output), comm_ptr, group, async_op)


def all_gather_sparse(input, output, group=None, async_op=False):
    """
    All-gather.

    Fast path (USE_SPARSE_AG=0): forwards directly to torch.distributed.
    Sparse path (USE_SPARSE_AG=1): uses sparse NCCL kernel on torch's internal
    NCCL stream, matching ProcessGroupNCCL's async launch pattern.

    input:  [sendcount]          — this rank's chunk
    output: [nranks * sendcount] — gathered result
    """
    if not _USE_SPARSE_AG:
        return dist.all_gather_into_tensor(output, input, group=group, async_op=async_op)

    if _LOG_SPARSITY:
        _log_sparsity("all_gather_sparse/input", input)
    comm_ptr = _get_comm_ptr(group)
    return _launch_sparse(_ext.all_gather_sparse, (input, output), comm_ptr, group, async_op)


def all_reduce_sparse(input, output=None, group=None, async_op=False):
    """
    All-reduce (sum). In-place if output is None or output is input.

    Fast path (USE_SPARSE_AR=0): forwards directly to torch.distributed.
    Sparse path (USE_SPARSE_AR=1): uses sparse NCCL kernel on torch's internal
    NCCL stream, matching ProcessGroupNCCL's async launch pattern.
    """
    if output is None:
        output = input

    if not _USE_SPARSE_AR:
        if output.data_ptr() != input.data_ptr():
            output.copy_(input)
        return dist.all_reduce(output, group=group, async_op=async_op)

    if _LOG_SPARSITY:
        _log_sparsity("all_reduce_sparse/input", input)
    comm_ptr = _get_comm_ptr(group)
    return _launch_sparse(_ext.all_reduce_sparse, (input, output), comm_ptr, group, async_op)
