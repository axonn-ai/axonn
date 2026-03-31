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


def _get_comm_ptr(group=None):
    pg = group or dist.distributed_c10d._get_default_group()
    # Unwrap to the ProcessGroupNCCL backend which exposes _comm_ptr().
    backend = pg._get_backend(torch.device("cuda", torch.cuda.current_device()))
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


def _make_handle(stream):
    event = torch.cuda.Event()
    event.record(stream)
    return event


def reduce_scatter_sparse(input, output, group=None, async_op=False):
    """
    Sparse reduce-scatter (sum).

    input:  [nranks * recvcount] — full tensor
    output: [recvcount]          — this rank's slice
    """
    if _LOG_SPARSITY:
        _log_sparsity(f"reduce_scatter_sparse/input, dense data size: {input.shape}", input)
    comm_ptr = _get_comm_ptr(group)
    _ext.reduce_scatter_sparse(input, output, comm_ptr)
    if async_op:
        return _make_handle(torch.cuda.current_stream())
    torch.cuda.current_stream().synchronize()


def all_gather_sparse(input, output, group=None, async_op=False):
    """
    Sparse all-gather.

    input:  [sendcount]          — this rank's chunk
    output: [nranks * sendcount] — gathered result
    """
    if _LOG_SPARSITY:
        _log_sparsity("all_gather_sparse/input", input)
    comm_ptr = _get_comm_ptr(group)
    _ext.all_gather_sparse(input, output, comm_ptr)
    if async_op:
        return _make_handle(torch.cuda.current_stream())
    torch.cuda.current_stream().synchronize()


def all_reduce_sparse(input, output=None, group=None, async_op=False):
    """Sparse all-reduce (sum). In-place if output is None."""
    if output is None:
        output = input
    if _LOG_SPARSITY:
        _log_sparsity("all_reduce_sparse/input", input)
    comm_ptr = _get_comm_ptr(group)
    _ext.all_reduce_sparse(input, output, comm_ptr)
    if async_op:
        return _make_handle(torch.cuda.current_stream())
    torch.cuda.current_stream().synchronize()


