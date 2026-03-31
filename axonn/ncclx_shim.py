# ncclx_shim.py
#
# NCCLx communicator backed by a dlmopen-isolated libnccl.so.2.
#
# Designed to drop in where TorchComms was used in AxoNN:
#
#   from axonn.ncclx_shim import NCCLxComm
#
#   # world communicator — call once after torch.distributed.init_process_group
#   world_tc = NCCLxComm()
#
#   # split into depth-parallel sub-communicators — collective across world comm
#   depth_tc = world_tc.split(my_depth_group_ranks)   # list of global ranks
#
#   # in _reduce_scatter — matches the TorchComms reduce_scatter_single signature
#   handle = depth_tc.reduce_scatter_single(output, input_, op=None, async_op=False)
#
# Environment:
#   NCCLX_LIB_PATH  — path to the NCCLx libnccl.so.2 (default below)
#   NCCLX_DEBUG=1   — enable verbose per-op logging (C++ prints to stderr;
#                     Python logs via the "ncclx_shim" logger at DEBUG level)
#   Do NOT LD_PRELOAD that library when using this shim; torch.distributed
#   continues to use PyTorch's own bundled libnccl unmodified.

import os
import logging
import torch
import torch.distributed as dist
from torch.utils.cpp_extension import load as _cpp_load

# ---------------------------------------------------------------------------
# Logger — activate with NCCLX_DEBUG=1 or by configuring the root logger
# ---------------------------------------------------------------------------
_log = logging.getLogger("ncclx_shim")
if os.environ.get("NCCLX_DEBUG", "0") not in ("0", ""):
    if not _log.handlers:
        _h = logging.StreamHandler()
        _h.setFormatter(logging.Formatter("[ncclx_shim py rank=%(rank)s] %(message)s"))
        _log.addHandler(_h)
    _log.setLevel(logging.DEBUG)
else:
    _log.setLevel(logging.WARNING)

_rank_str = os.environ.get("RANK", "?")


def _dbg(msg, *args):
    """Log at DEBUG with global rank injected into the format."""
    _log.debug(msg, *args, extra={"rank": _rank_str})


# ---------------------------------------------------------------------------
# JIT-build the C++ extension once (cached in $SCRATCH/ncclx_shim_build)
# ---------------------------------------------------------------------------
_SHIM_DIR  = os.path.dirname(os.path.abspath(__file__))
_BUILD_DIR = os.path.join(os.environ.get("SCRATCH", "/tmp"), "ncclx_shim_build")

_dbg("JIT-loading C++ extension from %s, build_dir=%s", _SHIM_DIR, _BUILD_DIR)
_ext = _cpp_load(
    name="_ncclx_shim",
    sources=[os.path.join(_SHIM_DIR, "ncclx_shim.cpp")],
    extra_ldflags=["-ldl"],
    build_directory=_BUILD_DIR,
    verbose=(os.environ.get("NCCLX_DEBUG", "0") not in ("0", "")),
)
_dbg("C++ extension loaded")

NCCLX_LIB_PATH: str = os.environ.get(
    "NCCLX_LIB_PATH",
    "/pscratch/sd/e/egencer/sparsecomms/torchcomms-sparse/build/ncclx/lib/libnccl.so.2",
)

# Load NCCLx at module import time — before CUDA or PyTorch's NCCL is active.
#
# When loaded mid-session (inside NCCLxComm.__init__), NCCLx's global
# constructors run with CUDA already initialized, causing partial init and
# NULL pointer crashes in ncclCommInitRank. Loading here replicates the
# sanity_check.cpp behaviour: NCCLx loads at process start, constructors
# run cleanly, everything works later when ncclCommInitRank is called.
_dbg("load_ncclx at import time from %s", NCCLX_LIB_PATH)
_ext.load_ncclx(NCCLX_LIB_PATH)

# ---------------------------------------------------------------------------
# dtype → NCCL datatype int  (stable NCCL 2.x enum values)
# ---------------------------------------------------------------------------
_DTYPE_TO_NCCL = {
    torch.float16:  6,
    torch.float32:  7,
    torch.float64:  8,
    torch.bfloat16: 9,
    torch.int32:    2,
    torch.int64:    4,
}


# ---------------------------------------------------------------------------
# Async handle (mimics torch.distributed.Work so axonn.register_handle works)
# ---------------------------------------------------------------------------
class _NCCLxWork:
    """Wraps a CUDA event so AxoNN's overlap mechanism can wait on it."""
    def __init__(self):
        self._event = torch.cuda.Event()
        self._event.record()

    def wait(self):
        self._event.wait()

    def is_completed(self) -> bool:
        return self._event.query()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------
class NCCLxComm:
    """
    NCCLx communicator that mirrors a torch.distributed process group.

    Unique-ID exchange is done over torch.distributed so no separate
    bootstrap is needed.  All collectives are issued on the current CUDA
    stream (asynchronous with respect to the host).

    Parameters
    ----------
    group : torch.distributed.ProcessGroup or None
        The process group this communicator covers.  None = world group.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, group: "dist.ProcessGroup | None" = None):
        _dbg("NCCLxComm.__init__: group init")

        self._group = group
        rank        = dist.get_rank(group)
        world_size  = dist.get_world_size(group)
        _dbg("NCCLxComm.__init__: group rank=%d size=%d", rank, world_size)

        # Exchange the NCCLx unique ID via MPI (not torch.distributed/NCCL).
        # Using dist.broadcast here would trigger lazy PyTorch NCCL world-comm
        # init, which races with NCCLx's own init and crashes rank 0.
        from mpi4py import MPI
        if not MPI.Is_initialized():
            MPI.Init()
        mpi_comm   = MPI.COMM_WORLD
        mpi_rank   = mpi_comm.Get_rank()
        mpi_size   = mpi_comm.Get_size()

        # For sub-group comms, find the global rank that is rank-0 in the group.
        group_ranks = list(range(world_size)) if group is None else [
            dist.get_global_rank(group, r) for r in range(world_size)
        ]
        src_mpi = group_ranks[0]   # MPI rank of group rank-0

        if mpi_rank == src_mpi:
            uid_bytes = _ext.get_unique_id()
            _dbg("NCCLxComm.__init__: generated unique ID (mpi_rank=%d)", mpi_rank)
        else:
            uid_bytes = b'\x00' * 128

        _dbg("NCCLxComm.__init__: broadcasting unique ID via MPI from mpi_rank=%d", src_mpi)
        uid_bytes = mpi_comm.bcast(uid_bytes, root=src_mpi)

        _dbg("NCCLxComm.__init__: calling ncclCommInitRank rank=%d size=%d", rank, world_size)
        self._comm = _ext.create_comm(uid_bytes, rank, world_size)
        self._rank = rank
        self._size = world_size
        _dbg("NCCLxComm.__init__: comm handle=0x%x", self._comm)

    @classmethod
    def _from_handle(cls, handle: int) -> "NCCLxComm":
        """Internal: wrap a raw comm handle returned by comm_split."""
        obj = cls.__new__(cls)
        obj._group = None
        obj._comm  = handle
        obj._rank  = _ext.comm_rank(handle)
        obj._size  = _ext.comm_size(handle)
        _dbg("NCCLxComm._from_handle: handle=0x%x rank=%d size=%d",
             handle, obj._rank, obj._size)
        return obj

    # ------------------------------------------------------------------
    # Split
    # ------------------------------------------------------------------

    def split(self, group_ranks: "list[int]", name: str = "") -> "NCCLxComm":
        """
        Create a sub-communicator for ``group_ranks``.

        **Collective** — every rank in *this* communicator must call split()
        simultaneously, each passing its own group's rank list.

        ``group_ranks`` must form a partition of the world ranks (each rank
        appears in exactly one group), which is always the case for AxoNN's
        tensor-parallel groups.

        Color is the minimum global rank in the group — guaranteed unique
        across non-overlapping groups and deterministic on all ranks.
        """
        world_rank = dist.get_rank()
        color = min(group_ranks)
        key   = sorted(group_ranks).index(world_rank)
        _dbg("NCCLxComm.split: world_rank=%d group_ranks=%s color=%d key=%d",
             world_rank, group_ranks, color, key)
        child_handle = _ext.comm_split(self._comm, color, key)
        child = NCCLxComm._from_handle(child_handle)
        _dbg("NCCLxComm.split: child handle=0x%x rank=%d size=%d",
             child_handle, child._rank, child._size)
        return child

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def get_rank(self) -> int:
        return self._rank

    def get_size(self) -> int:
        return self._size

    def get_backend(self) -> str:
        return "ncclx"

    # ------------------------------------------------------------------
    # Collectives
    # ------------------------------------------------------------------

    def reduce_scatter_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        op=None,                 # ignored — always SUM; kept for API compat
        async_op: bool = False,
    ) -> "_NCCLxWork | None":
        """
        ReduceScatter (sum).  Matches the TorchComms reduce_scatter_single
        signature: output first, then input.

        output shape : (..., N)               — per-rank shard
        input_ shape : (..., N * world_size)  — full tensor (all shards)
        """
        assert input_.is_contiguous() and output.is_contiguous(), \
            "ncclx_shim: tensors must be contiguous"
        assert input_.is_cuda and output.is_cuda, \
            "ncclx_shim: tensors must be on CUDA"
        dtype = _DTYPE_TO_NCCL.get(input_.dtype)
        if dtype is None:
            raise ValueError(f"ncclx_shim: unsupported dtype {input_.dtype}")
        _dbg("reduce_scatter_single: input=%s output=%s dtype=%s async=%s",
             tuple(input_.shape), tuple(output.shape), input_.dtype, async_op)
        stream = torch.cuda.current_stream().cuda_stream
        _ext.reduce_scatter(
            input_.data_ptr(), output.data_ptr(),
            output.numel(), dtype, self._comm, stream,
        )
        _dbg("reduce_scatter_single: enqueued on stream 0x%x", stream)
        return _NCCLxWork() if async_op else None

    def all_gather_single(
        self,
        output: torch.Tensor,
        input_: torch.Tensor,
        async_op: bool = False,
    ) -> "_NCCLxWork | None":
        """
        AllGather.  output first, then input (mirrors TorchComms convention).

        input_  shape : (..., N)               — this rank's shard
        output  shape : (..., N * world_size)  — gathered result
        """
        assert input_.is_contiguous() and output.is_contiguous(), \
            "ncclx_shim: tensors must be contiguous"
        assert input_.is_cuda and output.is_cuda, \
            "ncclx_shim: tensors must be on CUDA"
        dtype = _DTYPE_TO_NCCL.get(input_.dtype)
        if dtype is None:
            raise ValueError(f"ncclx_shim: unsupported dtype {input_.dtype}")
        _dbg("all_gather_single: input=%s output=%s dtype=%s async=%s",
             tuple(input_.shape), tuple(output.shape), input_.dtype, async_op)
        stream = torch.cuda.current_stream().cuda_stream
        _ext.all_gather(
            input_.data_ptr(), output.data_ptr(),
            input_.numel(), dtype, self._comm, stream,
        )
        _dbg("all_gather_single: enqueued on stream 0x%x", stream)
        return _NCCLxWork() if async_op else None

    def synchronize(self) -> None:
        torch.cuda.current_stream().synchronize()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def __del__(self):
        if hasattr(self, "_comm") and self._comm:
            _dbg("NCCLxComm.__del__: destroying comm handle=0x%x", self._comm)
            try:
                _ext.destroy_comm(self._comm)
            except Exception:
                pass
            self._comm = 0
