// Standalone Python extension for sparse NCCL collectives.
// Obtains ncclComm_t from PyTorch's process group via pg._comm_ptr().
//
// Environment variables (read once at first call):
//   USE_SPARSE_RS=1  — use ncclReduceScatterSparse, else ncclReduceScatter
//   USE_SPARSE_AR=1  — use ncclAllReduceSparse,     else ncclAllReduce
//   USE_SPARSE_AG=1  — use ncclAllGatherSparse,     else ncclAllGather

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <nccl.h>
#include <cstdlib>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Env-var flags (read once, thread-safe enough for our use case)
// ---------------------------------------------------------------------------

static int g_use_sparse_rs = -1;
static int g_use_sparse_ar = -1;
static int g_use_sparse_ag = -1;

static void init_flags() {
  if (g_use_sparse_rs >= 0) return;
  auto getflag = [](const char* name) -> int {
    const char* v = getenv(name);
    return (v && v[0] == '1') ? 1 : 0;
  };
  g_use_sparse_rs = getflag("USE_SPARSE_RS");
  g_use_sparse_ar = getflag("USE_SPARSE_AR");
  g_use_sparse_ag = getflag("USE_SPARSE_AG");
}

// ---------------------------------------------------------------------------
// Data type mapping
// ---------------------------------------------------------------------------

static ncclDataType_t getNcclDataType(at::ScalarType t) {
  switch (t) {
    case at::kChar:    return ncclInt8;
    case at::kByte:    return ncclUint8;
    case at::kFloat:   return ncclFloat;
    case at::kDouble:  return ncclDouble;
    case at::kInt:     return ncclInt32;
    case at::kLong:    return ncclInt64;
    case at::kHalf:    return ncclHalf;
    case at::kBFloat16: return ncclBfloat16;
    default:
      TORCH_CHECK(false, "Unsupported dtype for sparse NCCL collective: ",
                  c10::toString(t));
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static ncclComm_t toComm(int64_t comm_ptr) {
  TORCH_CHECK(comm_ptr != 0,
    "comm_ptr is 0 — call dist.barrier() to warm up the communicator first");
  return reinterpret_cast<ncclComm_t>(comm_ptr);
}

static void checkNccl(ncclResult_t r, const char* op) {
  TORCH_CHECK(r == ncclSuccess,
    op, " failed: ", ncclGetErrorString(r));
}

// ---------------------------------------------------------------------------
// Collectives
// ---------------------------------------------------------------------------

// reduce_scatter: sparse or dense depending on USE_SPARSE_RS
// input:  [nranks * recvcount] elements
// output: [recvcount] elements  (this rank's slice after reduce-scatter)
void reduce_scatter_sparse(
    const at::Tensor& input,
    at::Tensor& output,
    int64_t comm_ptr) {
  init_flags();
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "dtype mismatch");

  auto comm   = toComm(comm_ptr);
  auto stream = at::cuda::getCurrentCUDAStream(input.device().index()).stream();
  auto dtype  = getNcclDataType(input.scalar_type());
  auto count  = static_cast<size_t>(output.numel());

  if (g_use_sparse_rs) {
    checkNccl(
      ncclReduceScatterSparse(
        input.data_ptr(), output.data_ptr(),
        count, dtype, ncclSum, comm, stream),
      "ncclReduceScatterSparse");
  } else {
    checkNccl(
      ncclReduceScatter(
        input.data_ptr(), output.data_ptr(),
        count, dtype, ncclSum, comm, stream),
      "ncclReduceScatter");
  }
}

// all_gather: sparse or dense depending on USE_SPARSE_AG
// input:  [sendcount] elements  (this rank's chunk)
// output: [nranks * sendcount] elements
void all_gather_sparse(
    const at::Tensor& input,
    at::Tensor& output,
    int64_t comm_ptr) {
  init_flags();
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "dtype mismatch");

  auto comm   = toComm(comm_ptr);
  auto stream = at::cuda::getCurrentCUDAStream(input.device().index()).stream();
  auto dtype  = getNcclDataType(input.scalar_type());
  auto count  = static_cast<size_t>(input.numel());

  if (g_use_sparse_ag) {
    checkNccl(
      ncclAllGatherSparse(
        input.data_ptr(), output.data_ptr(),
        count, dtype, comm, stream),
      "ncclAllGatherSparse");
  } else {
    checkNccl(
      ncclAllGather(
        input.data_ptr(), output.data_ptr(),
        count, dtype, comm, stream),
      "ncclAllGather");
  }
}

// all_reduce: sparse or dense depending on USE_SPARSE_AR
// input/output: [count] elements (in-place OR out-of-place)
void all_reduce_sparse(
    const at::Tensor& input,
    at::Tensor& output,
    int64_t comm_ptr) {
  init_flags();
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "dtype mismatch");
  TORCH_CHECK(input.numel() == output.numel(), "size mismatch");

  auto comm   = toComm(comm_ptr);
  auto stream = at::cuda::getCurrentCUDAStream(input.device().index()).stream();
  auto dtype  = getNcclDataType(input.scalar_type());
  auto count  = static_cast<size_t>(input.numel());

  if (g_use_sparse_ar) {
    checkNccl(
      ncclAllReduceSparse(
        input.data_ptr(), output.data_ptr(),
        count, dtype, ncclSum, comm, stream),
      "ncclAllReduceSparse");
  } else {
    checkNccl(
      ncclAllReduce(
        input.data_ptr(), output.data_ptr(),
        count, dtype, ncclSum, comm, stream),
      "ncclAllReduce");
  }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

PYBIND11_MODULE(sparse_comms_ext, m) {
  m.doc() = "Sparse NCCL collectives via PyTorch comm_ptr";

  m.def("reduce_scatter_sparse", &reduce_scatter_sparse,
    R"(
Reduce-scatter: sparse (ncclReduceScatterSparse) if USE_SPARSE_RS=1, else dense.

Args:
    input:    CUDA tensor, shape [nranks * recvcount]
    output:   CUDA tensor, shape [recvcount] — this rank's slice
    comm_ptr: int64 from pg._comm_ptr()
)",
    py::arg("input"), py::arg("output"), py::arg("comm_ptr"),
    py::call_guard<py::gil_scoped_release>());

  m.def("all_gather_sparse", &all_gather_sparse,
    R"(
All-gather: sparse (ncclAllGatherSparse) if USE_SPARSE_AG=1, else dense.

Args:
    input:    CUDA tensor, shape [sendcount] — this rank's chunk
    output:   CUDA tensor, shape [nranks * sendcount]
    comm_ptr: int64 from pg._comm_ptr()
)",
    py::arg("input"), py::arg("output"), py::arg("comm_ptr"),
    py::call_guard<py::gil_scoped_release>());

  m.def("all_reduce_sparse", &all_reduce_sparse,
    R"(
All-reduce: sparse (ncclAllReduceSparse) if USE_SPARSE_AR=1, else dense.

Args:
    input:    CUDA tensor, shape [count]
    output:   CUDA tensor, shape [count]  (may alias input for in-place)
    comm_ptr: int64 from pg._comm_ptr()
)",
    py::arg("input"), py::arg("output"), py::arg("comm_ptr"),
    py::call_guard<py::gil_scoped_release>());

}
