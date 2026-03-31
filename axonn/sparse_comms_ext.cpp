// Standalone Python extension for sparse NCCL collectives.
// Obtains ncclComm_t from PyTorch's process group via pg._comm_ptr().
// Obtains the internal NCCL stream via get_nccl_stream_ptr(), using the
// #define private public trick to read ncclStreams_ without patching PyTorch.
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

// ---------------------------------------------------------------------------
// Access ncclStreams_ from ProcessGroupNCCL without patching PyTorch.
// #define private public makes all private members accessible; memory layout
// is unaffected by access specifiers so this is safe at runtime.
// USE_C10D_NCCL must be defined to unlock the #ifdef guard in the header.
// ---------------------------------------------------------------------------
#define USE_C10D_NCCL
#define private public
#define protected public
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#undef protected
#undef private

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
// Stream accessor
//
// Returns the raw cudaStream_t (as int64_t) that ProcessGroupNCCL uses for
// async collectives on the given device — the same stream internal to torch.
// Returns 0 if the communicator has not been warmed up yet.
// ---------------------------------------------------------------------------

int64_t get_nccl_stream_ptr(py::object backend_obj, int device_index) {
  auto* pg = backend_obj.cast<c10d::ProcessGroupNCCL*>();
  std::string key = std::to_string(device_index);
  auto it = pg->ncclStreams_.find(key);
  if (it == pg->ncclStreams_.end()) return 0;
  return reinterpret_cast<int64_t>(it->second.stream());
}

// ---------------------------------------------------------------------------
// Collectives
//
// stream_ptr is a cudaStream_t cast to int64_t — the caller selects the
// appropriate stream (torch's internal NCCL stream for async, current stream
// for sync) and passes it here, matching NCCL's own calling convention.
// ---------------------------------------------------------------------------

void reduce_scatter_sparse(
    const at::Tensor& input,
    at::Tensor& output,
    int64_t comm_ptr,
    int64_t stream_ptr) {
  init_flags();
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "dtype mismatch");

  auto comm   = toComm(comm_ptr);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
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

void all_gather_sparse(
    const at::Tensor& input,
    at::Tensor& output,
    int64_t comm_ptr,
    int64_t stream_ptr) {
  init_flags();
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "dtype mismatch");

  auto comm   = toComm(comm_ptr);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
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

void all_reduce_sparse(
    const at::Tensor& input,
    at::Tensor& output,
    int64_t comm_ptr,
    int64_t stream_ptr) {
  init_flags();
  TORCH_CHECK(input.is_cuda() && output.is_cuda(), "Tensors must be on CUDA");
  TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Tensors must be contiguous");
  TORCH_CHECK(input.scalar_type() == output.scalar_type(), "dtype mismatch");
  TORCH_CHECK(input.numel() == output.numel(), "size mismatch");

  auto comm   = toComm(comm_ptr);
  auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
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

  m.def("get_nccl_stream_ptr", &get_nccl_stream_ptr,
    R"(
Return the raw cudaStream_t (as int64_t) that ProcessGroupNCCL uses for
async collectives on device_index.  Returns 0 if not yet initialized.

Args:
    backend_obj: the ProcessGroupNCCL backend (pg._get_backend(device))
    device_index: int, CUDA device index
)",
    py::arg("backend_obj"), py::arg("device_index"));

  m.def("reduce_scatter_sparse", &reduce_scatter_sparse,
    R"(
Reduce-scatter: sparse (ncclReduceScatterSparse) if USE_SPARSE_RS=1, else dense.

Args:
    input:      CUDA tensor, shape [nranks * recvcount]
    output:     CUDA tensor, shape [recvcount]
    comm_ptr:   int64 from pg._comm_ptr()
    stream_ptr: int64 cudaStream_t
)",
    py::arg("input"), py::arg("output"), py::arg("comm_ptr"), py::arg("stream_ptr"),
    py::call_guard<py::gil_scoped_release>());

  m.def("all_gather_sparse", &all_gather_sparse,
    R"(
All-gather: sparse (ncclAllGatherSparse) if USE_SPARSE_AG=1, else dense.

Args:
    input:      CUDA tensor, shape [sendcount]
    output:     CUDA tensor, shape [nranks * sendcount]
    comm_ptr:   int64 from pg._comm_ptr()
    stream_ptr: int64 cudaStream_t
)",
    py::arg("input"), py::arg("output"), py::arg("comm_ptr"), py::arg("stream_ptr"),
    py::call_guard<py::gil_scoped_release>());

  m.def("all_reduce_sparse", &all_reduce_sparse,
    R"(
All-reduce: sparse (ncclAllReduceSparse) if USE_SPARSE_AR=1, else dense.

Args:
    input:      CUDA tensor, shape [count]
    output:     CUDA tensor, shape [count]  (may alias input for in-place)
    comm_ptr:   int64 from pg._comm_ptr()
    stream_ptr: int64 cudaStream_t
)",
    py::arg("input"), py::arg("output"), py::arg("comm_ptr"), py::arg("stream_ptr"),
    py::call_guard<py::gil_scoped_release>());
}
