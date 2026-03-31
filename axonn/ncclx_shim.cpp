// ncclx_shim.cpp
//
// Loads libnccl.so.2 (the NCCLx build) into a fresh dlmopen linker namespace so
// its symbols never collide with PyTorch's built-in libnccl.  All NCCL calls go
// through explicit function pointers obtained via dlsym inside that namespace.
//
// Python API (via pybind11):
//   load_ncclx(path)                              -- load once before anything
//   else get_unique_id()                    -> bytes    -- 128-byte
//   ncclUniqueId (rank 0) create_comm(uid, rank, size)       -> uint64   --
//   world communicator handle comm_split(parent, color, key)     -> uint64   --
//   sub-communicator handle comm_rank(comm)                    -> int
//   comm_size(comm)                    -> int
//   reduce_scatter(send, recv, recv_count, dtype_int, comm, stream)
//   all_gather(send, recv, send_count, dtype_int, comm, stream)
//   destroy_comm(comm)
//
// dtype_int values match stable NCCL 2.x enum: float16=6, float32=7, float64=8,
// bfloat16=9
//
// Debug: set NCCLX_DEBUG=1 in the environment for verbose per-op logging to
// stderr.

#define _GNU_SOURCE
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <link.h>
#include <stdexcept>
#include <string>

#include <pybind11/pybind11.h>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Debug helper — gated on NCCLX_DEBUG=1
// ---------------------------------------------------------------------------
static bool g_debug = false;

#define NCCLX_LOG(fmt, ...)                                                    \
  do {                                                                         \
    if (g_debug)                                                               \
      fprintf(stderr, "[ncclx_shim] " fmt "\n", ##__VA_ARGS__);                \
  } while (0)

// dtype_int → human-readable string
static const char *dtype_name(int d) {
  switch (d) {
  case 2:
    return "int32";
  case 4:
    return "int64";
  case 6:
    return "float16";
  case 7:
    return "float32";
  case 8:
    return "float64";
  case 9:
    return "bfloat16";
  default:
    return "unknown";
  }
}

// ---------------------------------------------------------------------------
// Minimal NCCL ABI types (stable across NCCL 2.x; no nccl.h required)
// ---------------------------------------------------------------------------
#define NCCL_UNIQUE_ID_BYTES 128
struct NcclUniqueId {
  char internal[NCCL_UNIQUE_ID_BYTES];
};
typedef void *NcclComm_t;
typedef int NcclResult_t;

// ncclCommSplit color value meaning "exclude this rank from any new comm"
static constexpr int NCCL_SPLIT_NOCOLOR = -1;

// ---------------------------------------------------------------------------
// Function pointer table
// ---------------------------------------------------------------------------
static void *g_handle = nullptr;

static NcclResult_t (*g_GetUniqueId)(NcclUniqueId *) = nullptr;
static NcclResult_t (*g_CommInitRank)(NcclComm_t *, int, NcclUniqueId,
                                      int) = nullptr;
static NcclResult_t (*g_CommSplit)(NcclComm_t, int, int, NcclComm_t *,
                                   void *) = nullptr;
static NcclResult_t (*g_CommUserRank)(NcclComm_t, int *) = nullptr;
static NcclResult_t (*g_CommCount)(NcclComm_t, int *) = nullptr;
static NcclResult_t (*g_ReduceScatter)(const void *, void *, size_t, int, int,
                                       NcclComm_t, void *) = nullptr;
static NcclResult_t (*g_AllGather)(const void *, void *, size_t, int,
                                   NcclComm_t, void *) = nullptr;
static NcclResult_t (*g_CommDestroy)(NcclComm_t) = nullptr;
static const char *(*g_GetErrorString)(NcclResult_t) = nullptr;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static void *load_sym(void *h, const char *name) {
  void *s = dlsym(h, name);
  if (!s)
    throw std::runtime_error(std::string("dlsym(") + name + "): " + dlerror());
  NCCLX_LOG("  resolved symbol: %s @ %p", name, s);
  return s;
}

static void check(NcclResult_t r, const char *op) {
  if (r != 0) {
    std::string msg = std::string(op) + " returned ";
    if (g_GetErrorString)
      msg += g_GetErrorString(r);
    else
      msg += std::to_string(r);
    fprintf(stderr, "[ncclx_shim] ERROR: %s\n", msg.c_str());
    throw std::runtime_error(msg);
  }
}

// Patch NCCLX's environ to point to the main program's environ.
// NCCLX is loaded with RTLD_DEEPBIND, so it resolves environ to its own
// libc copy which is uninitialized (NULL). This causes segfault in
// ncclCvarInit.
static void patch_ncclx_environ() {
  extern char **environ;
  char **main_environ = environ;

  NCCLX_LOG("patch_ncclx_environ: main environ = %p", main_environ);

  struct link_map *lm = nullptr;
  if (dlinfo(g_handle, RTLD_DI_LINKMAP, &lm) != 0) {
    NCCLX_LOG("patch_ncclx_environ: dlinfo failed: %s", dlerror());
    return;
  }

  // Debug: print all libraries in the link map
  NCCLX_LOG("patch_ncclx_environ: scanning link map...");
  for (struct link_map *cur = lm; cur != nullptr; cur = cur->l_next) {
    const char *name = cur->l_name ? cur->l_name : "(null)";
    NCCLX_LOG("  lib: %s @ %p", name, cur->l_addr);

    // Try broader pattern matching
    bool is_libc = (strstr(name, "libc") != nullptr) ||
                   (strstr(name, "libc-") != nullptr) ||
                   (strlen(name) == 0 &&
                    cur->l_addr != 0); // sometimes main exe has empty name

    if (is_libc || strstr(name, "ld-linux")) {
      NCCLX_LOG("  -> candidate: %s", name);

      void *libc_handle =
          dlopen(name[0] ? name : NULL, RTLD_NOW | RTLD_NOLOAD | RTLD_LOCAL);
      if (libc_handle) {
        // Try multiple symbol names
        char ***libc_environ_ptr = (char ***)dlsym(libc_handle, "__environ");
        char ***environ_ptr = (char ***)dlsym(libc_handle, "environ");

        NCCLX_LOG("  -> __environ = %p, environ = %p",
                  libc_environ_ptr ? *libc_environ_ptr : nullptr,
                  environ_ptr ? *environ_ptr : nullptr);

        if (libc_environ_ptr && *libc_environ_ptr != main_environ) {
          *libc_environ_ptr = main_environ;
          NCCLX_LOG("  -> PATCHED __environ");
          return; // Success
        }
        if (environ_ptr && *environ_ptr != main_environ) {
          *environ_ptr = main_environ;
          NCCLX_LOG("  -> PATCHED environ");
          return; // Success
        }
        dlclose(libc_handle);
      } else {
        NCCLX_LOG("  -> dlopen failed: %s", dlerror());
      }
    }
  }

  NCCLX_LOG("patch_ncclx_environ: could not find libc to patch");
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Load NCCLx into a brand-new linker namespace.  Idempotent.
void load_ncclx(const std::string &path) {
  extern char **environ; // The real, global environment
  g_debug = (std::getenv("NCCLX_DEBUG") != nullptr &&
             std::string(std::getenv("NCCLX_DEBUG")) != "0");

  if (g_handle) {
    NCCLX_LOG("load_ncclx: already loaded, skipping");
    return;
  }

  // Use dlopen with RTLD_LOCAL + RTLD_DEEPBIND instead of dlmopen(LM_ID_NEWLM).
  //
  // dlmopen creates a fully private linker namespace, which causes NCCLx to
  // load its own copy of libcudart.so — two CUDA runtimes in one process →
  // cudaErrorOperatingSystem (304) in NCCLx's constructor → rank 0 crash.
  //
  // RTLD_LOCAL:     NCCLx's nccl* symbols are NOT exported to the global
  //                 namespace, so they never shadow PyTorch's bundled NCCL.
  // RTLD_DEEPBIND:  NCCLx's own code looks up symbols in its own table first,
  //                 so NCCLx internal calls route to itself correctly.
  //                 Dependencies (libcudart, libcuda, libfabric) still resolve
  //                 against the already-loaded global copies → single CUDA
  //                 runtime.
  NCCLX_LOG("load_ncclx: dlopen(\"%s\") RTLD_LOCAL/* |RTLD_DEEPBIND */",
            path.c_str());
  g_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL /* | RTLD_DEEPBIND */);
  if (!g_handle) {
    std::string err = "dlopen(" + path + "): " + std::string(dlerror());
    fprintf(stderr, "[ncclx_shim] ERROR: %s\n", err.c_str());
    throw std::runtime_error(err);
  }
  NCCLX_LOG("load_ncclx: dlopen succeeded, handle=%p", g_handle);

  // NCCLX_LOG("Patching libc environ");
  // patch_ncclx_environ();

  // Inside load_ncclx, right after dlopen successfully returns g_handle:

  // // 1. Try to patch NCCLx's local reference to environ
  // char ***ncclx_environ = (char ***)dlsym(g_handle, "environ");
  // if (ncclx_environ) {
  //   NCCLX_LOG("load_ncclx: patched NCCLx's local environ pointer");
  //   *ncclx_environ = environ;
  // }

  // // 2. Try to patch NCCLx's local reference to __environ (used by getenv)
  // char ***ncclx_under_environ = (char ***)dlsym(g_handle, "__environ");
  // if (ncclx_under_environ) {
  //   NCCLX_LOG("load_ncclx: patched NCCLx's local __environ pointer");
  //   *ncclx_under_environ = environ;
  // }
  // NCCLX_LOG("Patched libc environ");

  g_GetUniqueId =
      (NcclResult_t (*)(NcclUniqueId *))load_sym(g_handle, "ncclGetUniqueId");
  g_CommInitRank =
      (NcclResult_t (*)(NcclComm_t *, int, NcclUniqueId, int))load_sym(
          g_handle, "ncclCommInitRank");
  g_CommSplit = (NcclResult_t (*)(NcclComm_t, int, int, NcclComm_t *,
                                  void *))load_sym(g_handle, "ncclCommSplit");
  g_CommUserRank = (NcclResult_t (*)(NcclComm_t, int *))load_sym(
      g_handle, "ncclCommUserRank");
  g_CommCount =
      (NcclResult_t (*)(NcclComm_t, int *))load_sym(g_handle, "ncclCommCount");
  g_ReduceScatter =
      (NcclResult_t (*)(const void *, void *, size_t, int, int, NcclComm_t,
                        void *))load_sym(g_handle, "ncclReduceScatter");
  g_AllGather = (NcclResult_t (*)(const void *, void *, size_t, int, NcclComm_t,
                                  void *))load_sym(g_handle, "ncclAllGather");
  g_CommDestroy =
      (NcclResult_t (*)(NcclComm_t))load_sym(g_handle, "ncclCommDestroy");
  g_GetErrorString =
      (const char *(*)(NcclResult_t))load_sym(g_handle, "ncclGetErrorString");

  NCCLX_LOG("load_ncclx: all symbols resolved");
}

py::bytes get_unique_id() {
  // Use PyTorch's bundled ncclGetUniqueId from the global symbol namespace.
  // NCCLx's own ncclGetUniqueId segfaults under RTLD_DEEPBIND; PyTorch's
  // libnccl.so.2 is already loaded globally, so dlsym(RTLD_DEFAULT) finds it.
  // The bootstrap TCP handshake is compatible — NCCLx's ncclCommInitRank will
  // connect to the socket PyTorch's NCCL opens.
  typedef NcclResult_t (*Fn)(NcclUniqueId *);
  auto fn = reinterpret_cast<Fn>(dlsym(RTLD_DEFAULT, "ncclGetUniqueId"));
  if (!fn)
    throw std::runtime_error(
        std::string("ncclGetUniqueId not found in global namespace: ") +
        dlerror());
  NcclUniqueId uid;
  check(fn(&uid), "ncclGetUniqueId (system)");
  NCCLX_LOG("get_unique_id: generated 128-byte unique ID via global NCCL");
  return py::bytes(uid.internal, NCCL_UNIQUE_ID_BYTES);
}

uint64_t create_comm(py::bytes uid_bytes, int rank, int size) {
  if (!g_handle)
    throw std::runtime_error("call load_ncclx() first");
  std::string s = uid_bytes;
  if ((int)s.size() != NCCL_UNIQUE_ID_BYTES)
    throw std::invalid_argument("uid must be exactly 128 bytes");
  NcclUniqueId uid;
  std::memcpy(uid.internal, s.data(), NCCL_UNIQUE_ID_BYTES);
  NcclComm_t comm = nullptr;
  NCCLX_LOG("create_comm: ncclCommInitRank rank=%d size=%d", rank, size);
  check(g_CommInitRank(&comm, size, uid, rank), "ncclCommInitRank");
  NCCLX_LOG("create_comm: success, comm=%p", comm);
  return reinterpret_cast<uint64_t>(comm);
}

// Collective: all ranks in the parent comm must call this simultaneously.
uint64_t comm_split(uint64_t parent_handle, int color, int key) {
  NCCLX_LOG("comm_split: parent=%p color=%d key=%d",
            reinterpret_cast<void *>(parent_handle), color, key);
  NcclComm_t newcomm = nullptr;
  check(g_CommSplit(reinterpret_cast<NcclComm_t>(parent_handle), color, key,
                    &newcomm,
                    /*config=*/nullptr),
        "ncclCommSplit");
  NCCLX_LOG("comm_split: success, child=%p", newcomm);
  return reinterpret_cast<uint64_t>(newcomm);
}

int comm_rank(uint64_t comm_handle) {
  int r = 0;
  check(g_CommUserRank(reinterpret_cast<NcclComm_t>(comm_handle), &r),
        "ncclCommUserRank");
  return r;
}

int comm_size(uint64_t comm_handle) {
  int n = 0;
  check(g_CommCount(reinterpret_cast<NcclComm_t>(comm_handle), &n),
        "ncclCommCount");
  return n;
}

// ReduceScatter: sendbuf = (recv_count * world_size) elems, recvbuf =
// recv_count elems.
void reduce_scatter(uint64_t sendbuf, uint64_t recvbuf, size_t recv_count,
                    int dtype_int, uint64_t comm_handle, uint64_t stream) {
  NCCLX_LOG("reduce_scatter: comm=%p recv_count=%zu dtype=%s stream=%p",
            reinterpret_cast<void *>(comm_handle), recv_count,
            dtype_name(dtype_int), reinterpret_cast<void *>(stream));
  check(g_ReduceScatter(reinterpret_cast<const void *>(sendbuf),
                        reinterpret_cast<void *>(recvbuf), recv_count,
                        dtype_int, /*ncclSum=*/0,
                        reinterpret_cast<NcclComm_t>(comm_handle),
                        reinterpret_cast<void *>(stream)),
        "ncclReduceScatter");
  NCCLX_LOG("reduce_scatter: kernel enqueued");
}

// AllGather: sendbuf = send_count elems, recvbuf = (send_count * world_size)
// elems.
void all_gather(uint64_t sendbuf, uint64_t recvbuf, size_t send_count,
                int dtype_int, uint64_t comm_handle, uint64_t stream) {
  NCCLX_LOG("all_gather: comm=%p send_count=%zu dtype=%s stream=%p",
            reinterpret_cast<void *>(comm_handle), send_count,
            dtype_name(dtype_int), reinterpret_cast<void *>(stream));
  check(g_AllGather(reinterpret_cast<const void *>(sendbuf),
                    reinterpret_cast<void *>(recvbuf), send_count, dtype_int,
                    reinterpret_cast<NcclComm_t>(comm_handle),
                    reinterpret_cast<void *>(stream)),
        "ncclAllGather");
  NCCLX_LOG("all_gather: kernel enqueued");
}

void destroy_comm(uint64_t comm_handle) {
  if (!comm_handle)
    return;
  NCCLX_LOG("destroy_comm: comm=%p", reinterpret_cast<void *>(comm_handle));
  check(g_CommDestroy(reinterpret_cast<NcclComm_t>(comm_handle)),
        "ncclCommDestroy");
}

// ---------------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "NCCLx shim: dlmopen-isolated NCCLx in a private linker namespace";
  m.def("load_ncclx", &load_ncclx,
        "Load NCCLx .so into a private linker namespace");
  m.def("get_unique_id", &get_unique_id,
        "Return 128-byte ncclUniqueId (rank-0 only)");
  m.def("create_comm", &create_comm,
        "Create world comm from uid bytes, rank, size");
  m.def("comm_split", &comm_split,
        "ncclCommSplit(parent, color, key) -> child handle");
  m.def("comm_rank", &comm_rank, "ncclCommUserRank -> int");
  m.def("comm_size", &comm_size, "ncclCommCount -> int");
  m.def("reduce_scatter", &reduce_scatter,
        "ncclReduceScatter(send, recv, recv_count, dtype, comm, stream)");
  m.def("all_gather", &all_gather,
        "ncclAllGather   (send, recv, send_count, dtype, comm, stream)");
  m.def("destroy_comm", &destroy_comm, "ncclCommDestroy");
}
