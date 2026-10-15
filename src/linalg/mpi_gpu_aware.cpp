// src/linalg/mpi_gpu_aware.cpp -- the shared GPU-aware MPI probe (Migration
// track, 2026-10-15; lead's layering decision, 2026-10-15 integration).
//
// BEFORE THIS FILE: `fuse::native::mpi_is_gpu_aware()`
// (src/backend/native_migration.cpp) and `fuse::transport::MpiTransport::
// is_gpu_aware()`'s own `transport_mpi_is_gpu_aware()`
// (src/transport/mpi_transport.cpp) each carried an IDENTICAL run-time
// probe, duplicated because `backend` could not depend on `transport`'s
// library and vice versa (cmake/FuseMpiGpuAware.cmake's own header
// comment states this explicitly). `fuse::linalg` sits BELOW both in the
// layering order (base < linalg < ... < search < backend < partition <
// transport < runtime, tools/check-layering.py) and `backend` already
// links `fuse::linalg` PRIVATE (src/CMakeLists.txt) -- so ONE probe here,
// in a layer both may depend on, replaces both duplicates; `backend` and
// `transport` now each call this instead of re-implementing it (see their
// own thin wrappers' comments for why the WRAPPER functions themselves
// stay, for source compatibility with every existing caller).
//
// Always compiled (added to fuse-coupler-linalg's sources unconditionally,
// like comm_serial.cpp), not only under FUSE_ENABLE_MPI: a build with no
// MPI still needs a definition of `fuse::linalg::mpi_is_gpu_aware()` that
// returns false, the same reason `fuse::native::mpi_is_gpu_aware()`
// (declared outside any `#if FUSE_HAVE_MPI` in native_migration.hpp) is
// itself unconditionally callable.

#include "fuse/linalg/comm.hpp"

#include "fuse/base/config.hpp"  // FUSE_HAVE_MPI, FUSE_MPI_GPU_AWARE

#include <cstdlib>

#if FUSE_HAVE_MPI
#include <mpi.h>
// Declares MPIX_Query_cuda_support()/MPIX_CUDA_AWARE_SUPPORT when this
// MPI was built with CUDA-aware support (cmake/FuseMpiGpuAware.cmake's
// own configure-time probe compiles against the same header). Absent on
// an MPI with no such extension, so the include is conditional.
#if defined(__has_include)
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#endif
#endif

namespace fuse::linalg {

bool mpi_is_gpu_aware() {
  // `FUSE_TRANSPORT_HOST_STAGING=1` forces this false regardless of what
  // either probe below reports -- the one lever that exercises the
  // host-staged fallback path on a machine whose MPI genuinely IS
  // GPU-aware, matching every existing caller's own override convention
  // (native_migration.cpp's and mpi_transport.cpp's own, now unified
  // here).
  if (const char* force = std::getenv("FUSE_TRANSPORT_HOST_STAGING");
      force != nullptr && force[0] != '\0' && force[0] != '0') {
    return false;
  }
#if FUSE_HAVE_MPI && defined(MPIX_CUDA_AWARE_SUPPORT)
  return MPIX_Query_cuda_support() != 0;
#else
  return FUSE_MPI_GPU_AWARE != 0;
#endif
}

}  // namespace fuse::linalg
