# fuse_probe_mpi_gpu_aware(<out-var>) -- sets <out-var> to 1 if the MPI
# library this build links reports CUDA- or HIP-aware buffers (the
# OpenMPI/MPICH `MPIX_CUDA_AWARE_SUPPORT` / `MPIX_ROCM_AWARE_SUPPORT`
# extension macros, from <mpi-ext.h>), else 0. Never fails the configure --
# an MPI without the extension (MPICH without it built in, or no MPI at
# all) simply probes to 0. This is the FALLBACK value only:
# fuse::native::mpi_is_gpu_aware() (src/backend/native_migration.cpp)
# prefers a run-time `MPIX_Query_cuda_support()` call where the extension
# declares it, and uses FUSE_MPI_GPU_AWARE (this macro's result) only
# where it does not; fuse::native's own migration exchange does not
# consult it at all and always stages through host memory, per the
# standing priority (robustness before efficiency) -- a device-resident
# native-mesh migration path is still future work.
# fuse::transport::MpiTransport::is_gpu_aware() (src/transport/
# mpi_transport.cpp, docs/dev/device-resident-arcs.md) decides
# whether a channel Transport exchange may hand device pointers straight
# to MPI from the IDENTICAL run-time probe, duplicated rather than
# shared: transport's library does not link backend's, so a call from
# there into fuse::native would need a link edge tools/check-layering.py's
# header-only check cannot see and the CMake target graph does not grant.
function(fuse_probe_mpi_gpu_aware out_var)
  set(_result 0)
  if(TARGET MPI::MPI_CXX)
    include(CheckCXXSourceCompiles)
    set(CMAKE_REQUIRED_LIBRARIES MPI::MPI_CXX)
    check_cxx_source_compiles(
      "#include <mpi.h>\n#include <mpi-ext.h>\nint main() {\n#if defined(MPIX_CUDA_AWARE_SUPPORT) || defined(MPIX_ROCM_AWARE_SUPPORT)\n  return 0;\n#else\n  static_assert(false, \"no GPU-aware extension macro\");\n#endif\n}\n"
      FUSE_MPI_HAS_GPU_AWARE_EXTENSION)
    if(FUSE_MPI_HAS_GPU_AWARE_EXTENSION)
      set(_result 1)
    endif()
    unset(CMAKE_REQUIRED_LIBRARIES)
  endif()
  set(${out_var} ${_result} PARENT_SCOPE)
endfunction()
