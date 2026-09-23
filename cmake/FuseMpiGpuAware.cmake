# fuse_probe_mpi_gpu_aware(<out-var>) -- sets <out-var> to 1 if the MPI
# library this build links reports CUDA- or HIP-aware buffers (the
# OpenMPI/MPICH `MPIX_CUDA_AWARE_SUPPORT` / `MPIX_ROCM_AWARE_SUPPORT`
# extension macros, from <mpi-ext.h>), else 0. Never fails the configure --
# an MPI without the extension (MPICH without it built in, or no MPI at
# all) simply probes to 0. See docs/dev/native-mesh.md: this result is
# recorded (fuse::native::mpi_is_gpu_aware()) but not yet ACTED on --
# fuse::native's migration exchange always stages through host memory,
# regardless of what this probe finds, per the standing priority
# (robustness before efficiency; a device-resident exchange path is
# future work).
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
