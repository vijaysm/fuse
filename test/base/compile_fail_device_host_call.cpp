// This must NOT compile: host_only() has no KOKKOS_FUNCTION annotation, so calling it
// from a device lambda is a cross-execution-space call, an error in every FUSE
// CUDA build (cmake/FuseDependencies.cmake).

#include <Kokkos_Core.hpp>

namespace {
double host_only(double x) { return 2.0 * x; }
}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  Kokkos::View<double*> v("v", 8);
  Kokkos::parallel_for(
      "fuse::test::device_host_call", 8,
      KOKKOS_LAMBDA(int i) { v(i) = host_only(static_cast<double>(i)); });
  Kokkos::fence();
  return 0;
}
