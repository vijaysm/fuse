// Control for base.compile_fail.device_host_call: the same kernel calling a
// KOKKOS_INLINE_FUNCTION, which MUST compile.

#include <Kokkos_Core.hpp>

namespace {
KOKKOS_INLINE_FUNCTION double device_ok(double x) { return 2.0 * x; }
}  // namespace

int main(int argc, char** argv) {
  Kokkos::ScopeGuard guard(argc, argv);
  Kokkos::View<double*> v("v", 8);
  Kokkos::parallel_for(
      "fuse::test::device_host_call", 8,
      KOKKOS_LAMBDA(int i) { v(i) = device_ok(static_cast<double>(i)); });
  Kokkos::fence();
  return 0;
}
