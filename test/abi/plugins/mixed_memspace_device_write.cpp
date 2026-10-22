/* D15 item 1 ("Mixed plugin" track, 2026-10-21): the one piece of
 * test/abi/plugins/mixed_memspace_provider.c's own p_advance() that
 * needs C++ instead of C -- Kokkos has no C binding (test/abi/plugins/
 * device_layout_provider.cpp's own header comment already states this
 * same reason for its own, separate plugin). Kept as its own
 * translation unit rather than converting the whole plugin to C++ (the
 * owner's own two shapes, Stage 2 of this track): mixed_memspace_
 * provider.c's own header comment already describes it as exercising
 * BOTH a host- and a device-resident field on one provider instance
 * FROM A C TRANSLATION UNIT, the one plugin in this suite that does;
 * converting the whole file to C++ would lose that coverage for no
 * benefit, since only this one function needs Kokkos at all.
 * include/fuse/abi/provider.h's own `#ifdef __cplusplus extern "C"`
 * wrapping already lets a C++-implemented function sit behind a
 * C-callable declaration -- this file applies that same mechanism to
 * one function instead of a whole plugin, and is compiled into each of
 * the three mixed_memspace_provider.c CMake targets (test/CMakeLists.txt)
 * alongside the .c file itself.
 *
 * fuse_test_mixed_memspace_write_device() launches a single-element
 * Kokkos::parallel_for on Kokkos::DefaultExecutionSpace to store value
 * through ptr -- the same approach device_layout_provider.cpp's own
 * write kernel uses for its own device-resident field, applied here to
 * a single double instead of a multi-component view.
 *
 * The kernel is fenced HERE, before this function returns (owner
 * decision 2026-10-22, KNOWN_ISSUES.md D15: the design must never depend
 * on a device pointer's visibility across streams or on the adapter's
 * own fence). src/abi/provider_adapter.cpp's ProviderAdapter::advance()
 * still issues an unconditional Kokkos::fence after advance() returns,
 * as a defensive measure; nothing in this library, and no row, is
 * specified in terms of it. Until 2026-10-22 this file relied on that
 * adapter fence and the row passed 10 of 10 on CUDA with it removed, so
 * the row never distinguished the two designs.
 */
#include <Kokkos_Core.hpp>

extern "C" void fuse_test_mixed_memspace_write_device(double* ptr, double value) {
  Kokkos::parallel_for(
      "mixed_memspace_provider.device_write",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(int) { *ptr = value; });
  Kokkos::fence("mixed_memspace_provider.device_write_fence");
}
