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
 * No Kokkos::fence() in this file: src/abi/provider_adapter.cpp's own
 * ProviderAdapter::advance() already issues an unconditional
 * Kokkos::fence("provider_adapter_advance_fence") immediately after
 * this plugin's own p_advance() returns, gated only on
 * has_device_fields_ (true whenever a device field was accepted).
 * Ordering this kernel's own completion before the Runner next reads
 * the field is exactly the property that existing fence exists to
 * guarantee; exercising it, rather than adding a second, redundant
 * fence here, is the point of the row this plugin serves -- see this
 * track's own measured_results for the host run that confirms a single
 * post-advance fence is sufficient (no fence added here, row passes).
 */
#include <Kokkos_Core.hpp>

extern "C" void fuse_test_mixed_memspace_write_device(double* ptr, double value) {
  Kokkos::parallel_for(
      "mixed_memspace_provider.device_write",
      Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0, 1),
      KOKKOS_LAMBDA(int) { *ptr = value; });
}
