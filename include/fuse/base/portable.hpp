#pragma once

/// What a header-only value type needs to be callable from a Kokkos kernel
/// without including Kokkos_Core.hpp: the `KOKKOS_INLINE_FUNCTION` annotation
/// and a bit reinterpretation that is defined on every execution space.
///
/// `std::memcpy` and `std::bit_cast` are host functions as far as nvcc is
/// concerned (a call from device code is a diagnostic that nvcc only warns
/// about, and the result is then undefined); `Kokkos::bit_cast` is annotated
/// for host and device and is implemented with the compiler's own builtin.

#include <Kokkos_BitManipulation.hpp>
#include <Kokkos_Macros.hpp>
