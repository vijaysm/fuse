#pragma once

/// Error-free transformations of binary64, callable from host and device code:
/// the building blocks of fuse::num::dd and fuse::num::qd.
///
///   two_sum(a, b)    s = fl(a + b), and r with  a + b = s + r  exactly (Knuth 1969)
///   quick_two_sum    the same when |a| >= |b| (Dekker 1971)
///   two_prod(a, b)   p = fl(a * b), and r with  a * b = p + r  exactly, by a fused
///                    multiply-add (Dekker 1971 split, Ogita, Rump and Oishi 2005)
///   three_sum        x + y + z  as  r0 + r1 + r2  exactly (Hida, Li and Bailey 2001)
///
/// Each returns the rounded result and writes the exact error through a
/// reference, and each writes a zero error when the rounded result is not
/// finite (an infinity or a NaN has no error to carry).
///
/// CONTRACTION. An error-free transformation is only as good as the rounding it
/// is told to do: if a compiler fuses a product into a following sum, the
/// rounded product the algorithm reasons about never exists. FMA contraction is
/// on by default for nvcc and for GCC, so every product that feeds a sum goes
/// through `mul()`, which is a single IEEE multiplication the compiler cannot
/// fold into a neighbouring add (`__dmul_rn` on a device, a volatile store on
/// the host). The sums themselves are not at risk, nothing reassociates them
/// without -ffast-math, which FUSE does not use.

#include <cstdint>

#include <Kokkos_Core.hpp>

namespace fuse::num::eft {

/// No libm, so it works on every execution space: the exponent field is not all ones.
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION bool is_finite(double x) noexcept {
  return (Kokkos::bit_cast<std::uint64_t>(x) & 0x7FF0000000000000ull) !=
         0x7FF0000000000000ull;
}
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION bool is_nan(double x) noexcept {
  return (Kokkos::bit_cast<std::uint64_t>(x) & 0x7FFFFFFFFFFFFFFFull) >
         0x7FF0000000000000ull;
}
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION bool is_inf(double x) noexcept {
  return (Kokkos::bit_cast<std::uint64_t>(x) & 0x7FFFFFFFFFFFFFFFull) ==
         0x7FF0000000000000ull;
}
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION bool signbit(double x) noexcept {
  return (Kokkos::bit_cast<std::uint64_t>(x) >> 63) != 0;
}

/// The quiet NaN and the positive infinity, from their bit patterns (no libm, no
/// constexpr host function called from device code).
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION double quiet_nan() noexcept {
  return Kokkos::bit_cast<double>(0x7FF8000000000000ull);
}
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION double infinity() noexcept {
  return Kokkos::bit_cast<double>(0x7FF0000000000000ull);
}

/// One IEEE multiplication, rounded to nearest, that is never contracted.
[[nodiscard]] KOKKOS_FORCEINLINE_FUNCTION double mul(double a, double b) noexcept {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __dmul_rn(a, b);
#else
  volatile double p = a * b;
  return p;
#endif
}

KOKKOS_FORCEINLINE_FUNCTION double two_sum(double a, double b, double& r) noexcept {
  const double s = a + b;
  if (is_finite(s)) {
    const double bb = s - a;
    r = (a - (s - bb)) + (b - bb);
  } else {
    r = 0.0;
  }
  return s;
}

/// Requires |a| >= |b| (or a == 0).
KOKKOS_FORCEINLINE_FUNCTION double quick_two_sum(double a, double b, double& r) noexcept {
  const double s = a + b;
  r = is_finite(s) ? b - (s - a) : 0.0;
  return s;
}

KOKKOS_FORCEINLINE_FUNCTION double two_prod(double a, double b, double& r) noexcept {
  const double p = mul(a, b);
  r = is_finite(p) ? Kokkos::fma(a, b, -p) : 0.0;
  return p;
}

KOKKOS_FORCEINLINE_FUNCTION double two_sqr(double a, double& r) noexcept {
  const double p = mul(a, a);
  r = is_finite(p) ? Kokkos::fma(a, a, -p) : 0.0;
  return p;
}

/// x + y + z = r0 + r1 + r2: x becomes r0, y becomes r1, z becomes r2.
KOKKOS_FORCEINLINE_FUNCTION void three_sum(double& x, double& y, double& z) noexcept {
  double u, v, w;
  u = two_sum(x, y, v);
  x = two_sum(z, u, w);
  y = two_sum(v, w, z);
}

/// x + y + z = r0 + r1 to the accuracy of the last sum: x becomes r0, y becomes r1.
KOKKOS_FORCEINLINE_FUNCTION void three_sum2(double& x, double& y, double z) noexcept {
  double u, v, w;
  u = two_sum(x, y, v);
  x = two_sum(z, u, w);
  y = v + w;
}

}  // namespace fuse::num::eft
