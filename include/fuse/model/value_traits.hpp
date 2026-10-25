#pragma once

/// The value type of a coupled field (plans/2026-10-22-universal-number-types.md,
/// stage U1): the compile-time trait `fuse::value_traits<T>`, the concept
/// `fuse::CoupledValue<T>` every field value type satisfies, and the run-time
/// descriptor `fuse::FieldValueType` a type-erased field carries so that a request
/// for the wrong type is refused by name instead of reinterpreting bytes.
///
/// WHY AN OPEN SET HERE, WHEN `ElementType` IS CLOSED. `fuse::ElementType`
/// (element_type.hpp) tags bytes on disk, which a reader built from an older
/// version must still be able to name, so it is a closed, append-only
/// enumeration. A field's value type inside one running program has no such
/// constraint: any `T` whose `value_traits<T>` says how it crosses the
/// non-template boundaries (virtual operators, the C ABI, MPI, persisted
/// files) can be a field value type, including a user's own type. What `T`
/// cannot do is cross one of those boundaries without saying how; the `kind`
/// below is that statement, and nothing in this header lets a `T` without a
/// `value_traits` specialization through.
///
/// THE THREE KINDS (section 5 of the plan):
///
///   ScalarArray  `T` is `scalar_count` consecutive scalars of type
///               `scalar_type`: std::complex<double> is two doubles, a
///               forward-mode dual number with N tangents is N+1 doubles. Every
///               operation the coupler applies to field values (a remap
///               y = W x whose weights do not depend on the values, a halo copy,
///               a transfer, time interpolation) acts on each scalar
///               separately, so a field of such values moves as a field of
///               scalars with `scalar_count` times as many components, and
///               tangents travel through a remap exactly: the tangent of W x is
///               W applied to the tangent. binary64 itself is the case of one
///               scalar.
///   Arithmetic  `T` needs its own arithmetic inside an operator (double-double,
///               quad-double): the operator is instantiated for `T` from a
///               closed list (stage U4).
///   Storage     `T` is a compact format (binary32, bfloat16, binary16, posit):
///               widened exactly to binary64 at a boundary and rounded once,
///               with round to nearest, ties to even, on the way back.
///
/// WHAT "EACH SCALAR SEPARATELY" LEAVES OUT. `derivative_count` of the scalars
/// (the last ones) are derivatives of the value, the tangents of a forward-mode
/// dual number; the first `scalar_count - derivative_count` are the value
/// itself. Two coupler operations are not independent per scalar and so are not
/// safe on derivatives:
///
///   * an affine map v -> s v + b (a unit conversion with an offset, such as
///     degrees Celsius to kelvin) adds b to the value, and its derivative is
///     s t: the offset must reach the value scalars only;
///   * an accumulation by minimum or maximum picks one sample, and the tangent
///     of the result is the tangent of the sample picked, which comparing the
///     tangents scalar by scalar does not give.
///
/// The compiler already refuses an affine step for every field (E-IR-UNLOWERABLE),
/// so only the second can occur today; the runner refuses it for a field with
/// derivatives (E-RT-DERIVATIVE-MINMAX). Scaling, remap, interpolation and the
/// accumulations by sum, mean and integral are linear, hence scalar by scalar.
///
/// `to_binary64` and `from_binary64` are the projection every boundary that
/// needs binary64 uses: exact widening for a Storage type, the value part for
/// a ScalarArray type with more than one scalar. They are not a licence to
/// compute in binary64 behind a field's back; where they are applied is a
/// decision of the stage that applies them.

#include <bit>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <type_traits>

#include "fuse/model/bfloat16.hpp"

namespace fuse {

/// How a value type crosses a boundary that is not a template (see the header
/// comment). The numeric values are not persisted anywhere; they may change.
enum class FieldValueKind : std::uint8_t {
  ScalarArray = 0,
  Arithmetic = 1,
  Storage = 2,
};

[[nodiscard]] constexpr const char* to_cstring(FieldValueKind k) noexcept {
  switch (k) {
    case FieldValueKind::ScalarArray: return "ScalarArray";
    case FieldValueKind::Arithmetic: return "Arithmetic";
    case FieldValueKind::Storage:    return "Storage";
  }
  return "Unknown";
}

/// Primary template deliberately left undefined: a type is not a field value
/// type until someone states, by specializing this, how it crosses the
/// boundaries. A specialization provides
///
///   static constexpr std::string_view name;   // unique, e.g. "binary64"
///   static constexpr FieldValueKind kind;
///   using scalar_type = ...;                  // T itself when scalar_count == 1
///   static constexpr int scalar_count;        // >= 1
///   static constexpr int derivative_count;    // 0 <= derivative_count < scalar_count
///   static double to_binary64(const T&);
///   static T from_binary64(double);
///
/// `name` identifies the type at run time (`FieldValueType` below). Two different
/// types must not share a name: the registry compares name, size and
/// alignment, and a collision between two types of equal size and alignment
/// is the one mismatch it cannot see.
template <class T>
struct value_traits;

template <>
struct value_traits<double> {
  static constexpr std::string_view name = "binary64";
  static constexpr FieldValueKind kind = FieldValueKind::ScalarArray;
  using scalar_type = double;
  static constexpr int scalar_count = 1;
  static constexpr int derivative_count = 0;
  [[nodiscard]] static constexpr double to_binary64(double v) noexcept {
    return v;
  }
  [[nodiscard]] static constexpr double from_binary64(double v) noexcept {
    return v;
  }
};

template <>
struct value_traits<float> {
  static constexpr std::string_view name = "binary32";
  static constexpr FieldValueKind kind = FieldValueKind::Storage;
  using scalar_type = float;
  static constexpr int scalar_count = 1;
  static constexpr int derivative_count = 0;
  /// Exact: every binary32 value is a binary64 value.
  [[nodiscard]] static constexpr double to_binary64(float v) noexcept {
    return static_cast<double>(v);
  }
  /// Round to nearest, ties to even, under the default floating-point
  /// environment (this project never changes the rounding mode).
  [[nodiscard]] static constexpr float from_binary64(double v) noexcept {
    return static_cast<float>(v);
  }
};

namespace detail {

/// binary64 -> binary32 rounded to odd: the exact result when it is
/// representable, otherwise whichever of the two neighbouring binary32 values
/// has an odd last significand bit. Rounding to odd in an intermediate format
/// with at least p + 2 significand bits, followed by one rounding to nearest
/// to the p-bit target, equals a single correct rounding to the target
/// (Boldo and Melquiond, "Emulation of FMA and correctly rounded sums: proved
/// algorithms using rounding to odd", IEEE Trans. Computers 57(4), 2008).
/// binary32 has 24 bits and bfloat16 8, so this makes binary64 -> bfloat16
/// correctly rounded. Rounding to nearest twice is not: a binary64 value just
/// above a bfloat16 midpoint can first round DOWN onto the midpoint in
/// binary32, and the tie then goes to even.
[[nodiscard]] inline float round_to_odd_binary32(double v) noexcept {
  float f = static_cast<float>(v);  // round to nearest, ties to even
  if (std::isnan(v) || static_cast<double>(f) == v) return f;
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(f);
  if ((bits & 1u) == 0u) {
    // f is the even neighbour; the odd neighbour lies on v's side of f.
    // When f overflowed to infinity this steps back to the largest finite
    // binary32 value, which is odd, as rounding to odd requires.
    f = std::nextafter(f, static_cast<double>(f) < v
                              ? std::numeric_limits<float>::infinity()
                              : -std::numeric_limits<float>::infinity());
  }
  return f;
}

}  // namespace detail

template <>
struct value_traits<bfloat16> {
  static constexpr std::string_view name = "bfloat16";
  static constexpr FieldValueKind kind = FieldValueKind::Storage;
  using scalar_type = bfloat16;
  static constexpr int scalar_count = 1;
  static constexpr int derivative_count = 0;
  /// Exact: bfloat16 -> binary32 is exact (bfloat16.hpp) and so is
  /// binary32 -> binary64.
  [[nodiscard]] static double to_binary64(bfloat16 v) noexcept {
    return static_cast<double>(v.to_float());
  }
  /// Correctly rounded (to nearest, ties to even); see
  /// detail::round_to_odd_binary32 for why the intermediate binary32 value is
  /// rounded to odd rather than to nearest.
  [[nodiscard]] static bfloat16 from_binary64(double v) noexcept {
    return bfloat16::from_float(detail::round_to_odd_binary32(v));
  }
};

namespace detail {

/// True when the scalar type of `T` is `T` itself or has a `value_traits` of its
/// own to take a name from (a dual number's scalar type is binary64, which has).
template <class T>
concept ScalarTypeNamed =
    std::is_same_v<typename value_traits<T>::scalar_type, T> ||
    requires {
      { value_traits<typename value_traits<T>::scalar_type>::name }
          -> std::convertible_to<std::string_view>;
    };

}  // namespace detail

/// A field value type: trivially copyable and standard layout (so its bytes
/// can move through MPI, Kokkos::deep_copy and a type-erased registry), with a
/// complete `value_traits<T>`. For a ScalarArray type the scalars must tile `T`
/// exactly, because viewing a field as scalars addresses scalar `k` of value
/// `v` as `reinterpret_cast<const scalar_type*>(&v)[k]`.
template <class T>
concept CoupledValue =
    std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> &&
    requires {
      typename value_traits<T>::scalar_type;
      { value_traits<T>::name } -> std::convertible_to<std::string_view>;
      { value_traits<T>::kind } -> std::convertible_to<FieldValueKind>;
      { value_traits<T>::scalar_count } -> std::convertible_to<int>;
      { value_traits<T>::derivative_count } -> std::convertible_to<int>;
    } &&
    detail::ScalarTypeNamed<T> &&
    requires(const T& v, double d) {
      { value_traits<T>::to_binary64(v) } -> std::same_as<double>;
      { value_traits<T>::from_binary64(d) } -> std::same_as<T>;
    } &&
    (value_traits<T>::scalar_count >= 1) &&
    (value_traits<T>::derivative_count >= 0) &&
    (value_traits<T>::derivative_count < value_traits<T>::scalar_count) &&
    (value_traits<T>::kind != FieldValueKind::ScalarArray ||
     sizeof(T) == static_cast<std::size_t>(value_traits<T>::scalar_count) *
                      sizeof(typename value_traits<T>::scalar_type));

/// The run-time identity of a value type, carried by a type-erased field.
/// Equality compares the name's characters, not its address, so the same type
/// named in two shared libraries compares equal.
struct FieldValueType {
  std::string_view name;
  std::size_t      size = 0;
  std::size_t      alignment = 0;
  FieldValueKind   kind = FieldValueKind::ScalarArray;
  int              scalar_count = 1;
  /// How many of those scalars are derivatives (tangents), the last ones; 0
  /// for an ordinary number. A field with any is refused by the operations
  /// listed above.
  int              derivative_count = 0;
  /// The scalar type's name and size, so a view as scalars is checked against
  /// what the storage really holds. For a one-scalar type these are the type's
  /// own.
  std::string_view scalar_name;
  std::size_t      scalar_size = 0;

  friend constexpr bool operator==(const FieldValueType&, const FieldValueType&) = default;
};

template <CoupledValue T>
[[nodiscard]] constexpr FieldValueType field_value_type_of() noexcept {
  using Scalar = typename value_traits<T>::scalar_type;
  constexpr std::string_view scalar_name = [] {
    if constexpr (std::is_same_v<Scalar, T>) {
      return value_traits<T>::name;
    } else {
      return value_traits<Scalar>::name;
    }
  }();
  return FieldValueType{value_traits<T>::name,
                        sizeof(T),
                        alignof(T),
                        value_traits<T>::kind,
                        value_traits<T>::scalar_count,
                        value_traits<T>::derivative_count,
                        scalar_name,
                        sizeof(Scalar)};
}

}  // namespace fuse
