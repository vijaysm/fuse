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
///               forward-mode dual number whose gradient has N entries is N+1
///               doubles. Every
///               operation the coupler applies to field values (a remap
///               y = W x whose weights do not depend on the values, a halo copy,
///               a transfer, time interpolation) acts on each scalar
///               separately, so a field of such values moves as a field of
///               scalars with `scalar_count` times as many components, and a
///               gradient travels through a remap exactly: the gradient of W x
///               is W applied to the gradient. binary64 itself is the case of
///               one scalar.
///   Arithmetic  `T` is `scalar_count` consecutive scalars that are the limbs
///               of an unevaluated sum (double-double: two binary64 values,
///               quad-double: four), so the arithmetic is not that of each
///               scalar separately. An operator that does arithmetic on the
///               values is instantiated for `T` from a closed list (stage U4);
///               one with no instantiation refuses the field by name, and an
///               operator that only moves values (a halo copy, a transfer)
///               moves the limbs. `FieldRegistry::get_limbs` is the view as
///               limbs; `get_scalars` refuses such a field, so that nothing
///               meant for scalars acts on the limbs one at a time.
///   Storage     `T` is a compact format (binary32, bfloat16, binary16, posit):
///               widened exactly to binary64 at a boundary and rounded once,
///               with round to nearest, ties to even, on the way back.
///
/// WHAT "EACH SCALAR SEPARATELY" LEAVES OUT. The last `gradient_size` of the
/// scalars are the gradient of the value (its derivatives with respect to the N
/// parameters of a forward-mode dual number); the first
/// `scalar_count - gradient_size` are the value itself. Two coupler operations
/// are not independent per scalar and so are not safe on a gradient:
///
///   * an affine map v -> s v + b (a unit conversion with an offset, such as
///     degrees Celsius to kelvin) adds b to the value, and its gradient is
///     s times the gradient of v: the offset must reach the value scalars
///     only;
///   * an accumulation by minimum or maximum picks one sample, and the gradient
///     of the result is the gradient of the sample picked, which comparing the
///     gradient scalar by scalar does not give.
///
/// The compiler already refuses an affine step for every field (E-IR-UNLOWERABLE),
/// so only the second can occur today; the runner refuses it for a field with
/// a gradient (E-RT-GRADIENT-MINMAX). Scaling, remap, interpolation and the
/// accumulations by sum, mean and integral are linear, hence scalar by scalar.
///
/// `to_binary64` and `from_binary64` are the projection every boundary that
/// needs binary64 uses: exact widening for a Storage type, the value part for
/// a ScalarArray type with more than one scalar. They are not a licence to
/// compute in binary64 behind a field's back; where they are applied is a
/// decision of the stage that applies them.

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "fuse/base/portable.hpp"
#include "fuse/model/bfloat16.hpp"
#include "fuse/model/binary16.hpp"

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
///   static constexpr int gradient_size;    // 0 <= gradient_size < scalar_count
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
  static constexpr int gradient_size = 0;
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr double to_binary64(double v) noexcept {
    return v;
  }
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr double from_binary64(double v) noexcept {
    return v;
  }
};

template <>
struct value_traits<float> {
  static constexpr std::string_view name = "binary32";
  static constexpr FieldValueKind kind = FieldValueKind::Storage;
  using scalar_type = float;
  static constexpr int scalar_count = 1;
  static constexpr int gradient_size = 0;
  /// Exact: every binary32 value is a binary64 value.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr double to_binary64(float v) noexcept {
    return static_cast<double>(v);
  }
  /// Round to nearest, ties to even, under the default floating-point
  /// environment (this project never changes the rounding mode).
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static constexpr float from_binary64(double v) noexcept {
    return static_cast<float>(v);
  }
};

template <>
struct value_traits<bfloat16> {
  static constexpr std::string_view name = "bfloat16";
  static constexpr FieldValueKind kind = FieldValueKind::Storage;
  using scalar_type = bfloat16;
  static constexpr int scalar_count = 1;
  static constexpr int gradient_size = 0;
  /// Exact: every bfloat16 value is a binary64 value.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double to_binary64(bfloat16 v) noexcept {
    return v.to_double();
  }
  /// Correctly rounded, to nearest with ties to even, once (bfloat16.hpp).
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static bfloat16 from_binary64(double v) noexcept {
    return bfloat16::from_double(v);
  }
};

template <>
struct value_traits<binary16> {
  static constexpr std::string_view name = "binary16";
  static constexpr FieldValueKind kind = FieldValueKind::Storage;
  using scalar_type = binary16;
  static constexpr int scalar_count = 1;
  static constexpr int gradient_size = 0;
  /// Exact: every binary16 value is a binary64 value.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static double to_binary64(binary16 v) noexcept {
    return v.to_double();
  }
  /// Correctly rounded, to nearest with ties to even, once (binary16.hpp).
  [[nodiscard]] KOKKOS_INLINE_FUNCTION static binary16 from_binary64(double v) noexcept {
    return binary16::from_double(v);
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
/// complete `value_traits<T>`. For a ScalarArray or an Arithmetic type the
/// scalars must tile `T` exactly, because viewing a field as scalars (or as
/// limbs) addresses scalar `k` of value `v` as
/// `reinterpret_cast<const scalar_type*>(&v)[k]`.
template <class T>
concept CoupledValue =
    std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T> &&
    requires {
      typename value_traits<T>::scalar_type;
      { value_traits<T>::name } -> std::convertible_to<std::string_view>;
      { value_traits<T>::kind } -> std::convertible_to<FieldValueKind>;
      { value_traits<T>::scalar_count } -> std::convertible_to<int>;
      { value_traits<T>::gradient_size } -> std::convertible_to<int>;
    } &&
    detail::ScalarTypeNamed<T> &&
    requires(const T& v, double d) {
      { value_traits<T>::to_binary64(v) } -> std::same_as<double>;
      { value_traits<T>::from_binary64(d) } -> std::same_as<T>;
    } &&
    (value_traits<T>::scalar_count >= 1) &&
    (value_traits<T>::gradient_size >= 0) &&
    (value_traits<T>::gradient_size < value_traits<T>::scalar_count) &&
    (value_traits<T>::kind == FieldValueKind::Storage ||
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
  /// How many of those scalars hold the gradient, the last ones; 0 for an
  /// ordinary number. A field with a gradient is refused by the operations
  /// listed above.
  int              gradient_size = 0;
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
                        value_traits<T>::gradient_size,
                        scalar_name,
                        sizeof(Scalar)};
}

}  // namespace fuse
