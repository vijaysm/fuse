#pragma once

/// A field of values of any coupled value type in a restart file
/// (plans/2026-10-22-universal-number-types.md, stage U4).
///
/// `RestartWriter::put_field<T>()` stores arrays of the types `ElementTypeOf`
/// names: the integer types, `bool`, binary32, binary64, `long double`, and the
/// two 16-bit formats `fuse::bfloat16` and `fuse::binary16`. A component whose
/// state is a field of another coupled value type -- a double-double or
/// quad-double number, a forward-mode dual number -- keeps it with the two
/// functions below, which store each value as its scalars: `components` values
/// per row become `components * scalar_count` scalars per row, the scalars of
/// one value consecutive, bit for bit.
///
/// NOTHING ABOUT THE FILE CHANGES. What is written is an ordinary binary64 (or
/// binary32, bfloat16, binary16) field with more components per row, so every
/// reader of every version reads it, and a double-double field read back by a
/// program that does not know the type is the pairs of binary64 numbers it was.
/// The cost is that the file does not say the pairs are double-double: the
/// reader names the type it expects (`get_values<dd>`), and the number of
/// components it asks for must be the number the writer used, which the
/// reader refuses otherwise (the restart reader's own check of `components`).
/// A value type that is already one of the stored types (binary64, binary32,
/// bfloat16, binary16) is stored as itself, so `put_values<float>` is
/// `put_field<float>`.

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

#include "fuse/base/result.hpp"
#include "fuse/model/element_type.hpp"
#include "fuse/model/restart.hpp"
#include "fuse/model/value_traits.hpp"

namespace fuse {

namespace detail {

/// True for the value types a restart file stores as themselves.
template <class T>
inline constexpr bool restart_stored_as_itself =
    std::is_same_v<T, double> || value_traits<T>::kind == FieldValueKind::Storage;

/// What a value of type `T` is stored as: itself when it is a stored type, otherwise
/// its `scalar_count` scalars, which must be a stored type themselves.
template <class T>
struct RestartScalars {
  using type = std::conditional_t<restart_stored_as_itself<T>, T, typename value_traits<T>::scalar_type>;
  static constexpr int count = restart_stored_as_itself<T> ? 1 : value_traits<T>::scalar_count;
  static_assert(restart_stored_as_itself<type>,
                "the scalars of this value type are not a type a restart file stores (binary64, "
                "binary32, bfloat16 or binary16)");
};

template <class T>
using restart_scalar_t = typename RestartScalars<T>::type;

template <class T>
inline constexpr int restart_scalar_count = RestartScalars<T>::count;

}  // namespace detail

/// Writes `values` (`ids.size() * components` of them) as the scalars of the
/// values: the same call as `RestartWriter::put_field`, with the value type
/// instead of the stored type.
template <CoupledValue T>
Result<void> put_values(RestartWriter& writer, std::string_view qualified_name,
                        std::span<const std::int64_t> ids, std::span<const T> values,
                        int components, std::int64_t declared_global_size) {
  using S = detail::restart_scalar_t<T>;
  constexpr int k = detail::restart_scalar_count<T>;
  static_assert(sizeof(T) == sizeof(S) * static_cast<std::size_t>(k),
                "the scalars of one value must be consecutive, with no padding, to be stored as such");
  static_assert(std::is_trivially_copyable_v<T>);
  const S* scalars = reinterpret_cast<const S*>(values.data());
  return writer.template put_field<S>(
      qualified_name, ids, std::span<const S>(scalars, values.size() * static_cast<std::size_t>(k)),
      components * k, declared_global_size);
}

/// Reads what `put_values<T>` wrote, with the same `components` (in values).
/// `conversion` is the restart reader's own, applied to the stored scalars.
template <CoupledValue T>
Result<ReadConversionReport> get_values(RestartReader& reader, std::string_view qualified_name,
                                        std::span<const std::int64_t> ids, std::span<T> out,
                                        int components,
                                        ReadConversion conversion = ReadConversion::Exact) {
  using S = detail::restart_scalar_t<T>;
  constexpr int k = detail::restart_scalar_count<T>;
  static_assert(sizeof(T) == sizeof(S) * static_cast<std::size_t>(k));
  static_assert(std::is_trivially_copyable_v<T>);
  S* scalars = reinterpret_cast<S*>(out.data());
  return reader.template get_field<S>(
      qualified_name, ids, std::span<S>(scalars, out.size() * static_cast<std::size_t>(k)),
      components * k, conversion);
}

}  // namespace fuse
