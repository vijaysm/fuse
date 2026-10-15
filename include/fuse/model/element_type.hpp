#pragma once

/// Restart track (docs/dev/restart.md), Stage B Slice 1b: the closed element-
/// type enumeration a typed restart field is tagged with, plus the compile-
/// time trait that maps a C++ type to it and the metadata every HDF5-backed
/// dataset records so a reader can verify what it is about to interpret
/// before it interprets it.
///
/// WHY A CLOSED ENUMERATION, NOT A TEMPLATE PARAMETER CARRIED TO DISK. The
/// enumerators below are LITERATURE names (IEEE 754-2019's own binary16/
/// binary32/binary64 for the standard floating formats; `docs/user/
/// glossary.md`'s own "Linear algebra" section is the project's convention
/// for borrowing a standard's vocabulary rather than coining a FUSE-specific
/// one). An `ElementType` value is itself part of a restart file's own
/// on-disk metadata (the `element_type` attribute every typed field dataset
/// carries, `fuse/restart_io/hdf5_restart_io.cpp`) -- a value a FUTURE
/// version of this enum appends a new enumerator to (never renumbers;
/// `entries.def`'s own "append-only" rule, reused here by the same
/// reasoning) must still be nameable by an OLDER reader as "I don't
/// recognise this", which a `std::uint8_t` discriminant with a known,
/// closed set of prior values supports and an open-ended type-erasure
/// scheme would not.
///
/// WHAT IS DELIBERATELY NOT HERE. `binary16` (IEEE half) is included only
/// because THIS build's compiler provides a native `_Float16` (measured:
/// Apple clang 21.0.0/arm64 defines `__FLT16_MANT_DIG__`, confirming
/// `_Float16` -- see `element_type_of<_Float16>()`'s own `#if` below). A
/// build on a compiler without it would still compile this header (every
/// enumerator OTHER than `Binary16` names a type every C++20 compiler has),
/// but `element_type_of<_Float16>()` itself would not exist, so no caller
/// could ask for it -- the enumerator stays in the closed set (a FILE
/// written by a compiler that has `_Float16` must still be nameable by a
/// reader that does not) but nothing in THIS build ever WRITES one. This is
/// the owner's own instruction: "only if the compiler provides _Float16;
/// otherwise omit it and say so -- do not invent."

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

#include "fuse/model/bfloat16.hpp"

namespace fuse {

/// The closed set. Values are part of the on-disk format (`element_type`
/// attribute, `fuse/restart_io/hdf5_restart_io.cpp`) -- append only, never
/// renumber, never remove; a reader encountering a numeric value with no
/// matching enumerator refuses by name (`E-RESTART-UNKNOWN-ELEMENT-TYPE`)
/// rather than guessing, because a newer writer's new enumerator is
/// (by definition) something this reader has never seen before.
enum class ElementType : std::uint8_t {
  Int8 = 0,
  Int16 = 1,
  Int32 = 2,
  Int64 = 3,
  UInt8 = 4,
  UInt16 = 5,
  UInt32 = 6,
  UInt64 = 7,
  Binary16 = 8,   ///< IEEE 754-2019 binary16 ("half"). Only ever WRITTEN by a
                  ///< build whose compiler provides `_Float16` (see the file
                  ///< header comment); always a valid enumerator to READ.
  BFloat16 = 9,   ///< `fuse::bfloat16` (fuse/model/bfloat16.hpp) -- no C++
                  ///< standard type backs this on any compiler this project
                  ///< has measured (`__bf16`/`std::bfloat16_t` both absent,
                  ///< Apple clang 21.0.0/arm64), so FUSE carries its own.
  Binary32 = 10,  ///< IEEE 754-2019 binary32 (`float`).
  Binary64 = 11,  ///< IEEE 754-2019 binary64 (`double`).
  LongDouble = 12,  ///< The PLATFORM's own `long double` -- NOT a fixed IEEE
                    ///< format. Measured on Apple clang 21.0.0/arm64:
                    ///< `sizeof(long double) == 8`, `LDBL_MANT_DIG == 53`,
                    ///< bit-identical to `Binary64` -- yet a DIFFERENT
                    ///< enumerator, because `ElementType` tags the C++ TYPE
                    ///< the caller asked for, not merely its bit layout (a
                    ///< caller requesting `Binary64` from a `LongDouble`
                    ///< field goes through `ReadConversion`, same as any
                    ///< other cross-type request, even on a platform where
                    ///< today it happens to be a no-op). On x86-64 Linux
                    ///< this is the 80-bit extended format (64-bit
                    ///< significand) -- see `element_type_of<long double>()`.
  Bool = 13,      ///< Stored as one `UInt8` byte, 0 or 1 -- `element_type`
                  ///< says `Bool`, not `UInt8`, so a reader can refuse a
                  ///< byte that is neither (docs/dev/restart.md Section 11).
};

/// How a `get_field`/`get_scalar` request's own C++ type may differ from
/// the STORED `ElementType`, owner decision (robustness first; docs/dev/
/// restart.md Section 11's own "READ CONVERSION" heading carries the full
/// reasoning this comment only summarises):
///   Exact           the requested type must equal the stored type (same
///                   enumerator) -- the default. Anything else refuses by
///                   name, both types named in the message.
///   Widening        additionally permits a conversion decided, from the
///                   TWO TYPES' OWN METADATA ALONE (never a per-value
///                   probe), to be exact for EVERY value the stored type
///                   can represent -- `fuse::detail::is_exact_widening()`
///                   (element_type.cpp) is the one place this decision is
///                   made. Never signed-integer -> unsigned-integer, at
///                   any width, regardless of whether a particular file's
///                   values happen to be non-negative.
///   AllowNarrowing  additionally permits a conversion that MAY lose
///                   precision for some values: INTEGER narrowing that does
///                   not fit still refuses (an integer value change is not
///                   a rounding -- owner's own words); FLOAT narrowing
///                   (reduced significand and/or exponent range) rounds
///                   to nearest, ties to even, and the call reports the
///                   largest change it actually made, measured over the
///                   values actually read, never assumed from the types
///                   alone.
enum class ReadConversion { Exact, Widening, AllowNarrowing };

/// Per-`ElementType` facts a restart dataset's own metadata records and a
/// reader's own type-agreement check compares against (docs/dev/restart.md
/// Section 11, item 5). `significand_bits`/`exponent_bits`/`exponent_bias`
/// are 0 for an integer or `Bool` type -- they describe a floating type's
/// `(sign, exponent, significand)` layout, the SAME three quantities
/// `H5Tget_fields()`/`H5Tget_ebias()` report for the dataset's own HDF5
/// datatype (`fuse/restart_io/hdf5_restart_io.cpp`'s read-side check
/// compares THESE, not merely the enumerator, so a corrupted or foreign
/// file's attribute/dataset-type disagreement is caught even if whichever
/// value was tampered with still happens to name a valid `ElementType`).
struct ElementTypeInfo {
  ElementType element_type{};
  std::string name;              ///< literature name, e.g. "binary32"
  std::size_t byte_size = 0;     ///< on-disk size, this platform
  bool        is_float = false;
  bool        is_signed_integer = false;    ///< meaningful only if !is_float
  bool        is_unsigned_integer = false;  ///< meaningful only if !is_float
  bool        is_bool = false;
  std::size_t significand_bits = 0;  ///< INCLUDING any implicit leading bit
  std::size_t exponent_bits = 0;
  std::int64_t exponent_bias = 0;
  /// True for the one platform-measured format this project has had to
  /// special-case so far: an EXPLICIT (stored) leading significand bit,
  /// x86's 80-bit extended `long double` (HDF5's `H5T_NORM_MSBSET`,
  /// `H5Tpublic.h`) rather than IEEE's implicit-leading-one convention
  /// (`H5T_NORM_IMPLIED`) every OTHER format here uses. See docs/dev/
  /// restart.md Section 11's own note on why HDF5's soft conversion
  /// cannot be trusted for this case (measured: `H5T__conv_f_f():
  /// normalization method not implemented yet`).
  bool        explicit_integer_bit = false;
};

/// The full table, indexed by `ElementType`'s own underlying value --
/// `fuse::element_type_info(t)` is the only way code outside this header
/// reads it, so the table itself can stay `static` to its translation unit
/// (`element_type.cpp`).
[[nodiscard]] const ElementTypeInfo& element_type_info(ElementType t);

/// `std::nullopt` for a numeric value with no matching enumerator -- the
/// caller (a restart reader encountering a newer file's `element_type`
/// attribute) refuses by name (`E-RESTART-UNKNOWN-ELEMENT-TYPE`) rather
/// than guessing; this function never throws and never asserts.
[[nodiscard]] std::optional<ElementType> element_type_from_name(
    const std::string& name);

/// The compile-time trait every `put_field<T>()`/`get_field<T>()` call
/// (fuse/model/restart.hpp) resolves through. Specialised below for every
/// `T` this project's typed restart interface supports; an UNSUPPORTED `T`
/// is a `static_assert` at the call site (owner's own instruction: "an
/// unsupported T is a static_assert, not a runtime error"), via the
/// primary template's own `false` assertion.
template <class T>
struct ElementTypeOf {
  static_assert(sizeof(T) == 0,
                "fuse::ElementTypeOf<T>: T is not one of the element types "
                "fuse::put_field<T>()/get_field<T>() support -- see "
                "fuse/model/element_type.hpp's own specialisation list. "
                "This is a compile-time refusal, not a runtime one, by the "
                "owner's own instruction.");
};

template <>
struct ElementTypeOf<std::int8_t> {
  static constexpr ElementType value = ElementType::Int8;
};
template <>
struct ElementTypeOf<std::int16_t> {
  static constexpr ElementType value = ElementType::Int16;
};
template <>
struct ElementTypeOf<std::int32_t> {
  static constexpr ElementType value = ElementType::Int32;
};
template <>
struct ElementTypeOf<std::int64_t> {
  static constexpr ElementType value = ElementType::Int64;
};
template <>
struct ElementTypeOf<std::uint8_t> {
  static constexpr ElementType value = ElementType::UInt8;
};
template <>
struct ElementTypeOf<std::uint16_t> {
  static constexpr ElementType value = ElementType::UInt16;
};
template <>
struct ElementTypeOf<std::uint32_t> {
  static constexpr ElementType value = ElementType::UInt32;
};
template <>
struct ElementTypeOf<std::uint64_t> {
  static constexpr ElementType value = ElementType::UInt64;
};
template <>
struct ElementTypeOf<fuse::bfloat16> {
  static constexpr ElementType value = ElementType::BFloat16;
};
template <>
struct ElementTypeOf<float> {
  static constexpr ElementType value = ElementType::Binary32;
};
template <>
struct ElementTypeOf<double> {
  static constexpr ElementType value = ElementType::Binary64;
};
template <>
struct ElementTypeOf<long double> {
  static constexpr ElementType value = ElementType::LongDouble;
};
template <>
struct ElementTypeOf<bool> {
  static constexpr ElementType value = ElementType::Bool;
};

#if defined(__FLT16_MANT_DIG__)
/// Only ever instantiated by a build whose compiler provides `_Float16`
/// (see the file header comment) -- `fuse::put_field<_Float16>()` simply
/// does not exist to call on a compiler without it, which is the
/// `static_assert` above doing its job, not a separate code path.
template <>
struct ElementTypeOf<_Float16> {
  static constexpr ElementType value = ElementType::Binary16;
};
#endif

template <class T>
inline constexpr ElementType kElementTypeOf = ElementTypeOf<T>::value;

/// Decides, from `from`'s and `to`'s own `ElementTypeInfo` ALONE (never a
/// per-value probe -- owner's own instruction), whether EVERY value `from`
/// can represent converts to `to` with no change at all. The one function
/// `ReadConversion::Widening` is defined in terms of; `fuse::restart_io`'s
/// reader calls this, never re-derives the rule itself.
///
/// TAKES `ElementTypeInfo`, NOT `ElementType`, for `from`: the STORED
/// side of a restart read is described by whatever metadata the FILE
/// ITSELF carries (`fuse::restart_io`'s own `read_and_verify_type_
/// metadata()` builds this directly from the file's own attributes,
/// cross-checked against the file's own actual HDF5 datatype -- NEVER
/// against this build's own canonical table for that enumerator). A
/// file written on a different platform (x86-64's 80-bit `long double`,
/// read on this Mac's 53-bit one) is internally self-consistent but
/// DIFFERS from `fuse::element_type_info(ElementType::LongDouble)` on
/// THIS build -- exactly docs/dev/restart.md Section 11, item 6's own
/// scenario, which this signature exists to make possible at all: if
/// this function instead re-fetched `from`'s metadata from the global
/// table by enumerator, reading a genuinely foreign file could never
/// succeed under ANY `ReadConversion`, not even `AllowNarrowing`, which
/// would misread the owner's OWN requirement backwards.
[[nodiscard]] bool is_exact_widening(const ElementTypeInfo& from,
                                     const ElementTypeInfo& to);

/// Convenience overload for the common case -- BOTH sides are THIS
/// build's own canonical metadata for the named enumerators (every
/// caller in this project except `fuse::restart_io`'s cross-platform
/// read path itself, which always has a concrete `ElementTypeInfo` for
/// the stored side already and calls the overload above directly).
[[nodiscard]] inline bool is_exact_widening(ElementType from, ElementType to) {
  return is_exact_widening(element_type_info(from), element_type_info(to));
}

}  // namespace fuse
