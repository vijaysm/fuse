#pragma once

/// The bit-level floating-point decode/round/encode pipeline every
/// CROSS-TYPE float conversion in `fuse::restart_io` goes through (see
/// docs/dev/restart.md) -- measured necessary
/// because HDF5's own soft conversion (`H5T__conv_f_f`) refuses a type
/// with an EXPLICIT (stored) leading significand bit (`H5T_NORM_MSBSET`,
/// x86's 80-bit extended `long double`) with "normalization method not
/// implemented yet" (measured directly against this project's own
/// `/opt/hdf5`, docs/dev/restart.md Section 11) -- and, having measured
/// that HDF5 cannot be trusted uniformly across every format this project
/// supports, used for EVERY float-to-float conversion here, not only the
/// one HDF5 happens to refuse, so there is exactly one tested code path
/// rather than one tested path plus one silently-trusted one.
///
/// ONE RULE: every function below operates on `fuse::ElementTypeInfo`'s
/// own `(significand_bits, exponent_bits, exponent_bias,
/// explicit_integer_bit)` fields ALONE -- never a C++ type, never an
/// `H5T` handle. `fuse::restart_io` (which DOES know both) is the only
/// caller; this header pulls in neither `<hdf5.h>` nor any C++ floating
/// type beyond what every translation unit already has.

#include <cstddef>
#include <cstdint>
#include <span>

#include "fuse/model/element_type.hpp"

namespace fuse::detail {

/// A floating-point value decomposed into integer parts, independent of
/// any particular on-disk bit layout. `significand` is normalised so a
/// finite nonzero value's own leading bit always sits at bit position
/// `significand_bits - 1` (i.e. `significand` is treated as if it were a
/// `significand_bits`-bit UNSIGNED INTEGER, not a fraction) -- `value =
/// (sign ? -1 : 1) * significand * 2^(unbiased_exponent - (significand_bits
/// - 1))`. This is the SAME convention `fuse::ElementTypeInfo::
/// significand_bits` already uses (it counts the implicit leading bit IN),
/// so a decode followed immediately by an encode at the SAME bit width is
/// the identity -- `test/model/float_bits_test.cpp` proves this for every
/// `ElementType` this project's typed restart interface supports.
struct DecodedFloat {
  bool sign = false;
  bool is_nan = false;
  bool is_inf = false;
  bool is_zero = false;
  std::uint64_t significand = 0;      ///< meaningless if is_nan/is_inf/is_zero
  std::int64_t unbiased_exponent = 0;  ///< meaningless if is_nan/is_inf/is_zero
};

/// Reads `raw` (exactly `info.byte_size` bytes, the type's own on-disk/
/// in-register layout, host byte order) and decodes it. `info` must be a
/// floating `ElementTypeInfo` (`info.is_float == true`); calling this with
/// an integer or `Bool` info is a programmer error (`FUSE_DEBUG_ASSERT`
/// fires in a Debug build, a Release build decodes garbage -- the caller,
/// `fuse::restart_io`, never reaches this call with the wrong `info`
/// because it branches on `ElementTypeInfo::is_float` first).
[[nodiscard]] DecodedFloat decode_float_bits(std::span<const std::byte> raw,
                                             const ElementTypeInfo& info);

/// The inverse of `decode_float_bits()` -- writes exactly `info.byte_size`
/// bytes to `out`. `v.significand` must already fit in `info.
/// significand_bits` bits (the caller rounds FIRST, via `round_
/// significand()` below, if it does not) -- this function does not itself
/// round or check range; it is a pure bit-packing step, matching `decode_
/// float_bits()`'s own "pure bit-UNpacking" scope.
void encode_float_bits(const DecodedFloat& v, const ElementTypeInfo& info,
                       std::span<std::byte> out);

/// Rounds `significand` (currently `from_bits` significant bits, leading
/// bit at position `from_bits - 1`, exactly `decode_float_bits()`'s own
/// convention) down to `to_bits` bits, round-to-nearest-even -- the ONE
/// rounding rule this entire pipeline uses, so `fuse::bfloat16::
/// from_float()` (fuse/model/bfloat16.hpp) and a `long double` ->
/// `bfloat16` restart read go through the IDENTICAL tie-breaking rule,
/// never two independently-written ones that could silently disagree at
/// a tie. `to_bits <= from_bits` (a WIDENING caller never reaches this at
/// all -- widening is a value-preserving zero-extend of the significand,
/// not a rounding, and `fuse::is_exact_widening()` is what established
/// there is nothing to round before this function is ever called).
struct RoundedSignificand {
  std::uint64_t significand = 0;  ///< exactly `to_bits` bits
  int exponent_increment = 0;     ///< 0 or 1 -- 1 iff rounding carried out
                                  ///< of the top bit (e.g. 0b111...1 -> a
                                  ///< new leading 0b1000...0 one exponent
                                  ///< higher); the caller adds this to the
                                  ///< UNBIASED exponent before re-encoding.
};
[[nodiscard]] RoundedSignificand round_significand(std::uint64_t significand,
                                                   int from_bits, int to_bits);

/// The full narrowing conversion `fuse::restart_io`'s `AllowNarrowing`
/// path calls: decodes `raw` as `from_info`'s own layout, rounds to
/// `to_info`'s own significand width (handling the exponent carry
/// `round_significand()` may report, and any further exponent-range
/// clamping this design does not define -- `to_info`'s own
/// `exponent_bits` is assumed wide enough to hold the result, which
/// every (from,to) pair this call is ever reached for in fact is, because
/// `fuse::restart_io` only calls this once `ReadConversion::AllowNarrowing`
/// is already in effect and a NARROWER significand's value came from a
/// WIDER-OR-EQUAL exponent range format in every case this project's own
/// `ElementTypeInfo` table can construct), then encodes the result into
/// `to_info`'s own layout. Returns the encoded bytes plus whether the
/// value actually changed and by how much (`docs/dev/restart.md` Section
/// 11's own "reports the largest change" requirement) -- `changed` is
/// `false` and `absolute_change` is `0.0` for NaN/Inf/zero (none of which
/// round) and for any finite value that happened to round exactly.
struct NarrowingResult {
  DecodedFloat decoded;    ///< the ROUNDED value, still in `to_info`'s layout
  bool changed = false;
  double absolute_change = 0.0;  ///< |exact source value - rounded value|,
                                 ///< in the SOURCE type's own units -- see
                                 ///< docs/dev/restart.md Section 11.
};
[[nodiscard]] NarrowingResult narrow_float(std::span<const std::byte> raw,
                                           const ElementTypeInfo& from_info,
                                           const ElementTypeInfo& to_info);

}  // namespace fuse::detail
