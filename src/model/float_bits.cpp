#include "fuse/model/float_bits.hpp"

#include <cmath>

namespace fuse::detail {

namespace {

/// Extracts `bit_count` (<= 64) bits starting at `bit_pos`, LSB-first
/// within each byte, byte 0 first -- the SAME convention `H5Tset_fields()`
/// documents (`spos`/`epos`/`mpos` are bit OFFSETS from the start of the
/// type's own precision, byte 0 holding bits [0,8)) and the convention
/// this project's own hand-built test fixture (docs/dev/restart.md
/// Section 11, item 6) constructs its raw bytes with.
[[nodiscard]] std::uint64_t extract_bits(std::span<const std::byte> raw,
                                         std::size_t bit_pos,
                                         std::size_t bit_count) {
  std::uint64_t result = 0;
  for (std::size_t i = 0; i < bit_count; ++i) {
    const std::size_t bit = bit_pos + i;
    const std::size_t byte_idx = bit / 8;
    const std::size_t bit_in_byte = bit % 8;
    if (byte_idx < raw.size()) {
      const auto byte_val = static_cast<unsigned char>(raw[byte_idx]);
      if ((byte_val >> bit_in_byte) & 1u) {
        result |= (std::uint64_t{1} << i);
      }
    }
  }
  return result;
}

/// The inverse of `extract_bits()` -- OR's `value`'s low `bit_count` bits
/// into `out` at `bit_pos`. `out` must already be zero-filled by the
/// caller before the first call for a given value (every `encode_float_
/// bits()` call below starts from a freshly zeroed buffer).
void pack_bits(std::span<std::byte> out, std::size_t bit_pos,
              std::size_t bit_count, std::uint64_t value) {
  for (std::size_t i = 0; i < bit_count; ++i) {
    if (((value >> i) & 1u) == 0) continue;
    const std::size_t bit = bit_pos + i;
    const std::size_t byte_idx = bit / 8;
    const std::size_t bit_in_byte = bit % 8;
    if (byte_idx < out.size()) {
      out[byte_idx] |= static_cast<std::byte>(1u << bit_in_byte);
    }
  }
}

struct Layout {
  std::size_t stored_mantissa_bits;  ///< bits actually ON DISK for the
                                     ///< mantissa field (excludes an
                                     ///< IMPLIED leading bit; includes an
                                     ///< explicit one).
  std::size_t mantissa_pos;
  std::size_t exponent_pos;
  std::size_t sign_pos;
};

[[nodiscard]] Layout layout_of(const ElementTypeInfo& info) {
  Layout l;
  l.stored_mantissa_bits = info.explicit_integer_bit
                               ? info.significand_bits
                               : info.significand_bits - 1;
  l.mantissa_pos = 0;
  l.exponent_pos = l.stored_mantissa_bits;
  l.sign_pos = l.stored_mantissa_bits + info.exponent_bits;
  return l;
}

}  // namespace

DecodedFloat decode_float_bits(std::span<const std::byte> raw,
                               const ElementTypeInfo& info) {
  const Layout l = layout_of(info);
  const std::uint64_t stored_mantissa =
      extract_bits(raw, l.mantissa_pos, l.stored_mantissa_bits);
  const std::uint64_t exponent_raw =
      extract_bits(raw, l.exponent_pos, info.exponent_bits);
  const bool sign = extract_bits(raw, l.sign_pos, 1) != 0;
  const std::uint64_t exp_all_ones =
      (std::uint64_t{1} << info.exponent_bits) - 1;

  DecodedFloat out;
  out.sign = sign;

  if (exponent_raw == exp_all_ones) {
    // Inf or NaN. For an EXPLICIT-integer-bit format the stored mantissa
    // still carries that bit; whether the value is Inf or NaN is decided
    // by the FRACTION bits only (the explicit integer bit is 1 for both,
    // by the format's own convention for valid encodings).
    const std::uint64_t fraction_mask =
        info.explicit_integer_bit
            ? (std::uint64_t{1} << (l.stored_mantissa_bits - 1)) - 1
            : (std::uint64_t{1} << l.stored_mantissa_bits) - 1;
    const bool fraction_nonzero = (stored_mantissa & fraction_mask) != 0;
    out.is_nan = fraction_nonzero;
    out.is_inf = !fraction_nonzero;
    return out;
  }

  if (exponent_raw == 0 && stored_mantissa == 0) {
    out.is_zero = true;
    return out;
  }

  if (exponent_raw == 0) {
    // Subnormal: normalise by left-shifting until the leading 1 sits at
    // bit (significand_bits - 1), our canonical `DecodedFloat` convention
    // -- see this file's own header comment on why this is not this
    // project's most exhaustively tested branch.
    std::uint64_t sig = stored_mantissa;
    std::int64_t shift = 0;
    while (((sig >> (info.significand_bits - 1)) & 1u) == 0) {
      sig <<= 1;
      ++shift;
    }
    out.significand = sig;
    out.unbiased_exponent = (1 - info.exponent_bias) - shift;
    return out;
  }

  // Normal.
  out.significand = info.explicit_integer_bit
                        ? stored_mantissa
                        : (stored_mantissa |
                           (std::uint64_t{1} << (info.significand_bits - 1)));
  out.unbiased_exponent =
      static_cast<std::int64_t>(exponent_raw) - info.exponent_bias;
  return out;
}

void encode_float_bits(const DecodedFloat& v, const ElementTypeInfo& info,
                       std::span<std::byte> out) {
  for (std::byte& b : out) b = std::byte{0};
  const Layout l = layout_of(info);
  const std::uint64_t exp_all_ones =
      (std::uint64_t{1} << info.exponent_bits) - 1;

  if (v.is_nan || v.is_inf) {
    pack_bits(out, l.sign_pos, 1, v.sign ? 1 : 0);
    pack_bits(out, l.exponent_pos, info.exponent_bits, exp_all_ones);
    if (info.explicit_integer_bit) {
      // The explicit integer bit is 1 for both Inf and a valid NaN; the
      // fraction (remaining bits) is 0 for Inf, nonzero (canonical: top
      // fraction bit set, a quiet NaN) for NaN.
      std::uint64_t mantissa = std::uint64_t{1}
                               << (l.stored_mantissa_bits - 1);
      if (v.is_nan) mantissa |= std::uint64_t{1}
                                << (l.stored_mantissa_bits - 2);
      pack_bits(out, l.mantissa_pos, l.stored_mantissa_bits, mantissa);
    } else if (v.is_nan) {
      // Canonical quiet NaN: top fraction bit set.
      pack_bits(out, l.mantissa_pos, l.stored_mantissa_bits,
               std::uint64_t{1} << (l.stored_mantissa_bits - 1));
    }
    return;
  }
  if (v.is_zero) {
    pack_bits(out, l.sign_pos, 1, v.sign ? 1 : 0);
    return;
  }

  std::int64_t exponent_raw = v.unbiased_exponent + info.exponent_bias;
  std::uint64_t significand = v.significand;

  if (exponent_raw <= 0) {
    // Underflow into subnormal (or to zero) -- shift the significand
    // right by however far below the normal range the exponent fell,
    // losing exactly that many low bits (this is the SAME rounding-free
    // "value is simply too small to represent at this precision at all"
    // case every IEEE format's own subnormal range exists for; this
    // project's own narrowing path always rounds the SIGNIFICAND first,
    // at the TARGET's own full normal precision, before this function is
    // reached, so reaching this branch means the target format's
    // exponent range itself could not hold the value, not that rounding
    // chose a bad width).
    const std::int64_t shift = 1 - exponent_raw;
    if (shift >= static_cast<std::int64_t>(info.significand_bits)) {
      pack_bits(out, l.sign_pos, 1, v.sign ? 1 : 0);
      return;  // flushes to zero
    }
    significand >>= shift;
    exponent_raw = 0;
  } else if (exponent_raw >= static_cast<std::int64_t>(exp_all_ones)) {
    // Overflow -- encode as Inf, the standard IEEE behaviour.
    pack_bits(out, l.sign_pos, 1, v.sign ? 1 : 0);
    pack_bits(out, l.exponent_pos, info.exponent_bits, exp_all_ones);
    return;
  }

  pack_bits(out, l.sign_pos, 1, v.sign ? 1 : 0);
  pack_bits(out, l.exponent_pos, info.exponent_bits,
           static_cast<std::uint64_t>(exponent_raw));
  const std::uint64_t stored_mantissa =
      info.explicit_integer_bit
          ? significand
          : (significand & ((std::uint64_t{1} << (info.significand_bits - 1)) -
                            1));
  pack_bits(out, l.mantissa_pos, l.stored_mantissa_bits, stored_mantissa);
}

RoundedSignificand round_significand(std::uint64_t significand, int from_bits,
                                     int to_bits) {
  if (to_bits >= from_bits) {
    // Not actually a narrowing (a widening caller should never reach
    // this, but a zero-extend is the well-defined, exact answer if one
    // does): shift the significand up so its leading bit is still at
    // position (to_bits - 1).
    return RoundedSignificand{significand
                                  << (static_cast<std::size_t>(to_bits) -
                                     static_cast<std::size_t>(from_bits)),
                              0};
  }
  const int drop = from_bits - to_bits;
  const std::uint64_t kept = significand >> drop;
  const std::uint64_t dropped_mask = (drop >= 64)
                                         ? ~std::uint64_t{0}
                                         : ((std::uint64_t{1} << drop) - 1);
  const std::uint64_t dropped = significand & dropped_mask;
  const std::uint64_t half = (drop == 0) ? 0 : (std::uint64_t{1} << (drop - 1));

  bool round_up;
  if (dropped > half) {
    round_up = true;
  } else if (dropped < half) {
    round_up = false;
  } else {
    // Exactly half: ties to even.
    round_up = (kept & 1u) != 0;
  }

  std::uint64_t rounded = kept;
  int exponent_increment = 0;
  if (round_up) {
    ++rounded;
    if (rounded == (std::uint64_t{1} << to_bits)) {
      // Carried out of the top bit: e.g. 0b1111 -> 0b10000 at the SAME
      // width is impossible to represent in `to_bits` bits as a leading-1
      // value, so renormalise: 0b1000...0 (to_bits bits, leading 1) at
      // one higher exponent -- this is the standard "rounding overflow"
      // case every correctly-rounded narrower converts handles.
      rounded = std::uint64_t{1} << (to_bits - 1);
      exponent_increment = 1;
    }
  }
  return RoundedSignificand{rounded, exponent_increment};
}

NarrowingResult narrow_float(std::span<const std::byte> raw,
                             const ElementTypeInfo& from_info,
                             const ElementTypeInfo& to_info) {
  const DecodedFloat src = decode_float_bits(raw, from_info);
  NarrowingResult result;

  if (src.is_nan || src.is_inf || src.is_zero) {
    result.decoded = src;  // no rounding applies to any of these
    return result;
  }

  const auto from_bits = static_cast<int>(from_info.significand_bits);
  const auto to_bits = static_cast<int>(to_info.significand_bits);
  const RoundedSignificand rounded =
      round_significand(src.significand, from_bits, to_bits);

  DecodedFloat dst;
  dst.sign = src.sign;
  dst.significand = rounded.significand;
  dst.unbiased_exponent = src.unbiased_exponent + rounded.exponent_increment;
  result.decoded = dst;

  if (to_bits >= from_bits) {
    // A caller asking to "narrow" into an equal-or-wider significand
    // never actually rounds -- exact, nothing changed.
    return result;
  }

  // Measure the change, in the SOURCE type's own units (docs/dev/
  // restart.md Section 11): both `src.significand` (from_bits wide) and
  // `rounded.significand` (to_bits wide, before any exponent_increment)
  // are integers at the SAME exponent scale when exponent_increment is 0;
  // when it is 1, the rounded value is exactly `2^from_bits` at the
  // ORIGINAL scale (the renormalised 0b1000...0 at one higher exponent,
  // shifted back up by `drop` bits, equals 2^from_bits in the original
  // significand's own units) -- both cases reduce to one integer
  // subtraction at a single shared scale.
  const int drop = from_bits - to_bits;
  const std::uint64_t rounded_at_source_scale =
      (rounded.exponent_increment == 0)
          ? (rounded.significand << drop)
          : (std::uint64_t{1} << from_bits);
  const std::int64_t delta_units =
      static_cast<std::int64_t>(rounded_at_source_scale) -
      static_cast<std::int64_t>(src.significand);
  if (delta_units != 0) {
    result.changed = true;
    // value = delta_units * 2^(unbiased_exponent - (from_bits - 1)) --
    // `delta_units` is always small (at most 2^drop in magnitude, far
    // under 53 bits for every (from,to) pair this project's own
    // ElementTypeInfo table can construct), so this ldexp is exact.
    result.absolute_change =
        std::fabs(std::ldexp(static_cast<double>(delta_units),
                             static_cast<int>(src.unbiased_exponent -
                                              (from_bits - 1))));
  }
  return result;
}

}  // namespace fuse::detail
