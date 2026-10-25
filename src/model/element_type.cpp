#include "fuse/model/element_type.hpp"

#include <cfloat>
#include <unordered_map>

namespace fuse {

namespace {

/// `std::numeric_limits<long double>::digits` is `LDBL_MANT_DIG` and
/// INCLUDES the implicit leading bit for a standard IEEE-style format --
/// exactly `significand_bits`'s own convention everywhere else in this
/// table. `digits10`/`max_exponent` are not used: `max_exponent` already
/// gives the unbiased exponent range directly (`exponent_bias = max_exponent
/// - 1`, the standard IEEE convention: bias centres the stored range so the
/// unbiased maximum is `2^(exponent_bits-1) - 1`).
///
/// EXPLICIT-INTEGER-BIT DETECTION. The only widely-deployed `long double`
/// with a STORED (not implicit) leading significand bit this project knows
/// of is x86/x86-64's 80-bit extended format, whose unmistakable signature
/// is `LDBL_MANT_DIG == 64` (64 stored significand bits, no implicit bit,
/// vs. an IEEE-style binaryN's `LDBL_MANT_DIG` always counting the implicit
/// bit IN). Measured on this project's own build host (Apple clang
/// 21.0.0/arm64): `LDBL_MANT_DIG == 53`, identical to `double` -- the
/// `else` branch below, which assumes the ordinary IEEE implicit-bit
/// convention, is what actually runs on every build this project has
/// measured. A `long double` that is neither of these two measured shapes
/// (e.g. PowerPC's double-double, whose `LDBL_MANT_DIG` is typically 106
/// and whose bits are not a single contiguous (sign,exponent,significand)
/// triple at all) is OUT OF SCOPE -- see docs/dev/restart.md Section 11's
/// own deviation note; this table's `long_double_exponent_bits()` below
/// returns a value computed from `LDBL_MAX_EXP` regardless, which would be
/// WRONG for such a platform, but nothing in this project builds on one.
constexpr bool kLongDoubleHasExplicitIntegerBit = (LDBL_MANT_DIG == 64);

constexpr std::size_t kLongDoubleSignificandBits =
    static_cast<std::size_t>(LDBL_MANT_DIG);

/// `LDBL_MAX_EXP` is the largest power-of-2 exponent + 1 (C's own
/// convention, matching `FLT_MAX_EXP`/`DBL_MAX_EXP`) -- the exponent FIELD
/// must represent `[1 - bias, LDBL_MAX_EXP - 1]` for normals plus the two
/// reserved all-zero/all-one encodings, i.e. `2*LDBL_MAX_EXP` distinct
/// values, which needs `ceil(log2(2*LDBL_MAX_EXP))` bits.
[[nodiscard]] constexpr std::size_t bits_for_exponent_range(int max_exp) {
  std::size_t bits = 0;
  // 2*max_exp distinct values needed; find the smallest n with 2^n >=
  // 2*max_exp. A plain loop, not std::log2, so this stays constexpr and
  // exact (no floating-point rounding in a function computing a type's
  // own exponent WIDTH).
  unsigned long long need = static_cast<unsigned long long>(max_exp) * 2ULL;
  unsigned long long cap = 1ULL;
  while (cap < need) {
    cap <<= 1;
    ++bits;
  }
  return bits;
}

constexpr std::size_t kLongDoubleExponentBits =
    bits_for_exponent_range(LDBL_MAX_EXP);
constexpr std::int64_t kLongDoubleExponentBias = LDBL_MAX_EXP - 1;

[[nodiscard]] ElementTypeInfo make_int(ElementType t, const char* name,
                                       std::size_t bytes, bool is_signed) {
  ElementTypeInfo info;
  info.element_type = t;
  info.name = name;
  info.byte_size = bytes;
  info.is_signed_integer = is_signed;
  info.is_unsigned_integer = !is_signed;
  return info;
}

[[nodiscard]] ElementTypeInfo make_float(ElementType t, const char* name,
                                         std::size_t bytes,
                                         std::size_t significand_bits,
                                         std::size_t exponent_bits,
                                         std::int64_t exponent_bias,
                                         bool explicit_integer_bit = false) {
  ElementTypeInfo info;
  info.element_type = t;
  info.name = name;
  info.byte_size = bytes;
  info.is_float = true;
  info.significand_bits = significand_bits;
  info.exponent_bits = exponent_bits;
  info.exponent_bias = exponent_bias;
  info.explicit_integer_bit = explicit_integer_bit;
  return info;
}

[[nodiscard]] const std::unordered_map<ElementType, ElementTypeInfo>& table() {
  static const std::unordered_map<ElementType, ElementTypeInfo> t = [] {
    std::unordered_map<ElementType, ElementTypeInfo> m;
    m.emplace(ElementType::Int8, make_int(ElementType::Int8, "int8", 1, true));
    m.emplace(ElementType::Int16,
              make_int(ElementType::Int16, "int16", 2, true));
    m.emplace(ElementType::Int32,
              make_int(ElementType::Int32, "int32", 4, true));
    m.emplace(ElementType::Int64,
              make_int(ElementType::Int64, "int64", 8, true));
    m.emplace(ElementType::UInt8,
              make_int(ElementType::UInt8, "uint8", 1, false));
    m.emplace(ElementType::UInt16,
              make_int(ElementType::UInt16, "uint16", 2, false));
    m.emplace(ElementType::UInt32,
              make_int(ElementType::UInt32, "uint32", 4, false));
    m.emplace(ElementType::UInt64,
              make_int(ElementType::UInt64, "uint64", 8, false));
    // IEEE 754-2019 binary16: 1 sign, 5 exponent (bias 15), 11 significand
    // (10 stored + implicit leading 1).
    m.emplace(ElementType::Binary16,
              make_float(ElementType::Binary16, "binary16", 2, 11, 5, 15));
    // bfloat16: 1 sign, 8 exponent (SAME width/bias as binary32 -- by
    // construction, every bfloat16 value's exponent fits binary32's
    // range exactly), 8 significand (7 stored + implicit leading 1).
    m.emplace(ElementType::BFloat16,
              make_float(ElementType::BFloat16, "bfloat16", 2, 8, 8, 127));
    // IEEE 754-2019 binary32: 1 sign, 8 exponent (bias 127), 24
    // significand (23 stored + implicit leading 1).
    m.emplace(ElementType::Binary32,
              make_float(ElementType::Binary32, "binary32", 4, 24, 8, 127));
    // IEEE 754-2019 binary64: 1 sign, 11 exponent (bias 1023), 53
    // significand (52 stored + implicit leading 1).
    m.emplace(ElementType::Binary64,
              make_float(ElementType::Binary64, "binary64", 8, 53, 11, 1023));
    // long double: THIS platform's own measured layout -- see the
    // kLongDoubleHasExplicitIntegerBit/kLongDoubleSignificandBits/
    // kLongDoubleExponentBits comments above.
    m.emplace(ElementType::LongDouble,
              make_float(ElementType::LongDouble, "long double",
                        sizeof(long double), kLongDoubleSignificandBits,
                        kLongDoubleExponentBits, kLongDoubleExponentBias,
                        kLongDoubleHasExplicitIntegerBit));
    ElementTypeInfo b;
    b.element_type = ElementType::Bool;
    b.name = "bool";
    b.byte_size = 1;
    b.is_bool = true;
    m.emplace(ElementType::Bool, b);
    return m;
  }();
  return t;
}

/// `[min_exp, max_exp]` (UNBIASED, the smallest/largest exponent a NORMAL
/// value of this type can carry) -- the containment test `is_exact_
/// widening()` needs. Subnormals are deliberately excluded from this
/// range: a subnormal's effective exponent is below `min_exp`, but every
/// subnormal of a NARROWER format is still representable (exactly or by
/// further rounding, which `AllowNarrowing`, not `Widening`, covers) in a
/// WIDER format's own normal range, so excluding them here only makes the
/// containment test for `Widening` conservative in the direction that
/// matters (never approves a widening that could lose a subnormal's
/// magnitude).
struct ExponentRange {
  std::int64_t min_exp;
  std::int64_t max_exp;
};

[[nodiscard]] ExponentRange exponent_range(const ElementTypeInfo& info) {
  // Normal exponent field values run 1 .. (2^exponent_bits - 2) (0 and
  // all-ones are reserved for subnormal/zero and inf/NaN); unbiased that
  // is [1 - bias, (2^exponent_bits - 2) - bias].
  const std::int64_t max_field =
      (std::int64_t{1} << info.exponent_bits) - 2;
  return ExponentRange{1 - info.exponent_bias, max_field - info.exponent_bias};
}

}  // namespace

const ElementTypeInfo& element_type_info(ElementType t) {
  return table().at(t);
}

std::optional<ElementType> element_type_from_name(const std::string& name) {
  for (const auto& [type, info] : table()) {
    if (info.name == name) return type;
  }
  return std::nullopt;
}

bool is_exact_widening(const ElementTypeInfo& f, const ElementTypeInfo& t) {
  if (f.element_type == t.element_type && f.byte_size == t.byte_size &&
      f.significand_bits == t.significand_bits &&
      f.exponent_bits == t.exponent_bits && f.exponent_bias == t.exponent_bias) {
    return true;  // identical: trivially exact, but callers needing
                 // Exact semantics never reach this function at all
                 // (Exact is enumerator equality, decided before this
                 // call) -- kept here only so this function is total.
  }

  if (f.is_bool) {
    // bool -> any integer type: always exact. bool ->
    // float is NOT authorized by the design and is deliberately
    // not added here -- see docs/dev/restart.md Section 11's open
    // question on this gap.
    return t.is_signed_integer || t.is_unsigned_integer;
  }

  if ((f.is_signed_integer || f.is_unsigned_integer) &&
      (t.is_signed_integer || t.is_unsigned_integer)) {
    // NEVER signed -> unsigned, at any width (a rule stated
    // unconditionally -- not a per-value check of whether the source
    // happens to be non-negative).
    if (f.is_signed_integer && t.is_unsigned_integer) return false;
    const std::size_t from_bits = f.byte_size * 8;
    const std::size_t to_bits = t.byte_size * 8;
    if (f.is_unsigned_integer && t.is_signed_integer) {
      // An N-bit unsigned value's range [0, 2^N - 1] fits an M-bit
      // signed type's range [-2^(M-1), 2^(M-1) - 1] exactly when N <
      // M - 1, i.e. N <= M - 2... but the named example,
      // uint32 -> int64 (N=32, M=64), needs N <= M-1 to hold (32 <= 63):
      // an N-bit unsigned max is 2^N - 1, which fits an M-bit signed
      // type's max (2^(M-1) - 1) exactly when 2^N - 1 <= 2^(M-1) - 1,
      // i.e. N <= M - 1.
      return from_bits <= to_bits - 1;
    }
    // Same signedness: a plain sign-/zero-extend is exact whenever the
    // destination is at least as wide.
    return from_bits <= to_bits;
  }

  if ((f.is_signed_integer || f.is_unsigned_integer) && t.is_float) {
    // "integer -> float only when the float's significand covers the
    // integer's width" -- an N-bit integer (signed
    // or unsigned; the rule does not distinguish) is exactly
    // representable in a float with >= N significand bits, because every
    // integer with magnitude < 2^(significand_bits) is exactly
    // representable in that format.
    const std::size_t from_bits = f.byte_size * 8;
    return from_bits <= t.significand_bits;
  }

  if (f.is_float && t.is_float) {
    // Exact float -> float widening: the destination's significand must
    // cover the source's (every value representable in P1 bits is
    // representable in P2 >= P1 bits with the SAME value), AND the
    // destination's normal exponent range must contain the source's (an
    // exponent outside the destination's range would overflow to
    // infinity or need subnormal handling the source format never
    // needed) -- see this file's own `exponent_range()`.
    if (t.significand_bits < f.significand_bits) return false;
    const ExponentRange fr = exponent_range(f);
    const ExponentRange tr = exponent_range(t);
    return tr.min_exp <= fr.min_exp && tr.max_exp >= fr.max_exp;
  }

  // float -> integer, or bool -> float: not authorized (see the comments
  // above at each relevant branch) -- neither Widening case exists.
  return false;
}

}  // namespace fuse
