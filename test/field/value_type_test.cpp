// Field value types (plans/2026-10-22-universal-number-types.md, stage U1):
// fuse::value_traits / fuse::CoupledValue, the type-erased registry, the typed
// pipeline views, and the correctly rounded binary64 -> bfloat16 conversion.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "fuse/base/ids.hpp"
#include "fuse/field/field_ref.hpp"
#include "fuse/field/pipeline.hpp"
#include "fuse/field/registry.hpp"
#include "fuse/model/value_traits.hpp"

namespace {

// A forward-mode dual number with three tangents: the case lane decomposition
// exists for. Its value_traits says LaneLinear over four binary64 lanes.
struct Dual3 {
  double v[4];
};

// A type with no value_traits at all.
struct NoTraits {
  double x;
};

// Claims to be LaneLinear over three doubles but holds four: the lanes do not
// tile the type, so lane decomposition would read the wrong bytes.
struct MisTiled {
  double v[4];
};

// Has a complete value_traits but is not trivially copyable, so its bytes
// cannot move through MPI or a type-erased registry.
struct NotTrivial {
  double x = 0.0;
  NotTrivial() = default;
  NotTrivial(const NotTrivial& o) : x(o.x) {}
  NotTrivial& operator=(const NotTrivial& o) {
    x = o.x;
    return *this;
  }
};

}  // namespace

template <>
struct fuse::value_traits<Dual3> {
  static constexpr std::string_view name = "test:dual3";
  static constexpr FieldValueKind kind = FieldValueKind::LaneLinear;
  using lane_type = double;
  static constexpr int lanes = 4;
  static double to_binary64(const Dual3& d) noexcept { return d.v[0]; }
  static Dual3 from_binary64(double x) noexcept { return Dual3{{x, 0, 0, 0}}; }
};

template <>
struct fuse::value_traits<MisTiled> {
  static constexpr std::string_view name = "test:mistiled";
  static constexpr FieldValueKind kind = FieldValueKind::LaneLinear;
  using lane_type = double;
  static constexpr int lanes = 3;
  static double to_binary64(const MisTiled& d) noexcept { return d.v[0]; }
  static MisTiled from_binary64(double x) noexcept {
    return MisTiled{{x, 0, 0, 0}};
  }
};

template <>
struct fuse::value_traits<NotTrivial> {
  static constexpr std::string_view name = "test:not-trivial";
  static constexpr FieldValueKind kind = FieldValueKind::Storage;
  using lane_type = NotTrivial;
  static constexpr int lanes = 1;
  static double to_binary64(const NotTrivial& d) noexcept { return d.x; }
  static NotTrivial from_binary64(double x) noexcept {
    NotTrivial n;
    n.x = x;
    return n;
  }
};

namespace {

static_assert(fuse::CoupledValue<double>);
static_assert(fuse::CoupledValue<float>);
static_assert(fuse::CoupledValue<fuse::bfloat16>);
static_assert(fuse::CoupledValue<Dual3>);
static_assert(!fuse::CoupledValue<NoTraits>,
              "a type with no value_traits must not be a field value type");
static_assert(!fuse::CoupledValue<MisTiled>,
              "a LaneLinear type whose lanes do not tile it must be refused");
static_assert(!fuse::CoupledValue<NotTrivial>,
              "a type that is not trivially copyable must be refused");
static_assert(!fuse::CoupledValue<int>,
              "no value_traits<int> exists: integers are identifiers, not "
              "coupled values");

static_assert(std::is_same_v<fuse::FieldRef, fuse::BasicFieldRef<double>>);
static_assert(std::is_same_v<fuse::FieldSpan<>,
                             fuse::FieldSpan<double, Kokkos::HostSpace>>);

fuse::Layout interleaved(std::size_t count, int components) {
  return fuse::Layout{static_cast<std::size_t>(components), 1, count,
                      components};
}

TEST(FieldValueType, DescriptorsNameSizeKindAndLanes) {
  constexpr auto b64 = fuse::field_value_type_of<double>();
  EXPECT_EQ(b64.name, "binary64");
  EXPECT_EQ(b64.size, 8u);
  EXPECT_EQ(b64.kind, fuse::FieldValueKind::LaneLinear);
  EXPECT_EQ(b64.lanes, 1);

  constexpr auto b32 = fuse::field_value_type_of<float>();
  EXPECT_EQ(b32.name, "binary32");
  EXPECT_EQ(b32.size, 4u);
  EXPECT_EQ(b32.kind, fuse::FieldValueKind::Storage);

  constexpr auto bf = fuse::field_value_type_of<fuse::bfloat16>();
  EXPECT_EQ(bf.name, "bfloat16");
  EXPECT_EQ(bf.size, 2u);

  constexpr auto d3 = fuse::field_value_type_of<Dual3>();
  EXPECT_EQ(d3.lanes, 4);
  EXPECT_EQ(d3.size, 32u);

  EXPECT_NE(b64, b32);
}

TEST(FieldValueType, EqualityComparesNameCharactersNotAddresses) {
  // The same type named from two shared libraries has two copies of the
  // literal; the descriptors must still compare equal.
  const std::string copy = "binary64";
  fuse::FieldValueType other = fuse::field_value_type_of<double>();
  other.name = copy;
  ASSERT_NE(static_cast<const void*>(other.name.data()),
            static_cast<const void*>(fuse::field_value_type_of<double>().name.data()));
  EXPECT_EQ(other, fuse::field_value_type_of<double>());
}

TEST(FieldRegistryValueType, TypedFieldRoundTripsAndKeepsItsType) {
  std::vector<float> data(6, 0.0f);
  fuse::FieldRegistry reg;
  const fuse::FieldId id{3};
  ASSERT_TRUE(reg.bind(id, fuse::BasicFieldRef<float>(
                               data.data(), interleaved(3, 2),
                               fuse::MemorySpace::Host)));

  auto typed = reg.get<float>(id);
  ASSERT_TRUE(typed);
  typed.value().at(1, 1) = 2.5f;
  EXPECT_EQ(data[3], 2.5f);  // the view writes through to the owner

  auto any = reg.get_any(id);
  ASSERT_TRUE(any);
  EXPECT_EQ(any.value().value_type(), fuse::field_value_type_of<float>());
  EXPECT_TRUE(any.value().holds<float>());
  EXPECT_FALSE(any.value().holds<double>());
}

TEST(FieldRegistryValueType, WrongTypeIsRefusedByNameAndNeverConverted) {
  std::vector<float> data(4, 1.0f);
  fuse::FieldRegistry reg;
  const fuse::FieldId id{7};
  ASSERT_TRUE(reg.bind(id, fuse::BasicFieldRef<float>(
                               data.data(), interleaved(4, 1),
                               fuse::MemorySpace::Host)));

  // get() with no template argument is get<double>(), what every caller that
  // predates value types asks for: it must refuse a binary32 field rather
  // than reinterpret four floats as two doubles.
  const auto as_default = reg.get(id);
  ASSERT_FALSE(as_default);
  EXPECT_EQ(as_default.error().code, "E-FIELD-VALUE-TYPE");
  EXPECT_NE(as_default.error().message.find("field 7"), std::string::npos);
  EXPECT_NE(as_default.error().message.find("binary32"), std::string::npos);
  EXPECT_NE(as_default.error().message.find("binary64"), std::string::npos);

  const auto as_bf16 = reg.get<fuse::bfloat16>(id);
  ASSERT_FALSE(as_bf16);
  EXPECT_EQ(as_bf16.error().code, "E-FIELD-VALUE-TYPE");

  const auto unbound = reg.get<float>(fuse::FieldId{8});
  ASSERT_FALSE(unbound);
  EXPECT_EQ(unbound.error().code, "E-FIELD-UNBOUND");
}

TEST(FieldRegistryValueType, BindRefusalsApplyToEveryValueType) {
  std::vector<float> data(6, 0.0f);
  fuse::FieldRegistry reg;
  // stride 1 with 3 interleaved components aliases element i+1 onto i.
  const auto bad = reg.bind(
      fuse::FieldId{1},
      fuse::BasicFieldRef<float>(data.data(), fuse::Layout{1, 1, 2, 3},
                                 fuse::MemorySpace::Host));
  ASSERT_FALSE(bad);
  EXPECT_EQ(bad.error().code, "E-FIELD-LAYOUT");

  const auto null_field = reg.bind(
      fuse::FieldId{2}, fuse::BasicFieldRef<float>(nullptr, interleaved(2, 1),
                                                   fuse::MemorySpace::Host));
  ASSERT_FALSE(null_field);
  EXPECT_EQ(null_field.error().code, "E-FIELD-NULL");

  ASSERT_TRUE(reg.bind(fuse::FieldId{3},
                       fuse::BasicFieldRef<float>(data.data(),
                                                  interleaved(2, 3),
                                                  fuse::MemorySpace::Host)));
  // Rebinding is refused even with a different value type.
  std::vector<double> other(6, 0.0);
  const auto rebind = reg.bind(
      fuse::FieldId{3},
      fuse::FieldRef(other.data(), interleaved(2, 3), fuse::MemorySpace::Host));
  ASSERT_FALSE(rebind);
  EXPECT_EQ(rebind.error().code, "E-FIELD-REBIND");
  EXPECT_EQ(reg.bound_count(), 1u);
}

TEST(FieldRegistryValueType, BinaryDoubleCallersAreUnchanged) {
  std::vector<double> data(4, 0.0);
  fuse::FieldRegistry reg;
  // The braced spelling that existing call sites use must still bind.
  ASSERT_TRUE(reg.bind(fuse::FieldId{1},
                       {data.data(), interleaved(4, 1), fuse::MemorySpace::Host}));
  auto ref = reg.get(fuse::FieldId{1});
  ASSERT_TRUE(ref);
  static_assert(std::is_same_v<decltype(ref.value()), fuse::FieldRef&>);
  ref.value().at(2) = 4.0;
  EXPECT_EQ(data[2], 4.0);
}

TEST(FieldRegistryValueType, UserLaneLinearTypeBindsAndRoundTrips) {
  std::vector<Dual3> data(2, Dual3{{0, 0, 0, 0}});
  fuse::FieldRegistry reg;
  ASSERT_TRUE(reg.bind(fuse::FieldId{9},
                       fuse::BasicFieldRef<Dual3>(data.data(),
                                                  interleaved(2, 1),
                                                  fuse::MemorySpace::Host)));
  auto ref = reg.get<Dual3>(fuse::FieldId{9});
  ASSERT_TRUE(ref);
  ref.value().at(1).v[2] = -3.0;
  EXPECT_EQ(data[1].v[2], -3.0);
  EXPECT_FALSE(reg.get(fuse::FieldId{9}));
}

TEST(FieldSpanValueType, StagesSelectWithoutConvertingAndSumInBinary64) {
  // 2^24 + 1 + 1: in binary32 each 1 is absorbed (2^24 + 1 rounds back to
  // 2^24), in binary64 both survive. local_sum() must give the binary64 sum.
  std::vector<float> data = {16777216.0f, 1.0f, 1.0f, 0.0f};
  const fuse::BasicFieldRef<float> ref(data.data(), interleaved(2, 2),
                                       fuse::MemorySpace::Host);
  const auto whole = fuse::FieldSpan<float>::from(ref);
  EXPECT_EQ(whole.local_sum(), 16777218.0);
  float naive = 0.0f;
  for (float x : data) naive += x;
  EXPECT_EQ(naive, 16777216.0f) << "the binary32 sum should lose both ones";

  const auto y = whole.component(1);
  ASSERT_EQ(y.element_count(), 2u);
  EXPECT_EQ(y.at(0), 1.0f);
  EXPECT_EQ(y.at(1), 0.0f);
  static_assert(std::is_same_v<decltype(y.at(0)), float>);
}

TEST(FieldSpanValueType, OwnedBfloat16FieldStartsAtZeroAndFills) {
  fuse::OwnedField<fuse::bfloat16> f("bf", 5, 2);
  EXPECT_EQ(f.span().local_sum(), 0.0);
  f.span().fill(fuse::value_traits<fuse::bfloat16>::from_binary64(0.5));
  EXPECT_EQ(f.span().local_sum(), 5.0);
  EXPECT_EQ(f.span().at(4, 1).bits(), 0x3F00u);  // 0.5 in bfloat16
}

// ---------------------------------------------------- binary64 -> bfloat16

std::uint16_t bf16_bits_from_binary64(double v) {
  return fuse::value_traits<fuse::bfloat16>::from_binary64(v).bits();
}

double bf16_value(std::uint16_t bits) {
  return fuse::value_traits<fuse::bfloat16>::to_binary64(
      fuse::bfloat16::from_bits(bits));
}

TEST(Bfloat16FromBinary64, RoundingTwiceToNearestIsWrongHereAndOnceIsRight) {
  // 1 and 1 + 2^-7 are adjacent bfloat16 values; their midpoint is 1 + 2^-8.
  // v lies 2^-30 ABOVE the midpoint, much less than half a binary32 unit in
  // the last place (2^-24), so rounding v to binary32 lands exactly on the
  // midpoint and the bfloat16 tie then goes to the even neighbour, 1.
  const double mid = 1.0 + std::ldexp(1.0, -8);
  const double above = mid + std::ldexp(1.0, -30);
  const double below = mid - std::ldexp(1.0, -30);
  ASSERT_EQ(static_cast<double>(static_cast<float>(above)), mid);
  const std::uint16_t twice =
      fuse::bfloat16::from_float(static_cast<float>(above)).bits();
  EXPECT_EQ(twice, 0x3F80u) << "the counterexample must exist for this test "
                               "to mean anything";
  EXPECT_EQ(bf16_bits_from_binary64(above), 0x3F81u);  // 1 + 2^-7
  EXPECT_EQ(bf16_bits_from_binary64(below), 0x3F80u);  // 1
  // Exact ties go to even.
  EXPECT_EQ(bf16_bits_from_binary64(mid), 0x3F80u);
  EXPECT_EQ(bf16_bits_from_binary64(1.0 + 3.0 * std::ldexp(1.0, -8)), 0x3F82u);
  // The same at negative sign.
  EXPECT_EQ(bf16_bits_from_binary64(-above), 0xBF81u);
}

TEST(Bfloat16FromBinary64, OverflowInfinityAndNaN) {
  const double max_bf16 = bf16_value(0x7F7Fu);  // (2 - 2^-7) * 2^127
  const double mid_to_inf = std::ldexp(2.0 - std::ldexp(1.0, -8), 127);
  EXPECT_EQ(bf16_bits_from_binary64(max_bf16), 0x7F7Fu);
  EXPECT_EQ(bf16_bits_from_binary64(std::nextafter(mid_to_inf, 0.0)), 0x7F7Fu);
  EXPECT_EQ(bf16_bits_from_binary64(mid_to_inf), 0x7F80u);  // tie to even: inf
  EXPECT_EQ(bf16_bits_from_binary64(1e300), 0x7F80u);
  EXPECT_EQ(bf16_bits_from_binary64(-1e300), 0xFF80u);
  EXPECT_EQ(bf16_bits_from_binary64(std::numeric_limits<double>::infinity()),
            0x7F80u);
  const std::uint16_t nan =
      bf16_bits_from_binary64(std::numeric_limits<double>::quiet_NaN());
  EXPECT_EQ(nan & 0x7F80u, 0x7F80u);
  EXPECT_NE(nan & 0x007Fu, 0u);
}

TEST(Bfloat16FromBinary64, RandomizedResultIsTheNearestWithTiesToEven) {
  // For each v, the result c must be at least as close to v as both of its
  // bfloat16 neighbours, and a tie must land on an even significand. v and
  // its candidates are within a factor of 2 of each other, so every
  // subtraction below is exact (Sterbenz's lemma) and the comparison is too.
  std::mt19937_64 rng(20261023);
  std::uniform_int_distribution<int> expo(-120, 120);
  std::uniform_real_distribution<double> frac(1.0, 2.0);
  int checked = 0;
  for (int k = 0; k < 200000; ++k) {
    double v = std::ldexp(frac(rng), expo(rng));
    if (k % 2) v = -v;
    const std::uint16_t c = bf16_bits_from_binary64(v);
    const double cv = bf16_value(c);
    ASSERT_TRUE(std::isfinite(cv)) << v;
    const double d = std::fabs(v - cv);
    for (const int step : {-1, 1}) {
      const std::uint16_t nb = static_cast<std::uint16_t>(c + step);
      const double nv = bf16_value(nb);
      if (!std::isfinite(nv) || std::signbit(nv) != std::signbit(cv) ||
          nv == 0.0) {
        continue;
      }
      const double dn = std::fabs(v - nv);
      ASSERT_LE(d, dn) << "v=" << v << " c=0x" << std::hex << c;
      if (d == dn) {
        ASSERT_EQ(c & 1u, 0u) << "tie not to even, v=" << v;
      }
    }
    ++checked;
  }
  EXPECT_EQ(checked, 200000);
}

}  // namespace
