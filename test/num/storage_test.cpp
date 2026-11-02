// The Storage conversions on every enabled execution space: widening and narrowing of binary32,
// bfloat16 and binary16 inside a Kokkos kernel give the same bits as the host
// functions, over every 16-bit pattern, over every midpoint between adjacent
// values of each 16-bit format (and one binary64 step either side of it), and
// over a log-uniform sample. The host functions are themselves checked against
// independent oracles in test/model/{binary16,small_float,bfloat16}_test.cpp, so
// this is the check that a device does not compute something else.

#include <gtest/gtest.h>

#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "fuse/model/value_traits.hpp"
#include "fuse/num/storage_convert.hpp"
#include "fuse_test_main.hpp"

namespace {

using fuse::bfloat16;
using fuse::binary16;

static_assert(fuse::CoupledValue<binary16>);
static_assert(fuse::value_traits<binary16>::kind == fuse::FieldValueKind::Storage);

template <class T>
std::uint64_t bits_of(const T& v) {
  std::uint64_t b = 0;
  std::memcpy(&b, &v, sizeof(T));
  return b;
}

// All finite and non-finite patterns of a 16-bit type, as doubles.
template <class T>
std::vector<double> every_pattern() {
  std::vector<double> v;
  for (unsigned p = 0; p <= 0xFFFFu; ++p) {
    v.push_back(fuse::value_traits<T>::to_binary64(
        T::from_bits(static_cast<std::uint16_t>(p))));
  }
  return v;
}

// Inputs that stress the rounding: the midpoints between adjacent finite values
// and a binary64 step each side, the values themselves, a log-uniform sample,
// and the special values.
template <class T>
std::vector<double> narrowing_inputs(double lo_exp, double hi_exp) {
  std::vector<double> in;
  std::vector<double> all = every_pattern<T>();
  for (unsigned p = 0; p < 0x7FFFu; ++p) {  // finite and infinite non-negative patterns
    const double a = all[p], b = all[p + 1];
    if (!std::isfinite(a) || !std::isfinite(b)) continue;
    const double mid = 0.5 * (a + b);
    for (const double x : {mid, std::nextafter(mid, 0.0), std::nextafter(mid, 1e300), a}) {
      in.push_back(x);
      in.push_back(-x);
    }
  }
  std::mt19937_64 rng(42);
  std::uniform_real_distribution<double> e(lo_exp, hi_exp);
  for (int i = 0; i < 100000; ++i) {
    const double x = std::exp2(e(rng));
    in.push_back(rng() & 1u ? -x : x);
  }
  for (const double x : {0.0, -0.0, std::numeric_limits<double>::infinity(),
                         -std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN(),
                         std::numeric_limits<double>::max(),
                         std::numeric_limits<double>::denorm_min(),
                         std::numeric_limits<double>::min(), 65519.99999999999, 65520.0}) {
    in.push_back(x);
  }
  return in;
}

template <class Space>
class StorageOnSpace : public ::testing::Test {
 protected:
  using Mem = typename Space::memory_space;
  template <class T>
  using Vec = Kokkos::View<T*, Mem>;

  template <class T>
  static Vec<T> upload(const std::vector<T>& v) {
    Vec<T> d(Kokkos::view_alloc(std::string("u"), Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }
  template <class T>
  static std::vector<T> download(const Vec<T>& d) {
    auto h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(h, d);
    std::vector<T> v(d.extent(0));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = h(i);
    return v;
  }

  // device widen of `patterns`, compared bit for bit with the host function
  template <class T>
  static void expect_widen_matches(const std::vector<T>& patterns) {
    auto din = upload(patterns);
    Vec<double> dout(Kokkos::view_alloc(std::string("w"), Kokkos::WithoutInitializing),
                     patterns.size());
    fuse::num::widen<Space, T>(
        fuse::num::ConstStorageSpanOf<Space, T>(din.data(), din.extent(0)),
        fuse::num::SpanOf<Space>(dout.data(), dout.extent(0)));
    Kokkos::fence();
    const auto got = download(dout);
    for (std::size_t i = 0; i < patterns.size(); ++i) {
      const double want = fuse::value_traits<T>::to_binary64(patterns[i]);
      ASSERT_EQ(bits_of(got[i]), bits_of(want)) << "pattern index " << i;
    }
  }

  // device narrow of `inputs`, compared bit for bit with the host function, and
  // the measured report compared with the same quantities computed on the host
  template <class T>
  static void expect_narrow_matches(const std::vector<double>& inputs) {
    auto din = upload(inputs);
    Vec<T> dout(Kokkos::view_alloc(std::string("n"), Kokkos::WithoutInitializing),
                inputs.size());
    fuse::num::narrow<Space, T>(
        fuse::num::ConstSpanOf<Space>(din.data(), din.extent(0)),
        fuse::num::StorageSpanOf<Space, T>(dout.data(), dout.extent(0)));
    Kokkos::fence();
    const auto got = download(dout);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      const T want = fuse::value_traits<T>::from_binary64(inputs[i]);
      if (std::isnan(inputs[i])) {
        ASSERT_TRUE(std::isnan(fuse::value_traits<T>::to_binary64(got[i]))) << i;
        continue;
      }
      ASSERT_EQ(bits_of(got[i]), bits_of(want)) << std::hexfloat << inputs[i];
    }

    // the report
    Vec<T> dout2(Kokkos::view_alloc(std::string("m"), Kokkos::WithoutInitializing),
                 inputs.size());
    const fuse::num::RoundingReport r = fuse::num::narrow_measured<Space, T>(
        fuse::num::ConstSpanOf<Space>(din.data(), din.extent(0)),
        fuse::num::StorageSpanOf<Space, T>(dout2.data(), dout2.extent(0)));
    fuse::num::RoundingReport want{};
    for (const double x : inputs) {
      if (!std::isfinite(x)) continue;
      ++want.values;
      const double w = fuse::value_traits<T>::to_binary64(fuse::value_traits<T>::from_binary64(x));
      if (!std::isfinite(w)) { ++want.overflow; continue; }
      if (x == 0.0) continue;
      if (w == 0.0) { ++want.underflow; continue; }
      want.max_relative_change = std::max(want.max_relative_change, std::fabs(w - x) / std::fabs(x));
    }
    EXPECT_EQ(r.values, want.values);
    EXPECT_EQ(r.overflow, want.overflow);
    EXPECT_EQ(r.underflow, want.underflow);
    EXPECT_EQ(r.max_relative_change, want.max_relative_change);
    const auto got2 = download(dout2);
    for (std::size_t i = 0; i < inputs.size(); ++i) {
      if (std::isnan(inputs[i])) continue;
      ASSERT_EQ(bits_of(got2[i]), bits_of(got[i])) << i;
    }
  }
};

}  // namespace

FUSE_TYPED_TEST_ON_SPACES(StorageOnSpace);

TYPED_TEST(StorageOnSpace, EveryBFloat16PatternWidensOnTheDeviceAsOnTheHost) {
  std::vector<bfloat16> p;
  for (unsigned v = 0; v <= 0xFFFFu; ++v) p.push_back(bfloat16::from_bits(static_cast<std::uint16_t>(v)));
  TestFixture::template expect_widen_matches<bfloat16>(p);
}

TYPED_TEST(StorageOnSpace, EveryBinary16PatternWidensOnTheDeviceAsOnTheHost) {
  std::vector<binary16> p;
  for (unsigned v = 0; v <= 0xFFFFu; ++v) p.push_back(binary16::from_bits(static_cast<std::uint16_t>(v)));
  TestFixture::template expect_widen_matches<binary16>(p);
}

TYPED_TEST(StorageOnSpace, Binary32WidensOnTheDeviceAsOnTheHost) {
  std::mt19937 rng(1);
  std::vector<float> p;
  for (int i = 0; i < 100000; ++i) p.push_back(std::bit_cast<float>(static_cast<std::uint32_t>(rng())));
  p.push_back(0.0f);
  p.push_back(-0.0f);
  TestFixture::template expect_widen_matches<float>(p);
}

TYPED_TEST(StorageOnSpace, BFloat16NarrowsOnTheDeviceAsOnTheHostAtEveryMidpoint) {
  TestFixture::template expect_narrow_matches<bfloat16>(narrowing_inputs<bfloat16>(-140.0, 129.0));
}

TYPED_TEST(StorageOnSpace, Binary16NarrowsOnTheDeviceAsOnTheHostAtEveryMidpoint) {
  TestFixture::template expect_narrow_matches<binary16>(narrowing_inputs<binary16>(-30.0, 17.5));
}

TYPED_TEST(StorageOnSpace, Binary32NarrowsOnTheDeviceAsOnTheHost) {
  std::mt19937_64 rng(2);
  std::uniform_real_distribution<double> e(-160.0, 130.0);
  std::vector<double> in;
  for (int i = 0; i < 200000; ++i) {
    const double x = std::exp2(e(rng)) * (1.0 + std::ldexp(static_cast<double>(rng() >> 11), -53));
    in.push_back(rng() & 1u ? -x : x);
  }
  in.push_back(0.0);
  in.push_back(-0.0);
  in.push_back(std::numeric_limits<double>::infinity());
  TestFixture::template expect_narrow_matches<float>(in);
}

TYPED_TEST(StorageOnSpace, ReportCountsOverflowAndUnderflowAndBoundsTheChangeByTheUnitRoundoff) {
  // values in binary16's normal range change by at most 2^-11
  std::mt19937_64 rng(3);
  std::uniform_real_distribution<double> e(-13.0, 15.0);
  std::vector<double> in;
  for (int i = 0; i < 50000; ++i) in.push_back(std::exp2(e(rng)));
  in.push_back(1.0e6);   // overflow
  in.push_back(-1.0e6);  // overflow
  in.push_back(1.0e-9);  // underflow to zero
  in.push_back(0.0);
  auto din = TestFixture::upload(in);
  typename TestFixture::template Vec<binary16> dout(
      Kokkos::view_alloc(std::string("r"), Kokkos::WithoutInitializing), in.size());
  const auto r = fuse::num::narrow_measured<TypeParam, binary16>(
      fuse::num::ConstSpanOf<TypeParam>(din.data(), din.extent(0)),
      fuse::num::StorageSpanOf<TypeParam, binary16>(dout.data(), dout.extent(0)));
  EXPECT_EQ(r.values, static_cast<std::int64_t>(in.size()));
  EXPECT_EQ(r.overflow, 2);
  EXPECT_EQ(r.underflow, 1);
  EXPECT_GT(r.max_relative_change, 0.0);
  EXPECT_LE(r.max_relative_change, std::ldexp(1.0, -11));
}
