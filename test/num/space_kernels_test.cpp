// The per-step kernels on EVERY execution space the build enabled, each
// checked against the plain-loop form in fuse/num/kernels.hpp as the oracle.
//
// One definition, every space: locally Serial and Threads, plus OpenMP in the
// CI leg and a device space wherever one is configured. A device test is then
// a build option rather than a second file that drifts from this one.
//
// NO PERFORMANCE CLAIM IS MADE HERE. What is asserted is that each kernel
// compiles for the space and returns the oracle's answer. Parameterisation is
// a precondition for device performance, not evidence of it.

#include "fuse/num/space_kernels.hpp"

#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/num/kernels.hpp"
#include "fuse_test_main.hpp"

namespace {

/// Host mirrors are used throughout rather than assuming the space's memory is
/// host-accessible: on a device space it is not, and a test that indexed the
/// view directly would compile here and fail there -- which is exactly the
/// portability defect these tests exist to prevent.
template <class Space>
class SpaceKernels : public ::testing::Test {
 protected:
  using MemSpace = typename Space::memory_space;
  using Owned = Kokkos::View<double*, MemSpace>;
  using OwnedIdx = Kokkos::View<std::size_t*, MemSpace>;

  static Owned upload(const std::vector<double>& v, const std::string& label) {
    Owned d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }
  static OwnedIdx upload_idx(const std::vector<std::size_t>& v,
                             const std::string& label) {
    OwnedIdx d(Kokkos::view_alloc(label, Kokkos::WithoutInitializing), v.size());
    auto h = Kokkos::create_mirror_view(d);
    for (std::size_t i = 0; i < v.size(); ++i) h(i) = v[i];
    Kokkos::deep_copy(d, h);
    return d;
  }
  static std::vector<double> download(const Owned& d) {
    auto h = Kokkos::create_mirror_view(d);
    Kokkos::deep_copy(h, d);
    std::vector<double> v(d.extent(0));
    for (std::size_t i = 0; i < v.size(); ++i) v[i] = h(i);
    return v;
  }
  /// A deterministic, non-trivial field: a constant input would let an
  /// off-by-one range or a wrong index list pass.
  static std::vector<double> ramp(std::size_t n, double a, double b) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
      v[i] = a * std::sin(0.37 * static_cast<double>(i)) +
             b * static_cast<double>(i % 7);
    }
    return v;
  }

  template <class V>
  static fuse::num::ConstSpanOf<Space> c(const V& v) {
    return fuse::num::ConstSpanOf<Space>(v.data(), v.extent(0));
  }
  template <class V>
  static fuse::num::SpanOf<Space> m(const V& v) {
    return fuse::num::SpanOf<Space>(v.data(), v.extent(0));
  }
  template <class V>
  static fuse::num::IndexSpanOf<Space> ix(const V& v) {
    return fuse::num::IndexSpanOf<Space>(v.data(), v.extent(0));
  }
};

}  // namespace

FUSE_TYPED_TEST_ON_SPACES(SpaceKernels);

TYPED_TEST(SpaceKernels, AccumulateMatchesTheSerialOracleForEveryOp) {
  using F = SpaceKernels<TypeParam>;
  constexpr std::size_t n = 4096;
  const auto sample = F::ramp(n, 1.0, 0.5);

  for (const auto op : {fuse::AccumOp::Instant, fuse::AccumOp::Sum,
                        fuse::AccumOp::Mean, fuse::AccumOp::Integral,
                        fuse::AccumOp::Min, fuse::AccumOp::Max}) {
    for (const bool first : {true, false}) {
      // Oracle: the plain loop, on a fresh accumulator seeded the same way.
      std::vector<double> want(n, 3.25);
      fuse::num::accumulate_into(op, sample.data(), want.data(), n,
                                 first ? 0u : 1u);

      auto acc = F::upload(std::vector<double>(n, 3.25), "acc");
      auto smp = F::upload(sample, "sample");
      fuse::num::accumulate<TypeParam>(op, F::c(smp), F::m(acc), first);

      const auto got = F::download(acc);
      for (std::size_t i = 0; i < n; ++i) {
        ASSERT_DOUBLE_EQ(got[i], want[i])
            << "op " << static_cast<int>(op) << " first=" << first
            << " element " << i;
      }
    }
  }
}

TYPED_TEST(SpaceKernels, RelaxMatchesTheSerialOracle) {
  using F = SpaceKernels<TypeParam>;
  constexpr std::size_t n = 4096;
  const auto prev = F::ramp(n, 1.0, 0.25);
  const auto curr = F::ramp(n, -0.5, 1.75);

  std::vector<double> want(n, 0.0);
  fuse::num::relax(0.37, prev.data(), curr.data(), want.data(), n);

  auto p = F::upload(prev, "prev");
  auto c = F::upload(curr, "curr");
  auto o = F::upload(std::vector<double>(n, 0.0), "out");
  fuse::num::relax<TypeParam>(0.37, F::c(p), F::c(c), F::m(o));

  const auto got = F::download(o);
  for (std::size_t i = 0; i < n; ++i) ASSERT_DOUBLE_EQ(got[i], want[i]) << i;
}

TYPED_TEST(SpaceKernels, GatherAndScatterRoundTripThroughAPermutation) {
  using F = SpaceKernels<TypeParam>;
  constexpr std::size_t n = 2048;
  const auto src = F::ramp(n, 1.0, 1.0);

  // A non-identity permutation, so a kernel that ignored the index list would
  // fail rather than coincidentally pass.
  std::vector<std::size_t> perm(n);
  for (std::size_t k = 0; k < n; ++k) perm[k] = (k * 977 + 13) % n;

  auto d_src = F::upload(src, "src");
  auto d_idx = F::upload_idx(perm, "idx");
  auto packed = F::upload(std::vector<double>(n, 0.0), "packed");
  fuse::num::gather<TypeParam>(F::c(d_src), F::ix(d_idx), F::m(packed));

  const auto got = F::download(packed);
  for (std::size_t k = 0; k < n; ++k) ASSERT_DOUBLE_EQ(got[k], src[perm[k]]) << k;

  // Scattering the packed values back through the same permutation must
  // reconstruct the original exactly: gather then scatter is the identity when
  // the index list is a permutation, which is the transport's whole contract.
  auto back = F::upload(std::vector<double>(n, -1.0), "back");
  fuse::num::scatter<TypeParam>(F::c(packed), F::ix(d_idx), F::m(back));
  const auto round = F::download(back);
  for (std::size_t i = 0; i < n; ++i) ASSERT_DOUBLE_EQ(round[i], src[i]) << i;
}

TYPED_TEST(SpaceKernels, RemapApplyMatchesADirectTripletProduct) {
  using F = SpaceKernels<TypeParam>;
  constexpr std::size_t rows = 512;
  constexpr std::size_t per_row = 4;
  constexpr std::size_t cols = 300;

  // Row-major sorted triplets with a prefix-scan row pointer, which is what
  // the kernel's no-atomics argument depends on.
  std::vector<std::size_t> row_begin(rows + 1, 0);
  std::vector<std::size_t> col(rows * per_row);
  std::vector<double>      val(rows * per_row);
  for (std::size_t j = 0; j < rows; ++j) {
    row_begin[j + 1] = row_begin[j] + per_row;
    for (std::size_t e = 0; e < per_row; ++e) {
      const std::size_t k = j * per_row + e;
      col[k] = (j * 7 + e * 53) % cols;
      val[k] = 0.25 + 0.1 * static_cast<double>(e);
    }
  }
  const auto x = F::ramp(cols, 1.0, 0.5);

  std::vector<double> want(rows, 0.0);
  for (std::size_t j = 0; j < rows; ++j) {
    for (std::size_t k = row_begin[j]; k < row_begin[j + 1]; ++k) {
      want[j] += val[k] * x[col[k]];
    }
  }

  auto d_rb = F::upload_idx(row_begin, "row_begin");
  auto d_col = F::upload_idx(col, "col");
  auto d_val = F::upload(val, "val");
  auto d_x = F::upload(x, "x");
  auto d_y = F::upload(std::vector<double>(rows, 0.0), "y");
  fuse::num::remap_apply<TypeParam>(F::ix(d_rb), F::ix(d_col), F::c(d_val),
                                    F::c(d_x), F::m(d_y));

  const auto got = F::download(d_y);
  for (std::size_t j = 0; j < rows; ++j) {
    ASSERT_NEAR(got[j], want[j], 1e-13 * (1.0 + std::abs(want[j]))) << j;
  }
}

TYPED_TEST(SpaceKernels, ReductionsAgreeWithTheSerialSumToRounding) {
  using F = SpaceKernels<TypeParam>;
  constexpr std::size_t n = 100'000;
  const auto a = F::ramp(n, 1.0, 0.5);
  const auto b = F::ramp(n, 0.75, -0.25);

  double want_dot = 0.0, want_wns = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    want_dot += a[i] * b[i];
    want_wns += b[i] * a[i] * a[i];
  }

  auto d_a = F::upload(a, "a");
  auto d_b = F::upload(b, "b");

  // NOT bitwise: a parallel reduction associates differently from a
  // sequential one, so the tolerance is the accumulated rounding of n terms
  // and is stated as such rather than tightened until it happens to pass.
  const double tol = 1e-12 * static_cast<double>(n);
  EXPECT_NEAR(fuse::num::dot<TypeParam>(F::c(d_a), F::c(d_b)), want_dot, tol);
  EXPECT_NEAR(
      fuse::num::weighted_norm_squared<TypeParam>(F::c(d_a), F::c(d_b)),
      want_wns, tol);
}

TYPED_TEST(SpaceKernels, TheSpaceUnderTestIsNamedInTheRecord) {
  // So a reader of CI output can see WHICH spaces ran, rather than inferring
  // it from the build flags. A typed test that silently instantiates once is
  // indistinguishable from one that covers everything.
  std::printf("[ space    ] %s\n", TypeParam::name());
  SUCCEED();
}
