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
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/base/config.hpp"
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

TYPED_TEST(SpaceKernels, CopyScaleAndResidualMatchThePlainArithmetic) {
  using F = SpaceKernels<TypeParam>;
  constexpr std::size_t n = 2048;
  const auto a = F::ramp(n, 1.0, 0.5);
  const auto b = F::ramp(n, -0.75, 1.25);

  auto d_a = F::upload(a, "a");
  auto d_b = F::upload(b, "b");
  auto d_out = F::upload(std::vector<double>(n, 0.0), "out");

  fuse::num::copy<TypeParam>(F::c(d_a), F::m(d_out));
  auto got_copy = F::download(d_out);
  for (std::size_t i = 0; i < n; ++i) ASSERT_DOUBLE_EQ(got_copy[i], a[i]) << i;

  fuse::num::scale<TypeParam>(2.75, F::c(d_a), F::m(d_out));
  auto got_scale = F::download(d_out);
  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_DOUBLE_EQ(got_scale[i], 2.75 * a[i]) << i;
  }

  fuse::num::residual<TypeParam>(F::c(d_a), F::c(d_b), F::m(d_out));
  auto got_residual = F::download(d_out);
  for (std::size_t i = 0; i < n; ++i) {
    ASSERT_DOUBLE_EQ(got_residual[i], a[i] - b[i]) << i;
  }
}

TYPED_TEST(SpaceKernels, RemapApplyMultiComponentMatchesApplyWeightsOrder) {
  using F = SpaceKernels<TypeParam>;
  constexpr std::size_t rows = 64;
  constexpr std::size_t per_row = 3;
  constexpr std::size_t cols = 40;
  constexpr std::size_t components = 3;

  // Row-major sorted triplets (row ascending, col ascending within a row) --
  // RemapWeights' own documented invariant -- with some rows deliberately
  // empty (row_begin(j) == row_begin(j+1)) so the CSR prefix scan's gap
  // handling is exercised, not just the dense case.
  std::vector<std::size_t> row_begin(rows + 1, 0);
  std::vector<std::size_t> col;
  std::vector<double>      val;
  for (std::size_t j = 0; j < rows; ++j) {
    const std::size_t k = (j % 5 == 4) ? 0 : per_row;  // every 5th row empty
    for (std::size_t e = 0; e < k; ++e) {
      col.push_back((j * 11 + e * 17) % cols);
      val.push_back(0.2 + 0.1 * static_cast<double>(e));
    }
    row_begin[j + 1] = row_begin[j] + k;
  }
  const auto x = F::ramp(cols * components, 1.0, 0.3);

  // The oracle: apply_weights' own COO scan (fuse/backend/remap_plan.hpp),
  // over the SAME triplets in the SAME (row-major, ascending-within-row)
  // order -- exactly what the CSR conversion the runtime does once at plan
  // time is supposed to preserve.
  std::vector<double> want(rows * components, 0.0);
  for (std::size_t j = 0; j < rows; ++j) {
    for (std::size_t k = row_begin[j]; k < row_begin[j + 1]; ++k) {
      for (std::size_t c = 0; c < components; ++c) {
        want[j * components + c] += val[k] * x[col[k] * components + c];
      }
    }
  }

  auto d_rb = F::upload_idx(row_begin, "row_begin");
  auto d_col = F::upload_idx(col, "col");
  auto d_val = F::upload(val, "val");
  auto d_x = F::upload(x, "x");
  auto d_y = F::upload(std::vector<double>(rows * components, -1.0), "y");
  fuse::num::remap_apply<TypeParam>(F::ix(d_rb), F::ix(d_col), F::c(d_val),
                                    F::c(d_x), F::m(d_y), components);

  const auto got = F::download(d_y);
  for (std::size_t j = 0; j < rows * components; ++j) {
    ASSERT_NEAR(got[j], want[j], 1e-13 * (1.0 + std::abs(want[j]))) << j;
  }
}

/// Matrix cell A-SPACES-TRANSPORT: the fixed-chunk
/// scheme `deterministic_dot`/`deterministic_sum`/`deterministic_norm_squared`
/// use so the runtime's own reductions (iteration.cpp's block residual/value
/// norms) stay bit-for-bit independent of the execution space's thread count.
///
/// The reference below is plain host C++ that implements the SAME algorithm
/// space_kernels.hpp documents -- chunk into `kDeterministicChunk`, sum each
/// chunk ascending, then sum the chunk partials ascending -- not
/// `fuse::num::sum`'s single-accumulator oracle, which re-associates
/// differently the moment a buffer spans more than one chunk (that header's
/// own comment on why the two are not the same bits in general). Comparing
/// against THIS reference with EXPECT_EQ (no tolerance) is possible only
/// because it is the identical computation, and it is what actually pins the
/// property this cell needs: a caller cannot get a different answer no
/// matter how the Kokkos runtime happened to partition the chunk loop across
/// threads, because the finalising sum never runs through Kokkos at all.
///
/// ADVERSARIAL DATA: alternating +1e16/1.0/-1e16/1e-16 magnitudes, so that
/// ANY re-association -- a different chunk boundary, a tree instead of a
/// scan, a plain running accumulator instead of chunking at all -- moves the
/// rounding and is caught, rather than the test coincidentally passing
/// because every candidate scheme rounds the same way on well-scaled data.
namespace {
std::vector<double> AdversarialMagnitudes(std::size_t n) {
  std::vector<double> v(n);
  static constexpr double kPattern[4] = {1e16, 1.0, -1e16, 1e-16};
  for (std::size_t i = 0; i < n; ++i) {
    // Not a clean period-4 repeat: a per-element sign/scale wobble on top of
    // the base pattern, so a scheme that only happens to agree with the
    // reference on an exactly-periodic input is not let off the hook.
    const double wobble = 1.0 + 0.125 * std::sin(0.91 * static_cast<double>(i));
    v[i] = kPattern[i % 4] * wobble;
  }
  return v;
}

double ChunkedSerialReference(const std::vector<double>& v,
                              std::size_t chunk) {
  const std::size_t n = v.size();
  const std::size_t n_chunks = (n + chunk - 1) / chunk;
  double total = 0.0;
  for (std::size_t c = 0; c < n_chunks; ++c) {
    const std::size_t begin = c * chunk;
    const std::size_t end = (begin + chunk < n) ? begin + chunk : n;
    double s = 0.0;
    for (std::size_t i = begin; i < end; ++i) s += v[i];
    total += s;
  }
  return total;
}

double ChunkedSerialDotReference(const std::vector<double>& a,
                                 const std::vector<double>& b,
                                 std::size_t chunk) {
  const std::size_t n = a.size();
  const std::size_t n_chunks = (n + chunk - 1) / chunk;
  double total = 0.0;
  for (std::size_t c = 0; c < n_chunks; ++c) {
    const std::size_t begin = c * chunk;
    const std::size_t end = (begin + chunk < n) ? begin + chunk : n;
    double s = 0.0;
    for (std::size_t i = begin; i < end; ++i) s += a[i] * b[i];
    total += s;
  }
  return total;
}

double ChunkedSerialDiffSquaredReference(const std::vector<double>& a,
                                         const std::vector<double>& b,
                                         std::size_t chunk) {
  const std::size_t n = a.size();
  const std::size_t n_chunks = (n + chunk - 1) / chunk;
  double total = 0.0;
  for (std::size_t c = 0; c < n_chunks; ++c) {
    const std::size_t begin = c * chunk;
    const std::size_t end = (begin + chunk < n) ? begin + chunk : n;
    double s = 0.0;
    for (std::size_t i = begin; i < end; ++i) {
      const double d = a[i] - b[i];
      s += d * d;
    }
    total += s;
  }
  return total;
}

// The same forward-error bound as
// test/linalg/dist_vector_test.cpp's forward_error_bound() (Higham,
// "Accuracy and Stability of Numerical Algorithms", 2nd ed., secs.
// 3.1/4.2) -- two IEEE-754 evaluations of sum_i x_i differ by at most
// 2*gamma_n*sum_i|x_i|, gamma_n = n*u/(1-n*u), u = 2^-53.
double forward_error_bound(std::size_t n, double sum_abs_terms) {
  constexpr double u = 0x1p-53;
  const double nu = static_cast<double>(n) * u;
  const double gamma_n = nu / (1.0 - nu);
  return 2.0 * gamma_n * sum_abs_terms;
}

// deterministic_dot (and deterministic_norm_squared/_diff, which call it)
// sum n products in chunks of up to kDeterministicChunk, THEN sum the
// n_chunks chunk totals -- two nested recursive sums, not one flat chain of
// n terms. Any root-to-leaf accumulation path in that computation is at
// most (b-1) + (n_chunks-1) additions long, where b = min(kDeterministicChunk,
// n) is the longest chunk and n_chunks = deterministic_chunks(n) the number
// of chunks. Bounding that by forward_error_bound(b + n_chunks, ...) -- one
// more term than the exact op count needs -- is a conservative, not a
// tuned, choice: it is simpler than deriving the tighter two-level bound
// (gamma_{b-1} + gamma_{n_chunks-1}, valid because each level's own partial
// sums are bounded by the SAME sum_abs_terms via the triangle inequality,
// since a chunk total's magnitude cannot exceed the sum of its terms'
// magnitudes) and gamma's monotonicity makes it safely larger. The result
// is still many orders of magnitude above every difference observed below
// (printed in each failure message as a ratio, which must stay <= 1).
double chunked_forward_error_bound(std::size_t n, double sum_abs_terms) {
  const std::size_t b =
      (n < fuse::num::kDeterministicChunk) ? n : fuse::num::kDeterministicChunk;
  const std::size_t n_chunks = fuse::num::deterministic_chunks(n);
  return forward_error_bound(b + n_chunks, sum_abs_terms);
}

double SumAbsProducts(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += std::fabs(a[i] * b[i]);
  return s;
}

double SumSquaredDiffs(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    s += d * d;
  }
  return s;
}
}  // namespace

// With FUSE_FP_REPRODUCIBLE == 0 the dot/norm_squared/norm_squared_diff
// comparisons below are not bit-for-bit, so the name says "Exactly Or Within
// The Fp Contraction Bound". deterministic_sum's own comparison (no multiply in
// `s += v[i]`, so no FMA for any contraction policy to fuse) and the
// single-chunk-equals-the-plain-oracle check right below it are NOT
// contraction-sensitive and stay exact in both configurations; the name
// covers what the test as a whole asserts in either one.
TYPED_TEST(SpaceKernels,
          DeterministicReductionsMatchTheChunkedReferenceExactlyOrWithinTheFpContractionBound) {
  using F = SpaceKernels<TypeParam>;
  // Sizes deliberately straddling the chunk boundary (256): one well below
  // it (single chunk -- must also equal a plain ascending sum), one that
  // lands exactly on a multiple, and two with a partial last chunk.
  for (const std::size_t n : {std::size_t{1}, std::size_t{37},
                              std::size_t{256}, std::size_t{257},
                              std::size_t{4096}, std::size_t{100'000}}) {
    const auto a = AdversarialMagnitudes(n);
    const auto b = AdversarialMagnitudes(n + 1);  // deliberately different
    std::vector<double> b_trim(b.begin(), b.begin() + static_cast<long>(n));

    auto d_a = F::upload(a, "a");
    auto d_b = F::upload(b_trim, "b");
    const std::size_t n_chunks = fuse::num::deterministic_chunks(n);
    auto d_scratch =
        F::upload(std::vector<double>(n_chunks, 0.0), "scratch");

    const double got_sum =
        fuse::num::deterministic_sum<TypeParam>(F::c(d_a), F::m(d_scratch));
    const double want_sum =
        ChunkedSerialReference(a, fuse::num::kDeterministicChunk);
    EXPECT_EQ(got_sum, want_sum) << "deterministic_sum, n=" << n;

    const double got_dot = fuse::num::deterministic_dot<TypeParam>(
        F::c(d_a), F::c(d_b), F::m(d_scratch));
    const double want_dot =
        ChunkedSerialDotReference(a, b_trim, fuse::num::kDeterministicChunk);
#if FUSE_FP_REPRODUCIBLE
    EXPECT_EQ(got_dot, want_dot) << "deterministic_dot, n=" << n;
#else
    // Both chunked product-sums (got_dot, and the serial reference it
    // is checked against) may each fuse their own a(i)*b(i) accumulation
    // into an FMA, independently -- measured to differ for deterministic_
    // norm_squared below at n=256/257; not observed for this a/b pair at
    // this n, but the SAME algorithmic pattern, so bounded rather than
    // assumed exact.
    const double bound_dot = chunked_forward_error_bound(n, SumAbsProducts(a, b_trim));
    const double diff_dot = std::fabs(got_dot - want_dot);
    const double ratio_dot = (bound_dot > 0.0) ? diff_dot / bound_dot : 0.0;
    std::printf(
        "[ fp_policy ] deterministic_dot n=%zu: observed=%.3e bound=%.3e ratio=%.6f\n", n,
        diff_dot, bound_dot, ratio_dot);
    EXPECT_LE(diff_dot, bound_dot)
        << "deterministic_dot, n=" << n << ": observed=" << diff_dot
        << " bound=" << bound_dot << " ratio=" << ratio_dot;
#endif

    const double got_ns = fuse::num::deterministic_norm_squared<TypeParam>(
        F::c(d_a), F::m(d_scratch));
    const double want_ns =
        ChunkedSerialDotReference(a, a, fuse::num::kDeterministicChunk);
#if FUSE_FP_REPRODUCIBLE
    EXPECT_EQ(got_ns, want_ns) << "deterministic_norm_squared, n=" << n;
#else
    // The measured failure (n=256, n=257): one ulp under the
    // compiler's default contraction, here bounded instead of asserted
    // bit-identical.
    const double bound_ns = chunked_forward_error_bound(n, SumAbsProducts(a, a));
    const double diff_ns = std::fabs(got_ns - want_ns);
    const double ratio_ns = (bound_ns > 0.0) ? diff_ns / bound_ns : 0.0;
    std::printf(
        "[ fp_policy ] deterministic_norm_squared n=%zu: observed=%.3e bound=%.3e ratio=%.6f\n",
        n, diff_ns, bound_ns, ratio_ns);
    EXPECT_LE(diff_ns, bound_ns)
        << "deterministic_norm_squared, n=" << n << ": observed=" << diff_ns
        << " bound=" << bound_ns << " ratio=" << ratio_ns;
#endif

    const double got_nsd =
        fuse::num::deterministic_norm_squared_diff<TypeParam>(
            F::c(d_a), F::c(d_b), F::m(d_scratch));
    const double want_nsd = ChunkedSerialDiffSquaredReference(
        a, b_trim, fuse::num::kDeterministicChunk);
#if FUSE_FP_REPRODUCIBLE
    EXPECT_EQ(got_nsd, want_nsd) << "deterministic_norm_squared_diff, n=" << n;
#else
    // Same pattern as deterministic_dot above: each per-element (a(i)-b(i))^2
    // term is itself a product (of the difference with itself), so the same
    // contraction sensitivity applies to its chunked accumulation.
    const double bound_nsd = chunked_forward_error_bound(n, SumSquaredDiffs(a, b_trim));
    const double diff_nsd = std::fabs(got_nsd - want_nsd);
    const double ratio_nsd = (bound_nsd > 0.0) ? diff_nsd / bound_nsd : 0.0;
    std::printf(
        "[ fp_policy ] deterministic_norm_squared_diff n=%zu: observed=%.3e bound=%.3e "
        "ratio=%.6f\n",
        n, diff_nsd, bound_nsd, ratio_nsd);
    EXPECT_LE(diff_nsd, bound_nsd)
        << "deterministic_norm_squared_diff, n=" << n << ": observed=" << diff_nsd
        << " bound=" << bound_nsd << " ratio=" << ratio_nsd;
#endif

    // n <= kDeterministicChunk: the chunked scheme is a SINGLE chunk, i.e.
    // exactly the plain ascending accumulator -- so it must also agree with
    // fuse::num::sum (kernels.hpp), not just with the chunked reference
    // above (which would trivially agree with itself for any chunk size).
    if (n <= fuse::num::kDeterministicChunk) {
      EXPECT_EQ(got_sum, fuse::num::sum(a.data(), n))
          << "single-chunk deterministic_sum must equal the plain "
          << "ascending oracle, n=" << n;
    }
  }
}

TYPED_TEST(SpaceKernels, TheSpaceUnderTestIsNamedInTheRecord) {
  // So a reader of CI output can see WHICH spaces ran, rather than inferring
  // it from the build flags. A typed test that silently instantiates once is
  // indistinguishable from one that covers everything.
  std::printf("[ space    ] %s\n", TypeParam::name());
  SUCCEED();
}
