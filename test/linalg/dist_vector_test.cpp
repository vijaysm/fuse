// DistVector at tier 0: a one-rank build (no MPI at all) against a plain
// vector, and a "fake" two-partition split of a single GLOBAL vector held
// entirely inside this one process -- two DistVectors, each the size of
// its own slab, wired together by local test doubles of GhostExchanger and
// Reducer that copy/combine directly between them. This is what the task
// means by measuring the PATTERN LOGIC here rather than only on the host:
// MPI cannot start in this sandbox (docs/dev/testing.md), so the real
// MpiExchanger/MpiReducer (src/linalg/comm_mpi.cpp) are built but never run
// here; the in-process fakes below exercise the SAME ghost-pattern and
// reduction contract DistVector itself depends on, with no MPI symbol
// anywhere in this file.

#include "fuse/linalg/dist_vector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/base/config.hpp"
#include "fuse/linalg/comm.hpp"

namespace {

// KNOWN_ISSUES.md D30: the standard forward-error bound for recursive
// summation of n floating-point products (Higham, "Accuracy and Stability
// of Numerical Algorithms", 2nd ed., secs. 3.1/4.2) -- any two IEEE-754
// evaluations of sum_i x_i, contracted (fused multiply-add, one rounding)
// or not (two roundings; an FMA is itself correctly rounded, so it cannot
// make an evaluation WORSE than this bound), differ from the exact sum by
// at most gamma_n * sum_i|x_i|, gamma_n = n*u/(1-n*u), u = 2^-53. Two such
// evaluations therefore differ from EACH OTHER by at most 2*gamma_n*
// sum_i|x_i| (triangle inequality through the exact value). n and the sum
// of magnitudes both come from the code under test below, never adjusted
// to make an observed ratio come out under 1.
double forward_error_bound(std::size_t n, double sum_abs_terms) {
  constexpr double u = 0x1p-53;  // the double-precision unit roundoff, 2^-53
  const double nu = static_cast<double>(n) * u;
  const double gamma_n = nu / (1.0 - nu);
  return 2.0 * gamma_n * sum_abs_terms;
}

using fuse::Result;
using fuse::linalg::DistVector;
using fuse::linalg::GhostExchanger;
using fuse::linalg::GhostPattern;
using fuse::linalg::Reducer;
using fuse::linalg::make_serial_exchanger;
using fuse::linalg::make_serial_reducer;

// ---------------------------------------------------------------------
// Requirement 1: one rank, no ghosts, behaves as a plain vector.
// ---------------------------------------------------------------------

// Renamed from ...BitForBit (KNOWN_ISSUES.md D30): with FUSE_FP_REPRODUCIBLE
// == 0 the dot/norm2 comparisons below are no longer bit-for-bit, so a name
// claiming that would overstate what this test checks in that configuration.
// The new name says what BOTH configurations actually assert.
TEST(DistVectorOneRank, ArithmeticMatchesAPlainLoopExactlyOrWithinTheFpContractionBound) {
  constexpr std::size_t n = 37;
  std::mt19937 rng(11u);
  std::uniform_real_distribution<double> u(-3.0, 3.0);

  std::vector<double> a(n), b(n);
  for (double& v : a) v = u(rng);
  for (double& v : b) v = u(rng);

  DistVector va(n, make_serial_exchanger(), make_serial_reducer());
  DistVector vb(n, make_serial_exchanger(), make_serial_reducer());
  std::copy(a.begin(), a.end(), va.data());
  std::copy(b.begin(), b.end(), vb.data());

  ASSERT_EQ(va.owned_size(), n);
  ASSERT_EQ(va.ghost_size(), 0u);
  ASSERT_EQ(va.size(), n);

  // Same loop order as DistVector::dot/norm_inf (src/linalg/dist_vector.cpp).
  // Under FUSE_FP_REPRODUCIBLE (-ffp-contract=off on both this file and
  // dist_vector.cpp) the comparison below is bit-for-bit; without it, D30
  // measured a one-ulp difference -- Apple clang may fuse `local +=
  // storage_[i]*other.storage_[i]` into a single-rounded FMA in one
  // translation unit and not (identically) in the other. sum_abs_* feeds
  // forward_error_bound() in that case.
  double expect_dot_ab = 0.0;
  double sum_abs_ab = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    expect_dot_ab += a[i] * b[i];
    sum_abs_ab += std::fabs(a[i] * b[i]);
  }
  double expect_dot_aa = 0.0;  // norm2() is sqrt(dot(*this, *this)), a self-dot, not dot(a, b).
  double sum_abs_aa = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    expect_dot_aa += a[i] * a[i];
    sum_abs_aa += std::fabs(a[i] * a[i]);
  }
  double expect_max = 0.0;
  for (std::size_t i = 0; i < n; ++i) expect_max = std::max(expect_max, std::fabs(a[i]));

#if FUSE_FP_REPRODUCIBLE
  EXPECT_EQ(va.dot(vb), expect_dot_ab);
  EXPECT_EQ(va.norm2(), std::sqrt(expect_dot_aa));
#else
  const double bound_ab = forward_error_bound(n, sum_abs_ab);
  const double diff_ab = std::fabs(va.dot(vb) - expect_dot_ab);
  const double ratio_ab = (bound_ab > 0.0) ? diff_ab / bound_ab : 0.0;
  // Printed unconditionally (not only on failure, as EXPECT_LE's own
  // streamed message would be) so the measured ratio is in the record of
  // every Release+OFF run, pass or fail -- docs/dev/testing.md's cost
  // table cites this output, not a number retyped from memory.
  std::printf("[ fp_policy ] dist_vector dot(vb): observed=%.3e bound=%.3e ratio=%.6f\n",
             diff_ab, bound_ab, ratio_ab);
  EXPECT_LE(diff_ab, bound_ab) << "dot(vb): observed=" << diff_ab << " bound=" << bound_ab
                              << " ratio=" << ratio_ab;
  // va.norm2() is sqrt(va.dot(va)) by definition (src/linalg/dist_vector.cpp)
  // -- that holds exactly regardless of contraction, since it is the same
  // function calling itself back, not a second evaluation order. The
  // substantive, contraction-sensitive comparison is dot(va, va) against
  // the independently summed expect_dot_aa, bounded the same way as above;
  // composing it with sqrt's own (negligible, correctly-rounded) error is
  // avoided by comparing the pre-sqrt value instead.
  const double bound_aa = forward_error_bound(n, sum_abs_aa);
  const double diff_aa = std::fabs(va.dot(va) - expect_dot_aa);
  const double ratio_aa = (bound_aa > 0.0) ? diff_aa / bound_aa : 0.0;
  std::printf("[ fp_policy ] dist_vector dot(va,va): observed=%.3e bound=%.3e ratio=%.6f\n",
             diff_aa, bound_aa, ratio_aa);
  EXPECT_LE(diff_aa, bound_aa) << "dot(va): observed=" << diff_aa << " bound=" << bound_aa
                              << " ratio=" << ratio_aa;
  EXPECT_EQ(va.norm2(), std::sqrt(va.dot(va)));
#endif
  EXPECT_EQ(va.norm_inf(), expect_max);

  // axpy, scale, set_zero, copy_from against the same plain-loop reference.
  std::vector<double> ref = a;
  const double alpha = 2.5;
  for (std::size_t i = 0; i < n; ++i) ref[i] += alpha * b[i];
  va.axpy(alpha, vb);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(va.data()[i], ref[i]) << "axpy i=" << i;

  for (double& v : ref) v *= 1.75;
  va.scale(1.75);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(va.data()[i], ref[i]) << "scale i=" << i;

  va.set_zero();
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(va.data()[i], 0.0) << "set_zero i=" << i;

  va.copy_from(vb);
  for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(va.data()[i], b[i]) << "copy_from i=" << i;

  ASSERT_TRUE(va.update_ghosts().ok());  // zero ghosts: a no-op that must still succeed.
}

// ---------------------------------------------------------------------
// Requirement 2: ghost update, measured with an in-process fake
// two-partition split (one process, two DistVectors, one GhostExchanger
// implementation per side that reads directly from its peer).
// ---------------------------------------------------------------------

/// Copies ghost values directly from a peer `DistVector` held in the SAME
/// process, driven by a `GhostPattern` exactly as `MpiExchanger::update`
/// would use one -- the difference is WHERE the owner's value comes from
/// (a pointer into this process's memory instead of a message), not how
/// the pattern is read. `bind_peers` is a second phase because the peer
/// `DistVector` does not exist yet when this exchanger must already be
/// handed to ITS OWN `DistVector`'s constructor.
class FakePeerExchanger : public GhostExchanger {
 public:
  explicit FakePeerExchanger(GhostPattern pattern) : pattern_(std::move(pattern)) {}

  void bind_peers(std::array<const DistVector*, 2> peers) { peers_ = peers; }

  [[nodiscard]] std::size_t ghost_size() const override { return pattern_.owner_rank.size(); }

  [[nodiscard]] Result<void> update(const double* owned, std::size_t owned_n, double* ghosts,
                                    std::size_t ghost_n) const override {
    (void)owned;
    (void)owned_n;
    for (std::size_t g = 0; g < ghost_n; ++g) {
      const auto owner = static_cast<std::size_t>(pattern_.owner_rank[g]);
      const DistVector* peer = peers_[owner];
      ghosts[g] = peer->data()[pattern_.remote_local_index[g]];
    }
    return Result<void>::success();
  }

 private:
  GhostPattern pattern_;
  std::array<const DistVector*, 2> peers_{nullptr, nullptr};
};

TEST(DistVectorTwoPartitionFake, GhostsEqualTheOwnersCurrentValuesAfterUpdate) {
  // A global vector of 10 entries, split [0,4) on partition 0 and [4,10) on
  // partition 1. Each partition's ghost region mirrors the OTHER
  // partition's full owned block, so the test exercises an interleaved
  // ghost order rather than a single contiguous run.
  constexpr std::size_t n0 = 4, n1 = 6;

  GhostPattern pattern0;  // partition 0's ghosts: all of partition 1's owned entries
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n1); ++i) {
    pattern0.owner_rank.push_back(1);
    pattern0.remote_local_index.push_back(i);
  }
  GhostPattern pattern1;  // partition 1's ghosts: all of partition 0's owned entries
  for (std::int64_t i = 0; i < static_cast<std::int64_t>(n0); ++i) {
    pattern1.owner_rank.push_back(0);
    pattern1.remote_local_index.push_back(i);
  }

  auto exch0 = std::make_shared<FakePeerExchanger>(pattern0);
  auto exch1 = std::make_shared<FakePeerExchanger>(pattern1);
  DistVector v0(n0, exch0, make_serial_reducer());
  DistVector v1(n1, exch1, make_serial_reducer());
  exch0->bind_peers({nullptr, &v1});
  exch1->bind_peers({&v0, nullptr});

  for (std::size_t i = 0; i < n0; ++i) v0.data()[i] = 100.0 + static_cast<double>(i);
  for (std::size_t i = 0; i < n1; ++i) v1.data()[i] = 200.0 + static_cast<double>(i);

  ASSERT_EQ(v0.ghost_size(), n1);
  ASSERT_EQ(v1.ghost_size(), n0);
  ASSERT_TRUE(v0.update_ghosts().ok());
  ASSERT_TRUE(v1.update_ghosts().ok());

  for (std::size_t g = 0; g < n1; ++g) {
    EXPECT_EQ(v0.data()[v0.owned_size() + g], v1.data()[pattern0.remote_local_index[g]])
        << "ghost " << g;
  }
  for (std::size_t g = 0; g < n0; ++g) {
    EXPECT_EQ(v1.data()[v1.owned_size() + g], v0.data()[pattern1.remote_local_index[g]])
        << "ghost " << g;
  }

  // The owner mutates after the first update: a second update must see the
  // NEW value, not a value cached from the first call -- update_ghosts is
  // "refreshed on request", not "synchronised once" (dist_vector.hpp).
  v1.data()[2] = -999.0;
  ASSERT_TRUE(v0.update_ghosts().ok());
  bool found = false;
  for (std::size_t g = 0; g < n1; ++g) {
    if (pattern0.remote_local_index[g] == 2) {
      EXPECT_EQ(v0.data()[v0.owned_size() + g], -999.0);
      found = true;
    }
  }
  ASSERT_TRUE(found);
}

// ---------------------------------------------------------------------
// Requirement 5: dot/norm2 reduce across partitions. The fake Reducer
// below bakes in the PEER partition's own local contribution, computed
// directly from the raw arrays the test built (an independent ground
// truth, not a call into the peer's own DistVector) -- so a call to
// dot()/norm_inf() on partition 0 ALONE must still return the true global
// answer, which is exactly what a real collective would do and what the
// "the dot reduction skipped" mutation (build-evidence/mutations/linalg.toml)
// must fail to survive: skipping the sum() call returns only the LOCAL
// partial, which disagrees with the global answer baked in here.
// ---------------------------------------------------------------------

class TwoPartitionReducer : public Reducer {
 public:
  TwoPartitionReducer(int rank, double peer_sum, double peer_max)
      : rank_(rank), peer_sum_(peer_sum), peer_max_(peer_max) {}

  [[nodiscard]] double sum(double local) const override { return local + peer_sum_; }
  void sum(double* inout, std::size_t n) const override {
    for (std::size_t i = 0; i < n; ++i) inout[i] += peer_sum_;
  }
  [[nodiscard]] double max(double local) const override { return std::max(local, peer_max_); }
  [[nodiscard]] int rank() const override { return rank_; }
  [[nodiscard]] int size() const override { return 2; }

  // Added by the "Schur-slab" track (docs/dev/schur-slab-solver.md) when
  // Reducer gained allgather() as a new pure virtual method (comm.hpp,
  // reported there as "the one change to the containers") -- not exercised
  // by this file's own tests (DistVector's dot/norm2/norm_inf never call
  // it), implemented only so this class stays instantiable. Writes this
  // participant's own `send` into its own slot of `recv`; the peer's own
  // slot is left whatever the caller initialised it to, since no peer
  // object exists for this fake to read from.
  void allgather(const double* send, std::size_t n_each, double* recv) const override {
    std::copy(send, send + n_each, recv + static_cast<std::size_t>(rank_) * n_each);
  }

 private:
  int rank_;
  double peer_sum_;
  double peer_max_;
};

TEST(DistVectorTwoPartitionFake, DotAndNorm2EqualTheGatheredSerialReduction) {
  constexpr std::size_t n0 = 4, n1 = 6, n = n0 + n1;
  std::mt19937 rng(23u);
  std::uniform_real_distribution<double> u(-5.0, 5.0);
  std::vector<double> a(n), b(n);
  for (double& v : a) v = u(rng);
  for (double& v : b) v = u(rng);

  // Ground truth, gathered: the global quantities a single-rank computation
  // over the WHOLE vector would produce.
  double expect_dot = 0.0, sum_abs_dot_terms = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    expect_dot += a[i] * b[i];
    sum_abs_dot_terms += std::fabs(a[i] * b[i]);
  }
  double expect_max_a = 0.0;
  for (std::size_t i = 0; i < n; ++i) expect_max_a = std::max(expect_max_a, std::fabs(a[i]));

  // Review 2026-10-11 NL-10: `a0.dot(b0)`/`a1.dot(b1)` below each sum a
  // 2-way split (this partition's own local partial plus the OTHER
  // partition's, handed in through `TwoPartitionReducer`) via one
  // `Reducer::sum`; `expect_dot` above sums all `n` terms in a single
  // ascending-index loop. Floating-point addition is not associative, so
  // the two groupings can disagree in the last bit even on a correct
  // implementation -- a bit-tight `EXPECT_DOUBLE_EQ` would pass today by
  // luck of magnitude (the same fragility class KNOWN_ISSUES.md's D30
  // already records for contraction, not reassociation), not by
  // construction. Bounded by a measured, term-count-scaled tolerance
  // instead, as dist_sparse_matrix_test.cpp's own
  // `check_two_partition_apply` already does for the analogous apply()
  // split.
  const double dot_tol =
      static_cast<double>(n) * std::numeric_limits<double>::epsilon() * sum_abs_dot_terms;

  // Each partition's own local partial, computed the SAME way (independent
  // of DistVector), so it can be handed to the OTHER partition's fake
  // Reducer as "the peer's contribution". local_selfdotN is a's own
  // partial dot with itself -- a DIFFERENT quantity from local_dotN (a
  // dotted with the SEPARATE vector b), needed because norm2() reduces
  // dot(*this, *this), not dot(a, b).
  double local_dot0 = 0.0, local_dot1 = 0.0;
  for (std::size_t i = 0; i < n0; ++i) local_dot0 += a[i] * b[i];
  for (std::size_t i = n0; i < n; ++i) local_dot1 += a[i] * b[i];
  double local_selfdot0 = 0.0, local_selfdot1 = 0.0;
  for (std::size_t i = 0; i < n0; ++i) local_selfdot0 += a[i] * a[i];
  for (std::size_t i = n0; i < n; ++i) local_selfdot1 += a[i] * a[i];
  double local_max0 = 0.0, local_max1 = 0.0;
  for (std::size_t i = 0; i < n0; ++i) local_max0 = std::max(local_max0, std::fabs(a[i]));
  for (std::size_t i = n0; i < n; ++i) local_max1 = std::max(local_max1, std::fabs(a[i]));

  DistVector a0(n0, make_serial_exchanger(), std::make_shared<TwoPartitionReducer>(
                                                 0, local_dot1, local_max1));
  DistVector b0(n0, make_serial_exchanger(), std::make_shared<TwoPartitionReducer>(
                                                 0, local_dot1, local_max1));
  std::copy(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n0), a0.data());
  std::copy(b.begin(), b.begin() + static_cast<std::ptrdiff_t>(n0), b0.data());

  EXPECT_NEAR(a0.dot(b0), expect_dot, dot_tol);

  DistVector a0_self(n0, make_serial_exchanger(), std::make_shared<TwoPartitionReducer>(
                                                      0, local_selfdot1, local_max1));
  std::copy(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n0), a0_self.data());
  double expect_norm2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) expect_norm2 += a[i] * a[i];
  // Every term of expect_norm2 is non-negative, so its own sum is its own
  // magnitude bound -- no separate sum_abs needed (NL-10, as above).
  const double norm2_tol =
      static_cast<double>(n) * std::numeric_limits<double>::epsilon() * expect_norm2;
  EXPECT_NEAR(a0_self.norm2(), std::sqrt(expect_norm2), norm2_tol);

  DistVector a0_max(n0, make_serial_exchanger(), std::make_shared<TwoPartitionReducer>(
                                                     0, local_dot1, local_max1));
  std::copy(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n0), a0_max.data());
  EXPECT_DOUBLE_EQ(a0_max.norm_inf(), expect_max_a);

  // And from partition 1's own side, with the SAME ground truth and the
  // peer contribution flipped -- the global answer does not depend on
  // which side asks.
  DistVector a1(n1, make_serial_exchanger(), std::make_shared<TwoPartitionReducer>(
                                                 1, local_dot0, local_max0));
  DistVector b1(n1, make_serial_exchanger(), std::make_shared<TwoPartitionReducer>(
                                                 1, local_dot0, local_max0));
  std::copy(a.begin() + static_cast<std::ptrdiff_t>(n0), a.end(), a1.data());
  std::copy(b.begin() + static_cast<std::ptrdiff_t>(n0), b.end(), b1.data());
  EXPECT_NEAR(a1.dot(b1), expect_dot, dot_tol);
}

// ---------------------------------------------------------------------
// Reducer::sum(double*, n) and max() directly: neither DistVector method
// above calls the vectorised sum overload (dot/norm2/norm_inf each reduce
// one scalar at a time), so it would otherwise go completely untested.
// ---------------------------------------------------------------------

TEST(LinalgComm, SerialReducerIsTheIdentityOnEveryOperation) {
  auto r = make_serial_reducer();
  EXPECT_EQ(r->rank(), 0);
  EXPECT_EQ(r->size(), 1);
  EXPECT_EQ(r->sum(3.5), 3.5);
  EXPECT_EQ(r->max(-2.0), -2.0);
  std::vector<double> v{1.0, -2.0, 3.0};
  const std::vector<double> before = v;
  r->sum(v.data(), v.size());
  EXPECT_EQ(v, before);  // one participant: the vectorised sum changes nothing.
}

}  // namespace
