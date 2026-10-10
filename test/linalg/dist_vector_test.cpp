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
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/linalg/comm.hpp"

namespace {

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

TEST(DistVectorOneRank, ArithmeticMatchesAPlainLoopBitForBit) {
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

  // Same loop order as DistVector::dot/norm_inf (src/linalg/dist_vector.cpp),
  // so the comparison below is bit-for-bit, not "close enough".
  double expect_dot_ab = 0.0;
  for (std::size_t i = 0; i < n; ++i) expect_dot_ab += a[i] * b[i];
  double expect_dot_aa = 0.0;  // norm2() is sqrt(dot(*this, *this)), a self-dot, not dot(a, b).
  for (std::size_t i = 0; i < n; ++i) expect_dot_aa += a[i] * a[i];
  double expect_max = 0.0;
  for (std::size_t i = 0; i < n; ++i) expect_max = std::max(expect_max, std::fabs(a[i]));

  EXPECT_EQ(va.dot(vb), expect_dot_ab);
  EXPECT_EQ(va.norm2(), std::sqrt(expect_dot_aa));
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
  double expect_dot = 0.0;
  for (std::size_t i = 0; i < n; ++i) expect_dot += a[i] * b[i];
  double expect_max_a = 0.0;
  for (std::size_t i = 0; i < n; ++i) expect_max_a = std::max(expect_max_a, std::fabs(a[i]));

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

  EXPECT_DOUBLE_EQ(a0.dot(b0), expect_dot);

  DistVector a0_self(n0, make_serial_exchanger(), std::make_shared<TwoPartitionReducer>(
                                                      0, local_selfdot1, local_max1));
  std::copy(a.begin(), a.begin() + static_cast<std::ptrdiff_t>(n0), a0_self.data());
  double expect_norm2 = 0.0;
  for (std::size_t i = 0; i < n; ++i) expect_norm2 += a[i] * a[i];
  EXPECT_DOUBLE_EQ(a0_self.norm2(), std::sqrt(expect_norm2));

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
  EXPECT_DOUBLE_EQ(a1.dot(b1), expect_dot);
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
