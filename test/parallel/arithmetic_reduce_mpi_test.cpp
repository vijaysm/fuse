// The ordered cross-rank sum of double-double and quad-double integrals
// (src/runtime/arithmetic_reduce.hpp; plans/2026-10-22-universal-number-types.md,
// stage U4), at 1 to 5 ranks. Each rank contributes integrals built to defeat a
// binary64 sum (terms of 1e6 that cancel across ranks and leave values of 1e-3
// down to 1e-18, which a binary64 sum of the leading limbs cannot hold); the total
// must be
//   * within the type's round-off of the exact sum, computed here with expansions
//     from the contributions every rank publishes through its own Allgather,
//   * the same bits on every rank,
//   * the sum in ascending rank order, bit for bit (the documented order), and
//   * flagged lost on every rank when any one rank reports a failed period.

#include <fuse/base/config.hpp>

#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "../num/exact_expansion.hpp"
#include "fuse_test_main.hpp"
#include "runtime/arithmetic_reduce.hpp"

namespace {

using fuse::num::dd;
using fuse::num::qd;
using fuse::test::Expansion;

std::uint64_t bits(double x) {
  std::uint64_t b;
  std::memcpy(&b, &x, sizeof b);
  return b;
}

int world_rank() { int r = 0; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }
int world_size() { int n = 0; MPI_Comm_size(MPI_COMM_WORLD, &n); return n; }

/// What rank `r` of `n` contributes: before and after integrals as values of `A`.
/// Large terms of alternating sign and small terms with a nonzero lo limb, so no
/// order of binary64 additions keeps the small part.
template <class A>
void contribution(int r, A& before, A& after) {
  const double big = (r % 2 == 0) ? 1.0e6 : -1.0e6;
  const double small = 0.001 * (r + 1) + 1.0 / (3.0 + r);
  const double tiny = 0x1p-60 * (r + 2);
  const double tinier = 0x1p-150 * (r + 3) / 7.0;
  before = A(big) + A(small) + A(tiny) * 0.1 + A(tinier) * (1.0 / 3.0);
  after = A(-big) + A(small * 0.5) + A(tiny) * 0.3 + A(1.0 / 7.0) + A(tinier) * 0.7;
}

template <class A>
std::vector<double> limbs(const A& a) {
  std::vector<double> v(static_cast<std::size_t>(fuse::num::limb_count<A>));
  fuse::num::detail::Limbs<A>::store(v.data(), a);
  return v;
}

template <class A>
Expansion exact_of(const A& a) {
  Expansion e;
  const auto l = limbs(a);
  for (std::size_t i = l.size(); i-- > 0;) e = fuse::test::grow(e, l[i]);
  return e;
}

template <class A>
void expect_ordered_total() {
  const int n = world_size(), r = world_rank();
  constexpr int L = fuse::num::limb_count<A>;
  A before, after;
  contribution(r, before, after);

  std::vector<double> gather;
  A tb = before, ta = after;
  bool lost = true;
  const auto status = fuse::detail::reduce_arithmetic_integrals<A>(MPI_Comm_c2f(MPI_COMM_WORLD), false, gather,
                                                                  7u, tb, ta, lost);
  ASSERT_TRUE(status.ok()) << (status.ok() ? "" : status.error().message);
  EXPECT_FALSE(lost);

  // every rank's own contribution, gathered independently of the code under test
  std::vector<double> mine = limbs(before), mine_after = limbs(after);
  mine.insert(mine.end(), mine_after.begin(), mine_after.end());
  std::vector<double> all(mine.size() * static_cast<std::size_t>(n));
  ASSERT_EQ(MPI_Allgather(mine.data(), static_cast<int>(mine.size()), MPI_DOUBLE, all.data(),
                          static_cast<int>(mine.size()), MPI_DOUBLE, MPI_COMM_WORLD), MPI_SUCCESS);

  // (1) the documented order: ascending rank, in the accumulator type
  A want_before(0.0), want_after(0.0);
  Expansion exact_before, exact_after;
  double mass = 0.0;
  for (int p = 0; p < n; ++p) {
    const double* g = all.data() + static_cast<std::size_t>(p) * 2 * L;
    const A b = fuse::num::detail::Limbs<A>::load(g), a = fuse::num::detail::Limbs<A>::load(g + L);
    want_before = want_before + b;
    want_after = want_after + a;
    exact_before = fuse::test::add(exact_before, exact_of(b));
    exact_after = fuse::test::add(exact_after, exact_of(a));
    mass += std::fabs(b.to_double()) + std::fabs(a.to_double());
  }
  const auto got_b = limbs(tb), got_a = limbs(ta), want_b = limbs(want_before), want_a = limbs(want_after);
  for (int i = 0; i < L; ++i) {
    EXPECT_EQ(bits(got_b[static_cast<std::size_t>(i)]), bits(want_b[static_cast<std::size_t>(i)])) << "before limb " << i;
    EXPECT_EQ(bits(got_a[static_cast<std::size_t>(i)]), bits(want_a[static_cast<std::size_t>(i)])) << "after limb " << i;
  }

  // (2) accurate to the type's round-off against the exact sum, relative to the
  // size of the terms
  const double u = L == 2 ? 0x1p-104 : 0x1p-208;
  const double eb = std::fabs(fuse::test::approx(fuse::test::sub(exact_before, exact_of(tb))));
  const double ea = std::fabs(fuse::test::approx(fuse::test::sub(exact_after, exact_of(ta))));
  EXPECT_LE(eb, 4 * (n + 1) * u * mass);
  EXPECT_LE(ea, 4 * (n + 1) * u * mass);

  // (3) the same bits on every rank
  std::vector<double> mine_total = got_b;
  mine_total.insert(mine_total.end(), got_a.begin(), got_a.end());
  std::vector<double> totals(mine_total.size() * static_cast<std::size_t>(n));
  ASSERT_EQ(MPI_Allgather(mine_total.data(), static_cast<int>(mine_total.size()), MPI_DOUBLE, totals.data(),
                          static_cast<int>(mine_total.size()), MPI_DOUBLE, MPI_COMM_WORLD), MPI_SUCCESS);
  for (int p = 1; p < n; ++p) {
    for (std::size_t k = 0; k < mine_total.size(); ++k) {
      EXPECT_EQ(bits(totals[static_cast<std::size_t>(p) * mine_total.size() + k]), bits(totals[k]))
          << "rank " << p << " differs from rank 0 at limb " << k;
    }
  }
}

template <class A>
void expect_failure_flag() {
  const int n = world_size(), r = world_rank();
  std::vector<double> gather;
  // no rank failed: nobody is told a partner is lost
  {
    A b(1.0), a(2.0);
    bool lost = true;
    ASSERT_TRUE(fuse::detail::reduce_arithmetic_integrals<A>(MPI_Comm_c2f(MPI_COMM_WORLD), false, gather, 1u, b, a, lost).ok());
    EXPECT_FALSE(lost);
    EXPECT_EQ(b.to_double(), static_cast<double>(n));
    EXPECT_EQ(a.to_double(), 2.0 * n);
  }
  // exactly one rank (the last) failed: every rank is told
  {
    A b(1.0), a(2.0);
    bool lost = false;
    ASSERT_TRUE(fuse::detail::reduce_arithmetic_integrals<A>(MPI_Comm_c2f(MPI_COMM_WORLD), r == n - 1, gather, 2u, b, a, lost).ok());
    EXPECT_TRUE(lost) << "rank " << r;
  }
}

TEST(ArithmeticReduce, DoubleDoubleTotalsAreOrderedAccurateAndIdenticalOnEveryRank) {
  expect_ordered_total<dd>();
}

TEST(ArithmeticReduce, QuadDoubleTotalsAreOrderedAccurateAndIdenticalOnEveryRank) {
  expect_ordered_total<qd>();
}

TEST(ArithmeticReduce, AFailedRankIsReportedToEveryRank) {
  expect_failure_flag<dd>();
  expect_failure_flag<qd>();
}

TEST(ArithmeticReduce, TheDataDefeatsABinary64SumOfTheLeadingLimbs) {
  // not a statement about the code under test: it shows the data is adversarial. The
  // binary64 sum of the leading limbs differs from the exact total by far more than
  // the dd round-off, so the accuracy check above has teeth.
  const int n = world_size();
  double plain = 0.0;
  Expansion exact;
  for (int p = 0; p < n; ++p) {
    dd b, a;
    contribution(p, b, a);
    plain += b.hi;
    exact = fuse::test::add(exact, exact_of(b));
  }
  const double err = std::fabs(fuse::test::approx(fuse::test::sub(exact, Expansion{plain})));
  if (n > 1) EXPECT_GT(err, 1e-14);
}

}  // namespace
