#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <fuse/base/error.hpp>
#include <fuse/model/component.hpp>

using fuse::ComponentDecl;
using fuse::ComponentId;
using fuse::RankSet;
using fuse::Rational;
using fuse::Window;

TEST(RankSet, ContainsAndOverlaps) {
  const RankSet a{0, 4};
  const RankSet b{4, 4};
  const RankSet c{2, 4};
  EXPECT_TRUE(a.contains(3));
  EXPECT_FALSE(a.contains(4));
  EXPECT_FALSE(a.overlaps(b));   // disjoint: the normal case
  EXPECT_TRUE(a.overlaps(c));
}

TEST(ComponentDecl, DefaultsToUnitIntervalOnRankZero) {
  const ComponentDecl c{.name = "atm"};
  EXPECT_EQ(c.interval, Rational{1});
  EXPECT_EQ(c.ranks.min_rank(), 0);
  EXPECT_EQ(c.ranks.size(), 1);
  EXPECT_TRUE(c.fields.empty());
}

TEST(ComponentDecl, CarriesItsOwnInterval) {
  const ComponentDecl c{.name = "ocn", .ranks = RankSet{4, 8},
                        .interval = Rational{1, 3}};
  EXPECT_EQ(c.interval, Rational(1, 3));
  EXPECT_EQ(c.ranks.size(), 8);
}

TEST(Window, ThreeKinds) {
  EXPECT_EQ(Window::coupling_period().kind, Window::Kind::CouplingPeriod);
  const Window w = Window::from(ComponentId{2});
  EXPECT_EQ(w.kind, Window::Kind::FromComponent);
  EXPECT_EQ(w.component, ComponentId{2});
  const Window e = Window::explicit_length(Rational{1, 4});
  EXPECT_EQ(e.kind, Window::Kind::Explicit);
  EXPECT_EQ(e.length, Rational(1, 4));
}

// ---- placements a range cannot express --------------------------------

TEST(RankSet, HoldsAStridedPlacement) {
  // One rank per socket, or every other core: a real layout, and not a block.
  const fuse::RankSet even{std::vector<int>{0, 2, 4, 6}};
  EXPECT_EQ(even.size(), 4u);
  EXPECT_FALSE(even.is_contiguous());
  EXPECT_TRUE(even.contains(4));
  EXPECT_FALSE(even.contains(3));
  EXPECT_EQ(even.min_rank(), 0);
  EXPECT_EQ(even.max_rank(), 6);
  // Position, not rank number, is what indexes per-rank data in a strided set.
  EXPECT_EQ(even.index_of(4), 2);
  EXPECT_EQ(even.index_of(3), -1);
}

TEST(RankSet, SortsAndDeduplicatesWhatItIsGiven) {
  // A launcher hands over ranks in its own order; every consumer depends on
  // sorted order (position-indexed matching, byte-comparable descriptors), so
  // the invariant is established here rather than asked of the caller.
  const fuse::RankSet messy{std::vector<int>{6, 0, 4, 0, 2, 6}};
  EXPECT_EQ(messy.ranks(), (std::vector<int>{0, 2, 4, 6}));
}

TEST(RankSet, DistinguishesSharedFromIdenticalFromDisjoint) {
  const fuse::RankSet a{0, 4};                       // 0,1,2,3
  const fuse::RankSet same{0, 4};
  const fuse::RankSet partial{std::vector<int>{2, 3, 4, 5}};
  const fuse::RankSet away{8, 2};

  // Identical: a channel between these needs no data move at all.
  EXPECT_TRUE(a.same_as(same));
  EXPECT_TRUE(a.overlaps(same));

  // Partially shared: needs a move for 0,1 -> 4,5 and a local copy on 2,3.
  EXPECT_FALSE(a.same_as(partial));
  EXPECT_TRUE(a.overlaps(partial));
  EXPECT_EQ(rank_intersection(a, partial).ranks(), (std::vector<int>{2, 3}));
  EXPECT_EQ(rank_union(a, partial).ranks(), (std::vector<int>{0, 1, 2, 3, 4, 5}));

  // Disjoint: the case that already worked.
  EXPECT_FALSE(a.overlaps(away));
  EXPECT_TRUE(rank_intersection(a, away).empty());
}

TEST(RankSet, AnEmptyPlacementIsLegal) {
  // A component declared but placed nowhere in this run. Not an error: the
  // graph is the same on every rank, so a component another rank hosts still
  // appears here.
  const fuse::RankSet none{0, 0};
  EXPECT_TRUE(none.empty());
  EXPECT_EQ(none.min_rank(), -1);
  EXPECT_FALSE(none.contains(0));
  EXPECT_TRUE(none.is_contiguous()) << "vacuously, and callers rely on it";
}

// ---- UB/OOM on an edge-input count -------------------------------------
//
// `RankSet(int, int)` must not `reserve(count)` and loop `count` times with
// no bound at all: a YAML document with `ranks: {first: 0, count:
// 2000000000}` would reserve ~16 GB before anything looked at it, and
// `first + count` overflowing `int` would be signed-overflow UB in the loop.
// Both are refused before either happens.

TEST(RankSet, RefusesAnAbsurdlyLargeCount) {
  EXPECT_THROW((void)RankSet(0, 1 << 30), fuse::Refusal);
}

TEST(RankSet, RefusesFirstPlusCountOverflow) {
  EXPECT_THROW((void)RankSet(std::numeric_limits<int>::max() - 10, 20),
              fuse::Refusal);
}

TEST(RankSet, RefusalCarriesTheDocumentedCode) {
  try {
    (void)RankSet(0, 1 << 30);
    FAIL() << "expected RankSet to refuse an absurd count";
  } catch (const fuse::Refusal& e) {
    EXPECT_EQ(e.code(), "E-RANKSET-RANGE");
  }
}

TEST(RankSet, AnOrdinaryLargeCountStillWorks) {
  // The refusal bound is far above any real placement; a merely large one
  // (every rank of a big job) must still construct normally.
  const RankSet big{0, 1 << 16};
  EXPECT_EQ(big.size(), std::size_t{1} << 16);
  EXPECT_EQ(big.min_rank(), 0);
  EXPECT_EQ(big.max_rank(), (1 << 16) - 1);
}
