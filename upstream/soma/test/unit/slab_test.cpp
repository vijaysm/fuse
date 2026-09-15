#include <gtest/gtest.h>
#include <soma/schedule/slab.hpp>

#include <cstddef>

using soma::decompose;
using soma::kMaxSlabsPerPeriod;
using soma::Rational;
using soma::Slab;

TEST(Slab, EqualIntervalsGiveOneDegenerateSlab) {
  const auto r = decompose(Rational{1}, Rational{1}, Rational{1});
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_TRUE(r.value().degenerate);
  ASSERT_EQ(r.value().slabs.size(), 1u);
  EXPECT_EQ(r.value().slabs[0].begin, Rational{0});
  EXPECT_EQ(r.value().slabs[0].end, Rational{1});
}

// degenerate means "the two partitions coincide", not "there is exactly one
// slab": with equal sub-period intervals every slab is a whole step of both
// sides, which is precisely the lockstep fast path of spec §8.2.
TEST(Slab, EqualSubPeriodIntervalsAreStillDegenerate) {
  const auto r = decompose(Rational{1, 2}, Rational{1, 2}, Rational{1});
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_TRUE(r.value().degenerate);
  ASSERT_EQ(r.value().slabs.size(), 2u);
  for (const Slab& s : r.value().slabs) {
    EXPECT_EQ(s.src_step, s.dst_step);
  }
}

TEST(Slab, IntegerSubcyclingGivesNSlabs) {
  // Source ticks 4x per target tick.
  const auto r = decompose(Rational{1, 4}, Rational{1}, Rational{1});
  ASSERT_TRUE(r.ok());
  EXPECT_FALSE(r.value().degenerate);
  ASSERT_EQ(r.value().slabs.size(), 4u);
  EXPECT_EQ(r.value().slabs[0].end, Rational(1, 4));
  EXPECT_EQ(r.value().slabs[3].end, Rational{1});
}

// The case that motivates exact rational time.
TEST(Slab, NonNestingIntervalsIntersectCorrectly) {
  // 1/2 and 1/3 over a period of 1 -> ticks at 0, 1/3, 1/2, 2/3, 1
  const auto r = decompose(Rational{1, 2}, Rational{1, 3}, Rational{1});
  ASSERT_TRUE(r.ok());
  const auto& s = r.value().slabs;
  ASSERT_EQ(s.size(), 4u);
  EXPECT_EQ(s[0].begin, Rational{0});
  EXPECT_EQ(s[0].end, Rational(1, 3));
  EXPECT_EQ(s[1].end, Rational(1, 2));
  EXPECT_EQ(s[2].end, Rational(2, 3));
  EXPECT_EQ(s[3].end, Rational{1});
}

TEST(Slab, SlabsTileThePeriodWithoutGapOrOverlap) {
  const auto r = decompose(Rational{1, 3}, Rational{1, 5}, Rational{1});
  ASSERT_TRUE(r.ok());
  const auto& s = r.value().slabs;
  ASSERT_FALSE(s.empty());
  EXPECT_EQ(s.front().begin, Rational{0});
  EXPECT_EQ(s.back().end, Rational{1});
  for (std::size_t i = 1; i < s.size(); ++i) {
    EXPECT_EQ(s[i - 1].end, s[i].begin) << "gap or overlap at slab " << i;
  }
  Rational total{0};
  for (const Slab& x : s) total += x.length();
  EXPECT_EQ(total, Rational{1});
}

TEST(Slab, RecordsSourceAndTargetStepPerSlab) {
  // src 1/2, dst 1: slab [0,1/2) is src step 0, slab [1/2,1) is src step 1;
  // both are dst step 0.
  const auto r = decompose(Rational{1, 2}, Rational{1}, Rational{1});
  ASSERT_TRUE(r.ok());
  const auto& s = r.value().slabs;
  ASSERT_EQ(s.size(), 2u);
  EXPECT_EQ(s[0].src_step, 0);
  EXPECT_EQ(s[1].src_step, 1);
  EXPECT_EQ(s[0].dst_step, 0);
  EXPECT_EQ(s[1].dst_step, 0);
}

// Every slab lies wholly inside one step of each side: the step index recorded
// for the slab start must also be the one containing an interior point.
TEST(Slab, StepIndicesContainTheWholeSlab) {
  const Rational src{1, 3};
  const Rational dst{1, 5};
  const auto r = decompose(src, dst, Rational{1});
  ASSERT_TRUE(r.ok());
  for (const Slab& s : r.value().slabs) {
    EXPECT_GE(s.begin, Rational{s.src_step} * src);
    EXPECT_LE(s.end, Rational{s.src_step + 1} * src);
    EXPECT_GE(s.begin, Rational{s.dst_step} * dst);
    EXPECT_LE(s.end, Rational{s.dst_step + 1} * dst);
  }
}

TEST(Slab, EveryLengthIsPositive) {
  const auto r = decompose(Rational{1, 6}, Rational{1, 4}, Rational{1});
  ASSERT_TRUE(r.ok());
  for (const Slab& s : r.value().slabs) {
    EXPECT_GT(s.length(), Rational{0});
  }
}

TEST(Slab, RejectsExplosivePeriod) {
  // Coprime large denominators would need ~10^12 slabs.
  const auto r =
      decompose(Rational{1, 999983}, Rational{1, 999979}, Rational{1});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-PERIOD");
}

// The per-side tick limit and the merged-boundary limit must agree: a
// decomposition of exactly kMaxSlabsPerPeriod slabs is legal, one more is not.
// Counting boundaries rather than slabs would reject the legal case.
TEST(Slab, AcceptsExactlyTheSlabLimitAndRejectsOneMore) {
  const auto ok = decompose(Rational{1, kMaxSlabsPerPeriod}, Rational{1},
                            Rational{1});
  ASSERT_TRUE(ok.ok()) << ok.error().message;
  EXPECT_EQ(static_cast<std::int64_t>(ok.value().slabs.size()),
            kMaxSlabsPerPeriod);

  const auto bad = decompose(Rational{1, kMaxSlabsPerPeriod + 1}, Rational{1},
                             Rational{1});
  ASSERT_FALSE(bad.ok());
  EXPECT_EQ(bad.error().code, "E-SCHED-PERIOD");
}

TEST(Slab, RejectsNonPositiveInterval) {
  EXPECT_FALSE(decompose(Rational{0}, Rational{1}, Rational{1}).ok());
  EXPECT_FALSE(decompose(Rational{1}, Rational{-1}, Rational{1}).ok());
}

TEST(Slab, RejectsNonPositivePeriod) {
  const auto zero = decompose(Rational{1}, Rational{1}, Rational{0});
  ASSERT_FALSE(zero.ok());
  EXPECT_EQ(zero.error().code, "E-SCHED-PERIOD");
  EXPECT_FALSE(decompose(Rational{1}, Rational{1}, Rational{-1}).ok());
}

TEST(Slab, RejectsPeriodThatIsNotAMultipleOfAnInterval) {
  const auto r = decompose(Rational{2, 3}, Rational{1}, Rational{1});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-PERIOD");
}

// period/interval here is 2^64-4, too large for an int64_t Rational. It must
// come back as an error: decompose() returns a Result, so Rational's overflow
// exception must not escape it.
TEST(Slab, ReportsUnrepresentableTickCountAsAnError) {
  const auto r = decompose(Rational{1, 4}, Rational{1},
                           Rational{4611686018427387903LL});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-PERIOD");
}

// Same contract one level down: here the tick *count* is fine (4 ticks) but
// tick 3 is 3*P/4 with P = 8999999999999999999, whose numerator exceeds 2^63.
// Rational would throw; decompose() must report instead.
TEST(Slab, ReportsUnrepresentableTickValueAsAnError) {
  constexpr std::int64_t kP = 8999999999999999999LL;  // odd, so gcd(kP, 4) == 1
  const auto r = decompose(Rational{kP, 4}, Rational{kP}, Rational{kP});
  ASSERT_FALSE(r.ok());
  EXPECT_EQ(r.error().code, "E-SCHED-PERIOD");
}

TEST(Slab, EqualityIsMemberwise) {
  const Slab a{Rational{0}, Rational{1, 2}, 0, 0};
  const Slab b{Rational{0}, Rational{1, 2}, 0, 0};
  const Slab c{Rational{0}, Rational{1, 2}, 1, 0};
  EXPECT_EQ(a, b);
  EXPECT_NE(a, c);
  EXPECT_EQ(a.length(), Rational(1, 2));
}
