#include <gtest/gtest.h>
#include <soma/base/time_level.hpp>

using soma::Anchor;
using soma::compatible;
using soma::Rational;
using soma::TimeLevel;

TEST(TimeLevel, DefaultsToCurrentWindowEnd) {
  const TimeLevel t;
  EXPECT_EQ(t.anchor, Anchor::WindowEnd);
  EXPECT_TRUE(t.offset.is_zero());
  EXPECT_EQ(t.iterate, 0);
}

TEST(TimeLevel, LaggedIsNegativeOffset) {
  const TimeLevel t = TimeLevel::lagged(1);
  EXPECT_EQ(t.offset, Rational(-1));
  EXPECT_EQ(t.anchor, Anchor::WindowEnd);
}

TEST(TimeLevel, PreviousIterate) {
  EXPECT_EQ(TimeLevel::previous_iterate().iterate, -1);
}

TEST(TimeLevel, CompatibilityIsExactEquality) {
  const TimeLevel a{Anchor::WindowEnd, Rational{0}, 0};
  const TimeLevel b{Anchor::WindowEnd, Rational{0}, 0};
  const TimeLevel c{Anchor::WindowStart, Rational{0}, 0};
  const TimeLevel d{Anchor::WindowEnd, Rational{1, 2}, 0};
  EXPECT_TRUE(compatible(a, b));
  EXPECT_FALSE(compatible(a, c));
  EXPECT_FALSE(compatible(a, d));
}

TEST(TimeLevel, ToStringShowsAnchorOffsetAndIterate) {
  const TimeLevel t{Anchor::WindowEnd, Rational{1, 2}, -1};
  const std::string s = t.to_string();
  EXPECT_NE(s.find("WindowEnd"), std::string::npos);
  EXPECT_NE(s.find("1/2"), std::string::npos);
  EXPECT_NE(s.find("k-1"), std::string::npos);
}

TEST(TimeLevel, ToStringOmitsZeroOffsetAndCurrentIterate) {
  EXPECT_EQ(TimeLevel::current().to_string(), "[WindowEnd, k]");
}
