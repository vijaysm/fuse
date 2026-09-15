#include <gtest/gtest.h>
#include <soma/graph/component.hpp>

using soma::ComponentDecl;
using soma::ComponentId;
using soma::RankSet;
using soma::Rational;
using soma::Window;

TEST(RankSet, ContainsAndOverlaps) {
  const RankSet a{.first = 0, .count = 4};
  const RankSet b{.first = 4, .count = 4};
  const RankSet c{.first = 2, .count = 4};
  EXPECT_TRUE(a.contains(3));
  EXPECT_FALSE(a.contains(4));
  EXPECT_FALSE(a.overlaps(b));   // disjoint: the normal case
  EXPECT_TRUE(a.overlaps(c));
}

TEST(ComponentDecl, DefaultsToUnitIntervalOnRankZero) {
  const ComponentDecl c{.name = "atm"};
  EXPECT_EQ(c.interval, Rational{1});
  EXPECT_EQ(c.ranks.first, 0);
  EXPECT_EQ(c.ranks.count, 1);
  EXPECT_TRUE(c.fields.empty());
}

TEST(ComponentDecl, CarriesItsOwnInterval) {
  const ComponentDecl c{.name = "ocn", .ranks = {.first = 4, .count = 8},
                        .interval = Rational{1, 3}};
  EXPECT_EQ(c.interval, Rational(1, 3));
  EXPECT_EQ(c.ranks.count, 8);
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
