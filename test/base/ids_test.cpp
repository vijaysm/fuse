#include <gtest/gtest.h>
#include <fuse/base/ids.hpp>

#include <unordered_set>

using fuse::ChannelId;
using fuse::ComponentId;

TEST(Id, DefaultIsInvalid) {
  constexpr ComponentId c;
  static_assert(!c.valid());
}

TEST(Id, ConstructedIsValid) {
  constexpr ComponentId c{3};
  static_assert(c.valid());
  static_assert(c.value() == 3u);
}

TEST(Id, ComparesWithinSameTag) {
  EXPECT_EQ(ComponentId{1}, ComponentId{1});
  EXPECT_NE(ComponentId{1}, ComponentId{2});
  EXPECT_LT(ComponentId{1}, ComponentId{2});
}

TEST(Id, IsHashable) {
  std::unordered_set<ComponentId> s;
  s.insert(ComponentId{1});
  s.insert(ComponentId{1});
  s.insert(ComponentId{2});
  EXPECT_EQ(s.size(), 2u);
}

// Distinct tags are distinct types: this must not compile if uncommented.
//   ComponentId c{1}; ChannelId ch = c;
TEST(Id, TagsAreDistinctTypes) {
  static_assert(!std::is_convertible_v<ComponentId, ChannelId>);
}

TEST(Enums, HaveStableNames) {
  EXPECT_STREQ(to_cstring(fuse::Discretization::FV), "FV");
  EXPECT_STREQ(to_cstring(fuse::Conservation::Extensive), "Extensive");
  EXPECT_STREQ(to_cstring(fuse::Intent::Provides), "Provides");
  EXPECT_STREQ(to_cstring(fuse::AccumOp::Mean), "Mean");
  EXPECT_STREQ(to_cstring(fuse::TemporalBasis::DG), "DG");
}
