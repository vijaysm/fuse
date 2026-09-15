#include <gtest/gtest.h>
#include <soma/ir/op.hpp>

using soma::Anchor;
using soma::assignable_from;
using soma::Conservation;
using soma::OpKind;
using soma::Rational;
using soma::RegionId;
using soma::TimeLevel;
using soma::ValueKind;
using soma::ValueType;
namespace u = soma::units;

TEST(OpKind, EveryKindHasAName) {
  EXPECT_STREQ(to_cstring(OpKind::Sample), "Sample");
  EXPECT_STREQ(to_cstring(OpKind::Accumulate), "Accumulate");
  EXPECT_STREQ(to_cstring(OpKind::Interpolate), "Interpolate");
  EXPECT_STREQ(to_cstring(OpKind::Extrapolate), "Extrapolate");
  EXPECT_STREQ(to_cstring(OpKind::Remap), "Remap");
  EXPECT_STREQ(to_cstring(OpKind::Merge), "Merge");
  EXPECT_STREQ(to_cstring(OpKind::Relax), "Relax");
  EXPECT_STREQ(to_cstring(OpKind::Transport), "Transport");
  EXPECT_STREQ(to_cstring(OpKind::Residual), "Residual");
  EXPECT_STREQ(to_cstring(OpKind::Accelerate), "Accelerate");
  EXPECT_STREQ(to_cstring(OpKind::Constrain), "Constrain");
  EXPECT_STREQ(to_cstring(OpKind::Diagnose), "Diagnose");
  EXPECT_STREQ(to_cstring(OpKind::Source), "Source");
  EXPECT_STREQ(to_cstring(OpKind::Sink), "Sink");
}

TEST(ValueType, DefaultsToInstantScalarDimensionless) {
  const ValueType v;
  EXPECT_EQ(v.kind, ValueKind::Instant);
  EXPECT_EQ(v.components, 1);
  EXPECT_TRUE(v.units.is_dimensionless());
  EXPECT_EQ(v.rank_set, -1);
}

TEST(ValueType, AssignableWhenIdentical) {
  const ValueType a{.region = RegionId{0}, .units = u::kKelvin};
  EXPECT_TRUE(assignable_from(a, a, /*allow_time_mismatch=*/false));
}

TEST(ValueType, NotAssignableAcrossUnits) {
  const ValueType a{.region = RegionId{0}, .units = u::kKelvin};
  const ValueType b{.region = RegionId{0}, .units = u::kHeatFluxDensity};
  EXPECT_FALSE(assignable_from(a, b, false));
}

TEST(ValueType, NotAssignableAcrossRegions) {
  const ValueType a{.region = RegionId{0}};
  const ValueType b{.region = RegionId{1}};
  EXPECT_FALSE(assignable_from(a, b, false));
}

TEST(ValueType, NotAssignableAcrossComponentCounts) {
  const ValueType a{.region = RegionId{0}, .components = 1};
  const ValueType b{.region = RegionId{0}, .components = 3};
  EXPECT_FALSE(assignable_from(a, b, false));
}

// Spec §4.3: strict by default.
TEST(ValueType, TimeLevelMismatchBlocksAssignmentByDefault) {
  ValueType a{.region = RegionId{0}};
  ValueType b{.region = RegionId{0}};
  a.time = TimeLevel{Anchor::WindowEnd, Rational{0}, 0};
  b.time = TimeLevel{Anchor::WindowEnd, Rational{1, 2}, 0};
  EXPECT_FALSE(assignable_from(a, b, false));
}

// ... but the channel-level escape hatch relaxes exactly this one check.
TEST(ValueType, TimeLevelMismatchPermittedWhenAuthorised) {
  ValueType a{.region = RegionId{0}};
  ValueType b{.region = RegionId{0}};
  a.time = TimeLevel{Anchor::WindowEnd, Rational{0}, 0};
  b.time = TimeLevel{Anchor::WindowEnd, Rational{1, 2}, 0};
  EXPECT_TRUE(assignable_from(a, b, /*allow_time_mismatch=*/true));
}

// The escape hatch must not launder a unit error.
TEST(ValueType, TimeEscapeHatchDoesNotExcuseUnitMismatch) {
  ValueType a{.region = RegionId{0}, .units = u::kKelvin};
  ValueType b{.region = RegionId{0}, .units = u::kWatt};
  b.time = TimeLevel{Anchor::WindowEnd, Rational{1, 2}, 0};
  EXPECT_FALSE(assignable_from(a, b, true));
}

TEST(ValueType, ToStringMentionsRegionUnitsAndTime) {
  ValueType v{.region = RegionId{7}, .components = 3, .units = u::kKelvin};
  v.time = TimeLevel::lagged(1);
  const std::string s = v.to_string();
  EXPECT_NE(s.find("r7"), std::string::npos);
  EXPECT_NE(s.find("K"), std::string::npos);
  EXPECT_NE(s.find("WindowEnd"), std::string::npos);
}
