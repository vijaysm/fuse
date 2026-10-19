#include <gtest/gtest.h>
#include <fuse/ir/op.hpp>

using fuse::Anchor;
using fuse::assignable_from;
using fuse::Conservation;
using fuse::OpKind;
using fuse::Rational;
using fuse::RegionId;
using fuse::TimeLevel;
using fuse::ValueKind;
using fuse::ValueType;
namespace u = fuse::units;

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
  EXPECT_STREQ(to_cstring(OpKind::WindowAdvance), "WindowAdvance");
  // Review 2026-10-11 IS-8: Scale, Epoch, Map, Reduce, Closure and
  // ExternalDelivery were all declared after this test was last extended and
  // none of the six was ever added here.
  EXPECT_STREQ(to_cstring(OpKind::Scale), "Scale");
  EXPECT_STREQ(to_cstring(OpKind::Epoch), "Epoch");
  EXPECT_STREQ(to_cstring(OpKind::Map), "Map");
  EXPECT_STREQ(to_cstring(OpKind::Reduce), "Reduce");
  EXPECT_STREQ(to_cstring(OpKind::Closure), "Closure");
  EXPECT_STREQ(to_cstring(OpKind::ExternalDelivery), "ExternalDelivery");
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

// Spec Section 4.3: strict by default.
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

// ---- what a failed rank may skip -------------------------------------

// `is_collective` decides what a rank keeps executing after it has failed
// mid-step. Getting one operator wrong here is a hang, not a wrong number, so
// the classification is pinned per operator and the enumeration is exhaustive:
// adding an operator without deciding fails this test rather than defaulting.
TEST(OpKind, CollectiveClassificationIsDeliberate) {
  using fuse::OpKind;
  using fuse::is_collective;

  // Involve the other ranks of the exchange: skipping one leaves partners
  // inside it.
  EXPECT_TRUE(is_collective(OpKind::Transport));
  EXPECT_TRUE(is_collective(OpKind::Epoch));
  // Constrain reduces its two conservation integrals over the base
  // communicator (KNOWN_ISSUES.md D10). Were it local, a rank that failed
  // earlier in the period would skip it and leave every partner inside the
  // Allreduce -- which is why it used to refuse a multi-rank target instead.
  EXPECT_TRUE(is_collective(OpKind::Constrain));

  // Rank-local arithmetic: skipping costs nothing once the period is lost.
  // Subcycling (WindowAdvance) is here too: it runs on the member's own
  // ranks only (docs/dev/coupling-windows.md; the exchange that delivers a
  // partner's waveform to it is the ordinary field-transfer chain, each
  // classified on its own merits). Review 2026-10-11 IS-8: ExternalDelivery
  // is the same shape -- the application-performed transfer that fills it
  // happens entirely outside this instruction stream, so it involves no
  // other rank set of THIS graph's own exchange -- and was never added here.
  // D73: BuildRemapHalo's own REAL geometry exchange
  // (`fuse::execute_remap_halo_instructions()`) is collective, but it
  // never runs through this instruction at all -- it has to happen
  // before `schedule()`, which has already run by the time this
  // instruction reaches the hot loop (see that enumerator's own doc
  // comment, fuse/ir/op.hpp). By the point this classification governs,
  // the instruction is a verified no-op; skipping it after a failure
  // touches no other rank.
  for (const OpKind k : {OpKind::Source, OpKind::Sink, OpKind::Sample,
                         OpKind::Accumulate, OpKind::Interpolate,
                         OpKind::Extrapolate, OpKind::Remap, OpKind::Merge,
                         OpKind::Relax, OpKind::Residual, OpKind::Accelerate,
                         OpKind::Diagnose, OpKind::Scale,
                         OpKind::WindowAdvance, OpKind::ExternalDelivery,
                         OpKind::BuildRemapHalo}) {
    EXPECT_FALSE(is_collective(k)) << fuse::to_cstring(k);
  }

  // Reserved for multiscale and refused by the scheduler. Collective is the
  // safe answer for an operator nobody has written: skipping one that turns
  // out to be collective hangs, executing one that is not merely wastes work.
  for (const OpKind k : {OpKind::Map, OpKind::Reduce, OpKind::Closure}) {
    EXPECT_TRUE(is_collective(k)) << fuse::to_cstring(k);
  }

  // Exhaustive: every enumerator, counted directly rather than as an offset
  // from the second-to-last one (review 2026-10-11 IS-8: `WindowAdvance + 1
  // == 20` silently stopped counting the enumerator declared right after
  // WindowAdvance -- ExternalDelivery, the 21st -- so a NEW, 22nd operator
  // would again escape this gate undetected, the exact "gate silently
  // stopped gating" class R-06/R-51 were fixed for). A new operator makes
  // this fail until it is added to one of the lists above.
  EXPECT_EQ(static_cast<int>(OpKind::ExternalDelivery) + 1, 21)  // 3+15+3
      << "OpKind gained or lost an operator; classify it above";
  // D73: BuildRemapHalo is the 22nd, added to the "false" list above.
  EXPECT_EQ(static_cast<int>(OpKind::BuildRemapHalo) + 1, 22)
      << "OpKind gained or lost an operator; classify it above";
}
