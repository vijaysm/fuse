// KNOWN_ISSUES.md B3: the named bit assignments fuse/model/capabilities.hpp
// gives to the 64-bit space fuse::ComponentSemantics::capabilities,
// fuse::FieldRequirement::required_capabilities and
// fuse::Component::capabilities() all share.

#include "fuse/model/capabilities.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using fuse::DeterminismLevel;

// ---------------------------------------------------------------------------
// Bit layout: each named bit is exactly one bit, and the three do not
// overlap. A future bit added in the wrong place would silently alias an
// existing one; this is what would catch that.
// ---------------------------------------------------------------------------

TEST(Capabilities, RollbackIsBitZero) {
  EXPECT_EQ(fuse::kCapabilityRollback, std::uint64_t{1});
}

TEST(Capabilities, HiddenStateIsBitThree) {
  EXPECT_EQ(fuse::kCapabilityHiddenState, std::uint64_t{1} << 3);
}

// S7.1's residual/linearisation seam (fuse/runtime/component.hpp): the two
// bits it took from what was "bits 4-63 reserved" before this track.
TEST(Capabilities, ResidualIsBitFour) {
  EXPECT_EQ(fuse::kCapabilityResidual, std::uint64_t{1} << 4);
}

TEST(Capabilities, LinearizationIsBitFive) {
  EXPECT_EQ(fuse::kCapabilityLinearization, std::uint64_t{1} << 5);
}

// Restart track (docs/dev/restart.md), Stage B Slice 1: bits 6-7, taken
// from what was "bits 6-63 reserved" before this track -- the same
// pattern ResidualIsBitFour/LinearizationIsBitFive above already pin for
// the S7.1 bits that preceded them.
TEST(Capabilities, RestartFormAIsBitSix) {
  EXPECT_EQ(fuse::kCapabilityRestartFormA, std::uint64_t{1} << 6);
}

TEST(Capabilities, RestartFormBIsBitSeven) {
  EXPECT_EQ(fuse::kCapabilityRestartFormB, std::uint64_t{1} << 7);
}

TEST(Capabilities, DeterminismMaskIsBitsOneAndTwo) {
  EXPECT_EQ(fuse::kCapabilityDeterminismMask, std::uint64_t{0b0110});
  EXPECT_EQ(fuse::kCapabilityDeterminismShift, std::uint64_t{1});
  EXPECT_EQ(fuse::kCapabilityDeterminismBits, std::uint64_t{2});
}

TEST(Capabilities, NamedBitsDoNotOverlap) {
  EXPECT_EQ(fuse::kCapabilityRollback & fuse::kCapabilityDeterminismMask, 0u);
  EXPECT_EQ(fuse::kCapabilityRollback & fuse::kCapabilityHiddenState, 0u);
  EXPECT_EQ(fuse::kCapabilityDeterminismMask & fuse::kCapabilityHiddenState, 0u);
  EXPECT_EQ(fuse::kCapabilityResidual & fuse::kCapabilityRollback, 0u);
  EXPECT_EQ(fuse::kCapabilityResidual & fuse::kCapabilityDeterminismMask, 0u);
  EXPECT_EQ(fuse::kCapabilityResidual & fuse::kCapabilityHiddenState, 0u);
  EXPECT_EQ(fuse::kCapabilityLinearization & fuse::kCapabilityResidual, 0u);
  EXPECT_EQ(fuse::kCapabilityLinearization & fuse::kCapabilityRollback, 0u);
  EXPECT_EQ(fuse::kCapabilityLinearization & fuse::kCapabilityDeterminismMask, 0u);
  EXPECT_EQ(fuse::kCapabilityLinearization & fuse::kCapabilityHiddenState, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormA & fuse::kCapabilityRestartFormB, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormA & fuse::kCapabilityLinearization, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormA & fuse::kCapabilityResidual, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormA & fuse::kCapabilityRollback, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormA & fuse::kCapabilityDeterminismMask, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormA & fuse::kCapabilityHiddenState, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormB & fuse::kCapabilityLinearization, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormB & fuse::kCapabilityResidual, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormB & fuse::kCapabilityRollback, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormB & fuse::kCapabilityDeterminismMask, 0u);
  EXPECT_EQ(fuse::kCapabilityRestartFormB & fuse::kCapabilityHiddenState, 0u);
}

TEST(Capabilities, ReservedMaskIsEverythingElse) {
  const std::uint64_t named = fuse::kCapabilityRollback |
                              fuse::kCapabilityDeterminismMask |
                              fuse::kCapabilityHiddenState |
                              fuse::kCapabilityResidual |
                              fuse::kCapabilityLinearization |
                              fuse::kCapabilityRestartFormA |
                              fuse::kCapabilityRestartFormB;
  EXPECT_EQ(fuse::kCapabilityReservedMask, ~named);
  EXPECT_EQ(fuse::kCapabilityReservedMask & named, 0u);
}

// ---------------------------------------------------------------------------
// encode_capability_determinism() / decode_capability_determinism(): the
// design constraint is an EXACT round trip for every enumerator.
// ---------------------------------------------------------------------------

TEST(Capabilities, DeterminismRoundTripsEveryEnumerator) {
  for (const DeterminismLevel level : {DeterminismLevel::Bitwise,
                                       DeterminismLevel::RunToRun,
                                       DeterminismLevel::Nondeterministic}) {
    const std::uint64_t encoded = fuse::encode_capability_determinism(level);
    // Encoding must not touch any bit outside the 2-bit field.
    EXPECT_EQ(encoded & ~fuse::kCapabilityDeterminismMask, 0u)
        << "level " << static_cast<int>(level);
    EXPECT_EQ(fuse::decode_capability_determinism(encoded), level)
        << "level " << static_cast<int>(level);
  }
}

// Silence is not a claim. capabilities() == 0 -- no override at all -- decodes
// to UNDECLARED, not to Bitwise: Bitwise is the strongest level the enum has
// ("bit-identical ... across the rank counts the coupler may use"), and a
// component that never mentioned determinism has not made it. Every
// enumerator therefore lives at a non-zero raw value. (The first version of
// this file pinned "zero decodes to Bitwise"; changed the same day, before
// anything serialised the field.)
TEST(Capabilities, ZeroDecodesToUndeclaredNotBitwise) {
  EXPECT_FALSE(fuse::decode_capability_determinism(0).has_value());
  for (const DeterminismLevel level : {DeterminismLevel::Bitwise,
                                       DeterminismLevel::RunToRun,
                                       DeterminismLevel::Nondeterministic}) {
    EXPECT_NE(fuse::encode_capability_determinism(level), 0u)
        << "level " << static_cast<int>(level) << " must not encode as silence";
  }
}

TEST(Capabilities, EncodingIsTheEnumeratorsRawValuePlusOneShiftedIntoPlace) {
  EXPECT_EQ(fuse::encode_capability_determinism(DeterminismLevel::Bitwise),
           std::uint64_t{1} << fuse::kCapabilityDeterminismShift);
  EXPECT_EQ(fuse::encode_capability_determinism(DeterminismLevel::RunToRun),
           std::uint64_t{2} << fuse::kCapabilityDeterminismShift);
  EXPECT_EQ(
      fuse::encode_capability_determinism(DeterminismLevel::Nondeterministic),
      std::uint64_t{3} << fuse::kCapabilityDeterminismShift);
}

// Bits outside the determinism field must not leak into the decode.
TEST(Capabilities, DecodeIgnoresBitsOutsideItsField) {
  const std::uint64_t polluted = fuse::encode_capability_determinism(
                                     DeterminismLevel::RunToRun) |
                                 fuse::kCapabilityRollback |
                                 fuse::kCapabilityHiddenState;
  EXPECT_EQ(fuse::decode_capability_determinism(polluted),
           DeterminismLevel::RunToRun);
}

// ---------------------------------------------------------------------------
// make_capabilities(): the three-question convenience.
// ---------------------------------------------------------------------------

TEST(Capabilities, MakeCapabilitiesCombinesAllThreeBits) {
  const std::uint64_t caps = fuse::make_capabilities(
      true, DeterminismLevel::RunToRun, true);
  EXPECT_NE(caps & fuse::kCapabilityRollback, 0u);
  EXPECT_EQ(fuse::decode_capability_determinism(caps),
           DeterminismLevel::RunToRun);
  EXPECT_NE(caps & fuse::kCapabilityHiddenState, 0u);
  EXPECT_EQ(caps & fuse::kCapabilityReservedMask, 0u);
}

TEST(Capabilities, MakeCapabilitiesAlwaysDeclaresItsLevel) {
  // Even with both flags off, make_capabilities() has been TOLD a level, so
  // the result is never the undeclared zero.
  const std::uint64_t caps =
      fuse::make_capabilities(false, DeterminismLevel::Bitwise, false);
  EXPECT_NE(caps, 0u);
  EXPECT_EQ(fuse::decode_capability_determinism(caps), DeterminismLevel::Bitwise);
  EXPECT_EQ(caps & ~fuse::kCapabilityDeterminismMask, 0u);
}
