#include <gtest/gtest.h>
#include <fuse/base/units.hpp>

using fuse::Units;
namespace u = fuse::units;

TEST(Units, DimensionlessIsIdentity) {
  static_assert(u::kDimensionless.is_dimensionless());
  static_assert(u::kMetre * u::kDimensionless == u::kMetre);
}

TEST(Units, MultiplicationAddsExponents) {
  static_assert(u::kMetre * u::kMetre == Units{.length = 2});
}

TEST(Units, DivisionSubtractsExponents) {
  static_assert(u::kMetre / u::kMetre == u::kDimensionless);
  static_assert(u::kJoule / u::kSecond == u::kWatt);
}

TEST(Units, HeatFluxDensityIsWattPerSquareMetre) {
  static_assert(u::kHeatFluxDensity == u::kWatt / (u::kMetre * u::kMetre));
}

// The check that makes §4.2 worth having.
TEST(Units, TemperatureIsNotHeatFlux) {
  static_assert(!(u::kKelvin == u::kHeatFluxDensity));
}

// Composition chain: metre to the 10th power via repeated multiplication.
// Exercises the int8_t exponent path well within [-128, 127] to confirm the
// narrowing assert does not spuriously fire on ordinary, legitimate use.
TEST(Units, CompositionChainToTenthPowerStaysCorrect) {
  Units m10 = u::kDimensionless;
  for (int i = 0; i < 10; ++i) m10 = m10 * u::kMetre;
  EXPECT_EQ(m10, (Units{.length = 10}));

  // And back down via division.
  Units back = m10;
  for (int i = 0; i < 10; ++i) back = back / u::kMetre;
  EXPECT_EQ(back, u::kDimensionless);
}

TEST(Units, ToStringRoundTripsFamiliarNames) {
  EXPECT_EQ(u::kDimensionless.to_string(), "1");
  EXPECT_EQ(u::kMetre.to_string(), "m");
  EXPECT_EQ(u::kKelvin.to_string(), "K");
  // W = kg·m^2·s^-3
  const std::string w = u::kWatt.to_string();
  EXPECT_NE(w.find("kg"), std::string::npos);
  EXPECT_NE(w.find("m^2"), std::string::npos);
  EXPECT_NE(w.find("s^-3"), std::string::npos);
}

// Transform (S1.2): the upstream `assert` on exponent narrowing is now a
// refusal that fires in every build, including Release. 100 * 100 = 200 does
// not fit int8_t; upstream wrapped it to -56 under NDEBUG.
TEST(Units, ExponentOverflowIsRefusedInEveryBuild) {
  fuse::ErrorPolicy quiet;
  quiet.on_error = fuse::OnError::Throw;
  fuse::set_error_policy(quiet);
  const Units big{.length = 100};
  EXPECT_THROW((void)(big * big), fuse::Refusal);
  const Units low{.length = -100};
  EXPECT_THROW((void)(low * low), fuse::Refusal);   // -200
  EXPECT_THROW((void)(big / low), fuse::Refusal);   // 200
  EXPECT_EQ((Units{.length = 60} * Units{.length = 67}).length, 127);  // boundary is representable
  fuse::set_error_policy(fuse::ErrorPolicy{});
}
