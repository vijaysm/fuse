// SubStepClock (integrator.hpp): "a Rational time accumulator with a recorded
// per-window quantum, last step clipped exactly" -- tested for the three properties
// that names: the accumulator stays EXACT (Rational equality, never a tolerance), the
// quantum is derived and recorded (the header's own doc comment on
// kDefaultQuantumCount), and the last step lands on the window end exactly.

#include "fuse/num/integrator.hpp"

#include <cmath>

#include <gtest/gtest.h>

namespace fuse::num {
namespace {

using fuse::Rational;

static_assert(IntegratorValue<double>, "double must satisfy IntegratorValue");
static_assert(std::is_same_v<integrator_scalar_t<double>, double>,
             "the primary integrator_scalar template must be the identity for double");

TEST(SubStepClock, QuantumDividesTheWindowExactly) {
  SubStepClock clk(Rational(0), Rational(1));
  // window_length / quantum == quantum_count, EXACTLY (Rational equality).
  EXPECT_EQ(clk.quantum() * Rational(clk.quantum_count()), Rational(1));
  EXPECT_EQ(clk.remaining_quanta(), clk.quantum_count());
  EXPECT_EQ(clk.now(), Rational(0));
  EXPECT_FALSE(clk.at_window_end());
}

TEST(SubStepClock, ANonTrivialWindowStaysExact) {
  // A window whose length is not a power of two, so a floating accumulation of the
  // same sub-steps would NOT stay exact -- this is exactly the case the
  // exact accumulator exists for.
  SubStepClock clk(Rational(1, 3), Rational(1, 2));
  std::int64_t total_advanced = 0;
  while (!clk.at_window_end()) {
    const std::int64_t n = clk.round_to_quantum(0.001);
    clk.advance(n);
    total_advanced += n;
  }
  EXPECT_EQ(total_advanced, clk.quantum_count());
  EXPECT_EQ(clk.now(), Rational(1, 2));
}

// "Last step clipped exactly": a controller that keeps proposing a
// step larger than what remains in the window is clipped by round_to_quantum, not by a
// separate post-hoc check, and the clock lands EXACTLY on the window end (Rational
// equality) rather than merely close to it.
TEST(SubStepClock, LastStepClipsExactlyToTheWindowEnd) {
  SubStepClock clk(Rational(0), Rational(1, 7));
  std::int64_t steps = 0;
  while (!clk.at_window_end()) {
    // A deliberately oversized proposal every time: round_to_quantum must clip it.
    const std::int64_t n = clk.round_to_quantum(10.0);
    ASSERT_GT(n, 0);
    clk.advance(n);
    ++steps;
    ASSERT_LE(steps, 2) << "an oversized proposal should finish the window in one step "
                           "(plus at most one more from rounding)";
  }
  EXPECT_EQ(clk.now(), Rational(1, 7));
  EXPECT_TRUE(clk.at_window_end());
}

// round_to_quantum ROUNDS (nearest), never truncates only -- a proposal just under one
// quantum rounds UP to one, rather than down to zero (the header's own doc comment: "a
// proposal that rounds to zero would never advance time").
TEST(SubStepClock, RoundToQuantumFloorsAtOneQuantumRatherThanStalling) {
  SubStepClock clk(Rational(0), Rational(1), /*quantum_count=*/1000);
  const double tiny = 0.0001 * clk.quantum().to_double();  // far below half a quantum
  const std::int64_t n = clk.round_to_quantum(tiny);
  EXPECT_EQ(n, 1);
}

TEST(SubStepClock, RoundToQuantumRoundsToNearestNotDown) {
  SubStepClock clk(Rational(0), Rational(1), /*quantum_count=*/1000);
  const double q = clk.quantum().to_double();
  // 7.6 quanta should round to 8, not truncate to 7.
  const std::int64_t n = clk.round_to_quantum(7.6 * q);
  EXPECT_EQ(n, 8);
}

TEST(SubStepClock, RejectsAnInvertedWindowAndANonPositiveQuantumCount) {
  EXPECT_THROW((SubStepClock{Rational(1), Rational(0)}), fuse::InvalidArgument);
  EXPECT_THROW((SubStepClock{Rational(0), Rational(1), 0}), fuse::InvalidArgument);
  EXPECT_THROW((SubStepClock{Rational(0), Rational(1), -5}), fuse::InvalidArgument);
}

TEST(SubStepClock, AdvanceRefusesToOverrunTheWindow) {
  SubStepClock clk(Rational(0), Rational(1), /*quantum_count=*/10);
  EXPECT_THROW(clk.advance(11), fuse::InvariantViolation);
  EXPECT_THROW(clk.advance(0), fuse::InvariantViolation);
  EXPECT_THROW(clk.advance(-1), fuse::InvariantViolation);
}

TEST(SubStepClock, AZeroLengthWindowIsLegalAndImmediatelyAtItsEnd) {
  SubStepClock clk(Rational(2, 5), Rational(2, 5));
  EXPECT_TRUE(clk.at_window_end());
  EXPECT_EQ(clk.remaining_quanta(), 0);
}

}  // namespace
}  // namespace fuse::num
