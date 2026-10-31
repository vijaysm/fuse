// A deliberate refusal counts as verified only when a test makes the refusal
// fire (docs/dev/testing.md, the five states). These are those tests: every
// declared-but-unimplemented entry point is called and asserted to refuse with
// its documented code. If one of them is ever implemented, its test here fails,
// which is the intended alarm -- the implementer has to come and say so.
#include "fuse/model/multiscale.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace fuse;

SurrogateContract complete_contract() {
  SurrogateContract c;
  c.name = "closure-emulator";
  FieldSemantics input;
  input.id = "strain";
  input.units = Units{};
  c.applicability = {input};
  c.ood_detector = "mahalanobis-on-strain-invariants";
  c.determinism = DeterminismLevel::Bitwise;
  c.derivative = DerivativeKind::Forward;
  return c;
}

TEST(Multiscale, EveryMultiscaleEntryPointRefuses) {
  const auto instance = declare_instance_set({.name = "rve", .component = ComponentId{0}, .count = 64});
  ASSERT_FALSE(instance.ok());
  EXPECT_EQ(instance.error().code, "E-MULTISCALE-NOT-IMPLEMENTED");

  const auto closure = declare_closure({.name = "mean-stress", .kind = ClosureKind::WeightedMean});
  ASSERT_FALSE(closure.ok());
  EXPECT_EQ(closure.error().code, "E-MULTISCALE-NOT-IMPLEMENTED");

  const auto pool = declare_sample_pool({.name = "rve-samples", .capacity = 1024});
  ASSERT_FALSE(pool.ok());
  EXPECT_EQ(pool.error().code, "E-MULTISCALE-NOT-IMPLEMENTED");
}

// Coverage that sums exactly to one is the case a permissive stub would let
// through. It must refuse too, or the accounting runs only where it is already
// right.
TEST(Multiscale, OverlapCheckRefusesEvenForValidCoverage) {
  const OverlapSpec exact{.coverage = {Rational{1, 4}, Rational{1, 4}, Rational{1, 2}}};
  EXPECT_EQ(exact.total(), Rational{1});

  const auto checked = check_overlap(exact);
  ASSERT_FALSE(checked.ok());
  EXPECT_EQ(checked.error().code, "E-MULTISCALE-NOT-IMPLEMENTED");
  // The message reports what it saw, so the refusal is still informative.
  EXPECT_NE(checked.error().message.find("summing to 1"), std::string::npos);

  const auto under = check_overlap({.coverage = {Rational{1, 3}, Rational{1, 3}}});
  ASSERT_FALSE(under.ok());
  EXPECT_EQ(under.error().code, "E-MULTISCALE-NOT-IMPLEMENTED");
}

// `total()` refuses an unrepresentable sum explicitly, by name, at its own
// call site via `try_add`, rather than from three stack frames down inside
// Rational::operator+= with a message naming an addition rather than this
// accumulation. The code is E-BASE-RATIONAL-OVERFLOW.
TEST(Multiscale, OverlapTotalRefusesAnUnrepresentableSum) {
  constexpr std::int64_t kMax = std::numeric_limits<std::int64_t>::max();
  const OverlapSpec huge{.coverage = {Rational{kMax}, Rational{kMax}}};
  try {
    (void)huge.total();
    FAIL() << "expected OverlapSpec::total() to refuse an unrepresentable sum";
  } catch (const fuse::Refusal& e) {
    EXPECT_EQ(e.code(), "E-BASE-RATIONAL-OVERFLOW");
  }
}

TEST(Multiscale, SurrogateSubstitutionRefusesWithACompleteContract) {
  const auto substituted = substitute_surrogate(ComponentId{0}, complete_contract());
  ASSERT_FALSE(substituted.ok());
  EXPECT_EQ(substituted.error().code, "E-SURROGATE-NOT-IMPLEMENTED");
}

// A contract that states nothing gets the contract error, not the
// not-implemented one: the user has something to fix, and telling them only
// that substitution is unavailable would send them away to fix nothing.
TEST(Multiscale, AnIncompleteContractIsReportedBeforeTheUnimplementedRefusal) {
  const auto substituted = substitute_surrogate(ComponentId{0}, SurrogateContract{});
  ASSERT_FALSE(substituted.ok());
  EXPECT_EQ(substituted.error().code, "E-SURROGATE-CONTRACT");
}

// The reviewer is real, not a stub, so it is tested as behaviour: every failure
// at once, and the two "usable but constrained" cases as warnings rather
// than errors.
TEST(Multiscale, ContractReviewReportsEveryGapAtOnce) {
  const Diagnostics empty = review_surrogate_contract(SurrogateContract{});
  EXPECT_TRUE(empty.has_errors());
  std::vector<std::string> codes;
  for (const auto& item : empty.items()) codes.push_back(item.code);
  EXPECT_NE(std::find(codes.begin(), codes.end(), "E-SURROGATE-UNNAMED"), codes.end());
  EXPECT_NE(std::find(codes.begin(), codes.end(), "E-SURROGATE-NO-APPLICABILITY"), codes.end());
  EXPECT_NE(std::find(codes.begin(), codes.end(), "E-SURROGATE-NO-OOD"), codes.end());
  EXPECT_NE(std::find(codes.begin(), codes.end(), "W-SURROGATE-NONDETERMINISTIC"), codes.end());

  EXPECT_FALSE(review_surrogate_contract(complete_contract()).has_errors());
}

TEST(Multiscale, HiddenStateAndNondeterminismAreWarningsNotErrors) {
  auto hidden = complete_contract();
  hidden.carries_hidden_state = true;
  const Diagnostics d = review_surrogate_contract(hidden);
  EXPECT_FALSE(d.has_errors()) << "hidden state is usable outside an iteration region";
  bool warned = false;
  for (const auto& item : d.items())
    warned = warned || (item.code == "W-SURROGATE-HIDDEN-STATE" && item.severity == Severity::Warning);
  EXPECT_TRUE(warned);
}

}  // namespace
