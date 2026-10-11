/// S7.1's residual/linearisation seam: fuse::ImplicitComponent,
/// fuse::LinearizationToken and Component::implicit() themselves --
/// test/model/newton_capability_test.cpp covers the model-level capability
/// refusal (E-NEWTON-CAPABILITY, E-NEWTON-SHAPE); this file covers the
/// runtime interface's own contract.
#include <gtest/gtest.h>
#include <fuse/runtime/component.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

using fuse::Component;
using fuse::Result;
using fuse::StepContext;

namespace {

/// A trivial component that does implement ImplicitComponent: its
/// evaluate_residual() is an affine map of the interface value, chosen so
/// the test can check the exact residual rather than merely "it ran".
/// advance()/evaluate_residual() share no mutable state but the one
/// deliberately-visible field below, so a no-side-effect test has
/// something concrete to compare.
class AffineImplicit final : public Component, public fuse::ImplicitComponent {
 public:
  explicit AffineImplicit(std::string name) : name_(std::move(name)) {}

  [[nodiscard]] std::string name() const override { return name_; }
  Result<void> register_fields(fuse::FieldRegistry&,
                               const fuse::CouplingGraph&) override {
    return Result<void>::success();
  }
  /// advance() DOES mutate state -- call_count -- so the no-side-effect
  /// test below is checking evaluate_residual(), not a component that
  /// never mutates anything at all.
  Result<void> advance(const StepContext&) override {
    ++advance_calls;
    return Result<void>::success();
  }
  [[nodiscard]] std::uint64_t capabilities() const override {
    return fuse::make_capabilities(/*rollback=*/false,
                                   fuse::DeterminismLevel::Bitwise, false) |
           fuse::kCapabilityResidual;
  }

  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return this; }

  /// residual_out[i] = scale_ * interface_in[i] - offset_[i]. Reads only
  /// its arguments and the two constants below, mutates neither: the
  /// pure-function contract evaluate_residual() itself documents.
  Result<void> evaluate_residual(const StepContext&,
                                 std::span<const double> interface_in,
                                 std::span<double> residual_out) override {
    ++residual_evaluations;  // observable on purpose, see the test below
    if (interface_in.size() != residual_out.size()) {
      return fuse::Failure{"E-TEST-SHAPE", "size mismatch"};
    }
    for (std::size_t i = 0; i < interface_in.size(); ++i) {
      residual_out[i] = scale_ * interface_in[i] - offset_;
    }
    return Result<void>::success();
  }

  double scale_ = 2.0;
  double offset_ = 1.0;
  int    advance_calls = 0;
  /// Deliberately NOT part of "component state" for the no-side-effect
  /// test: it is this TEST's own instrumentation, incremented by the
  /// override above to prove the call happened, and excluded from the
  /// snapshot the test compares.
  int    residual_evaluations = 0;

 private:
  std::string name_;
};

/// Implements ImplicitComponent but declares no LinearizationToken --
/// linearization() uses the interface's own default.
class NoTokenImplicit final : public Component, public fuse::ImplicitComponent {
 public:
  [[nodiscard]] std::string name() const override { return "notoken"; }
  Result<void> register_fields(fuse::FieldRegistry&,
                               const fuse::CouplingGraph&) override {
    return Result<void>::success();
  }
  Result<void> advance(const StepContext&) override {
    return Result<void>::success();
  }
  [[nodiscard]] fuse::ImplicitComponent* implicit() override { return this; }
  Result<void> evaluate_residual(const StepContext&,
                                 std::span<const double>,
                                 std::span<double> residual_out) override {
    for (double& v : residual_out) v = 0.0;
    return Result<void>::success();
  }
};

/// Ordinary component, no ImplicitComponent at all -- the common case this
/// seam must not disturb.
class PlainComponent final : public Component {
 public:
  [[nodiscard]] std::string name() const override { return "plain"; }
  Result<void> register_fields(fuse::FieldRegistry&,
                               const fuse::CouplingGraph&) override {
    return Result<void>::success();
  }
  Result<void> advance(const StepContext&) override {
    return Result<void>::success();
  }
};

}  // namespace

TEST(ImplicitComponent, DefaultComponentHasNoImplicitInterface) {
  PlainComponent c;
  EXPECT_EQ(c.implicit(), nullptr);
}

TEST(ImplicitComponent, ComponentImplementingItIsReachedThroughTheQuery) {
  AffineImplicit c("a");
  ASSERT_NE(c.implicit(), nullptr);
  EXPECT_EQ(c.implicit(), static_cast<fuse::ImplicitComponent*>(&c));
}

TEST(ImplicitComponent, CapabilityBitAgreesWithTheQuery) {
  AffineImplicit with_it("a");
  PlainComponent without_it;
  EXPECT_NE(with_it.capabilities() & fuse::kCapabilityResidual, 0u);
  EXPECT_NE(with_it.implicit(), nullptr);
  EXPECT_EQ(without_it.capabilities() & fuse::kCapabilityResidual, 0u);
  EXPECT_EQ(without_it.implicit(), nullptr);
}

TEST(ImplicitComponent, EvaluateResidualComputesTheExpectedValue) {
  AffineImplicit c("a");
  const std::vector<double> in = {1.0, 2.0, 3.0};
  std::vector<double>       out(3, 0.0);
  const fuse::StepContext   ctx{.t_begin = fuse::Rational{0},
                                .t_end = fuse::Rational{1}};
  const auto r = c.implicit()->evaluate_residual(ctx, in, out);
  ASSERT_TRUE(r.ok()) << r.error().message;
  EXPECT_DOUBLE_EQ(out[0], 2.0 * 1.0 - 1.0);
  EXPECT_DOUBLE_EQ(out[1], 2.0 * 2.0 - 1.0);
  EXPECT_DOUBLE_EQ(out[2], 2.0 * 3.0 - 1.0);
}

// The capability-rejection brief's own no-side-effect test: evaluate_residual()
// must leave the component's state bit-identical. "State" here is every
// member evaluate_residual() is contractually forbidden to touch -- scale_,
// offset_ and advance_calls -- read directly rather than through
// save_state()/restore_state(), so the test does not also depend on those
// being implemented correctly.
TEST(ImplicitComponent, EvaluateResidualLeavesComponentStateUnchanged) {
  AffineImplicit c("a");
  c.advance(fuse::StepContext{});  // give advance_calls a nonzero value first
  const double before_scale = c.scale_;
  const double before_offset = c.offset_;
  const int    before_advance_calls = c.advance_calls;

  const std::vector<double> in = {5.0, -3.0, 0.25};
  std::vector<double>       out(3, 0.0);
  const auto r = c.implicit()->evaluate_residual(fuse::StepContext{}, in, out);
  ASSERT_TRUE(r.ok());

  EXPECT_EQ(c.scale_, before_scale);
  EXPECT_EQ(c.offset_, before_offset);
  EXPECT_EQ(c.advance_calls, before_advance_calls)
      << "evaluate_residual() must not advance the component's own state";
}

// Repeated evaluation at the SAME input, including after an intervening
// advance(), must return bit-identical output -- the purity
// evaluate_residual()'s own contract requires (no RNG, no unordered
// reduction, no hidden counter feeding the arithmetic).
TEST(ImplicitComponent, EvaluateResidualIsRepeatable) {
  AffineImplicit c("a");
  const std::vector<double> in = {1.5, -2.5};
  std::vector<double>       out1(2, 0.0), out2(2, 0.0);
  ASSERT_TRUE(
      c.implicit()->evaluate_residual(fuse::StepContext{}, in, out1).ok());
  c.advance(fuse::StepContext{});
  ASSERT_TRUE(
      c.implicit()->evaluate_residual(fuse::StepContext{}, in, out2).ok());
  EXPECT_EQ(out1, out2);
}

TEST(ImplicitComponent, LinearizationDefaultsToNullWhenNotOverridden) {
  NoTokenImplicit c;
  EXPECT_EQ(c.implicit()->linearization(), nullptr);
}

TEST(ImplicitComponent, LinearizationIsIndependentOfResidualCapability) {
  // A component may answer evaluate_residual() without ever linearising it
  // -- kCapabilityLinearization is a SEPARATE bit from kCapabilityResidual,
  // and linearization() == nullptr is a legal, documented answer.
  NoTokenImplicit c;
  EXPECT_EQ(c.capabilities() & fuse::kCapabilityLinearization, 0u);
  EXPECT_EQ(c.implicit()->linearization(), nullptr);
}
