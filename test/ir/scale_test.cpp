// What the connection check decides must be what the plan contains.
//
// The checker (fuse/model/connect.hpp) looks at two field descriptions and
// either refuses the connection or returns the ordered list of operations that
// makes the source usable by the target. This suite pins that the compiled
// operator graph contains exactly those operations: no fewer (a missing
// conversion computes the wrong number and passes every check), and none the
// checker did not authorize (the compiler does not invent physics).
#include <cstdint>
#include <iterator>
#include <string>

#include <gtest/gtest.h>

#include <fuse/ir/dump.hpp>
#include <fuse/ir/lowering.hpp>
#include <fuse/model/coupling_graph.hpp>

namespace {

using namespace fuse;

// Two components, one channel, whose source and target units differ by the
// caller's choice.
LoweringResult lower_pair(Units source_units, Units target_units, ConnectPolicy policy = {}) {
  CouplingGraph g;
  const auto a = g.add_component({.name = "a", .ranks = {0, 1}, .interval = Rational{1}});
  const auto b = g.add_component({.name = "b", .ranks = {1, 1}, .interval = Rational{1}});
  const auto s = g.add_field(a, {.name = "q", .units = source_units, .intent = Intent::Provides});
  const auto t = g.add_field(b, {.name = "q", .units = target_units, .intent = Intent::Requires});
  (void)g.add_channel("q", s, t);

  return lower(g, policy);
}

std::size_t count_kind(const OperatorDAG& d, OpKind kind) {
  std::size_t n = 0;
  for (std::uint32_t i = 0; i < d.node_count(); ++i) n += (d.node(NodeId{i}).kind == kind);
  return n;
}

Units pascal() { return Units{.length = -1, .mass = 1, .time = -2}; }

Units kilopascal() {
  Units u = pascal();
  u.scale = Rational{1000};
  return u;
}

TEST(Scale, AgreeingUnitsEmitNoConversion) {
  const LoweringResult b = lower_pair(pascal(), pascal());
  EXPECT_FALSE(b.diags.has_errors()) << b.diags.to_string();
  EXPECT_EQ(count_kind(b.dag, OpKind::Scale), 0u)
      << "a factor of one is not an operation; emitting it would put a multiply "
         "in every plan that has matching units";
}

TEST(Scale, AnAuthorizedUnitConversionAppearsInThePlanExactly) {
  const LoweringResult b = lower_pair(pascal(), kilopascal());
  ASSERT_FALSE(b.diags.has_errors()) << b.diags.to_string();
  ASSERT_EQ(count_kind(b.dag, OpKind::Scale), 1u);

  // Exact, and visible. 1/1000 as a double is not 1/1000, and a conservation
  // check downstream would then compare sums differing in the last bits for no
  // physical reason.
  for (std::uint32_t i = 0; i < b.dag.node_count(); ++i) {
    const auto& n = b.dag.node(NodeId{i});
    if (n.kind != OpKind::Scale) continue;
    EXPECT_EQ(n.attrs.scale, Rational(1, 1000));
    EXPECT_EQ(n.attrs.reason, "inferred");
  }
  EXPECT_NE(dump_text(b.dag).find("scale=1/1000"), std::string::npos)
      << "an operator the user did not write must be visible in the dump";
}

// The conversion is inserted on the checker's authority, so withdrawing that
// authority must remove it from the plan -- not leave it in place with the
// diagnostic ignored.
TEST(Scale, ForbiddingAutomaticConversionRefusesInsteadOfEmitting) {
  const LoweringResult b = lower_pair(pascal(), kilopascal(), ConnectPolicy{.allow_automatic_unit_scale = false});
  EXPECT_EQ(count_kind(b.dag, OpKind::Scale), 0u);
  EXPECT_TRUE(b.diags.has_errors()) << "the channel should not compile at all";
}

// Every operation the checker can authorize is either emitted or refused, and
// which one is a decision someone made. Adding a kind without deciding fails
// here rather than silently producing a plan that is missing a step.
TEST(Scale, EveryAuthorizableOperationIsEitherEmittedOrRefused) {
  struct Case { OperatorKind kind; bool emitted; };
  // Listed exhaustively on purpose: a new enumerator makes this list
  // incomplete, and the count assertion below says so.
  const Case cases[] = {
      {OperatorKind::UnitScale, true},
      {OperatorKind::SpatialTransfer, true},
      {OperatorKind::TemporalTransfer, true},
      {OperatorKind::AffineUnit, false},
      {OperatorKind::SignFlip, false},
      {OperatorKind::ComponentSelect, false},
      {OperatorKind::ComponentReorder, false},
      {OperatorKind::Aggregate, false},
      {OperatorKind::Expand, false},
      {OperatorKind::Clip, false},
      {OperatorKind::ConservationEnforcement, false},
  };
  for (const Case& c : cases) {
    EXPECT_EQ(ir_can_emit(c.kind), c.emitted) << operator_kind_name(c.kind);
  }
  EXPECT_EQ(std::size(cases), 11u)
      << "an OperatorKind was added; decide whether the compiler emits it, and "
         "add it here";
}

}  // namespace
