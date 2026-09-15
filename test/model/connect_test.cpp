// The bridge between the graph's FieldSpec and the semantic model the
// compatibility stages evaluate. Its job is to state what the graph declared and
// NOTHING more: an axis the graph cannot express must stay Unspecified, because
// a defaulted axis passes a check the user never made.
#include "fuse/model/connect.hpp"

#include <gtest/gtest.h>

namespace {

using namespace fuse;

TEST(Connect, DiscretizationMapsToWhereTheValueActuallyLives) {
  EXPECT_EQ(to_semantics({.name = "p", .disc = Discretization::FV}).location,
            SemanticLocation::Element);

  // Order 1 GLL coefficients ARE vertex values; higher order ones are not, and
  // calling them vertex values is what makes a transfer silently wrong.
  EXPECT_EQ(to_semantics({.name = "T", .disc = Discretization::ContinuousGLL, .order = 1}).location,
            SemanticLocation::Vertex);
  EXPECT_EQ(to_semantics({.name = "T", .disc = Discretization::ContinuousGLL, .order = 4}).location,
            SemanticLocation::FiniteElementCoefficient);
  EXPECT_EQ(
      to_semantics({.name = "T", .disc = Discretization::DiscontinuousGLL, .order = 3}).location,
      SemanticLocation::FiniteElementCoefficient);

  // A point cloud is none of the three. Saying nothing is correct.
  EXPECT_EQ(to_semantics({.name = "x", .disc = Discretization::PointCloud}).location,
            SemanticLocation::Unspecified);
}

TEST(Connect, ContinuityIsClaimedOnlyWhereTheDiscretizationImpliesIt) {
  EXPECT_EQ(to_semantics({.name = "T", .disc = Discretization::ContinuousGLL}).continuity,
            SemanticContinuity::Continuous);
  EXPECT_EQ(to_semantics({.name = "T", .disc = Discretization::DiscontinuousGLL}).continuity,
            SemanticContinuity::Discontinuous);
  // Finite volume is neither continuous nor discontinuous in this sense.
  EXPECT_EQ(to_semantics({.name = "p", .disc = Discretization::FV}).continuity,
            SemanticContinuity::Unspecified);
}

TEST(Connect, AxesTheGraphCannotStateStayUnspecified) {
  const FieldSemantics s = to_semantics({.name = "q"});
  EXPECT_EQ(s.frame, SemanticFrame::Unspecified);
  EXPECT_EQ(s.temporal, SemanticTemporal::InstantaneousCommitted)
      << "the only defaulted axis, and it is the semantic model's own default";
  EXPECT_EQ(s.id, "q");
}

TEST(Connect, ShapeFollowsTheComponentCount) {
  EXPECT_EQ(to_semantics({.name = "s", .components = 1}).shape, SemanticShape::Scalar);
  EXPECT_EQ(to_semantics({.name = "v", .components = 3}).shape, SemanticShape::Vector);
  EXPECT_EQ(to_semantics({.name = "t", .components = 9}).shape, SemanticShape::Tensor2);
}

TEST(Connect, IntentBecomesRoleAndExtensiveFieldsRequireConservation) {
  EXPECT_EQ(to_semantics({.name = "q", .intent = Intent::Requires}).role, SemanticRole::Input);
  EXPECT_EQ(to_semantics({.name = "q", .intent = Intent::Provides}).role, SemanticRole::Prognostic);

  EXPECT_EQ(required_conservation({.name = "q", .conservation = Conservation::Extensive}),
            ConservationGuarantee::GlobalConservative);
  EXPECT_EQ(required_conservation({.name = "T", .conservation = Conservation::Intensive}),
            ConservationGuarantee::None);
}

}  // namespace
