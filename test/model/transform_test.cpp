// The compiled operator chain: phases, ordering, authorization, and identity.
//
// Needs neither PETSc nor MPI: a chain is a description.
#include "fuse/model/transform.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace fuse;

void require(bool condition, const char* message) { EXPECT_TRUE(condition) << message; }

OperatorStep step(OperatorKind kind, bool requested = true)
{
    OperatorStep value;
    value.kind = kind;
    value.explicitly_requested = requested;
    return value;
}

void phases()
{
    require(operator_phase(OperatorKind::UnitScale) == OperatorPhase::Transform, "unit scale is not a transform");
    require(operator_phase(OperatorKind::Expand) == OperatorPhase::Transform, "expansion is not a transform");
    require(operator_phase(OperatorKind::SpatialTransfer) == OperatorPhase::Spatial, "spatial phase is wrong");
    require(operator_phase(OperatorKind::TemporalTransfer) == OperatorPhase::Temporal, "temporal phase is wrong");
    require(operator_phase(OperatorKind::Clip) == OperatorPhase::Constraint, "clipping is not a constraint");
    require(operator_phase(OperatorKind::ConservationEnforcement) == OperatorPhase::Constraint,
            "conservation enforcement is not a constraint");

    // The declaration vocabulary and the chain vocabulary must agree.
    require(operator_kind_from_transform(TransformKind::Aggregate) == OperatorKind::Aggregate,
            "a declared transform did not lift to the same chain operator");
    require(operator_kind_from_transform(TransformKind::UnitScale) == OperatorKind::UnitScale,
            "a declared transform did not lift to the same chain operator");

    for (const auto kind : {OperatorKind::UnitScale, OperatorKind::AffineUnit, OperatorKind::SignFlip,
                            OperatorKind::ComponentSelect, OperatorKind::ComponentReorder, OperatorKind::Aggregate,
                            OperatorKind::Expand, OperatorKind::SpatialTransfer, OperatorKind::TemporalTransfer,
                            OperatorKind::Clip, OperatorKind::ConservationEnforcement})
        require(!operator_kind_name(kind).empty(), "an operator kind has no name");
    require(operator_kind_name(static_cast<OperatorKind>(99)).empty(), "an unknown operator acquired a name");
}

void loss_and_authorization()
{
    // Discarding values, inventing them, and resampling are all information
    // changes. A spatial or temporal transfer is only ever emitted when the
    // two sides genuinely differ, so it always resamples; the agreeing case
    // emits no operator at all.
    for (const auto kind : {OperatorKind::ComponentSelect, OperatorKind::Aggregate, OperatorKind::Expand,
                            OperatorKind::Clip, OperatorKind::SpatialTransfer, OperatorKind::TemporalTransfer})
        require(!operator_is_lossless(kind), "an information-changing operator claimed to be lossless");
    for (const auto kind : {OperatorKind::UnitScale, OperatorKind::AffineUnit, OperatorKind::SignFlip,
                            OperatorKind::ComponentReorder})
        require(operator_is_lossless(kind), "an information-preserving operator claimed to be lossy");

    // Exactly one operator may be inserted without being asked for.
    require(!operator_requires_explicit_request(OperatorKind::UnitScale),
            "exact unit scaling cannot be inserted automatically");
    for (const auto kind : {OperatorKind::AffineUnit, OperatorKind::SignFlip, OperatorKind::ComponentSelect,
                            OperatorKind::ComponentReorder, OperatorKind::Aggregate, OperatorKind::Expand,
                            OperatorKind::SpatialTransfer, OperatorKind::TemporalTransfer, OperatorKind::Clip,
                            OperatorKind::ConservationEnforcement})
        require(operator_requires_explicit_request(kind), "an operator may be inserted without a request");

    OperatorChain authorized{{step(OperatorKind::UnitScale, false), step(OperatorKind::SpatialTransfer)}};
    require(chain_requests_are_authorized(authorized), "an authorized chain was rejected");
    OperatorChain unauthorized{{step(OperatorKind::Aggregate, false)}};
    require(!chain_requests_are_authorized(unauthorized), "an unrequested lossy operator was authorized");

    require(chain_is_lossless(OperatorChain{{step(OperatorKind::UnitScale, false)}}), "a lossless chain read as lossy");
    require(!chain_is_lossless(authorized), "a chain containing a spatial transfer read as lossless");
    require(!chain_is_lossless(OperatorChain{{step(OperatorKind::Clip)}}), "a clipping chain read as lossless");
    require(chain_is_lossless({}), "an empty chain read as lossy");
}

void ordering()
{
    const OperatorChain canonical{{step(OperatorKind::UnitScale, false), step(OperatorKind::ComponentSelect),
                                   step(OperatorKind::SpatialTransfer), step(OperatorKind::TemporalTransfer),
                                   step(OperatorKind::Clip)}};
    require(chain_is_ordered(canonical), "the canonical chain was rejected as unordered");
    require(chain_is_ordered({}), "an empty chain was rejected as unordered");

    // Transferring in space before scaling units would map unscaled values.
    const OperatorChain transfer_first{{step(OperatorKind::SpatialTransfer), step(OperatorKind::UnitScale, false)}};
    require(!chain_is_ordered(transfer_first), "a chain transferring before transforming was accepted");

    // Projecting in time before mapping in space projects the wrong samples.
    const OperatorChain temporal_first{{step(OperatorKind::TemporalTransfer), step(OperatorKind::SpatialTransfer)}};
    require(!chain_is_ordered(temporal_first), "a chain projecting before transferring was accepted");

    // Constraints act on the values that will be published, so nothing may
    // follow them.
    const OperatorChain after_constraint{{step(OperatorKind::Clip), step(OperatorKind::TemporalTransfer)}};
    require(!chain_is_ordered(after_constraint), "a chain operating after a constraint was accepted");

    // Within the transform phase independent operators commute, so both
    // orders are ordered.
    require(chain_is_ordered({{step(OperatorKind::ComponentSelect), step(OperatorKind::UnitScale, false)}}),
            "two transform-phase operators were required to be in a fixed order");
}

void identity()
{
    const OperatorChain chain{{step(OperatorKind::UnitScale, false), step(OperatorKind::SpatialTransfer)}};
    require(operator_chain_hash(chain) == operator_chain_hash(chain), "the chain hash is not deterministic");

    // A reordered chain is a different chain, so its identity must differ. The
    // two steps must be identical in every other field, or the assertion would
    // pass on the differing request flags rather than on the operator kinds.
    const OperatorChain forward{{step(OperatorKind::Aggregate), step(OperatorKind::TemporalTransfer)}};
    const OperatorChain reordered{{step(OperatorKind::TemporalTransfer), step(OperatorKind::Aggregate)}};
    require(operator_chain_hash(reordered) != operator_chain_hash(forward), "chain identity ignores the operators");

    // The scale participates in identity: two chains differing only by a
    // conversion factor are not the same chain.
    OperatorChain scaled{chain};
    scaled.steps[0].scale = Rational{1000, 1};
    require(operator_chain_hash(scaled) != operator_chain_hash(chain), "chain identity ignores the scale");

    OperatorChain conserved{chain};
    conserved.steps[1].conservation = ConservationGuarantee::GlobalConservative;
    require(operator_chain_hash(conserved) != operator_chain_hash(chain),
            "chain identity ignores the conservation guarantee");

    // The algorithm participates in identity. Note that a boundary-collision
    // case is not constructible for this record: within a step the algorithm is
    // the only variable-length field and is followed by a fixed-width one, so
    // the length prefix is defense against a future second string rather than a
    // property this record can currently violate.
    OperatorChain named{{step(OperatorKind::SpatialTransfer)}};
    named.steps[0].algorithm = "intersection";
    OperatorChain other{{step(OperatorKind::SpatialTransfer)}};
    other.steps[0].algorithm = "nodal-sampling";
    require(operator_chain_hash(named) != operator_chain_hash(other), "chain identity ignores the algorithm");
    require(operator_chain_hash(named) != operator_chain_hash({{step(OperatorKind::SpatialTransfer)}}),
            "an empty algorithm hashes the same as a named one");
}
}

TEST(Transform, Phases) { phases(); }
TEST(Transform, LossAndAuthorization) { loss_and_authorization(); }
TEST(Transform, Ordering) { ordering(); }
TEST(Transform, Identity) { identity(); }
