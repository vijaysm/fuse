// Graded connection compatibility: the five verdicts, stage reporting, and the
// operator chain a compilable connection produces.
//
// Needs neither PETSc nor MPI: compatibility is decided from descriptions.
#include "karma_compatibility.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
using namespace KARMA;

void require(bool condition, const char* message)
{
    if (!condition) throw std::runtime_error(message);
}

FieldSemantics temperature()
{
    FieldSemantics field;
    field.id = "temperature";
    field.version = {1, 2, 0};
    field.units.temperature = 1;
    field.frame = SemanticFrame::Global;
    field.shape = SemanticShape::Scalar;
    field.components = 1;
    field.location = SemanticLocation::Vertex;
    field.continuity = SemanticContinuity::Continuous;
    field.basis_degree = 1;
    field.temporal = SemanticTemporal::InstantaneousCommitted;
    field.role = SemanticRole::Prognostic;
    return field;
}

ConnectionRequest agreeing()
{
    ConnectionRequest request;
    request.source = temperature();
    request.target = temperature();
    request.requirement.field_id = "temperature";
    request.requirement.minimum_version = {1, 0, 0};
    request.requirement.maximum_version = {2, 0, 0};
    request.source_interval = {{1, 4}, {1, 2}};
    request.target_interval = {{1, 4}, {1, 2}};
    return request;
}

bool has(const ConnectionCompatibility& result, ConnectionStage stage, ConnectionReason reason)
{
    return std::any_of(result.diagnostics.begin(), result.diagnostics.end(),
                       [&](const ConnectionDiagnostic& diagnostic) {
                           return diagnostic.stage == stage && diagnostic.reason == reason;
                       });
}

void exact_connection()
{
    const auto result = check_connection(agreeing());
    require(result.verdict == ConnectionVerdict::Exact, "two identical fields did not compile exactly");
    require(result.chain.empty(), "an exact connection emitted an operator");
    require(!result.first_incompatible_stage.has_value(), "an exact connection named an incompatible stage");
    require(result.diagnostics.empty(), "an exact connection produced diagnostics");
    require(result.compiles(), "an exact connection does not compile");
}

// A unit literal nobody could interpret degrades to zero dimensions, which
// is indistinguishable from a genuinely dimensionless field. Treating the two
// as equal is how a density gets connected to a bare count.
void undeclared_units_are_not_dimensionless()
{
    auto unknown_to_dimensionless = agreeing();
    unknown_to_dimensionless.source.metadata.push_back(
        {std::string(unrecognized_units_key), {std::byte{'k'}, std::byte{'g'}}});
    const auto refused = check_connection(unknown_to_dimensionless);
    require(!refused.compiles(), "a field with uninterpretable units was connected to a dimensionless one");
    require(refused.diagnostics.front().reason == ConnectionReason::UnitsUndeclared,
            "the refusal did not name the undeclared units");

    // Two sides carrying the SAME uninterpretable literal agree by spelling.
    auto both_unknown = agreeing();
    both_unknown.source.metadata.push_back({std::string(unrecognized_units_key), {std::byte{'k'}, std::byte{'g'}}});
    both_unknown.target.metadata.push_back({std::string(unrecognized_units_key), {std::byte{'k'}, std::byte{'g'}}});
    require(check_connection(both_unknown).compiles(),
            "two fields with the same unit literal were refused");

    // Two DIFFERENT uninterpretable literals must not agree just because both
    // degraded to the same zero dimensions.
    auto differing = agreeing();
    differing.source.metadata.push_back({std::string(unrecognized_units_key), {std::byte{'k'}, std::byte{'g'}}});
    differing.target.metadata.push_back({std::string(unrecognized_units_key), {std::byte{'m'}, std::byte{'s'}}});
    require(!check_connection(differing).compiles(),
            "two different uninterpretable unit literals were treated as equal");
}

void unit_rules()
{
    // A dimension mismatch cannot be repaired by any operator.
    auto dimensioned = agreeing();
    dimensioned.target.units.length = 1;
    const auto mismatched = check_connection(dimensioned);
    require(mismatched.verdict == ConnectionVerdict::Unsupported, "a dimension mismatch was not unsupported");
    require(mismatched.first_incompatible_stage == ConnectionStage::Unit, "the unit stage was not blamed");
    require(has(mismatched, ConnectionStage::Unit, ConnectionReason::DimensionMismatch), "wrong reason code");
    require(!mismatched.compiles(), "a dimension mismatch reported as compilable");

    // Same dimensions, different exact scale, no offset: the one operator that
    // may be inserted without being asked for.
    auto scaled = agreeing();
    scaled.source.units.scale = Rational{1, 1000};
    const auto automatic = check_connection(scaled);
    require(automatic.verdict == ConnectionVerdict::LosslessExplicit, "an exact unit scale was not lossless");
    require(automatic.chain.size() == 1 && automatic.chain.steps[0].kind == OperatorKind::UnitScale,
            "no unit scale was inserted");
    require(!automatic.chain.steps[0].explicitly_requested, "the inserted scale claimed to be requested");
    require(automatic.chain.steps[0].scale == Rational{1, 1000}, "the conversion factor is wrong");

    // Site policy can forbid even that, and then the caller must ask.
    auto forbidden = scaled;
    forbidden.allow_automatic_unit_scale = false;
    const auto refused = check_connection(forbidden);
    require(refused.verdict == ConnectionVerdict::Ambiguous, "a forbidden automatic scale was not ambiguous");
    require(has(refused, ConnectionStage::Unit, ConnectionReason::UnitScaleNotPermitted), "wrong reason code");

    auto requested = forbidden;
    requested.explicitly_requested = operator_request_bit(OperatorKind::UnitScale);
    const auto allowed = check_connection(requested);
    require(allowed.verdict == ConnectionVerdict::LosslessExplicit, "an explicitly requested scale was refused");
    require(allowed.chain.steps[0].explicitly_requested, "an explicit request was not recorded on the step");

    // An offset difference is an affine conversion and is never automatic:
    // Celsius to Kelvin changes what zero means.
    auto affine = agreeing();
    affine.source.units.offset = Rational{5463, 20};
    const auto unrequested = check_connection(affine);
    require(unrequested.verdict == ConnectionVerdict::Ambiguous, "an affine conversion was inserted automatically");
    require(has(unrequested, ConnectionStage::Unit, ConnectionReason::AffineConversionNotRequested), "wrong reason");

    auto affine_ok = affine;
    affine_ok.explicitly_requested = operator_request_bit(OperatorKind::AffineUnit);
    const auto affine_result = check_connection(affine_ok);
    require(affine_result.verdict == ConnectionVerdict::LosslessExplicit, "a requested affine conversion was refused");
    require(affine_result.chain.steps[0].kind == OperatorKind::AffineUnit, "no affine operator was emitted");
    require(affine_result.chain.steps[0].offset == Rational{5463, 20}, "the offset was not carried exactly");
}

void shape_rules()
{
    // Reducing three components to one could be selection or aggregation, and
    // choosing for the caller would silently change the physics.
    auto reduced = agreeing();
    reduced.source.shape = SemanticShape::Vector;
    reduced.source.components = 3;
    const auto unasked = check_connection(reduced);
    require(unasked.verdict == ConnectionVerdict::Ambiguous, "a component reduction was performed unasked");
    require(has(unasked, ConnectionStage::Shape, ConnectionReason::ComponentReductionNotRequested), "wrong reason");

    auto selected = reduced;
    selected.explicitly_requested = operator_request_bit(OperatorKind::ComponentSelect);
    const auto selection = check_connection(selected);
    require(selection.verdict == ConnectionVerdict::LossyExplicit, "a component selection was not lossy");
    require(selection.chain.size() == 1 && selection.chain.steps[0].kind == OperatorKind::ComponentSelect,
            "no selection operator was emitted");
    require(selection.compiles(), "an explicitly requested lossy connection does not compile");

    // Requesting both leaves the choice unmade.
    auto both = reduced;
    both.explicitly_requested =
        operator_request_bit(OperatorKind::ComponentSelect) | operator_request_bit(OperatorKind::Aggregate);
    const auto ambiguous = check_connection(both);
    require(ambiguous.verdict == ConnectionVerdict::Ambiguous, "two candidate reductions resolved silently");
    require(has(ambiguous, ConnectionStage::Shape, ConnectionReason::ComponentReductionAmbiguous), "wrong reason");

    // Widening invents values, so it must be asked for too.
    auto widened = agreeing();
    widened.target.shape = SemanticShape::Vector;
    widened.target.components = 3;
    require(check_connection(widened).verdict == ConnectionVerdict::Ambiguous, "an expansion was inserted unasked");
    require(has(check_connection(widened), ConnectionStage::Shape, ConnectionReason::ComponentExpansionNotRequested),
            "wrong reason");
}

void location_rules()
{
    // Vertex to element is a change of where the value lives, and a copy
    // cannot express it.
    auto relocated = agreeing();
    relocated.target.location = SemanticLocation::Element;
    const auto copied = check_connection(relocated);
    require(copied.verdict == ConnectionVerdict::Unsupported, "a copy was accepted across locations");
    require(has(copied, ConnectionStage::Location, ConnectionReason::LocationChangeRequiresTransfer), "wrong reason");

    auto interpolated = relocated;
    interpolated.transfer.kind = TransferKind::Interpolation;
    interpolated.transfer.algorithm = "nodal-sampling";
    interpolated.explicitly_requested = operator_request_bit(OperatorKind::SpatialTransfer);
    const auto transferred = check_connection(interpolated);
    require(transferred.verdict == ConnectionVerdict::LossyExplicit, "a resampling transfer was not lossy");
    require(transferred.chain.size() == 1 && transferred.chain.steps[0].kind == OperatorKind::SpatialTransfer,
            "no spatial transfer was emitted");
    require(transferred.chain.steps[0].algorithm == "nodal-sampling", "the transfer algorithm was not carried");

    // A continuity change also needs a transfer, not a copy.
    auto discontinuous = agreeing();
    discontinuous.source.continuity = SemanticContinuity::Discontinuous;
    require(has(check_connection(discontinuous), ConnectionStage::Location,
                ConnectionReason::ContinuityChangeRequiresTransfer), "a continuity change was copied");
}

void temporal_rules()
{
    // An instantaneous value is not an interval average, whatever the clock says.
    auto reinterpreted = agreeing();
    reinterpreted.target.temporal = SemanticTemporal::IntervalAverage;
    const auto mismatch = check_connection(reinterpreted);
    require(mismatch.verdict == ConnectionVerdict::Ambiguous, "a temporal reinterpretation happened silently");
    require(has(mismatch, ConnectionStage::Temporal, ConnectionReason::TemporalInterpretationMismatch), "wrong reason");

    auto projected = reinterpreted;
    projected.explicitly_requested = operator_request_bit(OperatorKind::TemporalTransfer);
    const auto projection = check_connection(projected);
    require(projection.verdict == ConnectionVerdict::LossyExplicit, "a temporal projection was not lossy");
    require(projection.chain.steps[0].kind == OperatorKind::TemporalTransfer, "no temporal operator was emitted");

    // Intervals are compared exactly. 1/3 is not representable as a double and
    // must still compare unequal to 1/2 without any tolerance.
    auto shifted = agreeing();
    shifted.target_interval = {{1, 4}, {1, 3}};
    require(has(check_connection(shifted), ConnectionStage::Temporal, ConnectionReason::TemporalIntervalMismatch),
            "differing exact intervals were accepted");
    auto equal_thirds = agreeing();
    equal_thirds.source_interval = {{0, 1}, {1, 3}};
    equal_thirds.target_interval = {{0, 1}, {2, 6}};
    require(check_connection(equal_thirds).verdict == ConnectionVerdict::Exact,
            "two equal rationals with different spellings compared unequal");

    auto reversed = agreeing();
    reversed.source_interval = {{1, 2}, {1, 4}};
    const auto backwards = check_connection(reversed);
    require(backwards.verdict == ConnectionVerdict::Unsupported, "a reversed interval was accepted");
    require(has(backwards, ConnectionStage::Temporal, ConnectionReason::TemporalIntervalReversed), "wrong reason");
}

void conservation_rules()
{
    // The failure this stage exists to prevent: a connection that must
    // conserve mass silently compiling onto a transfer that does not.
    auto demanded = agreeing();
    demanded.target.location = SemanticLocation::Element;
    demanded.transfer.kind = TransferKind::Interpolation;
    demanded.transfer.conservation = ConservationGuarantee::None;
    demanded.required_conservation = ConservationGuarantee::GlobalConservative;
    demanded.explicitly_requested = operator_request_bit(OperatorKind::SpatialTransfer);
    const auto downgraded = check_connection(demanded);
    require(downgraded.verdict == ConnectionVerdict::Unsupported, "a conservation downgrade compiled");
    require(has(downgraded, ConnectionStage::Conservation, ConnectionReason::ConservationNotProvidedByTransfer),
            "wrong reason code");

    auto conservative = demanded;
    conservative.transfer.kind = TransferKind::ConservativeRemap;
    conservative.transfer.conservation = ConservationGuarantee::GlobalConservative;
    const auto conserved = check_connection(conservative);
    require(conserved.compiles(), "a conservative transfer was refused");
    require(conserved.chain.steps[0].conservation == ConservationGuarantee::GlobalConservative,
            "the guarantee was not carried onto the step");

    // A stronger guarantee than required is acceptable; a weaker one is not.
    auto stronger = demanded;
    stronger.required_conservation = ConservationGuarantee::LocalConservative;
    stronger.transfer.conservation = ConservationGuarantee::GlobalConservative;
    require(check_connection(stronger).compiles(), "a stronger guarantee than required was refused");
    auto weaker = demanded;
    weaker.required_conservation = ConservationGuarantee::GlobalConservative;
    weaker.transfer.conservation = ConservationGuarantee::LocalConservative;
    require(!check_connection(weaker).compiles(), "a weaker guarantee than required was accepted");
}

void capability_and_version_closure()
{
    auto missing = agreeing();
    missing.requirement.required_capabilities = 0b1011;
    missing.available_capabilities = 0b0001;
    const auto result = check_connection(missing);
    require(result.verdict == ConnectionVerdict::Unsupported, "missing capabilities were tolerated");
    require(result.missing_capabilities == 0b1010, "the missing capability closure is wrong");
    require(has(result, ConnectionStage::Capability, ConnectionReason::MissingRequiredCapabilities), "wrong reason");

    // A version outside the requirement's interval is refused at the identity
    // stage, which precedes units.
    auto too_new = agreeing();
    too_new.requirement.maximum_version = {1, 1, 0};
    const auto refused = check_connection(too_new);
    require(refused.first_incompatible_stage == ConnectionStage::Identity, "the identity stage was not blamed");
    require(has(refused, ConnectionStage::Identity, ConnectionReason::RequirementVersionUnsatisfied), "wrong reason");

    // An unset upper bound is unbounded, not a refusal of everything. A zero
    // maximum treated as a real bound would reject every version silently.
    auto unbounded = agreeing();
    unbounded.requirement.maximum_version = {0, 0, 0};
    require(check_connection(unbounded).verdict == ConnectionVerdict::Exact,
            "an unset maximum version was treated as a bound");

    auto wrong_field = agreeing();
    wrong_field.requirement.field_id = "pressure";
    require(has(check_connection(wrong_field), ConnectionStage::Identity, ConnectionReason::RequirementFieldMismatch),
            "a requirement naming another field was accepted");

    auto anonymous = agreeing();
    anonymous.source.id.clear();
    require(has(check_connection(anonymous), ConnectionStage::Identity, ConnectionReason::EmptyFieldIdentifier),
            "an unnamed field was accepted");
}

void reports_every_stage_at_once()
{
    // A caller fixing units should not then discover a missing capability on
    // the next attempt: every stage runs and the earliest failure is named.
    auto broken = agreeing();
    broken.target.units.length = 1;
    broken.target.location = SemanticLocation::Element;
    broken.requirement.required_capabilities = 0b100;
    broken.available_capabilities = 0;
    const auto result = check_connection(broken);
    require(result.verdict == ConnectionVerdict::Unsupported, "a multiply broken connection compiled");
    require(result.first_incompatible_stage == ConnectionStage::Unit, "the earliest failing stage was not named");
    require(has(result, ConnectionStage::Unit, ConnectionReason::DimensionMismatch), "the unit problem was dropped");
    require(has(result, ConnectionStage::Location, ConnectionReason::LocationChangeRequiresTransfer),
            "the location problem was dropped");
    require(result.missing_capabilities == 0b100, "the capability closure was dropped after an earlier failure");
    require(result.diagnostics.size() >= 3, "diagnostics stopped at the first failure");

    // A refused connection emits no chain: there is no partial chain to fall
    // back onto. This needs a case where an EARLY stage succeeded and emitted
    // an operator before a LATER stage refused - otherwise the chain is empty
    // for want of any step rather than because it was discarded.
    require(result.chain.empty(), "a refused connection emitted an operator chain");

    auto late_refusal = agreeing();
    late_refusal.source.units.scale = Rational{1, 1000};
    late_refusal.requirement.required_capabilities = 0b1;
    late_refusal.available_capabilities = 0;
    const auto discarded = check_connection(late_refusal);
    require(discarded.verdict == ConnectionVerdict::Unsupported, "a missing capability was tolerated");
    require(discarded.chain.empty(),
            "a stage that refused after an operator was emitted left a partial chain");
}

void full_chain_is_ordered()
{
    auto everything = agreeing();
    everything.source.units.scale = Rational{1, 1000};
    everything.source.shape = SemanticShape::Vector;
    everything.source.components = 3;
    everything.target.location = SemanticLocation::Element;
    everything.target.temporal = SemanticTemporal::IntervalAverage;
    everything.transfer.kind = TransferKind::ConservativeRemap;
    everything.transfer.conservation = ConservationGuarantee::GlobalConservative;
    everything.transfer.algorithm = "intersection";
    everything.required_conservation = ConservationGuarantee::GlobalConservative;
    everything.explicitly_requested = operator_request_bit(OperatorKind::Aggregate) |
                                      operator_request_bit(OperatorKind::SpatialTransfer) |
                                      operator_request_bit(OperatorKind::TemporalTransfer) |
                                      operator_request_bit(OperatorKind::Clip);
    const auto result = check_connection(everything);
    require(result.compiles(), "a fully specified connection was refused");
    require(result.verdict == ConnectionVerdict::LossyExplicit, "an aggregating connection was not lossy");
    require(chain_is_ordered(result.chain), "the emitted chain is not in phase order");
    require(chain_requests_are_authorized(result.chain), "the emitted chain contains an unrequested operator");
    require(result.chain.size() == 5, "the emitted chain has the wrong number of steps");
    require(result.chain.steps[0].kind == OperatorKind::UnitScale, "the chain does not begin with the unit scale");
    require(result.chain.steps.back().kind == OperatorKind::Clip, "the chain does not end with the constraint");
}
}

int main()
{
    try
    {
        exact_connection();
        undeclared_units_are_not_dimensionless();
        unit_rules();
        shape_rules();
        location_rules();
        temporal_rules();
        conservation_rules();
        capability_and_version_closure();
        reports_every_stage_at_once();
        full_chain_is_ordered();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
