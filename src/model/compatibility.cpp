#include "fuse/model/compatibility.hpp"

#include <optional>

#include <algorithm>

namespace fuse
{
namespace
{
/** Every diagnostic this checker emits is a refusal; the two severities differ
 * in whether the caller can fix it by asking for an operator (`Ambiguous`) or
 * cannot express the connection at all (`Unsupported`). */
ConnectionVerdict severity_of(ConnectionReason reason) noexcept
{
    switch (reason)
    {
        case ConnectionReason::UnitScaleNotPermitted:
        case ConnectionReason::UnitScaleSignNotRequested:
        case ConnectionReason::AffineConversionNotRequested:
        case ConnectionReason::ShapeChangeNotRequested:
        case ConnectionReason::ComponentReductionNotRequested:
        case ConnectionReason::ComponentReductionAmbiguous:
        case ConnectionReason::ComponentExpansionNotRequested:
        case ConnectionReason::TemporalInterpretationMismatch:
        case ConnectionReason::TemporalIntervalMismatch: return ConnectionVerdict::Ambiguous;
        default: return ConnectionVerdict::Unsupported;
    }
}

class Evaluation
{
  public:
    explicit Evaluation(const ConnectionRequest& request) : request_(request) {}

    void refuse(ConnectionStage stage, ConnectionReason reason, std::string detail,
                std::uint64_t missing_capabilities = 0)
    {
        result_.diagnostics.push_back({stage, reason, missing_capabilities, std::move(detail)});
        const auto severity = severity_of(reason);
        if (severity > result_.verdict) result_.verdict = severity;
        if (!result_.first_incompatible_stage) result_.first_incompatible_stage = stage;
    }

    [[nodiscard]] bool requested(OperatorKind kind) const noexcept
    {
        return (request_.explicitly_requested & operator_request_bit(kind)) != 0;
    }

    void emit(OperatorStep step) { result_.chain.steps.push_back(std::move(step)); }

    [[nodiscard]] ConnectionCompatibility finish()
    {
        if (result_.verdict == ConnectionVerdict::Ambiguous || result_.verdict == ConnectionVerdict::Unsupported)
        {
            // A refused connection keeps no Spatial-, Temporal- or
            // Constraint-phase step: each of those moves or alters the actual
            // value, and emitting one without the connection it belongs to
            // would be exactly the runtime fallback that must not exist (a
            // fallback that substitutes an operator changes the meaning of
            // the coupled quantity while reporting success).
            //
            // BM-1 (review 2026-10-11): this used to clear the WHOLE chain,
            // including a Transform-phase step (an inferred `UnitScale`, most
            // often) that an EARLIER stage had already authorized before a
            // LATER stage refused -- Shape, Location, Temporal, Conservation
            // and Capability all run after Unit. Discarding it was a second,
            // independent silent loss on top of the refusal itself: a
            // channel whose target is `Conservation::Extensive` refuses at
            // the Conservation stage on every declared transfer that is not
            // conservative, so a scale-only unit mismatch (kPa -> Pa) on such
            // a channel lost its UnitScale here and surfaced, much later
            // and with no explanation, as a bare `E-IR-TYPE` in `lower()`
            // (src/ir/rules_channel.cpp reads this same chain). A
            // Transform-phase step only rescales, reorders or selects
            // components -- it carries none of the above risk, so keeping it
            // is not a partial chain in the sense the comment above warns
            // against; it is the one part of the chain that was already
            // correct before the later stage had anything to say.
            auto& steps = result_.chain.steps;
            steps.erase(std::remove_if(steps.begin(), steps.end(),
                                       [](const OperatorStep& step) {
                                           return operator_phase(step.kind) != OperatorPhase::Transform;
                                       }),
                       steps.end());
            return std::move(result_);
        }
        if (result_.chain.empty()) result_.verdict = ConnectionVerdict::Exact;
        else if (chain_is_lossless(result_.chain)) result_.verdict = ConnectionVerdict::LosslessExplicit;
        else result_.verdict = ConnectionVerdict::LossyExplicit;
        return std::move(result_);
    }

    [[nodiscard]] ConnectionCompatibility& result() noexcept { return result_; }

  private:
    const ConnectionRequest& request_;
    ConnectionCompatibility result_;
};

// The spine's rational helpers returned an error code; fuse::Rational compares
// and multiplies exactly and REFUSES rather than reporting overflow in-band. The
// overflow verdicts below therefore have to catch, because this function is
// declared total and must not throw out of a compatibility check.
bool equal(Rational left, Rational right) noexcept { return left == right; }

bool version_unbounded(SemanticVersion version) noexcept
{
    return version.major == 0 && version.minor == 0 && version.patch == 0;
}

void check_identity(const ConnectionRequest& request, Evaluation& evaluation)
{
    if (request.source.id.empty() || request.target.id.empty())
        evaluation.refuse(ConnectionStage::Identity, ConnectionReason::EmptyFieldIdentifier,
                          "a connected field has no identifier");
    if (!request.requirement.field_id.empty() && request.requirement.field_id != request.source.id)
        evaluation.refuse(ConnectionStage::Identity, ConnectionReason::RequirementFieldMismatch,
                          "the requirement names '" + request.requirement.field_id + "' but the source is '" +
                              request.source.id + "'");

    // A zero maximum is unbounded rather than a refusal of everything: an
    // unset bound must not silently reject every version.
    const auto& version = request.source.version;
    const bool below = version < request.requirement.minimum_version;
    const bool above = !version_unbounded(request.requirement.maximum_version) &&
                       request.requirement.maximum_version < version;
    if (below || above)
        evaluation.refuse(ConnectionStage::Identity, ConnectionReason::RequirementVersionUnsatisfied,
                          "source version " + serialize_semantic_version(version) +
                              " is outside the accepted interval");
}

namespace
{
/** Finds a metadata entry by key, or null. */
const MetadataEntry* find_metadata(const std::vector<MetadataEntry>& metadata, std::string_view key)
{
    for (const auto& entry : metadata)
        if (entry.key == key) return &entry;
    return nullptr;
}
} // namespace

void check_units(const ConnectionRequest& request, Evaluation& evaluation)
{
    // An uninterpretable unit literal degrades to zero dimensions, which
    // makes it indistinguishable from a genuinely dimensionless field. That
    // is why the adapter preserves the literal: a field whose units could not
    // be read is not dimensionless, it is unknown, and the two must not be
    // silently connected. If both sides carry the SAME literal they agree by
    // spelling and may proceed; anything else cannot be shown to agree.
    const auto* source_literal = find_metadata(request.source.metadata, unrecognized_units_key);
    const auto* target_literal = find_metadata(request.target.metadata, unrecognized_units_key);
    if (source_literal != nullptr || target_literal != nullptr)
    {
        const bool both = source_literal != nullptr && target_literal != nullptr;
        if (!both || source_literal->value != target_literal->value)
        {
            evaluation.refuse(ConnectionStage::Unit, ConnectionReason::UnitsUndeclared,
                              "a unit literal could not be interpreted, so the two units cannot be shown to agree");
            return;
        }
    }

    const auto& source = request.source.units;
    const auto& target = request.target.units;
    if (!unit_dimensions_match(source, target))
    {
        // No operator can repair a dimension mismatch: there is no conversion
        // between a temperature and a length.
        evaluation.refuse(ConnectionStage::Unit, ConnectionReason::DimensionMismatch,
                          "source and target dimensions differ");
        return;
    }

    const bool offsets_differ = !equal(source.offset, target.offset);
    const bool scales_differ = !equal(source.scale, target.scale);
    if (!offsets_differ && !scales_differ) return;

    // source_scale / target_scale, exactly. A zero target scale or an overflowing
    // product is a refusal, not an approximation.
    // Non-throwing by construction rather than by catching: an
    // unrepresentable conversion is a verdict on the channel, and building a
    // verdict is not something an exception should be routed through (it is
    // not even available in a no-exceptions build).
    const std::optional<Rational> factor = Rational::try_div(source.scale, target.scale);
    if (!factor)
    {
        evaluation.refuse(ConnectionStage::Unit, ConnectionReason::UnitConversionOverflow,
                          "the exact unit conversion factor is not representable");
        return;
    }

    if (offsets_differ)
    {
        // An offset change moves the zero of the scale - Celsius to Kelvin -
        // which is a change of meaning, never inserted on the caller's behalf.
        const std::optional<Rational> raw_delta =
            Rational::try_sub(source.offset, target.offset);
        if (!raw_delta)
        {
            evaluation.refuse(ConnectionStage::Unit, ConnectionReason::UnitConversionOverflow,
                              "the exact unit offset difference is not representable");
            return;
        }
        // BM-6: the affine conversion is `v_t = factor * v + (o_s - o_t) /
        // target.scale` -- the step's own OWN consumer (not yet landed;
        // `ir_can_emit(AffineUnit)` is false) will apply `step.scale` and
        // `step.offset` exactly that way (transform.hpp's `OperatorStep`).
        // The division by `target.scale` was missing here -- ported
        // verbatim from the parent, where it is only ever correct when
        // `target.scale == 1`, the one case `Compatibility.UnitRules`
        // happens to pin. `target.scale` is never zero (R-19: a unit's
        // scale is never non-positive), so `try_div` here fails only on
        // overflow, like `factor` above.
        const std::optional<Rational> delta = Rational::try_div(*raw_delta, target.scale);
        if (!delta)
        {
            evaluation.refuse(ConnectionStage::Unit, ConnectionReason::UnitConversionOverflow,
                              "the exact unit offset, divided by the target scale, is not "
                              "representable");
            return;
        }
        if (!evaluation.requested(OperatorKind::AffineUnit))
        {
            evaluation.refuse(ConnectionStage::Unit, ConnectionReason::AffineConversionNotRequested,
                              "an affine unit conversion is required and was not requested");
            return;
        }
        OperatorStep step;
        step.kind = OperatorKind::AffineUnit;
        step.explicitly_requested = true;
        step.scale = *factor;
        step.offset = *delta;
        evaluation.emit(std::move(step));
        return;
    }

    const bool explicit_request = evaluation.requested(OperatorKind::UnitScale);
    // REVIEW.md R-19: an exact, zero-offset unit scale is auto-insertable
    // because it is "a change of spelling, not of meaning" (this file's own
    // rationale for allow_automatic_unit_scale) -- but a NEGATIVE factor
    // negates every value crossing the channel, which is a change of
    // meaning. Gated exactly like the affine branch above: never inserted
    // without an explicit request, regardless of allow_automatic_unit_scale.
    if (*factor < Rational{0} && !explicit_request)
    {
        evaluation.refuse(ConnectionStage::Unit, ConnectionReason::UnitScaleSignNotRequested,
                          "the unit scale ratio is negative, which would flip the sign of "
                          "every value; request UnitScale explicitly if this is intended");
        return;
    }
    if (!explicit_request && !request.allow_automatic_unit_scale)
    {
        evaluation.refuse(ConnectionStage::Unit, ConnectionReason::UnitScaleNotPermitted,
                          "automatic unit scaling is disabled and no scale was requested");
        return;
    }
    OperatorStep step;
    step.kind = OperatorKind::UnitScale;
    step.explicitly_requested = explicit_request;
    step.scale = *factor;
    evaluation.emit(std::move(step));
}

void check_shape(const ConnectionRequest& request, Evaluation& evaluation)
{
    const auto source_components = request.source.components;
    const auto target_components = request.target.components;
    if (target_components < source_components)
    {
        const bool select = evaluation.requested(OperatorKind::ComponentSelect);
        const bool aggregate = evaluation.requested(OperatorKind::Aggregate);
        if (select && aggregate)
        {
            // Taking one component and averaging all of them are different
            // physics; with both offered the choice is not the checker's.
            evaluation.refuse(ConnectionStage::Shape, ConnectionReason::ComponentReductionAmbiguous,
                              "both selection and aggregation were requested for one reduction");
            return;
        }
        if (!select && !aggregate)
        {
            evaluation.refuse(ConnectionStage::Shape, ConnectionReason::ComponentReductionNotRequested,
                              "reducing components requires an explicit selection or aggregation");
            return;
        }
        OperatorStep step;
        step.kind = select ? OperatorKind::ComponentSelect : OperatorKind::Aggregate;
        step.explicitly_requested = true;
        evaluation.emit(std::move(step));
        return;
    }
    if (target_components > source_components)
    {
        if (!evaluation.requested(OperatorKind::Expand))
        {
            evaluation.refuse(ConnectionStage::Shape, ConnectionReason::ComponentExpansionNotRequested,
                              "widening components invents values and requires an explicit expansion");
            return;
        }
        OperatorStep step;
        step.kind = OperatorKind::Expand;
        step.explicitly_requested = true;
        evaluation.emit(std::move(step));
        return;
    }
    if (request.source.shape != request.target.shape)
    {
        if (!evaluation.requested(OperatorKind::ComponentReorder))
        {
            evaluation.refuse(ConnectionStage::Shape, ConnectionReason::ShapeChangeNotRequested,
                              "the shapes differ at equal component count and no reordering was requested");
            return;
        }
        OperatorStep step;
        step.kind = OperatorKind::ComponentReorder;
        step.explicitly_requested = true;
        evaluation.emit(std::move(step));
    }
}

void check_location(const ConnectionRequest& request, Evaluation& evaluation)
{
    const bool location_differs = request.source.location != request.target.location;
    const bool continuity_differs = request.source.continuity != request.target.continuity;
    const bool degree_differs = request.source.basis_degree != request.target.basis_degree;
    const bool declared_transfer = request.transfer.kind != TransferKind::Copy;

    if (!location_differs && !continuity_differs && !degree_differs && !declared_transfer) return;

    if ((location_differs || continuity_differs || degree_differs) && !declared_transfer)
    {
        evaluation.refuse(ConnectionStage::Location,
                          location_differs ? ConnectionReason::LocationChangeRequiresTransfer
                                           : ConnectionReason::ContinuityChangeRequiresTransfer,
                          "a copy cannot change where or how a value is represented");
        return;
    }

    // Declaring a transfer other than a copy *is* the explicit request: there
    // is no second place for the caller to say the same thing.
    OperatorStep step;
    step.kind = OperatorKind::SpatialTransfer;
    step.explicitly_requested = true;
    step.algorithm = request.transfer.algorithm;
    step.conservation = request.transfer.conservation;
    evaluation.emit(std::move(step));
}

void check_temporal(const ConnectionRequest& request, Evaluation& evaluation)
{
    if (!request.source_interval.ordered() || !request.target_interval.ordered())
    {
        evaluation.refuse(ConnectionStage::Temporal, ConnectionReason::TemporalIntervalReversed,
                          "a validity interval ends before it begins");
        return;
    }

    const bool interpretation_differs = request.source.temporal != request.target.temporal;
    // Compared exactly. A tolerance here would make rendezvous equality
    // depend on how a step size was spelled.
    const bool intervals_differ = !equal(request.source_interval.begin, request.target_interval.begin) ||
                                  !equal(request.source_interval.end, request.target_interval.end);
    if (!interpretation_differs && !intervals_differ) return;

    if (!evaluation.requested(OperatorKind::TemporalTransfer))
    {
        evaluation.refuse(ConnectionStage::Temporal,
                          interpretation_differs ? ConnectionReason::TemporalInterpretationMismatch
                                                 : ConnectionReason::TemporalIntervalMismatch,
                          interpretation_differs ? "an instantaneous value is not an interval quantity"
                                                 : "the exact validity intervals differ");
        return;
    }
    OperatorStep step;
    step.kind = OperatorKind::TemporalTransfer;
    step.explicitly_requested = true;
    evaluation.emit(std::move(step));
}

void check_conservation(const ConnectionRequest& request, Evaluation& evaluation)
{
    if (request.required_conservation == ConservationGuarantee::None) return;
    if (static_cast<std::uint8_t>(request.transfer.conservation) <
        static_cast<std::uint8_t>(request.required_conservation))
        evaluation.refuse(ConnectionStage::Conservation, ConnectionReason::ConservationNotProvidedByTransfer,
                          "the declared transfer does not provide the required conservation guarantee");
}

void check_capabilities(const ConnectionRequest& request, Evaluation& evaluation)
{
    const auto missing = request.requirement.required_capabilities & ~request.available_capabilities;
    if (missing == 0) return;
    // Collected even when an earlier stage already refused, so a caller fixing
    // units does not then discover this on the next attempt.
    evaluation.result().missing_capabilities |= missing;
    evaluation.refuse(ConnectionStage::Capability, ConnectionReason::MissingRequiredCapabilities,
                      "required capability bits are unavailable", missing);
}

void check_constraints(Evaluation& evaluation)
{
    // BM-13: `Clip` used to be the only requested constraint this
    // function honored; an explicit `operator_request_bit(OperatorKind::
    // ConservationEnforcement)` -- bit 41, a VALID request -- was
    // silently ignored, no step emitted and no diagnostic raised, the I4
    // shape (a caller's explicit request that silently does nothing).
    // Both operators are unimplemented in the IR today (`ir_can_emit()`,
    // src/ir/lowering.cpp, returns `false` for each), so emitting the
    // step -- exactly as `Clip` already does -- reaches the SAME
    // `E-IR-UNLOWERABLE` refusal `rules_channel.cpp` already raises for
    // `Clip`, rather than inventing a second way to say "not yet
    // implemented".
    if (evaluation.requested(OperatorKind::Clip))
    {
        OperatorStep step;
        step.kind = OperatorKind::Clip;
        step.explicitly_requested = true;
        evaluation.emit(std::move(step));
    }
    if (evaluation.requested(OperatorKind::ConservationEnforcement))
    {
        OperatorStep step;
        step.kind = OperatorKind::ConservationEnforcement;
        step.explicitly_requested = true;
        evaluation.emit(std::move(step));
    }
}
}

std::string_view connection_stage_name(ConnectionStage stage) noexcept
{
    switch (stage)
    {
        case ConnectionStage::Identity: return "identity";
        case ConnectionStage::Unit: return "unit";
        case ConnectionStage::Shape: return "shape";
        case ConnectionStage::Location: return "location";
        case ConnectionStage::Temporal: return "temporal";
        case ConnectionStage::Conservation: return "conservation";
        case ConnectionStage::Capability: return "capability";
    }
    return {};
}

std::string_view connection_reason_name(ConnectionReason reason) noexcept
{
    switch (reason)
    {
        case ConnectionReason::None: return "none";
        case ConnectionReason::EmptyFieldIdentifier: return "empty field identifier";
        case ConnectionReason::RequirementVersionUnsatisfied: return "requirement version unsatisfied";
        case ConnectionReason::RequirementFieldMismatch: return "requirement names another field";
        case ConnectionReason::DimensionMismatch: return "unit dimension mismatch";
        case ConnectionReason::UnitScaleNotPermitted: return "automatic unit scaling not permitted";
        case ConnectionReason::UnitScaleSignNotRequested: return "unit scale ratio is negative and was not explicitly requested";
        case ConnectionReason::AffineConversionNotRequested: return "affine unit conversion not requested";
        case ConnectionReason::UnitsUndeclared:
        case ConnectionReason::UnitConversionOverflow: return "unit conversion not representable";
        case ConnectionReason::ShapeChangeNotRequested: return "shape change not requested";
        case ConnectionReason::ComponentReductionNotRequested: return "component reduction not requested";
        case ConnectionReason::ComponentReductionAmbiguous: return "component reduction ambiguous";
        case ConnectionReason::ComponentExpansionNotRequested: return "component expansion not requested";
        case ConnectionReason::LocationChangeRequiresTransfer: return "location change requires a transfer";
        case ConnectionReason::ContinuityChangeRequiresTransfer: return "continuity change requires a transfer";
        case ConnectionReason::TemporalInterpretationMismatch: return "temporal interpretation mismatch";
        case ConnectionReason::TemporalIntervalMismatch: return "temporal interval mismatch";
        case ConnectionReason::TemporalIntervalReversed: return "temporal interval reversed";
        case ConnectionReason::ConservationNotProvidedByTransfer: return "conservation not provided by the transfer";
        case ConnectionReason::MissingRequiredCapabilities: return "required capabilities unavailable";
    }
    return {};
}

ConnectionCompatibility check_connection(const ConnectionRequest& request)
{
    Evaluation evaluation(request);
    // Stages run in this order and all of them run, so one call reports every
    // problem and the earliest failing stage is the first diagnostic.
    check_identity(request, evaluation);
    check_units(request, evaluation);
    check_shape(request, evaluation);
    check_location(request, evaluation);
    check_temporal(request, evaluation);
    check_conservation(request, evaluation);
    check_capabilities(request, evaluation);
    check_constraints(evaluation);
    return evaluation.finish();
}
} // namespace fuse
