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
    explicit Evaluation(const ConnectionRequest& request) : _request(request) {}

    void refuse(ConnectionStage stage, ConnectionReason reason, std::string detail,
                std::uint64_t missing_capabilities = 0)
    {
        _result.diagnostics.push_back({stage, reason, missing_capabilities, std::move(detail)});
        const auto severity = severity_of(reason);
        if (severity > _result.verdict) _result.verdict = severity;
        if (!_result.first_incompatible_stage) _result.first_incompatible_stage = stage;
    }

    [[nodiscard]] bool requested(OperatorKind kind) const noexcept
    {
        return (_request.explicitly_requested & operator_request_bit(kind)) != 0;
    }

    void emit(OperatorStep step) { _result.chain.steps.push_back(std::move(step)); }

    [[nodiscard]] ConnectionCompatibility finish()
    {
        if (_result.verdict == ConnectionVerdict::Ambiguous || _result.verdict == ConnectionVerdict::Unsupported)
        {
            // A refused connection has no chain. A partial chain would invite a
            // runtime fallback, which is exactly what must not exist: a
            // fallback that substitutes an operator changes the meaning of the
            // coupled quantity while reporting success.
            _result.chain.steps.clear();
            return std::move(_result);
        }
        if (_result.chain.empty()) _result.verdict = ConnectionVerdict::Exact;
        else if (chain_is_lossless(_result.chain)) _result.verdict = ConnectionVerdict::LosslessExplicit;
        else _result.verdict = ConnectionVerdict::LossyExplicit;
        return std::move(_result);
    }

    [[nodiscard]] ConnectionCompatibility& result() noexcept { return _result; }

  private:
    const ConnectionRequest& _request;
    ConnectionCompatibility _result;
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
    std::optional<Rational> factor;
    try {
        factor = source.scale / target.scale;
    } catch (const Error&) {
        factor.reset();
    }
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
        std::optional<Rational> delta;
        try {
            delta = source.offset - target.offset;
        } catch (const Error&) {
            delta.reset();
        }
        if (!delta)
        {
            evaluation.refuse(ConnectionStage::Unit, ConnectionReason::UnitConversionOverflow,
                              "the exact unit offset difference is not representable");
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
    if (!evaluation.requested(OperatorKind::Clip)) return;
    OperatorStep step;
    step.kind = OperatorKind::Clip;
    step.explicitly_requested = true;
    evaluation.emit(std::move(step));
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
