#include "fuse/model/transform.hpp"

#include "fuse/base/bytes.hpp"

#include <algorithm>

namespace fuse
{
namespace
{



}

std::string_view operator_kind_name(OperatorKind kind) noexcept
{
    switch (kind)
    {
        case OperatorKind::UnitScale: return "unit scale";
        case OperatorKind::AffineUnit: return "affine unit conversion";
        case OperatorKind::SignFlip: return "sign flip";
        case OperatorKind::ComponentSelect: return "component selection";
        case OperatorKind::ComponentReorder: return "component reordering";
        case OperatorKind::Aggregate: return "component aggregation";
        case OperatorKind::Expand: return "component expansion";
        case OperatorKind::SpatialTransfer: return "spatial transfer";
        case OperatorKind::TemporalTransfer: return "temporal transfer";
        case OperatorKind::Clip: return "clipping";
        case OperatorKind::ConservationEnforcement: return "conservation enforcement";
    }
    return {};
}

OperatorKind operator_kind_from_transform(TransformKind kind) noexcept
{
    return static_cast<OperatorKind>(static_cast<std::uint8_t>(kind));
}

OperatorPhase operator_phase(OperatorKind kind) noexcept
{
    const auto ordinal = static_cast<std::uint8_t>(kind);
    if (ordinal < static_cast<std::uint8_t>(OperatorKind::SpatialTransfer)) return OperatorPhase::Transform;
    if (ordinal < static_cast<std::uint8_t>(OperatorKind::TemporalTransfer)) return OperatorPhase::Spatial;
    if (ordinal < static_cast<std::uint8_t>(OperatorKind::Clip)) return OperatorPhase::Temporal;
    return OperatorPhase::Constraint;
}

bool operator_is_lossless(OperatorKind kind) noexcept
{
    switch (kind)
    {
        case OperatorKind::UnitScale:
        case OperatorKind::AffineUnit:
        case OperatorKind::SignFlip:
        case OperatorKind::ComponentReorder: return true;
        // Selection and aggregation discard values, expansion invents them,
        // clipping and conservation enforcement change them, and a spatial or
        // temporal transfer resamples. A transfer is only ever emitted when the
        // two sides genuinely differ; the agreeing case emits no operator.
        case OperatorKind::ComponentSelect:
        case OperatorKind::Aggregate:
        case OperatorKind::Expand:
        case OperatorKind::SpatialTransfer:
        case OperatorKind::TemporalTransfer:
        case OperatorKind::Clip:
        case OperatorKind::ConservationEnforcement: return false;
    }
    return false;
}

bool operator_requires_explicit_request(OperatorKind kind) noexcept
{
    return kind != OperatorKind::UnitScale;
}

std::uint64_t operator_request_bit(OperatorKind kind) noexcept
{
    const auto ordinal = static_cast<unsigned>(static_cast<std::uint8_t>(kind));
    if (ordinal >= 64) return 0;
    return std::uint64_t{1} << ordinal;
}

bool chain_is_ordered(const OperatorChain& chain) noexcept
{
    for (std::size_t index = 1; index < chain.steps.size(); ++index)
        if (operator_phase(chain.steps[index].kind) < operator_phase(chain.steps[index - 1].kind)) return false;
    return true;
}

bool chain_requests_are_authorized(const OperatorChain& chain) noexcept
{
    return std::all_of(chain.steps.begin(), chain.steps.end(), [](const OperatorStep& step) {
        return step.explicitly_requested || !operator_requires_explicit_request(step.kind);
    });
}

bool chain_is_lossless(const OperatorChain& chain) noexcept
{
    return std::all_of(chain.steps.begin(), chain.steps.end(),
                       [](const OperatorStep& step) { return operator_is_lossless(step.kind); });
}

std::vector<std::byte> serialize_operator_chain(const OperatorChain& chain)
{
    std::vector<std::byte> bytes;
    bytes::put_integer(bytes, chain.steps.size(), 4);
    for (const auto& step : chain.steps)
    {
        bytes::put_integer(bytes, static_cast<std::uint64_t>(step.kind), 1);
        bytes::put_integer(bytes, step.explicitly_requested ? 1U : 0U, 1);
        bytes::put_rational(bytes, step.scale);
        bytes::put_rational(bytes, step.offset);
        bytes::put_integer(bytes, step.component_map.size(), 4);
        for (const auto component : step.component_map) bytes::put_integer(bytes, component, 4);
        bytes::put_string(bytes, step.algorithm);
        bytes::put_integer(bytes, static_cast<std::uint64_t>(step.conservation), 1);
    }
    return bytes;
}

std::uint64_t operator_chain_hash(const OperatorChain& chain)
{
    const auto bytes = serialize_operator_chain(chain);
    return bytes::fnv1a(bytes);
}
} // namespace fuse
