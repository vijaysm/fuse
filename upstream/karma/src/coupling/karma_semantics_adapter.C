#include "karma_semantics_adapter.h"

#include <array>
#include <string_view>

namespace KARMA
{
namespace
{
/** The only unit symbols this adapter claims to understand. Anything else is
 * preserved verbatim and left undeclared rather than assumed dimensionless. */
std::vector<std::byte> literal_bytes(std::string_view text)
{
    std::vector<std::byte> bytes;
    bytes.reserve(text.size());
    for (const char character : text) bytes.push_back(static_cast<std::byte>(static_cast<unsigned char>(character)));
    return bytes;
}

SemanticLocation adapt_location(FieldLocation location)
{
    switch (location)
    {
        case FieldLocation::Vertex: return SemanticLocation::Vertex;
        case FieldLocation::Element: return SemanticLocation::Element;
        case FieldLocation::FiniteElementCoefficient: return SemanticLocation::FiniteElementCoefficient;
    }
    return SemanticLocation::Unspecified;
}

SemanticRole adapt_role(FieldRole role)
{
    switch (role)
    {
        case FieldRole::Prognostic: return SemanticRole::Prognostic;
        case FieldRole::Input: return SemanticRole::Input;
    }
    return SemanticRole::Input;
}

SemanticTemporal adapt_temporal(TimeSemantics temporal)
{
    switch (temporal)
    {
        case TimeSemantics::InstantaneousCommitted: return SemanticTemporal::InstantaneousCommitted;
        case TimeSemantics::InstantaneousTrial: return SemanticTemporal::InstantaneousTrial;
        case TimeSemantics::IntervalAverage: return SemanticTemporal::IntervalAverage;
        case TimeSemantics::IntervalIntegral: return SemanticTemporal::IntervalIntegral;
    }
    // Unreachable for a valid enumerator. Committed is the conservative answer:
    // it is the only interpretation that claims nothing about an interval.
    return SemanticTemporal::InstantaneousCommitted;
}
}

FieldSemantics field_semantics_from_spec(const FieldSpec& specification, PortDirection direction)
{
    FieldSemantics field;
    field.id = specification.id;
    // The existing records carry no version, so the adapted record declares the
    // first semantic version rather than inventing a higher one.
    field.version = {1, 0, 0};

    // Parsed as an expression, not matched against a list of spellings. The
    // list could not read `kg/m^3` at all, and called `m/s` and `m s^-1`
    // different quantities. Anything the grammar cannot read is recorded as an
    // undeclared literal rather than quietly becoming dimensionless - the
    // distinction the compatibility checker relies on.
    if (const auto parsed = parse_unit_expression(specification.units); parsed.has_value())
        field.units = *parsed;
    else
        field.metadata.push_back({std::string(unrecognized_units_key), literal_bytes(specification.units)});

    field.frame = SemanticFrame::Global;
    field.components = specification.components;
    field.shape = specification.components == 1 ? SemanticShape::Scalar : SemanticShape::Vector;
    field.location = adapt_location(specification.location);
    // The current coupler transfers only continuous scalar cG fields; degree is
    // not declared in the existing records and is therefore left unspecified.
    field.continuity = specification.location == FieldLocation::FiniteElementCoefficient
                           ? SemanticContinuity::Continuous
                           : SemanticContinuity::Unspecified;
    field.basis_degree = 0;
    field.temporal = adapt_temporal(specification.temporal);
    field.role = adapt_role(specification.role);
    field.ownership = SemanticOwnership::Owned;
    // Access is what the component does with the field: it writes its outputs
    // and reads its inputs.
    field.access = direction == PortDirection::Output ? SemanticAccess::WriteOnly : SemanticAccess::ReadOnly;
    field.update = SemanticUpdate::Replace;
    return field;
}

PortSemantics port_semantics_from_spec(const PortSpec& specification)
{
    PortSemantics port;
    port.id = specification.id;
    port.direction = specification.direction == PortDirection::Output ? PortSemanticDirection::Output
                                                                      : PortSemanticDirection::Input;
    port.field = field_semantics_from_spec(specification.field, specification.direction);
    port.requirement.field_id = specification.field.id;
    port.requirement.required = specification.required;
    port.requirement.minimum_version = {1, 0, 0};
    port.requirement.maximum_version = {1, 99, 99};
    return port;
}

ComponentSemantics component_semantics_from_spec(const ComponentSpec& specification)
{
    ComponentSemantics component;
    component.id = specification.id;
    component.version = {1, 0, 0};
    component.supports_fractional_steps = specification.supports_fractional_steps;
    component.supports_checkpoint_restore = specification.supports_checkpoint_restore;
    component.requires_communicator = true;
    component.ports.reserve(specification.ports.size());
    for (const auto& port : specification.ports) component.ports.push_back(port_semantics_from_spec(port));
    return component;
}

RationalResult exact_time(Real position, unsigned maximum_denominator_bits) noexcept
{
    return rational_from_dyadic_bounded(static_cast<double>(position), maximum_denominator_bits);
}

ExactIntervalResult exact_interval(const TimeInterval& interval) noexcept
{
    const auto begin = exact_time(interval.begin);
    if (!begin.value) return {{}, begin.error};
    const auto end = exact_time(interval.end);
    if (!end.value) return {{}, end.error};
    return {ExactInterval{*begin.value, *end.value}, RationalError::None};
}
} // namespace KARMA
