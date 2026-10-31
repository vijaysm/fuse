#include "fuse/model/semantics.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <limits>
#include <span>

#include "fuse/base/bytes.hpp"

/// Ported from the spine, less its rational arithmetic (`fuse::Rational` is the one
/// exact-time type) and less its own unit struct (`fuse::Units` is). What survives
/// is the substance nothing in this library had: the unit EXPRESSION parser, the
/// canonical semantic serializers, the semantic hash, and the per-field validators.

namespace fuse
{
namespace
{
void put_version(std::vector<std::byte>& bytes, SemanticVersion version)
{
    bytes::put_integer(bytes, version.major, 4);
    bytes::put_integer(bytes, version.minor, 4);
    bytes::put_integer(bytes, version.patch, 4);
}

void put_metadata(std::vector<std::byte>& bytes, std::vector<MetadataEntry> metadata)
{
    // Sorted by key so container order cannot change identity. Unknown keys are
    // written verbatim rather than filtered.
    std::sort(metadata.begin(), metadata.end(),
              [](const MetadataEntry& left, const MetadataEntry& right) { return left.key < right.key; });
    bytes::put_integer(bytes, metadata.size(), 4);
    for (const auto& entry : metadata)
    {
        bytes::put_string(bytes, entry.key);
        bytes::put_bytes(bytes, entry.value);
    }
}

void put_units(std::vector<std::byte>& bytes, const Units& units)
{
    for (const auto exponent : {units.length, units.mass, units.time, units.current, units.temperature,
                                units.amount, units.luminosity})
        bytes::put_signed(bytes, exponent, 1);
    bytes::put_rational(bytes, units.scale);
    bytes::put_rational(bytes, units.offset);
}

void put_field(std::vector<std::byte>& bytes, const FieldSemantics& field)
{
    bytes::put_string(bytes, field.id);
    put_version(bytes, field.version);
    put_units(bytes, field.units);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.frame), 1);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.shape), 1);
    bytes::put_integer(bytes, field.components, 4);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.location), 1);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.continuity), 1);
    bytes::put_integer(bytes, field.basis_degree, 4);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.temporal), 1);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.role), 1);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.ownership), 1);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.access), 1);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(field.update), 1);
    put_metadata(bytes, field.metadata);
}

void put_requirement(std::vector<std::byte>& bytes, const FieldRequirement& requirement)
{
    bytes::put_string(bytes, requirement.field_id);
    bytes::put_integer(bytes, requirement.required ? 1U : 0U, 1);
    put_version(bytes, requirement.minimum_version);
    put_version(bytes, requirement.maximum_version);
    bytes::put_integer(bytes, requirement.required_capabilities, 8);
    put_metadata(bytes, requirement.metadata);
}

void put_port(std::vector<std::byte>& bytes, const PortSemantics& port)
{
    bytes::put_string(bytes, port.id);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(port.direction), 1);
    put_field(bytes, port.field);
    put_requirement(bytes, port.requirement);
    put_metadata(bytes, port.metadata);
}

template <typename Enum>
bool in_range(Enum value, std::uint64_t limit) noexcept
{
    return static_cast<std::uint64_t>(value) <= limit;
}

void validate_metadata(const std::vector<MetadataEntry>& metadata, std::string_view path,
                       std::vector<SemanticDiagnostic>& diagnostics)
{
    std::vector<std::string> keys;
    keys.reserve(metadata.size());
    for (const auto& entry : metadata)
    {
        if (entry.key.empty())
            diagnostics.push_back({SemanticDiagnosticCode::EmptyMetadataKey, std::string(path) + ".metadata"});
        keys.push_back(entry.key);
    }
    std::sort(keys.begin(), keys.end());
    if (std::adjacent_find(keys.begin(), keys.end()) != keys.end())
        diagnostics.push_back({SemanticDiagnosticCode::DuplicateMetadataKey, std::string(path) + ".metadata"});
}

bool version_greater(SemanticVersion left, SemanticVersion right) noexcept
{
    if (left.major != right.major) return left.major > right.major;
    if (left.minor != right.minor) return left.minor > right.minor;
    return left.patch > right.patch;
}
}

bool unit_is_dimensionless(const Units& units) noexcept
{
    return units.length == 0 && units.mass == 0 && units.time == 0 && units.current == 0 &&
           units.temperature == 0 && units.amount == 0 && units.luminosity == 0;
}

namespace
{
struct UnitSymbolEntry
{
    std::string_view symbol;
    std::int8_t length, mass, time, current, temperature, amount, luminosity;
};

/** The seven SI base units, the dimensionless unit, and the derived units
 * whose dimensions are fixed by definition. Nothing here carries a scale: a
 * derived SI unit is exactly one in base SI, which is why these are safe to
 * expand and metric prefixes are not. */
constexpr std::array<UnitSymbolEntry, 21> unit_symbols{{
    {"1", 0, 0, 0, 0, 0, 0, 0},
    {"m", 1, 0, 0, 0, 0, 0, 0},
    {"kg", 0, 1, 0, 0, 0, 0, 0},
    {"s", 0, 0, 1, 0, 0, 0, 0},
    {"A", 0, 0, 0, 1, 0, 0, 0},
    {"K", 0, 0, 0, 0, 1, 0, 0},
    {"mol", 0, 0, 0, 0, 0, 1, 0},
    {"cd", 0, 0, 0, 0, 0, 0, 1},
    {"rad", 0, 0, 0, 0, 0, 0, 0},
    {"sr", 0, 0, 0, 0, 0, 0, 0},
    {"Hz", 0, 0, -1, 0, 0, 0, 0},
    {"N", 1, 1, -2, 0, 0, 0, 0},
    {"Pa", -1, 1, -2, 0, 0, 0, 0},
    {"J", 2, 1, -2, 0, 0, 0, 0},
    {"W", 2, 1, -3, 0, 0, 0, 0},
    {"C", 0, 0, 1, 1, 0, 0, 0},
    {"V", 2, 1, -3, -1, 0, 0, 0},
    {"Ohm", 2, 1, -3, -2, 0, 0, 0},
    {"F", -2, -1, 4, 2, 0, 0, 0},
    {"T", 0, 1, -2, -1, 0, 0, 0},
    {"Wb", 2, 1, -2, -1, 0, 0, 0},
}};

const UnitSymbolEntry* lookup_unit_symbol(std::string_view symbol)
{
    for (const auto& entry : unit_symbols)
        if (entry.symbol == symbol) return &entry;
    return nullptr;
}

/** Accumulates one term's contribution with the sign the surrounding operator
 * gives it. Exponents are int8_t, so a product deep enough to overflow one is
 * refused rather than wrapped - an exponent that wraps would silently declare
 * a different physical quantity. */
bool accumulate_term(Units& units, const UnitSymbolEntry& entry, int exponent)
{
    const auto add = [](std::int8_t& target, std::int8_t base_exponent, int power) {
        const int result = static_cast<int>(target) + static_cast<int>(base_exponent) * power;
        if (result < -128 || result > 127) return false;
        target = static_cast<std::int8_t>(result);
        return true;
    };
    return add(units.length, entry.length, exponent) && add(units.mass, entry.mass, exponent) &&
           add(units.time, entry.time, exponent) && add(units.current, entry.current, exponent) &&
           add(units.temperature, entry.temperature, exponent) && add(units.amount, entry.amount, exponent) &&
           add(units.luminosity, entry.luminosity, exponent);
}
} // namespace

std::optional<Units> parse_unit_expression(std::string_view expression)
{
    Units units;
    std::size_t index = 0;
    int sign = 1;
    bool saw_term = false;

    const auto skip_spaces = [&] {
        while (index != expression.size() && expression[index] == ' ') ++index;
    };

    skip_spaces();
    if (index == expression.size()) return std::nullopt; // an empty expression is not dimensionless

    while (index != expression.size())
    {
        // Symbol: a run of letters, or the single digit 1.
        const std::size_t symbol_begin = index;
        if (expression[index] == '1')
            ++index;
        else
            while (index != expression.size() &&
                   ((expression[index] >= 'a' && expression[index] <= 'z') ||
                    (expression[index] >= 'A' && expression[index] <= 'Z')))
                ++index;
        if (index == symbol_begin) return std::nullopt;
        const auto* entry = lookup_unit_symbol(expression.substr(symbol_begin, index - symbol_begin));
        if (entry == nullptr) return std::nullopt;

        // Optional exponent.
        int exponent = 1;
        if (index != expression.size() && expression[index] == '^')
        {
            ++index;
            int exponent_sign = 1;
            if (index != expression.size() && (expression[index] == '-' || expression[index] == '+'))
            {
                exponent_sign = expression[index] == '-' ? -1 : 1;
                ++index;
            }
            const std::size_t digits_begin = index;
            int magnitude = 0;
            while (index != expression.size() && expression[index] >= '0' && expression[index] <= '9')
            {
                magnitude = magnitude * 10 + (expression[index] - '0');
                if (magnitude > 127) return std::nullopt; // refuse rather than wrap
                ++index;
            }
            if (index == digits_begin) return std::nullopt; // `^` with no digits
            exponent = exponent_sign * magnitude;
        }

        if (!accumulate_term(units, *entry, sign * exponent)) return std::nullopt;
        saw_term = true;

        // Operator, or the end.
        skip_spaces();
        if (index == expression.size()) break;
        if (expression[index] == '*')
        {
            sign = 1;
            ++index;
        }
        else if (expression[index] == '/')
        {
            sign = -1;
            ++index;
        }
        else
        {
            // A space already separated the terms, which means multiplication.
            sign = 1;
        }
        skip_spaces();
        if (index == expression.size()) return std::nullopt; // a trailing operator
    }
    if (!saw_term) return std::nullopt;
    return units;
}

bool unit_dimensions_match(const Units& left, const Units& right) noexcept
{
    return left.length == right.length && left.mass == right.mass && left.time == right.time &&
           left.current == right.current && left.temperature == right.temperature &&
           left.amount == right.amount && left.luminosity == right.luminosity;
}

std::vector<std::byte> serialize_field_semantics(const FieldSemantics& field)
{
    std::vector<std::byte> bytes;
    put_field(bytes, field);
    return bytes;
}

std::vector<std::byte> serialize_port_semantics(const PortSemantics& port)
{
    std::vector<std::byte> bytes;
    put_port(bytes, port);
    return bytes;
}

std::vector<std::byte> serialize_component_semantics(const ComponentSemantics& component)
{
    std::vector<std::byte> bytes;
    bytes::put_string(bytes, component.id);
    put_version(bytes, component.version);
    bytes::put_integer(bytes, component.capabilities, 8);
    bytes::put_integer(bytes, component.supports_fractional_steps ? 1U : 0U, 1);
    bytes::put_integer(bytes, component.supports_checkpoint_restore ? 1U : 0U, 1);
    bytes::put_integer(bytes, component.requires_communicator ? 1U : 0U, 1);

    auto ports = component.ports;
    std::sort(ports.begin(), ports.end(),
              [](const PortSemantics& left, const PortSemantics& right) { return left.id < right.id; });
    bytes::put_integer(bytes, ports.size(), 4);
    for (const auto& port : ports) put_port(bytes, port);
    put_metadata(bytes, component.metadata);
    return bytes;
}

std::vector<std::byte> serialize_transform_semantics(const TransformSemantics& transform)
{
    std::vector<std::byte> bytes;
    bytes::put_integer(bytes, static_cast<std::uint64_t>(transform.kind), 1);
    bytes::put_integer(bytes, transform.explicitly_requested ? 1U : 0U, 1);
    bytes::put_rational(bytes, transform.scale);
    bytes::put_rational(bytes, transform.offset);
    bytes::put_integer(bytes, transform.component_map.size(), 4);
    for (const auto entry : transform.component_map) bytes::put_integer(bytes, entry, 4);
    put_metadata(bytes, transform.metadata);
    return bytes;
}

std::vector<std::byte> serialize_transfer_semantics(const TransferSemantics& transfer)
{
    std::vector<std::byte> bytes;
    bytes::put_integer(bytes, static_cast<std::uint64_t>(transfer.kind), 1);
    bytes::put_integer(bytes, static_cast<std::uint64_t>(transfer.conservation), 1);
    bytes::put_string(bytes, transfer.algorithm);
    put_metadata(bytes, transfer.metadata);
    return bytes;
}

std::vector<std::byte> serialize_schedule_node_semantics(const ScheduleNodeSemantics& node)
{
    std::vector<std::byte> bytes;
    bytes::put_string(bytes, node.id);
    bytes::put_string(bytes, node.kind);
    bytes::put_rational(bytes, node.interval.begin);
    bytes::put_rational(bytes, node.interval.end);
    put_metadata(bytes, node.metadata);
    return bytes;
}

std::uint64_t semantic_hash(std::span<const std::byte> bytes) noexcept
{
    return bytes::fnv1a(bytes);
}

SemanticValidation validate_field_semantics(const FieldSemantics& field, std::string_view path)
{
    SemanticValidation validation;
    auto& diagnostics = validation.diagnostics;
    const std::string prefix(path);
    if (field.id.empty()) diagnostics.push_back({SemanticDiagnosticCode::EmptyIdentifier, prefix + ".id"});
    if (field.components == 0) diagnostics.push_back({SemanticDiagnosticCode::ZeroComponents, prefix + ".components"});
    if (field.shape == SemanticShape::Scalar && field.components != 1)
        diagnostics.push_back({SemanticDiagnosticCode::ShapeComponentMismatch, prefix + ".shape"});
    // No zero-denominator check is needed here: fuse::Rational cannot have one
    // (its constructor refuses it). The real defect is a ZERO scale: it is not a unit, it is an
    // annihilator, and it would turn every converted value into 0 while passing every dimensional
    // check.
    if (field.units.scale == Rational{0})
        diagnostics.push_back({SemanticDiagnosticCode::NonfiniteScale, prefix + ".units"});
    // A scale's SIGN is checked too: a unit's scale relative to SI base
    // units is never negative. Left
    // unchecked, check_units() (compatibility.cpp) could auto-insert an
    // unrequested UnitScale operator that flips the sign of every value
    // crossing the channel, while reporting the connection as an ordinary
    // successful conversion.
    else if (field.units.scale < Rational{0})
        diagnostics.push_back({SemanticDiagnosticCode::NegativeScale, prefix + ".units"});
    if (!in_range(field.shape, 2) || !in_range(field.location, 3) || !in_range(field.continuity, 2) ||
        !in_range(field.temporal, 3) || !in_range(field.frame, 2) || !in_range(field.ownership, 2) ||
        !in_range(field.access, 2) || !in_range(field.update, 1) || !in_range(field.role, 3))
        diagnostics.push_back({SemanticDiagnosticCode::InvalidEnumValue, prefix});
    validate_metadata(field.metadata, prefix, diagnostics);
    return validation;
}

SemanticValidation validate_component_semantics(const ComponentSemantics& component)
{
    SemanticValidation validation;
    auto& diagnostics = validation.diagnostics;
    if (component.id.empty())
        diagnostics.push_back({SemanticDiagnosticCode::EmptyIdentifier, component.id + ".id"});

    std::vector<std::string> port_ids;
    port_ids.reserve(component.ports.size());
    for (const auto& port : component.ports)
    {
        const std::string path = component.id + ".ports[" + port.id + "]";
        if (port.id.empty()) diagnostics.push_back({SemanticDiagnosticCode::EmptyIdentifier, path + ".id"});
        port_ids.push_back(port.id);
        if (!in_range(port.direction, 1))
            diagnostics.push_back({SemanticDiagnosticCode::InvalidEnumValue, path + ".direction"});

        const auto field_validation = validate_field_semantics(port.field, path + ".field");
        diagnostics.insert(diagnostics.end(), field_validation.diagnostics.begin(),
                           field_validation.diagnostics.end());

        if (version_greater(port.requirement.minimum_version, port.requirement.maximum_version))
            diagnostics.push_back({SemanticDiagnosticCode::RequirementVersionInverted, path + ".requirement"});
        // A requirement names the field it constrains; naming another field
        // would let a port advertise one meaning and demand a different one.
        if (!port.requirement.field_id.empty() && port.requirement.field_id != port.field.id)
            diagnostics.push_back({SemanticDiagnosticCode::RequirementFieldMismatch, path + ".requirement"});
        validate_metadata(port.requirement.metadata, path + ".requirement", diagnostics);
        validate_metadata(port.metadata, path, diagnostics);
    }
    std::sort(port_ids.begin(), port_ids.end());
    if (std::adjacent_find(port_ids.begin(), port_ids.end()) != port_ids.end())
        diagnostics.push_back({SemanticDiagnosticCode::DuplicatePortId, component.id + ".ports"});
    validate_metadata(component.metadata, component.id, diagnostics);
    return validation;
}
} // namespace fuse
