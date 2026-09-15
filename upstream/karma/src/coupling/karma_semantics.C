#include "karma_semantics.h"

#include <array>

#include "karma_byte_writer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace KARMA
{
namespace
{
static_assert(sizeof(__int128) == 16, "exact rational comparison requires a 128-bit integer type");

constexpr std::int64_t int64_min = std::numeric_limits<std::int64_t>::min();


RationalResult normalized(std::int64_t numerator, std::int64_t denominator) noexcept
{
    if (denominator == 0) return {std::nullopt, RationalError::ZeroDenominator};
    // Negating int64_min overflows, so a value that would need it is rejected
    // rather than silently wrapped into a positive denominator.
    if (numerator == int64_min || denominator == int64_min) return {std::nullopt, RationalError::Overflow};
    if (denominator < 0)
    {
        numerator = -numerator;
        denominator = -denominator;
    }
    const auto divisor = static_cast<std::int64_t>(
        std::gcd(static_cast<std::uint64_t>(numerator < 0 ? -numerator : numerator),
                 static_cast<std::uint64_t>(denominator)));
    if (divisor > 1)
    {
        numerator /= divisor;
        denominator /= divisor;
    }
    return {Rational{numerator, denominator}, RationalError::None};
}

/** Cross-multiply in 128 bits, then require the reduced result to fit in 64. */
RationalResult combine(Rational left, Rational right, bool subtract) noexcept
{
    std::int64_t cross_left = 0, cross_right = 0, denominator = 0;
    if (__builtin_mul_overflow(left.numerator, right.denominator, &cross_left) ||
        __builtin_mul_overflow(right.numerator, left.denominator, &cross_right) ||
        __builtin_mul_overflow(left.denominator, right.denominator, &denominator))
        return {std::nullopt, RationalError::Overflow};
    std::int64_t numerator = 0;
    if (subtract ? __builtin_sub_overflow(cross_left, cross_right, &numerator)
                 : __builtin_add_overflow(cross_left, cross_right, &numerator))
        return {std::nullopt, RationalError::Overflow};
    return normalized(numerator, denominator);
}






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

void put_units(std::vector<std::byte>& bytes, const UnitDimensions& units)
{
    for (const auto exponent : {units.length, units.mass, units.time, units.current, units.temperature,
                                units.amount, units.luminous})
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

RationalResult make_rational(std::int64_t numerator, std::int64_t denominator) noexcept
{
    return normalized(numerator, denominator);
}

RationalResult rational_add(Rational left, Rational right) noexcept { return combine(left, right, false); }

RationalResult rational_subtract(Rational left, Rational right) noexcept { return combine(left, right, true); }

RationalResult rational_multiply(Rational left, Rational right) noexcept
{
    std::int64_t numerator = 0, denominator = 0;
    if (__builtin_mul_overflow(left.numerator, right.numerator, &numerator) ||
        __builtin_mul_overflow(left.denominator, right.denominator, &denominator))
        return {std::nullopt, RationalError::Overflow};
    return normalized(numerator, denominator);
}

std::strong_ordering rational_compare(Rational left, Rational right) noexcept
{
    const auto cross_left = static_cast<__int128>(left.numerator) * static_cast<__int128>(right.denominator);
    const auto cross_right = static_cast<__int128>(right.numerator) * static_cast<__int128>(left.denominator);
    if (cross_left < cross_right) return std::strong_ordering::less;
    if (cross_left > cross_right) return std::strong_ordering::greater;
    return std::strong_ordering::equal;
}

RationalResult rational_from_dyadic(double value) noexcept
{
    if (!std::isfinite(value)) return {std::nullopt, RationalError::NotFinite};
    if (value == 0.0) return {Rational{0, 1}, RationalError::None};

    int exponent = 0;
    const double fraction = std::frexp(value, &exponent);
    // frexp gives value = fraction * 2^exponent with |fraction| in [0.5, 1).
    // Scaling by 2^53 makes the mantissa an exact integer for every finite
    // double, which is why this conversion never rounds.
    constexpr int mantissa_bits = 53;
    const double scaled = std::ldexp(fraction, mantissa_bits);
    auto mantissa = static_cast<std::int64_t>(scaled);
    int power = exponent - mantissa_bits;

    // Reduce the mantissa's trailing zeros into the exponent so that a value
    // needing no denominator does not consume denominator range.
    while (mantissa != 0 && mantissa % 2 == 0)
    {
        mantissa /= 2;
        ++power;
    }
    if (power >= 0)
    {
        std::int64_t numerator = mantissa;
        for (int index = 0; index != power; ++index)
            if (__builtin_mul_overflow(numerator, static_cast<std::int64_t>(2), &numerator))
                return {std::nullopt, RationalError::Overflow};
        return normalized(numerator, 1);
    }
    if (-power > 62) return {std::nullopt, RationalError::Overflow};
    const std::int64_t denominator = static_cast<std::int64_t>(1) << (-power);
    return normalized(mantissa, denominator);
}

RationalResult rational_from_dyadic_bounded(double value, unsigned maximum_denominator_bits) noexcept
{
    const auto exact = rational_from_dyadic(value);
    if (!exact.value) return exact;
    if (maximum_denominator_bits >= 63) return exact;
    const std::int64_t limit = static_cast<std::int64_t>(1) << maximum_denominator_bits;
    if (exact.value->denominator > limit) return {std::nullopt, RationalError::NotDyadic};
    return exact;
}

double rational_to_double(Rational value) noexcept
{
    return static_cast<double>(value.numerator) / static_cast<double>(value.denominator);
}

std::string serialize_rational(Rational value)
{
    return std::to_string(value.numerator) + "/" + std::to_string(value.denominator);
}

bool exact_interval_ordered(const ExactInterval& interval) noexcept
{
    return rational_compare(interval.begin, interval.end) != std::strong_ordering::greater;
}

RationalResult exact_interval_length(const ExactInterval& interval) noexcept
{
    return rational_subtract(interval.end, interval.begin);
}

bool unit_is_dimensionless(const UnitDimensions& units) noexcept
{
    return units.length == 0 && units.mass == 0 && units.time == 0 && units.current == 0 &&
           units.temperature == 0 && units.amount == 0 && units.luminous == 0;
}

namespace
{
struct UnitSymbolEntry
{
    std::string_view symbol;
    std::int8_t length, mass, time, current, temperature, amount, luminous;
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
bool accumulate_term(UnitDimensions& units, const UnitSymbolEntry& entry, int exponent)
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
           add(units.luminous, entry.luminous, exponent);
}
} // namespace

std::optional<UnitDimensions> parse_unit_expression(std::string_view expression)
{
    UnitDimensions units;
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

bool unit_dimensions_match(const UnitDimensions& left, const UnitDimensions& right) noexcept
{
    return left.length == right.length && left.mass == right.mass && left.time == right.time &&
           left.current == right.current && left.temperature == right.temperature &&
           left.amount == right.amount && left.luminous == right.luminous;
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
    if (field.units.scale.denominator == 0 || field.units.offset.denominator == 0)
        diagnostics.push_back({SemanticDiagnosticCode::NonfiniteScale, prefix + ".units"});
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
} // namespace KARMA
