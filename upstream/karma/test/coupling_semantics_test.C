// Versioned semantic model: exact rational time, canonical serialization,
// structural validation, and adaptation of the current the existing specification specs.
//
// Initializes neither PETSc nor MPI: the semantic model is deliberately free of
// storage handles, and the the existing specification adapter operates on plain specification
// records.
#include "karma_semantics.h"
#include "karma_semantics_adapter.h"

#include <cstdint>
#include <iostream>
#include <limits>
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

Rational rational(std::int64_t numerator, std::int64_t denominator)
{
    const auto result = make_rational(numerator, denominator);
    require(result.value.has_value(), "fixture rational was rejected");
    return *result.value;
}

void rational_arithmetic()
{
    require(rational(4, 8) == rational(1, 2), "rationals are not normalized on construction");
    require(rational(1, -2) == rational(-1, 2), "negative denominators are not canonicalized");
    require(rational(0, 5) == rational(0, 1), "zero is not canonicalized");
    require(make_rational(1, 0).error == RationalError::ZeroDenominator, "zero denominator accepted");
    require(!make_rational(1, 0).value.has_value(), "rejected rational still produced a value");

    require(*rational_add(rational(1, 3), rational(1, 6)).value == rational(1, 2), "addition is wrong");
    require(*rational_subtract(rational(1, 2), rational(1, 8)).value == rational(3, 8), "subtraction is wrong");
    require(*rational_multiply(rational(2, 3), rational(3, 4)).value == rational(1, 2), "multiplication is wrong");

    // Overflow is reported, never wrapped.
    const auto large = rational(std::numeric_limits<std::int64_t>::max(), 1);
    require(rational_add(large, rational(1, 1)).error == RationalError::Overflow, "addition overflow not detected");
    require(rational_multiply(large, large).error == RationalError::Overflow,
            "multiplication overflow not detected");
    require(rational_add(rational(1, 3037000493LL), rational(1, 3037000507LL)).error == RationalError::Overflow,
            "denominator product overflow not detected");
    // 2^62 + (2^62 + 1) overflows to int64_min + 1, a value normalization would
    // accept, so only an explicit check on the addition itself catches it.
    require(rational_add(rational(4611686018427387904LL, 1), rational(4611686018427387905LL, 1)).error ==
            RationalError::Overflow, "numerator addition overflow not detected");
    require(rational_subtract(rational(-4611686018427387904LL, 1), rational(4611686018427387905LL, 1)).error ==
            RationalError::Overflow, "numerator subtraction overflow not detected");

    // Comparison is exact where the cross products genuinely exceed 64 bits:
    // 1/2 against a value a hair above it whose cross product is int64 max + 1.
    const auto just_above_half = rational(4611686018427387904LL, 9223372036854775807LL);
    require(rational_compare(rational(1, 2), just_above_half) == std::strong_ordering::less,
            "comparison overflowed instead of comparing exactly");
    require(rational_compare(just_above_half, rational(1, 2)) == std::strong_ordering::greater,
            "comparison is not antisymmetric across the 64-bit boundary");
    require(rational_compare(rational(1, 3037000493LL), rational(1, 3037000507LL)) == std::strong_ordering::greater,
            "large-denominator comparison is wrong");
    require(rational_compare(rational(-1, 2), rational(1, 3)) == std::strong_ordering::less,
            "signed comparison is wrong");
    require(rational_compare(rational(2, 4), rational(1, 2)) == std::strong_ordering::equal,
            "equal values do not compare equal");
}

void dyadic_conversion()
{
    // Every scheduler position is dyadic, so conversion is lossless.
    require(*rational_from_dyadic(0.125).value == rational(1, 8), "0.125 did not convert exactly");
    require(*rational_from_dyadic(0.0625).value == rational(1, 16), "0.0625 did not convert exactly");
    require(*rational_from_dyadic(0.0).value == rational(0, 1), "zero did not convert");
    require(*rational_from_dyadic(-3.0).value == rational(-3, 1), "negative integer did not convert");
    require(*rational_from_dyadic(2.5).value == rational(5, 2), "2.5 did not convert exactly");

    // Every finite double is dyadic, so the decimal literal converts to its
    // true value rather than to the 1/10 the author of the literal meant.
    require(*rational_from_dyadic(0.1).value == rational(3602879701896397LL, 36028797018963968LL),
            "0.1 did not convert to the exact value of its double");
    // The bounded form is the guard that matters: a denominator near 2^55
    // overflows on the first addition, so it is refused up front.
    require(rational_from_dyadic_bounded(0.1, 32).error == RationalError::NotDyadic,
            "a 55-bit denominator passed a 32-bit bound");
    require(*rational_from_dyadic_bounded(0.125, 32).value == rational(1, 8),
            "a clean dyadic position was refused by the bound");
    require(*rational_from_dyadic_bounded(3.0 / 1024.0, 32).value == rational(3, 1024),
            "a halved dyadic step was refused by the bound");
    require(rational_add(*rational_from_dyadic(0.1).value, *rational_from_dyadic(0.1).value).error ==
            RationalError::Overflow, "adding two 55-bit-denominator times did not overflow");
    require(rational_from_dyadic(std::numeric_limits<double>::quiet_NaN()).error == RationalError::NotFinite,
            "NaN was accepted");
    require(rational_from_dyadic(std::numeric_limits<double>::infinity()).error == RationalError::NotFinite,
            "infinity was accepted");
    require(rational_from_dyadic(0x1p-70).error == RationalError::Overflow,
            "a denominator too large for int64 was accepted");

    require(rational_to_double(rational(1, 8)) == 0.125, "diagnostic conversion back to double is wrong");
}

void intervals()
{
    const ExactInterval ordered{rational(1, 8), rational(1, 4)};
    require(exact_interval_ordered(ordered), "an ordered interval was rejected");
    require(*exact_interval_length(ordered).value == rational(1, 8), "interval length is wrong");
    require(!exact_interval_ordered({rational(1, 4), rational(1, 8)}), "a reversed interval was accepted");
    const ExactInterval degenerate{rational(1, 4), rational(1, 4)};
    require(exact_interval_ordered(degenerate) && *exact_interval_length(degenerate).value == rational(0, 1),
            "a zero-length interval was rejected");
}

FieldSemantics temperature_field()
{
    FieldSemantics field;
    field.id = "solid.temperature";
    field.version = {1, 0, 0};
    field.units.temperature = 1;
    field.frame = SemanticFrame::Global;
    field.shape = SemanticShape::Scalar;
    field.components = 1;
    field.location = SemanticLocation::FiniteElementCoefficient;
    field.continuity = SemanticContinuity::Continuous;
    field.basis_degree = 1;
    field.temporal = SemanticTemporal::InstantaneousCommitted;
    field.role = SemanticRole::Prognostic;
    field.ownership = SemanticOwnership::Owned;
    field.access = SemanticAccess::WriteOnly;
    field.update = SemanticUpdate::Replace;
    return field;
}

ComponentSemantics two_port_component()
{
    ComponentSemantics component;
    component.id = "solid";
    component.version = {2, 1, 0};
    component.supports_fractional_steps = true;
    component.supports_checkpoint_restore = true;
    PortSemantics output;
    output.id = "state";
    output.direction = PortSemanticDirection::Output;
    output.field = temperature_field();
    PortSemantics input;
    input.id = "coupled";
    input.direction = PortSemanticDirection::Input;
    input.field = temperature_field();
    input.field.id = "fluid.temperature";
    input.field.role = SemanticRole::Input;
    input.field.access = SemanticAccess::ReadOnly;
    input.requirement = {"fluid.temperature", true, {1, 0, 0}, {1, 99, 99}, 0, {}};
    component.ports = {output, input};
    return component;
}

void canonical_identity()
{
    const auto component = two_port_component();
    auto reordered = component;
    std::swap(reordered.ports[0], reordered.ports[1]);
    require(semantic_hash(serialize_component_semantics(component)) ==
            semantic_hash(serialize_component_semantics(reordered)),
            "port container order changed semantic identity");

    // Identity is defined by meaning, so every semantic field participates.
    auto renamed = component;
    renamed.ports[0].field.id = "solid.temperature.renamed";
    auto rescaled = component;
    rescaled.ports[0].field.units.scale = rational(1000, 1);
    auto redimensioned = component;
    redimensioned.ports[0].field.units.temperature = 2;
    auto reversioned = component;
    reversioned.ports[0].field.version = {1, 1, 0};
    auto retimed = component;
    retimed.ports[0].field.temporal = SemanticTemporal::IntervalAverage;
    auto reaccessed = component;
    reaccessed.ports[0].field.access = SemanticAccess::ReadWrite;
    const auto baseline = semantic_hash(serialize_component_semantics(component));
    for (const auto& variant : {renamed, rescaled, redimensioned, reversioned, retimed, reaccessed})
        require(semantic_hash(serialize_component_semantics(variant)) != baseline,
                "a semantic difference did not change identity");

    // Length prefixes must prevent collisions between adjacent strings. A
    // schedule node writes its id directly before its kind, so these two nodes
    // concatenate to identical bytes unless both strings are prefixed.
    const ExactInterval unit{rational(0, 1), rational(1, 1)};
    const ScheduleNodeSemantics split_left{"ab", "c", unit, {}};
    const ScheduleNodeSemantics split_right{"a", "bc", unit, {}};
    require(semantic_hash(serialize_schedule_node_semantics(split_left)) !=
            semantic_hash(serialize_schedule_node_semantics(split_right)),
            "adjacent strings are not length-prefixed");
    require(semantic_hash(serialize_schedule_node_semantics(split_left)) ==
            semantic_hash(serialize_schedule_node_semantics({"ab", "c", unit, {}})),
            "schedule node serialization is not deterministic");
}

void optional_metadata()
{
    auto component = two_port_component();
    const auto baseline = semantic_hash(serialize_component_semantics(component));

    // An unknown key is preserved verbatim and participates in identity.
    component.ports[0].metadata.push_back({"vendor.unknown", {std::byte{1}, std::byte{2}}});
    component.ports[0].metadata.push_back({"another.unknown", {std::byte{3}}});
    const auto extended = semantic_hash(serialize_component_semantics(component));
    require(extended != baseline, "unknown optional metadata was dropped from identity");
    require(validate_component_semantics(component).ok(), "unknown optional metadata failed validation");

    auto shuffled = component;
    std::swap(shuffled.ports[0].metadata[0], shuffled.ports[0].metadata[1]);
    require(semantic_hash(serialize_component_semantics(shuffled)) == extended,
            "metadata container order changed identity");
}

void structural_validation()
{
    require(validate_component_semantics(two_port_component()).ok(), "a valid component was rejected");

    const auto rejects = [](const ComponentSemantics& component, SemanticDiagnosticCode code, const char* message) {
        const auto validation = validate_component_semantics(component);
        require(!validation.ok(), message);
        bool found = false;
        for (const auto& diagnostic : validation.diagnostics) found = found || diagnostic.code == code;
        require(found, message);
    };

    auto empty_id = two_port_component();
    empty_id.id.clear();
    rejects(empty_id, SemanticDiagnosticCode::EmptyIdentifier, "empty component id accepted");

    auto duplicate_ports = two_port_component();
    duplicate_ports.ports[1].id = duplicate_ports.ports[0].id;
    rejects(duplicate_ports, SemanticDiagnosticCode::DuplicatePortId, "duplicate port ids accepted");

    auto zero_components = two_port_component();
    zero_components.ports[0].field.components = 0;
    rejects(zero_components, SemanticDiagnosticCode::ZeroComponents, "zero-component field accepted");

    auto shape_mismatch = two_port_component();
    shape_mismatch.ports[0].field.components = 3;
    rejects(shape_mismatch, SemanticDiagnosticCode::ShapeComponentMismatch, "scalar with 3 components accepted");

    auto bad_scale = two_port_component();
    bad_scale.ports[0].field.units.scale = {1, 0};
    rejects(bad_scale, SemanticDiagnosticCode::NonfiniteScale, "zero-denominator unit scale accepted");

    auto inverted = two_port_component();
    inverted.ports[1].requirement.minimum_version = {2, 0, 0};
    inverted.ports[1].requirement.maximum_version = {1, 0, 0};
    rejects(inverted, SemanticDiagnosticCode::RequirementVersionInverted, "inverted version interval accepted");

    auto empty_key = two_port_component();
    empty_key.ports[0].metadata.push_back({"", {}});
    rejects(empty_key, SemanticDiagnosticCode::EmptyMetadataKey, "empty metadata key accepted");

    auto duplicate_key = two_port_component();
    duplicate_key.ports[0].metadata.push_back({"vendor.k", {}});
    duplicate_key.ports[0].metadata.push_back({"vendor.k", {std::byte{1}}});
    rejects(duplicate_key, SemanticDiagnosticCode::DuplicateMetadataKey, "duplicate metadata key accepted");

    auto bad_enum = two_port_component();
    bad_enum.ports[0].field.location = static_cast<SemanticLocation>(200);
    rejects(bad_enum, SemanticDiagnosticCode::InvalidEnumValue, "out-of-range enum accepted");

    auto unmatched = two_port_component();
    unmatched.ports[1].requirement.field_id = "absent.field";
    rejects(unmatched, SemanticDiagnosticCode::RequirementFieldMismatch,
            "requirement naming another field accepted");
}

// The unit grammar. What matters is not that it parses a long list, but that
// two spellings of one quantity agree, that a compound expression is readable
// at all, and that anything it cannot read FAILS rather than degrading to
// dimensionless - because dimensionless is a claim, and a wrong one.
void unit_expressions()
{
    const auto parse = [](std::string_view text) { return parse_unit_expression(text); };
    const auto dims = [&](std::string_view text) {
        const auto parsed = parse(text);
        require(parsed.has_value(), "a unit expression that should parse did not");
        return *parsed;
    };

    // The reason this exists: a list of spellings could not read this one.
    const auto density = dims("kg/m^3");
    require(density.mass == 1 && density.length == -3 && density.time == 0,
            "kg/m^3 did not parse as a density");

    // Two spellings of one quantity must agree. Comparing strings called these
    // different, which is the defect the dimensional model removes.
    require(unit_dimensions_match(dims("m/s"), dims("m s^-1")), "m/s and m s^-1 disagree");
    require(unit_dimensions_match(dims("m*s^-1"), dims("m s^-1")), "explicit and implicit product disagree");

    // A derived unit expands to base SI, so it compares equal to its expansion.
    require(unit_dimensions_match(dims("N"), dims("kg m s^-2")), "N is not kg m s^-2");
    require(unit_dimensions_match(dims("Pa"), dims("kg m^-1 s^-2")), "Pa is not kg/(m s^2)");
    require(unit_dimensions_match(dims("J"), dims("N m")), "J is not N m");
    require(unit_dimensions_match(dims("W"), dims("J/s")), "W is not J/s");

    // Each `/` divides the term that follows it, not everything after it.
    const auto coefficient = dims("W/m^2/K");
    require(coefficient.mass == 1 && coefficient.time == -3 && coefficient.temperature == -1 &&
                coefficient.length == 0,
            "W/m^2/K did not parse as a heat transfer coefficient");

    // Dimensionless, and a reciprocal built from it.
    require(unit_dimensions_match(dims("1"), UnitDimensions{}), "1 is not dimensionless");
    require(dims("1/s").time == -1, "1/s is not a reciprocal time");
    require(unit_dimensions_match(dims("1/s"), dims("Hz")), "1/s and Hz disagree");

    // Refusals. Each of these previously became dimensionless, which is a
    // claim about the quantity rather than an admission of ignorance.
    for (const auto* rejected : {"", " ", "furlong", "m^", "m^x", "m/", "m*", "m2", "kg/", "^2"})
        require(!parse(rejected).has_value(), "an unreadable unit expression was accepted");

    // Metric prefixes are refused deliberately: `m` is both metre and milli-,
    // so reading `km` as metres would be wrong by a factor of a thousand.
    require(!parse("km").has_value(), "a metric prefix was silently accepted");

    // An exponent large enough to overflow the int8_t dimension is refused
    // rather than wrapped, because a wrapped exponent names another quantity.
    require(!parse("m^300").has_value(), "an out-of-range exponent was accepted");
    // The dimensionless unit is the case the accumulation guard CANNOT catch:
    // every exponent of `1` is still zero, so only the magnitude bound refuses
    // this. Found by mutating that bound away and watching nothing fail.
    require(!parse("1^300").has_value(), "an out-of-range exponent on a dimensionless unit was accepted");

    // A `/` applies to the term that follows it and to nothing else, so a
    // space-separated term after a quotient multiplies rather than divides.
    require(unit_dimensions_match(dims("m/s m"), dims("m^2/s")), "division leaked past the term that followed it");
    require(!parse("m^40*m^40*m^40*m^40").has_value(), "an accumulated exponent overflow was accepted");
}

// Every temporal interpretation the specification records can now express must
// reach the semantic model distinctly; a collapse here would make a real
// disagreement unrefusable.
void temporal_adaptation_is_total()
{
    const std::pair<TimeSemantics, SemanticTemporal> pairs[] = {
        {TimeSemantics::InstantaneousCommitted, SemanticTemporal::InstantaneousCommitted},
        {TimeSemantics::InstantaneousTrial, SemanticTemporal::InstantaneousTrial},
        {TimeSemantics::IntervalAverage, SemanticTemporal::IntervalAverage},
        {TimeSemantics::IntervalIntegral, SemanticTemporal::IntervalIntegral},
    };
    for (const auto& [specified, expected] : pairs)
    {
        FieldSpec specification{"x", "1", 1, FieldRole::Prognostic, FieldLocation::FiniteElementCoefficient,
                                specified};
        const auto adapted = field_semantics_from_spec(specification, PortDirection::Output);
        require(adapted.temporal == expected, "a temporal interpretation did not survive adaptation");
    }
}

void specification_adaptation()
{
    // The specification the current coupler builds for a Gray-Scott component.
    FieldSpec state{"U.state", "1", 1, FieldRole::Prognostic, FieldLocation::FiniteElementCoefficient,
                    TimeSemantics::InstantaneousCommitted};
    FieldSpec coupled{"U.coupled", "1", 1, FieldRole::Input, FieldLocation::FiniteElementCoefficient,
                      TimeSemantics::InstantaneousCommitted};
    ComponentSpec specification{"U",
                                {{"state", PortDirection::Output, false, state},
                                 {"coupled", PortDirection::Input, true, coupled}},
                                true, true};

    const auto adapted = component_semantics_from_spec(specification);
    require(adapted.id == "U" && adapted.ports.size() == 2, "adapter lost component identity");
    require(adapted.supports_fractional_steps && adapted.supports_checkpoint_restore,
            "adapter lost declared capabilities");
    require(validate_component_semantics(adapted).ok(), "adapted the existing specification component failed validation");

    const auto& output = adapted.ports[0].direction == PortSemanticDirection::Output ? adapted.ports[0]
                                                                                     : adapted.ports[1];
    const auto& input = adapted.ports[0].direction == PortSemanticDirection::Output ? adapted.ports[1]
                                                                                    : adapted.ports[0];
    require(output.id == "state" && output.field.id == "U.state", "adapter lost port or field identity");
    require(output.field.role == SemanticRole::Prognostic && input.field.role == SemanticRole::Input,
            "adapter lost the field role");
    require(output.field.access == SemanticAccess::WriteOnly && input.field.access == SemanticAccess::ReadOnly,
            "adapter assigned the wrong access direction");
    require(output.field.location == SemanticLocation::FiniteElementCoefficient, "adapter lost the location");
    require(output.field.shape == SemanticShape::Scalar && output.field.components == 1,
            "adapter lost the shape");
    require(unit_is_dimensionless(output.field.units), "\"1\" did not adapt to a dimensionless unit");
    require(input.requirement.required && input.requirement.field_id == "U.coupled",
            "adapter lost the required-input requirement");
    require(!output.requirement.required, "an optional output became required");

    // An unrecognized unit string is preserved verbatim rather than guessed at.
    auto kelvin = specification;
    kelvin.ports[0].field.units = "K";
    auto exotic = specification;
    exotic.ports[0].field.units = "furlong/fortnight";
    const auto adapted_kelvin = component_semantics_from_spec(kelvin);
    const auto adapted_exotic = component_semantics_from_spec(exotic);
    const auto& kelvin_port = adapted_kelvin.ports[0].id == "state" ? adapted_kelvin.ports[0] : adapted_kelvin.ports[1];
    const auto& exotic_port = adapted_exotic.ports[0].id == "state" ? adapted_exotic.ports[0] : adapted_exotic.ports[1];
    require(kelvin_port.field.units.temperature == 1, "\"K\" did not adapt to a temperature dimension");
    bool preserved = false;
    for (const auto& entry : exotic_port.field.metadata)
        preserved = preserved || (entry.key == "karma.units.unrecognized" && entry.value.size() == 17);
    require(preserved, "an unrecognized unit literal was not preserved");
    require(exotic_port.field.units.length == 0 && exotic_port.field.units.time == 0,
            "an unrecognized unit literal was guessed at");

    // Adaptation is deterministic.
    require(semantic_hash(serialize_component_semantics(adapted)) ==
            semantic_hash(serialize_component_semantics(component_semantics_from_spec(specification))),
            "adaptation is not deterministic");

    // scheduler positions convert exactly.
    require(*exact_time(0.125).value == rational(1, 8), "a the existing specification position did not convert exactly");
    const auto interval = exact_interval({0.0, 0.125});
    require(interval.error == RationalError::None && interval.value.begin == rational(0, 1) &&
            interval.value.end == rational(1, 8), "a the existing specification interval did not convert exactly");
    require(exact_time(0.1).error == RationalError::NotDyadic,
            "a non-dyadic position was accepted by the adapter");
}
}

int main()
{
    try
    {
        rational_arithmetic();
        dyadic_conversion();
        intervals();
        canonical_identity();
        optional_metadata();
        structural_validation();
        unit_expressions();
        temporal_adaptation_is_total();
        specification_adaptation();
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
