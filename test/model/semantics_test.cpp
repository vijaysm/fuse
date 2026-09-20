// Ported from the spine's semantics test. Five of its nine groups survive: its
// rational-arithmetic and dyadic-conversion groups tested a rational type this
// library replaced (fuse::Rational carries its own suite), and its two adaptation
// groups test an adapter header that arrives with compatibility inference at S2.4.
//
// DROPPED CAPABILITY, recorded rather than lost: the spine could convert a double
// to an exact rational dyadically (`rational_from_dyadic`), which is how a legacy
// code's floating-point dt would enter. FUSE has no double time at any ABI (I1), so
// nothing calls it today; if a participant ABI ever needs it, it returns as an
// explicit, refusing conversion in fuse::base -- not as a silent narrowing.
#include "fuse/model/semantics.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace fuse;

// Upstream's require() threw on first failure; EXPECT_TRUE reports every failure
// in the group (U2) and carries the same message.
void require(bool condition, const char* message) { EXPECT_TRUE(condition) << message; }

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

void intervals()
{
    // The spine's helper returned a RationalResult; fuse::Rational's arithmetic is
    // total for these values and refuses instead of returning an error code, so the
    // checks read directly. Same three properties: ordered, length, degenerate.
    const ExactInterval ordered{Rational(1, 8), Rational(1, 4)};
    require(ordered.ordered(), "an ordered interval was rejected");
    require(ordered.length() == Rational(1, 8), "interval length is wrong");
    require(!(ExactInterval{Rational(1, 4), Rational(1, 8)}).ordered(),
            "a reversed interval was accepted");
    const ExactInterval degenerate{Rational(1, 4), Rational(1, 4)};
    require(degenerate.ordered() && degenerate.length() == Rational(0),
            "a zero-length interval was rejected");
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
    rescaled.ports[0].field.units.scale = Rational(1000, 1);
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
    const ExactInterval unit{Rational(0, 1), Rational(1, 1)};
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

    // Upstream wrote {1, 0} here, a zero denominator. fuse::Rational cannot hold one
    // -- its constructor refuses it -- so that case is now unconstructible and the
    // validator's check for it was unreachable. The reachable defect is a ZERO
    // SCALE: dimensionally valid, and it turns every converted value into zero.
    auto bad_scale = two_port_component();
    bad_scale.ports[0].field.units.scale = Rational{0};
    rejects(bad_scale, SemanticDiagnosticCode::NonfiniteScale, "zero unit scale accepted");

    // REVIEW.md R-19: a unit's scale relative to SI base units is never
    // negative; distinct from the zero/annihilator case just above.
    auto negative_scale = two_port_component();
    negative_scale.ports[0].field.units.scale = Rational{-1};
    rejects(negative_scale, SemanticDiagnosticCode::NegativeScale, "negative unit scale accepted");

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
    require(unit_dimensions_match(dims("1"), Units{}), "1 is not dimensionless");
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

TEST(Semantics, Intervals) { intervals(); }
TEST(Semantics, CanonicalIdentity) { canonical_identity(); }
TEST(Semantics, OptionalMetadata) { optional_metadata(); }
TEST(Semantics, StructuralValidation) { structural_validation(); }
TEST(Semantics, UnitExpressions) { unit_expressions(); }

}  // namespace
