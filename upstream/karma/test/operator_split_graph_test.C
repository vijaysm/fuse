#include "karma_operator_split_graph.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
KARMA::ComponentSpec component(const std::string& id,
                               const std::string& output,
                               const std::string& input,
                               bool fractional = true)
{
    const KARMA::FieldSpec field{id + ".state", "1", 1, KARMA::FieldRole::Prognostic,
                                 KARMA::FieldLocation::FiniteElementCoefficient,
                                 KARMA::TimeSemantics::InstantaneousCommitted};
    const KARMA::FieldSpec coupled{id + ".coupled", "1", 1, KARMA::FieldRole::Input,
                                   KARMA::FieldLocation::FiniteElementCoefficient,
                                   KARMA::TimeSemantics::InstantaneousCommitted};
    return {id,
            {{output, KARMA::PortDirection::Output, false, field},
             {input, KARMA::PortDirection::Input, true, coupled}},
            fractional,
            true};
}

KARMA::OperatorSplitGraphSpec graph(KARMA::SplitScheme scheme = KARMA::SplitScheme::Lie)
{
    KARMA::OperatorSplitGraphSpec specification;
    specification.id = "gray-scott";
    specification.components = std::array<KARMA::ComponentInstanceSpec, 2>{
        KARMA::ComponentInstanceSpec{"U", component("U", "state", "coupled_v")},
        KARMA::ComponentInstanceSpec{"V", component("V", "state", "coupled_u")}};
    specification.connections = std::array<KARMA::ConnectionSpec, 2>{
        KARMA::ConnectionSpec{"U", "state", "V", "coupled_u"},
        KARMA::ConnectionSpec{"V", "state", "U", "coupled_v"}};
    specification.scheme = scheme;
    return specification;
}

void expect_invalid(const KARMA::OperatorSplitGraphSpec& spec)
{
    try
    {
        (void)KARMA::compile_operator_split_graph(spec);
    }
    catch (const std::invalid_argument&)
    {
        return;
    }
    throw std::runtime_error("invalid graph was accepted");
}
}

// The semantic model is consulted when a connection is admitted, so a graph
// whose two ends disagree about MEANING is refused at compile time rather
// than producing wrong numbers at run time.
void semantically_incompatible_connections_are_refused()
{
    // Units: the source is a concentration per second, the target a bare
    // scalar. Nothing about the names, shapes or locations differs.
    auto units = graph();
    units.components[0].component.ports[1].field.units = "kg/m^3";
    expect_invalid(units);

    // Component count: a 3-component vector cannot silently feed a scalar.
    auto components = graph();
    components.components[0].component.ports[1].field.components = 3;
    expect_invalid(components);

    // Location: a per-element value is not a finite-element coefficient, and
    // reading one as the other is exactly the error this check exists for.
    auto location = graph();
    location.components[0].component.ports[1].field.location = KARMA::FieldLocation::Element;
    expect_invalid(location);

    // Temporal interpretation: an interval average is not an instantaneous
    // value even when every other property agrees. This case was previously
    // inexpressible - `TimeSemantics` had a single enumerator - and the
    // vocabulary was widened so that a real disagreement could be refused
    // rather than merely described.
    auto temporal = graph();
    temporal.components[0].component.ports[1].field.temporal = KARMA::TimeSemantics::IntervalAverage;
    expect_invalid(temporal);

    // And a trial iterate is not a committed value, which is the distinction
    // a fixed-point sweep depends on.
    auto trial = graph();
    trial.components[0].component.ports[1].field.temporal = KARMA::TimeSemantics::InstantaneousTrial;
    expect_invalid(trial);

    // Units now compare as dimensions, so two spellings of one quantity must
    // NOT be refused - the check that replaced string equality has to be
    // shown not to have inherited its defect.
    auto spelled_differently = graph();
    spelled_differently.components[0].component.ports[0].field.units = "m/s";
    spelled_differently.components[1].component.ports[1].field.units = "m s^-1";
    (void)KARMA::compile_operator_split_graph(spelled_differently);

    // And the baseline still compiles, so the four above are refusals of a
    // real disagreement rather than a check that refuses everything.
    (void)KARMA::compile_operator_split_graph(graph());
}

// The fingerprint reports what a component's ports MEAN, which the declared
// mesh/layout/basis identifiers do not.
void the_contract_fingerprint_tracks_meaning()
{
    const auto baseline = KARMA::compile_operator_split_graph(graph()).report();
    const auto again = KARMA::compile_operator_split_graph(graph()).report();
    if (baseline.components[0].contract_fingerprint != again.components[0].contract_fingerprint)
        throw std::runtime_error("the contract fingerprint is not deterministic");
    if (baseline.components[0].contract_fingerprint == 0)
        throw std::runtime_error("the contract fingerprint was never computed");
    if (baseline.components[0].contract_fingerprint == baseline.components[1].contract_fingerprint)
        throw std::runtime_error("two different components share a contract fingerprint");

    // A declared identifier that the graph does not interpret must NOT move
    // it; a change of meaning must.
    auto relabelled = graph();
    relabelled.components[0].mesh_id = "a completely different mesh";
    const auto relabelled_report = KARMA::compile_operator_split_graph(relabelled).report();
    if (relabelled_report.components[0].contract_fingerprint != baseline.components[0].contract_fingerprint)
        throw std::runtime_error("a non-semantic identifier moved the contract fingerprint");

    auto rerolled = graph();
    rerolled.components[0].component.ports[0].field.units = "kg/m^3";
    rerolled.components[1].component.ports[1].field.units = "kg/m^3";
    const auto rerolled_report = KARMA::compile_operator_split_graph(rerolled).report();
    if (rerolled_report.components[0].contract_fingerprint == baseline.components[0].contract_fingerprint)
        throw std::runtime_error("a change of units left the contract fingerprint unmoved");
}

int main()
{
    try
    {
        semantically_incompatible_connections_are_refused();
        the_contract_fingerprint_tracks_meaning();
        const auto lie = KARMA::compile_operator_split_graph(graph());
        if (lie.metadata().id != "gray-scott" || lie.instructions().size() != 7) return 1;
        if (lie.instructions()[1].kind != KARMA::InstructionKind::Exchange ||
            lie.instructions()[2].kind != KARMA::InstructionKind::Advance ||
            lie.instructions()[2].interval != KARMA::AdvanceInterval::Full ||
            lie.instructions()[4].component != "V") return 1;

        const auto strang = KARMA::compile_operator_split_graph(KARMA::OperatorSplitGraphSpec{
            graph(KARMA::SplitScheme::Strang)});
        if (strang.instructions().size() != 9 ||
            strang.instructions()[2].interval != KARMA::AdvanceInterval::HalfFirst ||
            strang.instructions()[4].interval != KARMA::AdvanceInterval::Full ||
            strang.instructions()[6].interval != KARMA::AdvanceInterval::HalfSecond)
            return 1;

        auto swapped = graph();
        std::swap(swapped.connections[0], swapped.connections[1]);
        const auto swapped_compiled = KARMA::compile_operator_split_graph(swapped);
        if (swapped_compiled.instructions()[1].source_component != "V" ||
            swapped_compiled.instructions()[1].target_component != "U" ||
            swapped_compiled.instructions()[3].source_component != "U" ||
            swapped_compiled.instructions()[3].target_component != "V")
            return 1;

        auto mutable_graph = graph();
        const auto compiled = KARMA::compile_operator_split_graph(mutable_graph);
        mutable_graph.id = "changed";
        mutable_graph.components[0].id = "changed";
        if (compiled.metadata().id != "gray-scott" || compiled.metadata().components[0].id != "U") return 1;

        auto bad = graph();
        bad.connections[1].source_component = "U";
        expect_invalid(bad);
        bad = graph();
        bad.components[1].component.supports_fractional_steps = false;
        bad.scheme = KARMA::SplitScheme::Strang;
        expect_invalid(bad);
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
