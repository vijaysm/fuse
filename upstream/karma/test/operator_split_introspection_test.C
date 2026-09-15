#include "karma_operator_split_graph.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace
{
KARMA::ComponentSpec component(const std::string& id, const std::string& output, const std::string& input)
{
    const KARMA::FieldSpec state{id + ".state", "1", 1, KARMA::FieldRole::Prognostic,
                                 KARMA::FieldLocation::FiniteElementCoefficient,
                                 KARMA::TimeSemantics::InstantaneousCommitted};
    const KARMA::FieldSpec coupled{id + ".coupled", "1", 1, KARMA::FieldRole::Input,
                                   KARMA::FieldLocation::FiniteElementCoefficient,
                                   KARMA::TimeSemantics::InstantaneousCommitted};
    return {id, {{output, KARMA::PortDirection::Output, false, state}, {input, KARMA::PortDirection::Input, true, coupled}}, true, true};
}

KARMA::OperatorSplitGraphSpec graph()
{
    auto result = KARMA::OperatorSplitGraphSpec{
        "gray-scott",
        {KARMA::ComponentInstanceSpec{"U", component("U", "state", "coupled_v")},
         KARMA::ComponentInstanceSpec{"V", component("V", "state", "coupled_u")}},
        {KARMA::ConnectionSpec{"U", "state", "V", "coupled_u"},
         KARMA::ConnectionSpec{"V", "state", "U", "coupled_v"}},
        KARMA::SplitScheme::Strang};
    result.components[0].mesh_id = "generated-16x16";
    result.components[0].layout_id = "u-scalar-cg";
    result.components[0].basis_id = "lagrange-p1";
    result.components[0].solver_policy_id = "snes-newtonls";
    result.components[1].mesh_id = "generated-19x17";
    result.components[1].layout_id = "v-scalar-cg";
    result.components[1].basis_id = "lagrange-p1";
    result.components[1].solver_policy_id = "snes-newtonls";
    for (auto& connection : result.connections)
    {
        connection.transfer_id = "persistent-cg-projection-q5";
        connection.solver_policy_id = "cg-jacobi";
    }
    return result;
}

void require_category(const std::vector<std::string>& differences, const std::string& category)
{
    if (std::find(differences.begin(), differences.end(), category) == differences.end())
        throw std::runtime_error("missing graph diff category: " + category);
}
}

int main()
{
    try
    {
        const auto compiled = KARMA::compile_operator_split_graph(graph());
        auto reordered = graph();
        std::swap(reordered.components[0], reordered.components[1]);
        std::swap(reordered.connections[0], reordered.connections[1]);
        for (auto& instance : reordered.components)
            std::reverse(instance.component.ports.begin(), instance.component.ports.end());
        const auto reordered_compiled = KARMA::compile_operator_split_graph(reordered);
        if (compiled.signature() != reordered_compiled.signature()) return 1;
        for (const auto& instance : reordered_compiled.metadata().components)
            if (!std::is_sorted(instance.component.ports.begin(), instance.component.ports.end(),
                                [](const auto& left, const auto& right) { return left.id < right.id; }))
                return 1;

        const auto report = compiled.report();
        if (report.signature != compiled.signature() || report.components.size() != 2 || report.connections.size() != 2 ||
            report.instructions.size() != compiled.instructions().size() || report.deferred_features.empty() ||
            report.components[0].mesh_id == "unknown" || report.components[0].layout_id == "unknown" ||
            report.components[0].basis_id == "unknown" || report.components[0].solver_policy_id == "unknown" ||
            report.connections[0].transfer_id == "unknown" || report.connections[0].solver_policy_id == "unknown" ||
            report.connections[0].setup != "not initialized" || report.connections[0].coverage != "unknown")
            return 1;

        auto changed = graph();
        changed.scheme = KARMA::SplitScheme::Lie;
        require_category(compiled.diff(KARMA::compile_operator_split_graph(changed)), "schedule differs");
        changed = graph();
        changed.components[0].id = "W";
        changed.components[0].component.id = "W";
        changed.connections[0].source_component = "W";
        changed.connections[1].target_component = "W";
        require_category(compiled.diff(KARMA::compile_operator_split_graph(changed)), "component metadata differs");
        changed = graph();
        changed.components[0].component.ports[0].required = true;
        require_category(compiled.diff(KARMA::compile_operator_split_graph(changed)), "port metadata differs");
        changed = graph();
        changed.components[0].component.ports[0].field.id = "U.renamed";
        require_category(compiled.diff(KARMA::compile_operator_split_graph(changed)), "field metadata differs");
        changed = graph();
        changed.components[0].mesh_id = "another-mesh";
        require_category(compiled.diff(KARMA::compile_operator_split_graph(changed)), "mesh/layout metadata differs");
        changed = graph();
        changed.connections[0].transfer_id = "nearest-neighbor";
        require_category(compiled.diff(KARMA::compile_operator_split_graph(changed)), "transfer metadata differs");
        changed = graph();
        changed.components[0].solver_policy_id = "snes-trust-region";
        require_category(compiled.diff(KARMA::compile_operator_split_graph(changed)), "solver policy differs");
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
