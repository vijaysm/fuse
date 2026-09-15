#pragma once

#include <string>
#include <string_view>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"

namespace soma {

/// Emit the graph as YAML. Names, not indices, are used for references.
[[nodiscard]] Result<std::string> to_yaml(const CouplingGraph& g);

/// Parse YAML into a graph. Reference and unit resolution failures are
/// reported as E-YAML-REF / E-YAML-UNITS.
[[nodiscard]] Result<CouplingGraph> from_yaml(std::string_view text);

[[nodiscard]] Result<CouplingGraph> load_yaml_file(const std::string& path);

}  // namespace soma
