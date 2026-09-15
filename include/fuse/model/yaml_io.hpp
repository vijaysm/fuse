#pragma once

#include <string>
#include <string_view>

#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"

namespace fuse {

/// Emit the graph as YAML. Names, not indices, are used for references.
[[nodiscard]] Result<std::string> to_yaml(const CouplingGraph& g);

/// Parse YAML into a graph. Reference and unit resolution failures are
/// reported as E-YAML-REF / E-YAML-UNITS.
[[nodiscard]] Result<CouplingGraph> from_yaml(std::string_view text);

[[nodiscard]] Result<CouplingGraph> load_yaml_file(const std::string& path);

}  // namespace fuse
