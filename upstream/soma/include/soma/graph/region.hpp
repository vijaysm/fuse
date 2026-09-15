#pragma once

#include <string>

#include "soma/base/ids.hpp"

namespace soma {

/// A mesh the coupler is told about. `source` is opaque to the graph layer;
/// the mesh backend (Phase 4 serial, Phase 5 MOAB) interprets it.
struct MeshDecl {
  std::string name;
  std::string source;  // file path, or "generated:<spec>"
};

/// Spec §4.1: the connectable unit. Coupling happens on interfaces, not whole
/// meshes, so channels connect Regions.
struct Region {
  std::string name;
  MeshId      mesh;
  std::string entity_set;   // empty = whole mesh
  int         dimension = -1;  // topological dim of coupled entities; -1 = mesh dim
};

}  // namespace soma
