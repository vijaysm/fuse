#pragma once

/// Mesh boundary-extraction helpers, adapted from KARMA's
/// karma_mesh_extract_edge.C / karma_mesh_extract_face.C
/// (docs/dev/physics-kit-disposition.md Appendix A, port-adapt).
///
/// WAVE 1 SCOPE: only extract_boundary_faces (the 3-D-volume-to-2-D-
/// surface half, karma_mesh_extract_face.C) is ported, against this wave's
/// native Hex8 spatial::Mesh (mesh.hpp). karma_mesh_extract_edge.C (the
/// 2-D-surface-to-1-D-edge half) is NOT ported: it needs a 2-D (quad)
/// surface mesh source, and this wave's spatial::Mesh only generates a
/// 3-D Hex8 box (mesh.hpp's own scope note) -- there is no 2-D mesh yet to
/// extract an edge FROM. Named as an open gap (design_notes/
/// open_questions) rather than stubbed against a mesh type that does not
/// exist.

#include <vector>

#include "fuse_physics/spatial/mesh.hpp"

namespace fuse::physics::spatial {

/// Every boundary side (spatial::Mesh::is_boundary_side) of `volume`, as
/// (element, side) pairs -- the SideId KARMA's own karma_mesh.h already
/// names for exactly this purpose.
[[nodiscard]] std::vector<SideId> extract_boundary_faces(const Mesh& volume);

}  // namespace fuse::physics::spatial
