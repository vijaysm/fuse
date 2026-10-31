#pragma once

/// Mesh boundary-extraction helpers, adapted from KARMA's
/// karma_mesh_extract_edge.C / karma_mesh_extract_face.C
/// (docs/dev/physics-kit-disposition.md Appendix A, port-adapt).
///
/// (item 4): extract_edge_at_coordinate
/// now ports karma_mesh_extract_edge.C's own operation -- collect the 1-D
/// edges of a 2-D mesh lying on a line of constant coordinate -- against
/// this wave's new 2-D spatial::Mesh::generate_2d_box (mesh.hpp), which
/// did not exist in. ADAPTED, not a
/// literal port: KARMA's own karma_mesh_extract_edge.C is legacy libMesh
/// code (Node/Elem/MeshBase iterators, add_point with a processor id) in a
/// visibly older style than the rest of numerics/spatial; this reproduces
/// its OPERATION -- the same coordinate-equality filter, the same
/// deduplicated 1-D output mesh -- against this port's own Mesh/
/// EntityHandle idiom, generalized slightly: every edge of the source mesh
/// on the line is collected (KARMA's own loop is equivalent: it visits
/// every element and keeps only the nodes at the target coordinate, which
/// for a structured quad mesh is exactly the edges lying on that line).

#include <vector>

#include "fuse_physics/spatial/mesh.hpp"

namespace fuse::physics::spatial {

/// Every boundary side (spatial::Mesh::is_boundary_side) of `volume`, as
/// (element, side) pairs -- the SideId KARMA's own karma_mesh.h already
/// names for exactly this purpose. Works for a volume mesh of any
/// supported element kind (Hex8, Tet4, Prism6, Pyramid5 -- element_kind()
/// per element), not only Hex8.
[[nodiscard]] std::vector<SideId> extract_boundary_faces(const Mesh& volume);

/// The 1-D boundary at `coordinate(direction) == value` (within
/// `tolerance`) of a 2-D `surface` mesh, as a new, independent Edge2 Mesh
/// -- karma_mesh_extract_edge.C's own operation ("Extract a 1-D edge from
/// a 2-D surface mesh"; `direction` is 0 for x, 1 for y, matching this
/// port's 0-based Point::operator()() rather than KARMA's own 1-based
/// `direction-1` convention, stated here so a caller porting a KARMA call
/// site does not carry the off-by-one across). Every edge of `surface`
/// (not only its outer boundary: an interior line at the given coordinate
/// is extracted too, matching KARMA's own coordinate filter, which has no
/// boundary concept at all) whose both endpoints satisfy the filter is
/// kept; vertices shared by more than one kept edge are NOT duplicated in
/// the output (karma_mesh_extract_edge.C's own `node_map`/`node_ids_global`
/// deduplication, ported as a vertex-position map here).
[[nodiscard]] Mesh extract_edge_at_coordinate(const Mesh& surface, unsigned direction, double value,
                                               double tolerance);

}  // namespace fuse::physics::spatial
