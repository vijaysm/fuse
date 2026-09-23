#pragma once

/// One file format for the native mesh: a subset of the legacy VTK ASCII
/// `UNSTRUCTURED_GRID` format (docs/dev/native-mesh.md explains the
/// choice). Every cell type this mesh supports has a direct VTK legacy
/// counterpart with an IDENTICAL corner-order convention (VTK's
/// Hexahedron/Wedge/Pyramid orderings are the same base-then-opposite-face
/// convention `cell_topology.hpp` already documents), so no reordering is
/// needed on either side -- except `Polyhedron`, which uses VTK's
/// `VTK_POLYHEDRON` (42) face-stream cell format directly.
///
/// This is a WRITER-DEFINED SUBSET, not a general VTK reader: it reads
/// only what `write_native_mesh_file` produces. A real VTK/ParaView reader
/// can still open the geometry (points, cell connectivity, cell types) --
/// every token used is standard legacy VTK syntax -- but the round-trip
/// this backend needs also carries global ids and owner/ghost flags, which
/// VTK has no convention for, so they ride as ordinary named
/// `POINT_DATA`/`CELL_DATA` scalars (`GlobalPointID`, `PointOwner`,
/// `PointIsOwned`, `GlobalCellID`, `CellOwner`, `CellIsOwned`) plus one
/// FUSE-specific comment line naming the mesh's topological dimension
/// (a `#`-prefixed line, which a real VTK reader skips as a comment
/// anywhere in the file, so this does not break compatibility).
///
/// Ids are written as `double`: VTK legacy has no signed 64-bit integer
/// scalar type, and `double` represents every integer up to 2^53 exactly,
/// which covers every mesh size this library is used at.

#include <string>

#include "fuse/backend/native_mesh.hpp"

namespace fuse::native {

/// Writes `mesh` to `path` in the format above. Errors: `E-MESH-IO` if the
/// file cannot be opened for writing.
[[nodiscard]] Result<void> write_native_mesh_file(const std::string& path,
                                                  const NativeMeshView& mesh, int rank);

/// Reads a file this module wrote. Errors: `E-MESH-IO` (cannot open, or the
/// file is not in this subset), `E-MESH-CONNECTIVITY`/`E-MESH-DIM`/
/// `E-MESH-POLYGON-*`/`E-MESH-CELL-NOT-STAR-SHAPED` from the same
/// validation `NativeMeshBuilder::build` always runs.
[[nodiscard]] Result<MeshArrays<Kokkos::HostSpace>> read_native_mesh_file(
    const std::string& path);

}  // namespace fuse::native
