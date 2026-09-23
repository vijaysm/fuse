#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/native_mesh.hpp"

namespace fuse::native {

/// `fuse::MeshBackend` over the native distributed mesh. No MPI, no files
/// beyond the native VTK-subset reader (native_mesh_io.hpp); a caller with a
/// real communicator passes its `(rank, num_ranks)` in as plain ints, the
/// same way every other backend layer stays MPI-free in its public header
/// (docs/dev/layering.md).
///
/// SOURCES. `decl.source`:
///
///   "generated:line:N"                 the unit interval, N segments
///   "generated:tri:NxM"                unit square, N*M quads split to
///                                       2*N*M triangles
///   "generated:quad:NxM"               unit square, N*M quads
///   "generated:cubedsphere:N"          6 panels, N*N quads each, on the
///                                       unit sphere
///   "generated:hexbox:NxMxP"           unit cube, N*M*P hexes
///   "generated:tetbox:NxMxP"           unit cube, N*M*P*6 tets (Kuhn split
///                                       of each hex)
///   "generated:prismpyramidmix:NxMxP"  unit cube; a hex at even (i+j+k) is
///                                       split into 2 prisms, odd into 6
///                                       pyramids about an added centroid
///                                       vertex -- one mesh, two 3D shapes
///   "generated:polyhedral:NxMxP"       unit cube, N*M*P hexes, each stored
///                                       as a general Polyhedron (explicit
///                                       face CSR) rather than the fixed
///                                       Hex shape
///   anything else                      a path read via native_mesh_io.hpp
///
/// Every generator is DISTRIBUTED: it takes `(rank, num_ranks)` from the
/// backend's own construction and builds only this rank's slice, with
/// global ids that do not depend on `num_ranks` -- the same cell splits the
/// domain into the same global ids whether run at 1 rank or 8. Distribution
/// is by the outermost extent's index range, split as evenly as possible
/// (see native_generators.cpp's `rank_slice`).
class NativeMeshBackend final : public MeshBackend {
 public:
  NativeMeshBackend(int rank, int num_ranks);
  ~NativeMeshBackend() override;
  NativeMeshBackend(NativeMeshBackend&&) noexcept;
  NativeMeshBackend& operator=(NativeMeshBackend&&) noexcept;

  Result<void> load(const MeshDecl& decl) override;
  [[nodiscard]] Result<MeshId> mesh_id(std::string_view name) const override;
  [[nodiscard]] Result<const MeshView*> view(MeshId id) const override;
  [[nodiscard]] Result<std::size_t> region_size(const Region& r) const override;
  [[nodiscard]] std::string name() const override { return "native"; }

  /// Registers `view` as mesh `id`'s decomposition for `placement`
  /// (mesh_view.hpp's per-placement contract) -- how a caller that just
  /// migrated a mesh onto a component's own placement makes that result
  /// reachable through `placed_view`/`placed_region_size`. `placement`
  /// must be valid; use plain `load` (an unplaced, default decomposition)
  /// otherwise.
  Result<void> set_placed_view(MeshId id, ComponentId placement,
                               std::unique_ptr<NativeMeshView> view);

  [[nodiscard]] Result<const MeshView*> placed_view(MeshId id,
                                                    ComponentId placement) const override;
  [[nodiscard]] Result<std::size_t> placed_region_size(const Region& r,
                                                       ComponentId placement) const override;

  [[nodiscard]] int rank() const noexcept { return rank_; }
  [[nodiscard]] int num_ranks() const noexcept { return num_ranks_; }

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int rank_;
  int num_ranks_;
};

/// The distributed slice of index range `[0, n)` this rank owns, as a
/// half-open `[lo, hi)`, split as evenly as possible (the first
/// `n % num_ranks` ranks get one extra). Exposed so migration/driver code
/// and tests can reproduce the same split a generator used.
[[nodiscard]] std::pair<std::size_t, std::size_t> rank_slice(std::size_t n, int rank,
                                                             int num_ranks);

/// Builds this rank's slice of each generated source named above.
/// `spec` is the part after "generated:" (e.g. "hexbox:4x4x4"). Errors:
/// `E-MESH-SOURCE` (unknown kind or malformed dims), `E-MESH-SIZE`
/// (a count would not fit `std::size_t`).
[[nodiscard]] Result<MeshArrays<Kokkos::HostSpace>> generate_native_mesh(
    const std::string& spec, int rank, int num_ranks);

}  // namespace fuse::native
