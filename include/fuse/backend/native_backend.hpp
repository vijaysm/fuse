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
/// same MPI-free-public-header convention `fuse::partition::detail`'s own
/// `collectives.hpp` documents and `native_migration.hpp` follows.
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

  /// Register a placed decomposition, on mesh `id` (already loaded, by
  /// either overload of `load()` above), for EVERY component in the
  /// graph -- not just the ones that look like they need one.
  ///
  /// WHY EVERY COMPONENT, UNCONDITIONALLY. A narrower rule -- a
  /// component that owns a field on this mesh, or a `ComponentKind::
  /// Coupler` some channel's `remap_on` routes through -- was tried
  /// first and found WRONG: an ordinary (no `remap_on`) cross-mesh
  /// Remap resolves its weights on `dag.arc(ins.out).type.rank_set`
  /// (`Runner::Impl::resolve_remap()`, src/runtime/runner.cpp), which
  /// is a DEDUPLICATED RankSet index, not necessarily the owning
  /// component's own `ComponentId` -- two components sharing an
  /// IDENTICAL `RankSet` (the common tier-0 case: every component on
  /// rank 0) collapse to the SAME index, which can then numerically
  /// coincide with a different component's own id and ask for THAT
  /// component's placement on a mesh it owns no field on at all
  /// (confirmed this session: an ABI test's own two-component lag1
  /// fixture, both components on the same `RankSet{0,1}`, hit exactly
  /// this with the narrower rule). Reverse-engineering every path that
  /// can derive a placement this way is not worth it when every
  /// placement this backend can ever generate is the SAME cheap,
  /// deterministic `(rank_, num_ranks_)` view regardless of which
  /// component asks for it (tier-0: one process IS every placement) --
  /// so hand out all of them.
  ///
  /// WHY THIS METHOD EXISTS AT ALL. A caller with the graph in hand
  /// used to have to do this BY HAND, once per generated mesh, right
  /// after loading it -- examples/brusselator/brusselator_graph.hpp's
  /// own (now-removed) `register_native_window_placements()` was
  /// exactly that, and every caller without its own hand-rolled copy
  /// got `E-MESH-PLACEMENT` the moment any component's own placement
  /// needed resolving (confirmed: the participant ABI's own
  /// `fuse_initialize()`, Conformance Unit 2, hit this before this
  /// method existed). A caller that cannot see library internals -- a
  /// real participant process naming a generated mesh in its own graph
  /// -- has no way to reproduce that by-hand logic at all, so it
  /// belongs in the backend.
  ///
  /// A no-op for a FILE-read mesh (`graph.mesh(id).source` not prefixed
  /// "generated:"): only a generated mesh can be rebuilt, cheaply and
  /// identically, on an arbitrary RankSet; a file mesh keeps relying on
  /// the unplaced default view for every placement, exactly as before.
  Result<void> register_default_placements(const CouplingGraph& graph, MeshId id);

  /// `load(decl)`, then `register_default_placements(graph, <the id just
  /// loaded>)` -- the convenience spelling for a caller that has not
  /// loaded this mesh yet at all (most callers; `register_default_
  /// placements` on its own is for the few, like examples/brusselator/
  /// main.cpp, that load through a different path first -- NativeGrid::
  /// make's own canonical-order bookkeeping -- and only need the
  /// placement half afterwards).
  Result<void> load(const MeshDecl& decl, const CouplingGraph& graph);

  [[nodiscard]] Result<MeshId> mesh_id(std::string_view name) const override;
  [[nodiscard]] Result<const MeshView*> view(MeshId id) const override;
  [[nodiscard]] Result<std::size_t> region_size(const Region& r) const override;
  [[nodiscard]] std::string name() const override { return "native"; }

  /// Registers `view` as mesh `id`'s decomposition for `placement`
  /// (mesh_view.hpp's per-placement contract) -- how a caller that just
  /// migrated a mesh onto a component's own processor set makes that result
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
