#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/native_mesh.hpp"
#include "fuse/backend/native_migration.hpp"  // CommHandle, kNullComm
#include "fuse/linalg/comm.hpp"               // GhostPattern

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
  /// `fuse_initialize()` hit this before this
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

  // =========================================================================
  // THE PLANAR/VOLUMETRIC REMAP HALO. See docs/dev/native-halo.md for the
  // construction and src/backend/native_halo.cpp for the implementation;
  // this is the bookkeeping half, the same split moab_backend.hpp draws
  // between its own `adopt_remap_halo`/`has_remap_halo` (bookkeeping, here)
  // and `build_remap_halo`'s covering-set geometry (src/backend/
  // intx_weights.cpp, which dispatches to `fuse::native::build_native_
  // remap_halo` for THIS backend).
  // =========================================================================

  /// Adopts a real MPI communicator for the covering-set exchange. A
  /// Fortran handle (`MPI_Comm_c2f`), never an MPI type, per this header's
  /// own no-mpi.h convention. `kNullComm` (the default before this is
  /// called) means no covering-set exchange is possible; `build_remap_halo`
  /// then treats every mesh here as serial, exactly as `MoabMeshBackend`
  /// treats a null `ParallelComm`.
  void set_communicator(CommHandle comm) noexcept;
  [[nodiscard]] CommHandle communicator() const noexcept;

  /// `1` with no communicator adopted (every cell is local, so the serial
  /// construction is exactly right -- the same "undivided mesh" case
  /// `MoabMeshBackend::communicator_size()` documents), `num_ranks()`
  /// otherwise. NOT `num_ranks()` unconditionally: `(rank, num_ranks)` is
  /// supplied at construction for mesh GENERATION, which needs no MPI
  /// symbol at all, so a caller that never adopts a communicator must still
  /// get the serial answer here.
  [[nodiscard]] int communicator_size() const noexcept;

  /// Whether `build_remap_halo(source, target, dim)` (remap_plan.hpp, via
  /// `fuse::native::build_native_remap_halo`) has run for this triple AT
  /// this `placement` (default: the unplaced decomposition, matching
  /// every caller that does not pass one today). Zero cells adopted is a
  /// legitimate, already-built answer -- this is "was it asked for", not
  /// "did it find anything" -- matching `MoabMeshBackend::has_remap_halo`'s
  /// own contract exactly. NM-4: `placement` is part of the bookkeeping
  /// key precisely because `build_native_remap_halo` can run once per
  /// placement on the SAME (source, target, dim) triple (a component's own
  /// decomposition is its own halo); omitting it would make two placements
  /// clobber each other's recorded pattern.
  [[nodiscard]] bool has_remap_halo(MeshId source, MeshId target, int dim,
                                    ComponentId placement = ComponentId{}) const noexcept;

  /// Where each of `source`'s adopted (non-owned) cells at dimension `dim`
  /// came from, in the order they were appended to the mesh (after every
  /// owned cell -- see `native_halo.cpp`'s own comment on why that is the
  /// order and not an accident of it). `owner_rank[g]` is the rank that
  /// OWNS the cell; `remote_local_index[g]` is that rank's own position for
  /// it among ITS OWNED cells, ascending local-index order -- the "owned
  /// block" position `fuse::linalg::DistVector`/`GhostExchanger` already use
  /// (comm.hpp), so this pattern plugs directly into
  /// `fuse::linalg::make_mpi_exchanger` to fill the adopted cells' field
  /// values before `apply_weights`. UNCHANGED return type (deliberately:
  /// `examples/brusselator/species_core.hpp`'s `GhostExchangerFactory`
  /// takes exactly `const fuse::linalg::GhostPattern&`, outside this
  /// library) -- see `remap_halo_ghost_offset` below for the
  /// piece this type has no room for.
  ///
  /// `E-MESH-HALO` if no halo was built for this triple (at this
  /// `placement`) at all (distinct from the legitimate zero-cells answer,
  /// which returns an empty pattern successfully).
  [[nodiscard]] Result<fuse::linalg::GhostPattern> remap_halo_pattern(
      MeshId source, MeshId target, int dim, ComponentId placement = ComponentId{}) const;

  /// NM-3: where THIS triple's own ghost block (the `remap_halo_pattern`
  /// above) begins within `source`'s ghost region, i.e. in `source`'s
  /// local cell numbering, `remap_halo_pattern(...).owner_rank[g]`'s
  /// value lives at local cell index `owned_n + remap_halo_ghost_offset(...)
  /// + g`, NOT `owned_n + g`. `merge_halo_cells` appends each call's newly
  /// adopted cells STRICTLY AFTER every cell `source` already held
  /// (`native_halo.cpp`'s own comment), so a SECOND halo built on the same
  /// `source` for a DIFFERENT target starts its own ghosts after whatever
  /// an EARLIER call already appended -- `0` for the first halo ever built
  /// on a given `source` (every consumer in this tree today builds only
  /// one), and for the identical reason `remap_halo_pattern` is, `E-MESH-
  /// HALO` if no halo was built for this triple at all.
  [[nodiscard]] Result<std::size_t> remap_halo_ghost_offset(
      MeshId source, MeshId target, int dim, ComponentId placement = ComponentId{}) const;

  /// INTERNAL SEAM for `native_halo.cpp`'s covering-set construction, which
  /// needs `NativeMeshView`'s own extended accessors (`cell_vertices`,
  /// `vertex_x`, ...) that the abstract `MeshView`/`placed_view` do not
  /// expose. The pointer's lifetime is the backend's, like `view()`'s --
  /// EXCEPT across a call to `commit_remap_halo` below, which replaces it
  /// (see that method's own comment). Nothing but the native halo
  /// machinery may call this.
  [[nodiscard]] Result<const NativeMeshView*> native_view(MeshId id,
                                                          ComponentId placement) const;

  /// INTERNAL SEAM, the other half of `native_view` above: replaces mesh
  /// `source`'s view at `placement` (the default decomposition when
  /// `placement` is invalid) with `new_source_view` -- which already
  /// contains every existing cell/vertex PLUS the newly adopted ones,
  /// built by `fuse::native::merge_halo_cells` and
  /// `NativeMeshView::create_from_host` -- and records `pattern` against
  /// `(source, target, dim)` so `has_remap_halo`/`remap_halo_pattern`
  /// answer it afterward. `new_source_view == nullptr` means the
  /// zero-cells-adopted no-op (one rank, no communicator, or source and
  /// target are the identity): nothing to replace, `pattern` is expected
  /// to be empty, and only the bookkeeping is recorded.
  ///
  /// DEVIATION FROM THE MOAB BACKEND'S OWN POINTER-STABILITY CONTRACT
  /// (`mesh_view.hpp`'s "the pointer is owned by the backend and stays
  /// valid until the backend is destroyed or moved from"): `MoabMeshBackend
  /// ::adopt_remap_halo` grows MOAB's own entity set IN PLACE, so its
  /// `MeshView*` never changes identity. Growing a `NativeMeshView` in
  /// place would mean re-deriving its per-cell simplex decomposition,
  /// centroid and measure by hand for just the new cells -- exactly the
  /// logic `NativeMeshView::create_from_host` already has, tested, for
  /// every cell shape this mesh supports. Rebuilding the WHOLE view through
  /// that one tested path and swapping the stored pointer was judged the
  /// more robust choice (standing priority: robustness over efficiency),
  /// at the cost that a caller must not cache a raw `MeshView*` obtained
  /// before `build_remap_halo` and keep using it afterward -- which
  /// `remap_plan.hpp`'s own "ORDER MATTERS" discipline (halo before any
  /// buffer is sized) already requires of every caller. `resolve()`/
  /// `supermesh_side()` in `intx_weights.cpp` always re-fetch the view by
  /// `(MeshId, ComponentId)`, never hold one across a halo build, so this
  /// is safe for every path this library drives itself. Recorded here
  /// rather than left to be discovered as an undocumented diff.
  Result<void> commit_remap_halo(MeshId source, ComponentId placement,
                                 std::unique_ptr<NativeMeshView> new_source_view,
                                 MeshId target, int dim, fuse::linalg::GhostPattern pattern);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  int rank_;
  int num_ranks_;
  CommHandle communicator_ = kNullComm;
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
