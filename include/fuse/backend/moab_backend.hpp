#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "fuse/backend/mesh_view.hpp"

namespace fuse {

/// What `MoabMeshBackend::sphere_radius` measured about one mesh's vertices,
/// read as a sphere about the origin -- the two numbers a DRIVER needs to
/// decide whether (and how) to rescale it before a geometry-sensitive call
/// (`build_remap_halo`/`build_weights`) reaches `E-REMAP-GEOMETRY`. See
/// `MoabMeshBackend::sphere_radius`'s own doc for how both are derived.
struct SphereRadiusReport {
  /// `0.5 * (rmin + rmax)` over every vertex's distance from the origin,
  /// reduced across every rank holding a piece of the mesh.
  double mean_radius = 0.0;
  /// `0.5 * (rmax - rmin)` -- the largest distance any single vertex's
  /// radius is from `mean_radius`, the same reduced extent. Zero for a
  /// mathematically perfect sphere; a few times `1e-16 * mean_radius` for
  /// a real mesh file's own floating-point round-off (measured on
  /// `wholeATM_T.h5m`/`recMeshOcn.h5m`).
  double max_deviation = 0.0;
};

/// What `MoabMeshBackend::project_onto_unit_sphere` did to one mesh's
/// vertices. See that function's own doc for the no-op rule and the
/// tolerance that decides it.
struct SphereProjectionReport {
  /// The mesh's own `SphereRadiusReport::mean_radius`, measured BEFORE any
  /// rescale -- `1.0` exactly when `already_unit` (see below), since no
  /// measurement is needed to confirm a mesh already reporting a mean
  /// radius of~1 within tolerance is the one being normalised onto.
  double mean_radius = 0.0;
  /// The scalar factor actually multiplied into every vertex coordinate:
  /// `1.0` when `already_unit` (a true no-op, coordinates untouched), else
  /// `1.0 / mean_radius`.
  double applied_factor = 1.0;
  /// `true` when `mean_radius` was already within `kUnitSphereTol` of 1.0
  /// -- no vertex coordinate was read back or rewritten in that case, only
  /// measured.
  bool already_unit = false;
};

/// The MOAB objects behind one loaded mesh, as opaque integers so this header
/// still names no MOAB type. See `MoabMeshBackend::native_mesh`.
struct MoabNativeMesh {
  /// The backend's `moab::Interface*`. One instance holds EVERY mesh the
  /// backend has loaded, which is why `file_set` below is also needed.
  void* interface = nullptr;
  /// The `moab::EntityHandle` of this mesh's file set. The dimension-`d`
  /// entities of `get_entities_by_dimension(file_set, d, r, true)` are, in
  /// that Range's ascending handle order, exactly the `MeshView` local indices
  /// `0 .. entity_count(d) - 1`.
  std::uint64_t file_set = 0;
  /// The backend's `moab::ParallelComm*`, or null when no communicator was
  /// adopted. Added for `build_remap_halo`, which has to hand it to MOAB's
  /// covering-set construction; nothing else may use it.
  void* parallel_comm = nullptr;
};

/// MOAB-backed topology. Per the Global Constraints dependency rule, no MOAB
/// type appears here; `moab/Core.hpp` is included only by
/// `src/backend/moab_backend.cpp`.
///
/// SOURCES
///   any path MOAB can read      `.h5m`, `.exo`, `.g`, `.vtk`, `.cub`, ...
///   "generated:line:N"          the unit interval [0,1], N edge cells
///   "generated:box:NxMxP"       the unit cube [0,1]^3, N*M*P hexes
/// The generated forms match `SerialMeshBackend`'s exactly -- same domain,
/// same counts, same entity-set names -- so a test can swap the backends and
/// compare. They exist so the MOAB path is exercised without a data file.
///
/// ENTITIES. `entity_count(d)` is the number of dimension-`d` entities MOAB
/// holds for this mesh ON THIS RANK, owned plus ghost. For a loaded file that
/// is whatever the file (and the partition, if one was read) contains: a file
/// of hexes with no explicit faces reports `entity_count(2) == 0`, and no
/// faces are invented. For a generated box the boundary skin IS created, so
/// dimension 2 is the 2*(N*M + M*P + N*P) boundary quads -- matching the
/// serial backend -- and interior faces are still not enumerated.
///
/// ENTITY SETS, resolved in this order:
///   ""  or  "all"        the whole mesh at the requested dimension
///   <NAME tag match>     a set whose MOAB `NAME` tag equals the string
///   "sideset_<n>"        a set whose `NEUMANN_SET`   tag equals <n>
///   "neumann_<n>"        likewise
///   "nodeset_<n>"        a set whose `DIRICHLET_SET` tag equals <n>
///   "dirichlet_<n>"      likewise
///   "block_<n>"          a set whose `MATERIAL_SET`  tag equals <n>
///   "material_<n>"       likewise
/// Anything else, or a prefix whose integer matches no set, is `E-MESH-SET`.
/// A generated box additionally defines the `NAME`-tagged sets "left",
/// "right", "front", "back", "bottom", "top" and "wall", the same names the
/// serial backend answers to.
///
/// MEASURES are physical, unsigned, additive and local, exactly as
/// `MeshView::measure` specifies. See the implementation notes in
/// `src/backend/moab_backend.cpp` for how each element type is integrated
/// and which ones are rejected.
///
/// PARALLEL. Without `set_communicator` the backend is serial: every rank
/// that runs it reads the whole file independently and owns every entity.
/// With a communicator the read is partitioned, `entity_count` reports the
/// LOCAL count, and `is_owned` / `owned_count` / `owned_entities` are the
/// only correct way to sum anything.
///
/// KNOWN LIMITATION, stated rather than hidden: `MeshView::dimension()`
/// reports the highest dimension this RANK holds entities of. A rank whose
/// partition is empty, or which holds only faces of a volume mesh, therefore
/// reports a smaller number than its peers. Agreeing on one answer needs a
/// collective reduction, and `load` deliberately performs no communication of
/// its own beyond what MOAB's partitioned read already does. Callers that
/// need a mesh-wide dimension must reduce it themselves; every other method
/// here is honestly local and needs no such care.
class MoabMeshBackend final : public MeshBackend {
 public:
  MoabMeshBackend();
  ~MoabMeshBackend() override;

  /// Moving transfers the loaded meshes and the communicator, and leaves the
  /// source empty but usable: a moved-from backend holds no meshes and
  /// reports `E-MESH-MOVED` from every method rather than dereferencing a
  /// null pimpl. Pointers previously returned by `view()` remain valid --
  /// they point into the moved-to backend's MOAB instance, which did not
  /// move, only the owning pointer did.
  MoabMeshBackend(MoabMeshBackend&&) noexcept;
  MoabMeshBackend& operator=(MoabMeshBackend&&) noexcept;

  Result<void> load(const MeshDecl& decl) override;
  [[nodiscard]] Result<MeshId> mesh_id(std::string_view name) const override;
  [[nodiscard]] Result<const MeshView*> view(MeshId id) const override;
  [[nodiscard]] Result<std::size_t> region_size(const Region& r) const override;
  [[nodiscard]] std::string name() const override { return "moab"; }

  /// Declare one `RankSet` per graph
  /// component, indexed by `ComponentId` -- mirrors
  /// `PartitionedTestBackend::set_placements`
  /// (test/support/partitioned_test_backend.hpp) and `NativeMeshBackend::
  /// set_placed_view`'s own per-placement contract, so a caller that reads
  /// processor sets off a `CouplingGraph` (`graph.component_count()`, including
  /// `Coupler`-kind entries -- never listed by hand) hands them to every
  /// mesh backend the same way.
  ///
  /// MUST PRECEDE THE FIRST `load()`: each `load()` call afterwards reads
  /// the SAME source into every declared processor set's own decomposition, in
  /// addition to the backend's default one, so a processor set declared after a
  /// mesh already exists would leave that mesh without one. Rejects
  /// `E-MOAB-PLACEMENT` in that case, and if called twice.
  ///
  /// TWO MODES, both usable without ever calling `set_communicator`:
  ///   * NO communicator adopted -- "this rank" is rank 0 of an effective
  ///     size-1 communicator (`partition_rank_of`'s own convention). A
  ///     processor set containing rank 0 IS the default decomposition (there is
  ///     only one rank to hold it); one that does not gets an empty view.
  ///     No MOAB Core, no MPI call, is needed or possible, so this mode
  ///     -- including its rejections -- is exercised at the base build configuration.
  ///   * A communicator IS adopted -- one `moab::ParallelComm`, on its own
  ///     `moab::Core`, split from the BASE communicator over exactly each
  ///     processor set's ranks (iMOAB's application-id model).
  ///     Requires `FUSE_ENABLE_MPI` and an MPI-enabled MOAB;
  ///     `E-MOAB-NOMPI` otherwise, the same boundary `set_communicator`
  ///     itself draws.
  ///
  /// COLLECTIVE over the base communicator when one is adopted: every rank
  /// calls this with the SAME `placements` content, in the same order
  /// relative to its other collectives (the graph is identical on every
  /// rank, docs/user/placement.md). The processor set count and every named
  /// rank are agreed BEFORE the first `MPI_Comm_split`, so a caller bug that
  /// hands ranks a different-length or out-of-range `placements` is a
  /// rejection every rank returns identically, never a hang inside the split
  /// sequence that follows (a rank that disagreed would otherwise return
  /// early while its peers still expect it at the next collective).
  ///
  /// Errors, never throws: `E-MESH-MOVED`; `E-MOAB-PLACEMENT` for a mesh
  /// already loaded, processor sets already declared, a rank named outside the
  /// base communicator's `[0, size)`, or an MPI failure while splitting. No
  /// communicator adopted (the no-communicator mode above) is never an
  /// error, whether or not this build has MPI at all -- that mode needs
  /// none.
  Result<void> set_placements(std::vector<RankSet> placements);

  [[nodiscard]] Result<const MeshView*> placed_view(
      MeshId id, ComponentId placement) const override;
  [[nodiscard]] Result<std::size_t> placed_region_size(
      const Region& r, ComponentId placement) const override;

  /// Adopt an MPI communicator for partitioned reads.
  ///
  /// Pass `MPI_Comm_c2f(comm)`, not the `MPI_Comm` itself, so this header
  /// names no MPI type (Global Constraints). The handle is converted back
  /// with `MPI_Comm_f2c` inside the one translation unit that may see
  /// `mpi.h`. MPI must already be initialised -- converting a Fortran handle
  /// before `MPI_Init` is undefined -- and the value must be a live handle
  /// from the SAME MPI implementation this library was built against; there
  /// is no way to validate a foreign integer, so a wrong one is a crash, not
  /// an error return.
  ///
  /// Must be called before the first `load`; the read options depend on it.
  /// Errors: `E-MOAB-NOMPI` if this build has no MPI, `E-MOAB-MPIINIT` if
  /// MPI is not initialised, `E-MOAB-COMM` for `MPI_COMM_NULL`, for a call
  /// made after a mesh has already been loaded, or when duplicating the
  /// communicator fails.
  ///
  /// COLLECTIVE over the given communicator: the backend works on its own
  /// duplicate with `MPI_ERRORS_RETURN` set, so that its MPI failures come
  /// back as `Failure`s instead of aborting inside the MPI call. Every rank
  /// of the communicator must call this, in the same order relative to its
  /// other collectives. The duplicate is released when the backend is
  /// destroyed, unless MPI has already been finalised.
  Result<void> set_communicator(int fortran_comm_handle);

  /// How many ranks share the meshes this backend has read: 1 unless
  /// `set_communicator` adopted a communicator, and then that communicator's
  /// size.
  ///
  /// USED BY THE REMAP PATH, which needs it to reject work it cannot do
  /// correctly rather than to do it differently. `build_weights`
  /// intersects the cells THIS RANK holds; on a partitioned read that is a
  /// piece of the domain with no halo coverage, so a source cell straddling a
  /// partition boundary would have the part living on a neighbour silently
  /// dropped -- wrong numbers, not a crash, and conservative-looking ones at
  /// that, because the truncation is invisible to every local check. Until the
  /// covering-set machinery exists, a size above 1 with no halo built is
  /// `E-REMAP-PARALLEL`. `build_remap_halo` (`fuse/backend/remap_plan.hpp`) is
  /// what closes it, and once it has run the weights are conservative on a
  /// partitioned read.
  ///
  /// THIS VALUE IS LOAD BEARING AND ITS BOUNDARY SHOULD BE TESTED. `> 1`
  /// rejects; `== 1` must NOT, because a one-rank communicator is an
  /// ordinary MPI job whose mesh happens to be undivided, every cell is
  /// local, and the serial construction is exactly right. Rejecting it would
  /// take remap away from every MPI run and no serial test could see the
  /// difference. Guard on the property that breaks correctness -- the mesh
  /// being DIVIDED -- not on "a communicator exists".
  /// `test/parallel/remap_halo_mpi_test.cpp`'s
  /// `RemapHaloMpi.ERemapParallelRefusesAboveOneRankAndNotAtOne` exercises
  /// both sides of the boundary at every registered rank count 1..5:
  /// `== 1` must succeed, `> 1` must refuse `E-REMAP-PARALLEL`.
  ///
  /// 1 on a moved-from backend, which holds no meshes for anyone to remap.
  /// Never throws.
  [[nodiscard]] int communicator_size() const noexcept;

  /// Local indices of the owned (non-ghost) entities of mesh entity set `r`.
  ///
  /// The indices are `MeshView` local indices for the mesh entity set's dimension --
  /// the same `i` you pass to `measure(dim, i)` -- restricted to entities
  /// that are both in the mesh entity set's entity set and owned by this rank. They
  /// are returned in increasing order. A conservation sum uses this to avoid
  /// counting a shared entity once per rank that can see it.
  ///
  /// Same error codes as `region_size`.
  [[nodiscard]] Result<std::vector<std::size_t>> owned_entities(
      const Region& r) const;

  /// The rank that owns entity `local_index` of mesh entity set `r`, in the
  /// communicator given to `set_communicator`.
  ///
  /// `local_index` is a `MeshView` local index for the mesh entity set's dimension,
  /// as returned by `owned_entities`, NOT a position within that vector.
  /// With no communicator set, the mesh was read serially -- the effective
  /// communicator is `MPI_COMM_SELF`, whose only rank is 0 -- so the answer
  /// is 0.
  ///
  /// Errors: those of `region_size`, plus `E-MESH-INDEX` if `local_index` is
  /// out of range for the mesh entity set's dimension.
  [[nodiscard]] Result<int> partition_rank_of(const Region& r,
                                              std::size_t local_index) const;

  /// DRIVER-SIDE GEOMETRY SEAM. This should not be assumed at the library
  /// level; the driver needs to explicitly invoke it. `build_remap_halo`'s
  /// covering-set construction rejects a spherical pair whose two meshes fit to
  /// DIFFERENT radii (`E-REMAP-GEOMETRY`, "not the same domain") --
  /// correctly, and this library-level refusal is NEVER bypassed by
  /// anything below: grep `src/` for `project_onto_unit_sphere` finds no
  /// call outside a driver (`examples/`) or test fixture. MOAB's own
  /// non-matching spherical fixture pair (`wholeATM_T.h5m`, radius 1;
  /// `recMeshOcn.h5m`, radius 6371229, i.e. Earth radius in metres) is
  /// exactly the case this seam exists for: two REAL meshes, each
  /// individually a well-formed sphere, at two different nominal radii
  /// that a DRIVER -- not the library -- knows are meant to be the same
  /// domain in different units.
  ///
  /// `sphere_radius` MEASURES; `project_onto_unit_sphere` MEASURES THEN,
  /// UNLESS ALREADY ON THE UNIT SPHERE, RESCALES. Both exist in two forms:
  /// a `MeshId`-only overload resolving the backend's own DEFAULT
  /// `moab::Core` (`native_mesh(id)`'s own target), and a `(MeshId,
  /// ComponentId)` overload resolving the NAMED placement's own Core
  /// (`native_mesh(id, placement)`'s own target, `moab_backend.cpp`) --
  /// `set_placements()` gives every declared processor set its own,
  /// SEPARATE Core, with its own independently-read copy of every loaded
  /// mesh, so rescaling the default Core's copy does nothing to any
  /// placement's own copy and vice versa. Both overloads of both functions
  /// share ONE measurement implementation (`moab_backend.cpp`'s file-local
  /// `measure_sphere_radius()`), so they can never disagree about the
  /// formula. The placement overload refuses `E-MESH-PLACEMENT` on a rank
  /// that does not participate in the named placement -- the identical
  /// refusal `native_mesh(id, placement)` itself already gives there,
  /// since there is no Core on that rank to measure.
  ///
  /// `sphere_radius` returns `SphereRadiusReport{mean_radius,
  /// max_deviation}` (see that struct's own doc) after checking the mesh
  /// is a well-formed sphere AT ALL (a generous RELATIVE tolerance, the
  /// same `kProjectSphereTol` the rescale below uses: `E-MESH-GEOMETRY`
  /// if not) -- a pure query, no vertex coordinate is ever read back.
  ///
  /// `project_onto_unit_sphere` additionally decides whether `mean_radius`
  /// is "ALREADY the unit sphere": `max(|rmax - 1|, |rmin - 1|) <=
  /// kUnitSphereTol`, where `kUnitSphereTol = 1e-9` -- derived beside the
  /// constant (`moab_backend.cpp`) as roughly `4e6` times IEEE double's
  /// machine epsilon (`~2.22e-16`), enough headroom to absorb round-off
  /// accumulated summing millions of vertex coordinates while remaining
  /// many orders of magnitude below any real sphere's genuine eccentricity
  /// (Earth's own flattening is `~3e-3`). When already unit, NO vertex
  /// coordinate is read back or rewritten -- a measured, reported no-op
  /// (`SphereProjectionReport::already_unit == true`,
  /// `applied_factor == 1.0`), not merely a factor numerically close to
  /// one. Otherwise every vertex is rescaled by `1.0 / mean_radius`,
  /// reported in `applied_factor`, and `already_unit == false`.
  ///
  /// COLLECTIVE over the backend's communicator when one is set (for the
  /// placement overload, over THAT placement's own split communicator):
  /// a partitioned read may leave this rank holding only a subset of the
  /// mesh's vertices, so `mean_radius`/`max_deviation` and the decision to
  /// rescale are reduced over every rank holding a piece of it, the same
  /// reconciliation discipline `build_remap_halo` itself uses for
  /// `fit_surface`. Every rank holding this mesh (or this placement's own
  /// share of it) must call the same one of these functions for the same
  /// mesh, in the same order, exactly once before the first geometry-
  /// sensitive call (`build_remap_halo`, `build_weights`).
  ///
  /// Errors, never throws: `E-MESH-MOVED`, `E-MESH-ID`, `E-MESH-PLACEMENT`
  /// (placement overload, unknown or non-participating), `E-MESH-GEOMETRY`
  /// (the mesh's vertices are not at one common radius about the origin,
  /// so there is no single sphere to measure or project), `E-MESH-HALO`
  /// (the collective reduction failed), and MOAB read/write failures as
  /// `E-MOAB-QUERY`.
  [[nodiscard]] Result<SphereRadiusReport> sphere_radius(MeshId id) const;
  [[nodiscard]] Result<SphereRadiusReport> sphere_radius(MeshId id, ComponentId placement) const;
  [[nodiscard]] Result<SphereProjectionReport> project_onto_unit_sphere(MeshId id);
  [[nodiscard]] Result<SphereProjectionReport> project_onto_unit_sphere(MeshId id,
                                                                        ComponentId placement);

  // =========================================================================
  // THE REMAP HALO.
  //
  // On a partitioned read, the cells of the SOURCE mesh that overlap this
  // rank's target cells are, in general, spread over every rank. A conservative
  // remap needs them locally. `MeshView` already has the vocabulary for that:
  // its local entity set is "owned AND ghost" and `entity_count` is "the size
  // a per-entity field buffer over the whole mesh must have" -- so a halo cell
  // is an ordinary non-owned local entity, addressed by an ordinary local
  // index, and needs no new type anywhere. `RemapWeights` columns and
  // `MeshBackend::region_size` both follow `entity_count`, so they widen with
  // it automatically.
  //
  // ORDER MATTERS AND IS NOT NEGOTIABLE. `build_remap_halo` CHANGES
  // `entity_count(dim)`, and therefore `region_size`, and therefore every
  // buffer sized from it. It must be called after `load` and BEFORE anything
  // sizes a buffer on the mesh -- before `make_region_sizes`, `plan_buffers`,
  // `schedule()`, and before a component registers storage for a field on this
  // mesh. Calling it later is not detected here; it surfaces as the buffer
  // plan and the weights disagreeing (`E-RT-REMAP-SIZE`).
  // =========================================================================

  /// Whether `build_remap_halo(source, target, dim)` (declared in
  /// `fuse/backend/remap_plan.hpp`, because its geometry half belongs to the
  /// remap layer) has been run for this triple AT `placement` (default: the
  /// unplaced/default-Core decomposition, matching every caller that does
  /// not pass one). Never throws; false on a moved-from backend or an
  /// unknown id. `placement` is part of the bookkeeping
  /// key precisely because `build_remap_halo` can run once per placement on
  /// the SAME (source, target, dim) triple (a component's own decomposition
  /// is its own halo); omitting it would make the default Core and every
  /// declared placement clobber each other's recorded answer -- matching
  /// `fuse::native::NativeMeshBackend::has_remap_halo`'s own contract
  /// exactly (native_backend.hpp).
  [[nodiscard]] bool has_remap_halo(MeshId source, MeshId target, int dim,
                                    ComponentId placement = ComponentId{}) const noexcept;

  /// How many of mesh `source`'s dimension-`dim` local entities are halo cells
  /// adopted by `build_remap_halo` AT `placement` (default: the unplaced
  /// decomposition). They are the last `remap_halo_size` local indices only
  /// by accident of MOAB's handle ordering -- use `MeshView::is_owned`, not
  /// arithmetic on this number, to tell them apart.
  [[nodiscard]] std::size_t remap_halo_size(
      MeshId source, int dim, ComponentId placement = ComponentId{}) const noexcept;

  /// COLLECTIVE over `placement`'s own communicator (default: the backend's
  /// own, unplaced one), and every rank that participates in it must call
  /// this the same number of times in the same order. Fill the halo slots
  /// of `buffer` with the values their owners hold.
  ///
  /// `buffer` is `entity_count(dim) * components` doubles, entity-major with
  /// the component index fastest -- the layout `apply_weights` takes. Slots
  /// for owned entities are read, never written; halo slots are overwritten.
  /// A no-op, returning success, when the mesh carries no halo.
  ///
  /// This is the piece that makes a halo useful: `build_remap_halo` gives the
  /// weights somewhere to put their off-rank columns, and this fills them.
  ///
  /// Errors: `E-MESH-MOVED`, `E-MESH-ID`, `E-MESH-DIM`, `E-MESH-ARG` for a
  /// null buffer or `components < 1`, `E-MESH-HALO` for an MPI failure.
  [[nodiscard]] Result<void> exchange_remap_halo(
      MeshId source, int dim, double* buffer, int components,
      ComponentId placement = ComponentId{}) const;

  /// COLLECTIVE, the reverse of `exchange_remap_halo`: each halo slot's value
  /// is ADDED to the owner's slot, and the halo slots are then zeroed.
  /// `placement` is the SAME `ComponentId` `exchange_remap_halo` takes, for
  /// the identical reason.
  ///
  /// Exists for the conservation audit, which is the one quantity a halo makes
  /// non-local: a source cell straddling a target partition boundary has part
  /// of its overlap accounted on the owner and part on a rank holding it as a
  /// halo copy, and only their sum is the cell's real coverage. Summing
  /// locally and comparing per rank is how a correct map gets reported as a
  /// coverage failure -- and, worse, gets reported as one on SOME ranks only.
  [[nodiscard]] Result<void> accumulate_remap_halo(
      MeshId source, int dim, double* buffer, int components,
      ComponentId placement = ComponentId{}) const;

  /// INTERNAL SEAM for `build_remap_halo`'s geometry half, which lives in
  /// `src/backend/intx_weights.cpp` because that is where the surface
  /// classification and the orientation pass already are. Adopts `cells`
  /// (`moab::EntityHandle`s, as opaque integers) as halo cells of `source` at
  /// `dim`, skipping any whose global id this rank already holds. Nothing but
  /// `build_remap_halo` may call it.
  ///
  /// `placement` (default: the
  /// unplaced/default-Core decomposition) names WHICH Core/communicator
  /// `cells` were built in -- the caller's own `native_mesh(source,
  /// placement)` resolution, which it must pass the SAME `placement` to, so
  /// the `moab::EntityHandle`s in `cells` are read back through the Core
  /// that actually allocated them. Reading them through a DIFFERENT Core
  /// is undefined by MOAB's own contract, not
  /// merely wrong bookkeeping.
  Result<std::size_t> adopt_remap_halo(MeshId source, MeshId target, int dim,
                                       const std::uint64_t* cells,
                                       std::size_t count,
                                       ComponentId placement = ComponentId{});

  /// INTERNAL SEAM for `src/backend/intx_weights.cpp`, used
  /// nowhere else. `MeshView` deliberately exposes no connectivity and no
  /// coordinates, which is right for the runtime but leaves a supermesh
  /// builder unable to do its job: an intersection needs the actual polygons.
  /// Rather than widen the abstract `MeshBackend` -- every other consumer would
  /// then have to care -- the MOAB-specific remap path reaches through here,
  /// and it is the only thing that may.
  ///
  /// A bare `moab::Core*` would not be enough to identify a mesh: this
  /// backend keeps every loaded mesh in ONE `moab::Core`, distinguished only by
  /// its file set, so a bare interface pointer would leave the caller intersecting the union of everything
  /// loaded. The file set is returned alongside it.
  ///
  /// The pointer's lifetime is the backend's, exactly as for `view()`.
  /// `E-MESH-ID` for an unknown id, `E-MESH-MOVED` on a moved-from backend.
  /// Never throws.
  [[nodiscard]] Result<MoabNativeMesh> native_mesh(MeshId id) const;

  /// The mismatch `resolve()` (intx_weights.cpp) would otherwise have:
  /// `native_mesh(id)` above ALWAYS answers
  /// the backend's DEFAULT `moab::Core` (`impl_->mb`), regardless of
  /// `placed_view(id, placement)` having answered a DIFFERENT processor
  /// set's OWN `moab::Core` (`set_placements`'s own "one moab::Core per
  /// processor set" contract). A caller that resolves its VIEW through a
  /// placement but its Core/file_set/communicator through this overload's
  /// placement-oblivious sibling queries the WRONG Core -- the right file
  /// set handle, read against the wrong Interface, which is not even
  /// guaranteed to be a refusal (a same-numbered file set can easily exist,
  /// by coincidence, in both Cores) rather than a silently wrong answer.
  /// `placement` invalid, or no processor sets declared, answers EXACTLY
  /// `native_mesh(id)` above (mesh_view.hpp's own default-answer contract);
  /// otherwise, when this rank participates in `placement`, the triple
  /// belongs to THAT processor set's own `core`/`pcomm` and `meshes[id]`'s
  /// own `file_set()`. `E-MESH-PLACEMENT` when this rank does NOT
  /// participate in `placement` at all -- there is no Core to answer a
  /// handle into, the same reason `placed_view` answers the empty view
  /// rather than a handle from a Core that was never built here.
  [[nodiscard]] Result<MoabNativeMesh> native_mesh(MeshId id, ComponentId placement) const;

  /// Whether THIS RANK is inside `placement`'s own declared processor set --
  /// the same test `placed_view`/`native_mesh(id, placement)` already make
  /// internally, exposed so a caller (intx_weights.cpp's
  /// `resolve()`) can derive "structurally outside this placement" from the
  /// PLACEMENT itself rather than from a local entity count, which a
  /// genuinely empty (but in-placement) mesh could also produce. `false`
  /// for an invalid `placement` or when none were declared -- there is
  /// nothing to be "outside" of, which is a different question from
  /// whether this rank participates in a GIVEN one.
  [[nodiscard]] bool placement_participates(ComponentId placement) const noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fuse
