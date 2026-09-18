#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "fuse/backend/mesh_view.hpp"

namespace fuse {

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
/// `src/mesh/moab_backend.cpp`.
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
/// `src/mesh/moab_backend.cpp` for how each element type is integrated and
/// which ones are refused.
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
  /// MPI is not initialised, `E-MOAB-COMM` for `MPI_COMM_NULL` or for a call
  /// made after a mesh has already been loaded.
  Result<void> set_communicator(int fortran_comm_handle);

  /// How many ranks share the meshes this backend has read: 1 unless
  /// `set_communicator` adopted a communicator, and then that communicator's
  /// size.
  ///
  /// ADDED FOR THE REMAP PATH (Task 5.5), which needs it to refuse work it
  /// cannot do correctly rather than to do it differently. `build_weights`
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
  /// THIS VALUE IS LOAD BEARING AND ITS BOUNDARY IS TESTED. `> 1` refuses;
  /// `== 1` must NOT, because a one-rank communicator is an ordinary MPI job
  /// whose mesh happens to be undivided, every cell is local, and the serial
  /// construction is exactly right. Refusing it would take remap away from
  /// every MPI run and no serial test could see the difference. Guard on the
  /// property that breaks correctness -- the mesh being DIVIDED -- not on
  /// "a communicator exists". `test/unit/remap_parallel_test.cpp` pins both
  /// sides, the one-rank leg being the positive control.
  ///
  /// 1 on a moved-from backend, which holds no meshes for anyone to remap.
  /// Never throws.
  [[nodiscard]] int communicator_size() const noexcept;

  /// Local indices of the owned (non-ghost) entities of region `r`.
  ///
  /// The indices are `MeshView` local indices for the region's dimension --
  /// the same `i` you pass to `measure(dim, i)` -- restricted to entities
  /// that are both in the region's entity set and owned by this rank. They
  /// are returned in increasing order. A conservation sum uses this to avoid
  /// counting a shared entity once per rank that can see it.
  ///
  /// Same error codes as `region_size`.
  [[nodiscard]] Result<std::vector<std::size_t>> owned_entities(
      const Region& r) const;

  /// The rank that owns entity `local_index` of region `r`, in the
  /// communicator given to `set_communicator`.
  ///
  /// `local_index` is a `MeshView` local index for the region's dimension,
  /// as returned by `owned_entities`, NOT a position within that vector.
  /// With no communicator set, the mesh was read serially -- the effective
  /// communicator is `MPI_COMM_SELF`, whose only rank is 0 -- so the answer
  /// is 0.
  ///
  /// Errors: those of `region_size`, plus `E-MESH-INDEX` if `local_index` is
  /// out of range for the region's dimension.
  [[nodiscard]] Result<int> partition_rank_of(const Region& r,
                                              std::size_t local_index) const;

  // =========================================================================
  // THE REMAP HALO (R3).
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
  /// remap layer) has been run for this triple. Never
  /// throws; false on a moved-from backend or an unknown id.
  [[nodiscard]] bool has_remap_halo(MeshId source, MeshId target,
                                    int dim) const noexcept;

  /// How many of mesh `source`'s dimension-`dim` local entities are halo cells
  /// adopted by `build_remap_halo`. They are the last `remap_halo_size` local
  /// indices only by accident of MOAB's handle ordering -- use
  /// `MeshView::is_owned`, not arithmetic on this number, to tell them apart.
  [[nodiscard]] std::size_t remap_halo_size(MeshId source,
                                            int dim) const noexcept;

  /// COLLECTIVE over the backend's communicator, and every rank must call it
  /// the same number of times in the same order. Fill the halo slots of
  /// `buffer` with the values their owners hold.
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
  [[nodiscard]] Result<void> exchange_remap_halo(MeshId source, int dim,
                                                 double* buffer,
                                                 int components) const;

  /// COLLECTIVE, the reverse of `exchange_remap_halo`: each halo slot's value
  /// is ADDED to the owner's slot, and the halo slots are then zeroed.
  ///
  /// Exists for the conservation audit, which is the one quantity a halo makes
  /// non-local: a source cell straddling a target partition boundary has part
  /// of its overlap accounted on the owner and part on a rank holding it as a
  /// halo copy, and only their sum is the cell's real coverage. Summing
  /// locally and comparing per rank is how a correct map gets reported as a
  /// coverage failure -- and, worse, gets reported as one on SOME ranks only.
  [[nodiscard]] Result<void> accumulate_remap_halo(MeshId source, int dim,
                                                   double* buffer,
                                                   int components) const;

  /// INTERNAL SEAM for `build_remap_halo`'s geometry half, which lives in
  /// `src/remap/intx_weights.cpp` because that is where the surface
  /// classification and the orientation pass already are. Adopts `cells`
  /// (`moab::EntityHandle`s, as opaque integers) as halo cells of `source` at
  /// `dim`, skipping any whose global id this rank already holds. Nothing but
  /// `build_remap_halo` may call it.
  Result<std::size_t> adopt_remap_halo(MeshId source, MeshId target, int dim,
                                       const std::uint64_t* cells,
                                       std::size_t count);

  /// INTERNAL SEAM, added for Task 5.2 (`src/remap/intx_weights.cpp`) and used
  /// nowhere else. `MeshView` deliberately exposes no connectivity and no
  /// coordinates, which is right for the runtime but leaves a supermesh
  /// builder unable to do its job: an intersection needs the actual polygons.
  /// Rather than widen the abstract `MeshBackend` -- every other consumer would
  /// then have to care -- the MOAB-specific remap path reaches through here,
  /// and it is the only thing that may.
  ///
  /// DEVIATION FROM THE TASK 5.2 BRIEF, which specified `Result<void*>
  /// native_handle(MeshId)` returning the `moab::Core*` alone. That is not
  /// enough to identify a mesh: this backend keeps every loaded mesh in ONE
  /// `moab::Core`, distinguished only by its file set, so a bare interface
  /// pointer would leave the caller intersecting the union of everything
  /// loaded. The file set is returned alongside it.
  ///
  /// The pointer's lifetime is the backend's, exactly as for `view()`.
  /// `E-MESH-ID` for an unknown id, `E-MESH-MOVED` on a moved-from backend.
  /// Never throws.
  [[nodiscard]] Result<MoabNativeMesh> native_mesh(MeshId id) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace fuse
