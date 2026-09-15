#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "soma/base/result.hpp"
#include "soma/graph/coupling_graph.hpp"
#include "soma/schedule/buffers.hpp"  // RegionSizes

namespace soma {

/// A point in physical space, in the mesh's coordinate units. Always three
/// components: a 1D or 2D mesh is embedded in R^3 with the unused coordinates
/// zero, so no caller has to branch on `dimension()` to read a position.
struct Point {
  double x = 0.0, y = 0.0, z = 0.0;
};

/// Spec §4.1: the core knows only counts, ownership, ids, and geometry.
/// No third-party type appears here (Global Constraints).
///
/// DESIGNED FOR THE DISTRIBUTED, UNSTRUCTURED CASE. The Phase 4 backend is a
/// serial structured grid, but Phase 5's MOAB backend must fit this interface
/// without changing the runtime, so the contract below deliberately refuses
/// four assumptions that a structured serial mesh would otherwise smuggle in:
///
///   1. `entity_count(dim)` is a LOCAL count -- the entities this rank can
///      see -- not a global one. A rank sees a partition, never the whole
///      mesh. Nothing here reports a global count; a global quantity is an
///      MPI reduction over owned entities, which is the caller's business.
///   2. The local entity set contains owned AND ghost entities, and the owned
///      ones are NOT required to be a prefix, or contiguous, or sorted. Use
///      `is_owned(dim, i)` to test an entity; `owned_count(dim)` only tells
///      you how many there are. Summing over `0 <= i < owned_count(dim)` is a
///      bug that happens to work on the serial backend.
///   3. `global_id(dim, i)` is NOT an index. Global ids are typically sparse
///      after partitioning, may exceed `entity_count(dim)` by orders of
///      magnitude, and must never be used to subscript anything. They are for
///      matching entities across ranks and across meshes.
///   4. `measure(dim, i)` genuinely depends on `i`. Cell volumes are not
///      uniform on an unstructured mesh, and a conservation check written
///      against a backend whose `measure` ignores `i` would pass for the
///      wrong reason.
///
/// INDEXING. A local index `i` is an ephemeral handle into this view, valid
/// for `0 <= i < entity_count(dim)` and meaningful only to this `MeshView`
/// instance. It carries no ordering guarantee across dimensions, ranks, or
/// reloads.
class MeshView {
 public:
  virtual ~MeshView() = default;

  /// The top topological dimension the mesh represents: 1 for a line, 3 for a
  /// volume mesh. Entities of lower dimension may also exist -- see
  /// `entity_count`.
  [[nodiscard]] virtual int dimension() const = 0;

  /// Number of dimension-`dim` entities visible on this rank, owned plus
  /// ghost. This is the size a per-entity field buffer over the whole mesh
  /// must have.
  ///
  /// Total, never throws, for any `int`: a dimension the mesh does not
  /// represent (including a negative one, or one above `dimension()`) yields
  /// 0. It is a query, so callers can use it to discover what a mesh offers.
  /// Note the converse does not hold: 0 does not prove the dimension is
  /// unrepresented, because a rank may legitimately hold no entities of it.
  [[nodiscard]] virtual std::size_t entity_count(int dim) const = 0;

  /// How many of the `entity_count(dim)` local entities this rank owns.
  /// Always `<= entity_count(dim)`. The remainder are ghosts, whose values
  /// are copies of another rank's and must be excluded from any global
  /// reduction to avoid double counting. Never throws.
  [[nodiscard]] virtual std::size_t owned_count(int dim) const = 0;

  /// True if local entity `i` is owned by this rank, false if it is a ghost.
  /// This, not an index range, is how a conservation sum skips ghosts.
  ///
  /// Throws `std::out_of_range` if `dim` is not represented or
  /// `i >= entity_count(dim)`.
  [[nodiscard]] virtual bool is_owned(int dim, std::size_t i) const = 0;

  /// Geometric centre of entity `i`, in physical coordinates.
  ///
  /// Throws `std::out_of_range` if `dim` is not represented, `i` is out of
  /// range, or the backend has no centroid for that dimension.
  [[nodiscard]] virtual Point centroid(int dim, std::size_t i) const = 0;

  /// The measure of entity `i`: its length (`dim == 1`), area (`dim == 2`) or
  /// volume (`dim == 3`).
  ///
  /// PRECISE DEFINITION -- conservation depends on every word of it:
  ///
  ///   * PHYSICAL, not reference. It is the `dim`-dimensional measure of the
  ///     element as it sits in physical space, in the mesh's coordinate units
  ///     raised to the power `dim`. It is not the measure of a reference
  ///     element, and it is not multiplied by any quadrature weight. For a
  ///     curved or high-order element it is the integral of the Jacobian
  ///     determinant over the reference element, i.e. the true measure of the
  ///     curved image, not the measure of its straight-sided approximation.
  ///   * UNSIGNED. Always `>= 0`. An inverted or negatively-oriented element
  ///     reports its absolute measure. A signed measure would let two bad
  ///     cells cancel inside a global sum and turn a broken mesh into a
  ///     conservation check that passes.
  ///   * ADDITIVE AND NON-OVERLAPPING. Summing `measure(dim, i)` over a set
  ///     of entities that tiles a domain without overlap reproduces that
  ///     domain's `dim`-measure, to rounding. Two consequences the backends
  ///     must respect: the dimension-`dimension()` entities tile the domain,
  ///     so their measures sum to its volume; and a boundary face set tiles
  ///     the boundary, so its measures sum to the surface area.
  ///   * LOCAL. Over a distributed mesh the sum must run over OWNED entities
  ///     only (`is_owned`) before the MPI reduction; including ghosts double
  ///     counts.
  ///
  /// `dim == 0` has no geometric measure. A backend that defines a dual or
  /// control volume for vertices may return it; one that does not must throw
  /// rather than return 0, because a silent zero weight removes a vertex from
  /// a conservation sum without a diagnostic.
  ///
  /// Throws `std::out_of_range` if `dim` is not represented, `i` is out of
  /// range, or the backend defines no measure for that dimension.
  [[nodiscard]] virtual double measure(int dim, std::size_t i) const = 0;

  /// The globally unique identifier of local entity `i`.
  ///
  /// Unique among dimension-`dim` entities across ALL ranks -- this is the
  /// MOAB/iMOAB `GLOBAL_ID` convention, which numbers each entity type
  /// separately, so ids from different `dim` may collide and must not be
  /// mixed in one map. Not contiguous, not ordered, not bounded by
  /// `entity_count(dim)`, and NOT an index. A rank's local entities are an
  /// arbitrary subset of the global numbering; an owned entity and its ghost
  /// copies on other ranks share one id, which is exactly what makes the id
  /// usable for matching.
  ///
  /// Throws `std::out_of_range` if `dim` is not represented or
  /// `i >= entity_count(dim)`.
  [[nodiscard]] virtual std::size_t global_id(int dim, std::size_t i) const = 0;
};

/// Loads meshes and answers questions about them. Phase 4 provides the serial
/// structured backend; Phase 5 adds `MoabMeshBackend` behind this same
/// interface.
class MeshBackend {
 public:
  virtual ~MeshBackend() = default;

  /// Load one mesh, interpreting `decl.source` however the backend sees fit
  /// (spec §4.1: the source string is opaque above this layer).
  ///
  /// MESH ID CONTRACT. The n-th SUCCESSFUL `load` defines `MeshId{n}`,
  /// counting from zero; a failed load consumes no id. `CouplingGraph`
  /// assigns its `MeshId`s the same way in `add_mesh`, so loading the graph's
  /// meshes in declaration order -- and stopping at the first failure -- makes
  /// the two numberings agree. Callers that cannot guarantee that order must
  /// use `mesh_id(name)` instead of assuming it; `make_region_sizes` does.
  ///
  /// `decl.name` is remembered so `mesh_id` can find the mesh again. It is
  /// not required to be unique, but a duplicate makes `mesh_id` ambiguous
  /// and therefore an error -- see there.
  virtual Result<void> load(const MeshDecl& decl) = 0;

  /// The id of the loaded mesh whose `MeshDecl::name` is `name`.
  ///
  /// WHY THIS EXISTS. `MeshId` is positional on both sides -- the n-th
  /// `add_mesh` on a `CouplingGraph` and the n-th successful `load` on a
  /// backend -- so the graph's ids and the backend's ids agree only when the
  /// backend was fed the graph's meshes in declaration order with no gaps.
  /// Nothing in either type enforces that. A backend that loads in a
  /// different order (MOAB reading a partitioned set of files, a driver that
  /// loads on demand, a caller that skips a mesh it does not need) silently
  /// shifts every id, and a positional lookup then sizes a region against
  /// the WRONG MESH: a plausible number, no diagnostic, and a buffer plan
  /// that is quietly too small or too large. Name resolution is the only
  /// thing that ties the two id spaces together, so `make_region_sizes`
  /// goes through here rather than trusting the ordering.
  ///
  /// Errors, never throws:
  ///   * `E-MESH-NAME` -- no loaded mesh carries that name. An empty `name`
  ///     never matches anything, even a mesh loaded with an empty name,
  ///     because "the mesh with no name" is not a thing a caller can mean.
  ///   * `E-MESH-NAME-AMBIGUOUS` -- more than one does. Returning the first
  ///     would silently choose between two different meshes.
  [[nodiscard]] virtual Result<MeshId> mesh_id(std::string_view name) const = 0;

  /// The view for `id`, or `E-MESH-ID`. The pointer is owned by the backend
  /// and is valid until the backend is destroyed or moved from; it must not
  /// be deleted. Loading further meshes does not invalidate views already
  /// handed out.
  [[nodiscard]] virtual Result<const MeshView*> view(MeshId id) const = 0;

  /// How many entities region `r` designates on this rank -- owned plus
  /// ghost, matching `MeshView::entity_count`, because a field buffer needs a
  /// slot for every locally visible entity. (Conservation sums restrict to
  /// owned entities afterwards via `MeshView::is_owned`; the two counts are
  /// deliberately separate and must not be conflated.)
  ///
  /// Errors rather than guessing: `E-MESH-ID` for an unknown mesh,
  /// `E-MESH-SET` for an entity set the mesh does not define, `E-MESH-DIM`
  /// for a dimension it does not represent or one inconsistent with the set.
  /// Never throws.
  [[nodiscard]] virtual Result<std::size_t> region_size(const Region& r) const = 0;

  /// Short backend identifier for diagnostics, e.g. "serial", "moab".
  [[nodiscard]] virtual std::string name() const = 0;
};

/// Adapt a backend to the callback Phase 3 Task 3.5 defined.
///
/// MESH IDS ARE RESOLVED BY NAME, NOT BY POSITION. A `Region` carries the
/// GRAPH's `MeshId`, which indexes `g`'s meshes. The backend has its own,
/// independent, also-positional id space. Passing one to the other works
/// only while the two happen to line up. This adapter therefore reads
/// `g.mesh(r.mesh).name` and asks the backend for ITS id for that name,
/// so a backend that loaded the meshes in a different order (or loaded
/// extra ones) still sizes the right region instead of a wrong-but-plausible
/// one. `MeshBackend::mesh_id` documents why that ordering cannot be
/// assumed.
///
/// LIFETIME. The returned callable stores references to `g` and `backend`.
/// Both must outlive it, and `g` must not have regions added or removed while
/// it is in use.
///
/// FAILURE IS LOUD. `RegionSizes` returns a bare `std::size_t`, with no way to
/// report an error, so the adapter THROWS rather than substituting a
/// plausible number:
///
///   * `std::out_of_range` if the id is invalid or outside `g`. `buffers.hpp`
///     promises `plan_buffers` never passes one, but an unchecked invariant is
///     not enforced, and this is the check that would have caught it.
///   * `std::runtime_error`, carrying the backend's error code and message, if
///     the region cannot be sized -- a mesh name the backend never loaded, an
///     ambiguous mesh name, an unknown entity set, a dimension the mesh does
///     not represent.
///
/// Returning 0 instead, as a size-returning callback naturally tempts one to
/// do, would allocate an empty buffer for a real region and let the run
/// proceed to produce silently empty exchanges. `plan_buffers` returns a plain
/// value and is documented to throw, so an exception propagates to the caller
/// intact.
[[nodiscard]] RegionSizes make_region_sizes(const CouplingGraph& g,
                                            const MeshBackend& backend);

}  // namespace soma
