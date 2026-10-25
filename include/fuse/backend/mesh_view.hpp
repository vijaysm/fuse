#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "fuse/backend/cell_simplices.hpp"  // CellShape
#include "fuse/base/result.hpp"
#include "fuse/model/coupling_graph.hpp"
#include "fuse/schedule/buffers.hpp"  // RegionSizes

namespace fuse {

/// A point in physical space, in the mesh's coordinate units. Always three
/// components: a 1D or 2D mesh is embedded in R^3 with the unused coordinates
/// zero, so no caller has to branch on `dimension()` to read a position.
struct Point {
  double x = 0.0, y = 0.0, z = 0.0;
};

/// Spec Section 4.1: the core knows only counts, ownership, ids, and geometry.
/// No third-party type appears here (Global Constraints).
///
/// DESIGNED FOR THE DISTRIBUTED, UNSTRUCTURED CASE. The native backend is a
/// serial structured grid, but the MOAB backend must fit this interface
/// without changing the runtime, so the contract below deliberately rejects
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

  /// --- cell geometry, as simplices -----------------------------------------
  ///
  /// The supermesh kernel clips polytopes, so it needs real geometry, and a
  /// centroid and a measure are not geometry. The contract is a **simplex
  /// decomposition**: triangles in 2D, tetrahedra in 3D. It was chosen over
  /// "cell type plus vertex list" and over a general boundary representation
  /// deliberately:
  ///
  ///   * clipping a simplex needs no convexity assumption, so a non-convex
  ///     cell, or a hexahedron whose quadrilateral faces are warped and
  ///     therefore not planar, is handled EXACTLY -- the intersection of a
  ///     cell is exactly the sum of the intersections of its simplices;
  ///   * there is one geometric case in the kernel rather than six, so one
  ///     thing to test rather than a table of canonical vertex orderings that
  ///     every backend would have to agree with silently;
  ///   * a degenerate cell shows up as a degenerate simplex, which has zero
  ///     measure and is detectable, instead of as an inverted face.
  ///
  /// The cost is that a backend must decompose, and that a cell costs several
  /// clips instead of one. That is the trade that was taken.
  ///
  /// `simplex_count` returns 0 for a view that carries no geometry at all --
  /// which is legal, and which the weight builder turns into a rejection naming
  /// this backend rather than into a silently empty supermesh.
  [[nodiscard]] virtual std::size_t simplex_count(int dim,
                                                  std::size_t i) const = 0;

  /// Writes the `dim + 1` corners of simplex `s` of entity `i` into `out`,
  /// which must have room for at least `dim + 1` points. Orientation is not
  /// required: the kernel uses unsigned measures, so a backend need not know
  /// or maintain a winding convention.
  virtual void simplex_points(int dim, std::size_t i, std::size_t s,
                              Point* out) const = 0;

  /// --- cell-to-vertex connectivity, for a general nodal basis --------------
  ///
  /// WHY THIS EXISTS. The simplex decomposition above is
  /// exactly what the supermesh/clipping kernel needs, but it is NOT enough
  /// to interpolate a Q1 field on a quadrilateral or hexahedron: a hex's
  /// trilinear basis needs its 8 ACTUAL corners in a known order, not the 6
  /// tetrahedra `simplex_points` decomposes it into (whose "corners" are
  /// physical points with no local vertex index attached, and which do not
  /// preserve the cell's own multilinear shape). General consistent
  /// interpolation (`fuse/backend/general_interp.hpp`) needs, for a cell of
  /// dimension-`dim` index `i`: which SHAPE it is, and which `MeshView`
  /// dimension-0 (vertex) local indices are its corners, in
  /// `cell_simplices.hpp`'s canonical corner order (the MOAB/Exodus
  /// convention every `CellShape` table already uses) -- because a nodal
  /// field's degrees of freedom ARE `MeshView` dimension-0 local indices
  /// (mesh_view.hpp's own field-extent contract), and the weight this
  /// function's caller computes for corner `k` must land in the sparse
  /// matrix's column `vertices[k]` for the result to be indexed consistently
  /// with everything else that reads a nodal field's buffer.
  ///
  /// DEFAULT: rejects `E-MESH-NO-CELL-VERTICES`. Most existing `MeshView`s
  /// have no answer -- a structured serial grid keeps no explicit
  /// connectivity list at all (its cells are addressed by index arithmetic),
  /// and a hand-built 1D test double (`ArbitraryLine`,
  /// `test/backend/operator_conformance_test.cpp`) orders its vertices by
  /// coordinate rather than storing a corner list. Implemented for
  /// `fuse::native::NativeMeshView` (its own `CellType`/corner-list storage
  /// already matches this convention exactly), the structured
  /// `SerialMeshBackend`'s hexahedral cells (`ix`/`iy`/`iz` arithmetic gives
  /// the 8 corners directly), and `MoabMeshView` (MOAB's own
  /// `get_connectivity`, translated through each dimension's existing
  /// handle<->local-index tables).
  ///
  /// Rejects (never throws), beyond the default:
  ///   * `E-MESH-NO-CELL-VERTICES` -- this entity's cell type has no
  ///     `CellShape` equivalent (a `Point`, or a general `Polyhedron` whose
  ///     faces are not a fixed corner list) -- named so a caller can tell
  ///     "this backend never answers" apart from "this specific cell has no
  ///     answer", though both are rejected the same way today.
  struct CellVertices {
    CellShape shape = CellShape::Edge;
    /// `MeshView` dimension-0 local indices, in `cell_simplices.hpp`'s
    /// canonical corner order for `shape` -- length `2` for `Edge`, `3` for
    /// `Triangle`, `4` for `Quad`/`Tet`, `5` for `Pyramid`, `6` for `Prism`,
    /// `8` for `Hex`; variable (`>= 3`) for `Polygon`.
    std::vector<std::size_t> vertices;
  };
  [[nodiscard]] virtual Result<CellVertices> cell_vertices(int dim,
                                                           std::size_t i) const {
    (void)i;
    return Failure{
        "E-MESH-NO-CELL-VERTICES",
        "MeshView::cell_vertices: this view carries no cell-to-vertex "
        "connectivity at dimension " +
            std::to_string(dim) +
            " (the default MeshView implementation always refuses); general "
            "nodal interpolation needs a concrete backend that overrides it"};
  }
};

/// Loads meshes and answers questions about them. The native backend provides
/// the serial structured case; `MoabMeshBackend` sits behind this same
/// interface.
class MeshBackend {
 public:
  virtual ~MeshBackend() = default;

  /// Load one mesh, interpreting `decl.source` however the backend sees fit
  /// (spec Section 4.1: the source string is opaque above this layer).
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
  /// shifts every id, and a positional lookup then sizes a mesh entity set against
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

  /// How many entities mesh entity set `r` designates on this rank -- owned plus
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

  // ---- per-placement decomposition ----
  //
  // Each component holds its OWN decomposition of a mesh it shares with
  // another: the mesh laid out over that component's ranks, not over the
  // job's. Without this a rank holds exactly one piece of each mesh, shared by
  // every component placed on it, and two PARTIALLY overlapping processor sets
  // cannot both hold the whole mesh entity set -- A on {0,1,2} and B on {1,2,3} over one
  // 4-way partition leaves A without piece 3, so B's rank 3 has no source.
  // Disjoint and identical processor sets happened to work; partial could not be
  // expressed at all.
  //
  // So these answer "this rank's piece of mesh `id` AS COMPONENT `placement`
  // SEES IT". A rank in two processor sets has two pieces, one each. That is
  // inherent to partial overlap, not a cost of the interface.
  //
  // DEFAULTS. A backend with one decomposition per mesh -- serial, or a
  // parallel backend all of whose components share one processor set -- need not
  // override either: the defaults return the processor-set-free answer, which is
  // then correct for every component. An invalid `placement` (`ComponentId{}`)
  // always means "the backend's default decomposition".
  //
  // CONTRACT FOR AN OVERRIDING BACKEND:
  //   * two components with IDENTICAL processor sets must get the same
  //     decomposition (the same entities in the same local order). The IR
  //     emits no Transport between identical processor sets (rules_channel.cpp,
  //     `same_as`), so a backend that partitioned them differently would have
  //     the value read at the wrong local indices with no data move to fix it;
  //   * the pointer is owned by the backend and stays valid for its lifetime,
  //     like `view`'s;
  //   * errors are the same codes `view` / `region_size` use, plus
  //     `E-MESH-PLACEMENT` for a processor set the backend was never told about,
  //     rather than a silent fallback to a different decomposition.
  //
  // Named rather than overloaded on purpose: an overload of `view` would be
  // HIDDEN in every derived class that overrides `view(MeshId)` (C++ name
  // lookup stops at the first scope that declares the name), so a direct call
  // on a concrete backend would fail to compile or pick the wrong function.
  [[nodiscard]] virtual Result<const MeshView*> placed_view(
      MeshId id, ComponentId placement) const {
    (void)placement;
    return view(id);
  }

  [[nodiscard]] virtual Result<std::size_t> placed_region_size(
      const Region& r, ComponentId placement) const {
    (void)placement;
    return region_size(r);
  }

  /// Short backend identifier for diagnostics, e.g. "serial", "moab".
  [[nodiscard]] virtual std::string name() const = 0;
};

/// ---------------------------------------------------------------------------
/// The one definition of a field's extent (docs/dev/nodal-fields.md)
/// ---------------------------------------------------------------------------
///
/// How many scalar slots ONE FIELD needs, per entity of the block it is
/// declared over, decided from `(disc, order)` ALONE -- never re-derived a
/// second way at a second call site. Every place that used to size a field's
/// buffer, or check one, by `view.entity_count(region_dim)` unconditionally
/// now goes through this function instead: `make_region_sizes`
/// (serial_backend.cpp), the Runner's conservation-measure builder
/// (`weights_for`, runner.cpp), and the parallel Transport dimension
/// selection (`attach_comm_graphs`, runner.cpp).
///
///   * FV, and `DiscontinuousGLL` order 0, read one value per `region_dim`
///     entity -- a cell. This is `region_size`'s existing, UNCHANGED
///     behaviour; nothing here alters what `region_size`/`placed_region_size`
///     mean on their own (both keep sizing by the mesh entity set's own dimension for
///     every caller that asks them directly, with no field in view at all).
///   * `ContinuousGLL` order 1 is genuinely nodal -- `connect.hpp`'s
///     `to_semantics` already maps that combination to
///     `SemanticLocation::Vertex`, and `consistent_interp.hpp`'s H/H^T
///     already index `MeshView` dimension-0 local indices for it -- so its
///     extent is the mesh's VERTEX count, `view.entity_count(0)`, regardless
///     of `region_dim`.
///   * Everything else -- `ContinuousGLL` order != 1, `DiscontinuousGLL`
///     order >= 1, `PointCloud` -- is a NAMED rejection, `E-EXTENT-DISC|`
///     rather than a guess: a spectral element needs as many DOFs per cell as
///     `order` implies, which nothing here computes, and a point cloud has no
///     cell topology to size a field by at all.
[[nodiscard]] inline Result<std::size_t> field_extent(Discretization disc,
                                                       int order,
                                                       const MeshView& view,
                                                       int region_dim) {
  if (disc == Discretization::FV ||
      (disc == Discretization::DiscontinuousGLL && order == 0)) {
    return view.entity_count(region_dim);
  }
  if (disc == Discretization::ContinuousGLL && order == 1) {
    return view.entity_count(0);
  }
  return Failure{
      "E-EXTENT-DISC",
      "field_extent: " + std::string(to_cstring(disc)) + " order " +
          std::to_string(order) +
          " has no field-extent definition; only FV/DiscontinuousGLL order "
          "0 (cell count) and ContinuousGLL order 1 (vertex count) are "
          "implemented"};
}

/// Which of `field_extent`'s two implemented shapes every field DECLARED
/// over `region` implies, read off the graph rather than assumed.
///
/// A mesh entity set with no field on it yet (an unused declaration, or one probed
/// before any `add_field` call) defaults to `Cell` -- `region_size`'s
/// existing behaviour -- since there is no field to disagree with it.
enum class RegionExtentKind { Cell, Vertex };

/// Classifies the SAME `(disc, order)` combinations `field_extent` accepts,
/// without needing a `MeshView` at all -- the caller supplies one only once
/// a `Vertex`/`Cell` outcome is in hand and the real dimension is known.
/// Rejects `E-EXTENT-DISC` for a field whose `(disc, order)` `field_extent`
/// itself would reject, and `E-EXTENT-MIXED` when the mesh entity set carries BOTH a
/// cell field and a nodal one: one mesh entity set cannot be sized two different ways
/// at once, and choosing either silently would size the other field wrong
/// with no diagnostic.
[[nodiscard]] inline Result<RegionExtentKind> region_extent_kind(
    const CouplingGraph& g, RegionId region) {
  bool saw_cell = false;
  bool saw_vertex = false;
  std::string cell_field;
  std::string vertex_field;
  for (std::size_t i = 0; i < g.field_count(); ++i) {
    const FieldId fid{static_cast<std::uint32_t>(i)};
    const FieldSpec& f = g.field(fid);
    if (!(f.region == region)) continue;
    if (f.disc == Discretization::ContinuousGLL && f.order == 1) {
      saw_vertex = true;
      vertex_field = f.name;
      continue;
    }
    if (f.disc == Discretization::FV ||
        (f.disc == Discretization::DiscontinuousGLL && f.order == 0)) {
      saw_cell = true;
      cell_field = f.name;
      continue;
    }
    return Failure{"E-EXTENT-DISC",
                   "region_extent_kind: field '" + f.name + "' on region " +
                       std::to_string(region.value()) + " uses " +
                       std::string(to_cstring(f.disc)) + " order " +
                       std::to_string(f.order) +
                       ", which has no field-extent definition (only FV/"
                       "DiscontinuousGLL order 0 and ContinuousGLL order 1 "
                       "are implemented)"};
  }
  if (saw_cell && saw_vertex) {
    return Failure{"E-EXTENT-MIXED",
                   "region_extent_kind: region " +
                       std::to_string(region.value()) +
                       " is used by a cell field ('" + cell_field +
                       "') and a nodal field ('" + vertex_field +
                       "'); one region cannot be sized two different ways "
                       "at once"};
  }
  return saw_vertex ? RegionExtentKind::Vertex : RegionExtentKind::Cell;
}

/// Adapt a backend to the `RegionSizes` callback (fuse/schedule/buffers.hpp).
///
/// BEFORE doing anything else, this function executes every `OpKind::BuildRemapHalo` instruction
/// `dag` carries (`fuse::execute_remap_halo_instructions()`,
/// fuse/backend/remap_halo_instructions.hpp) -- COLLECTIVELY, structural
/// conditions only (every rank compiles the identical `dag` from the
/// identical `CouplingGraph`, so the set and order of instructions
/// already agree without any data-dependent branch) -- and only THEN
/// reads the entity counts the returned callback reports. This is the
/// ONE call site a remap halo's real geometry exchange can run from: the
/// returned `RegionSizes` callback is invoked once, synchronously, inside
/// `schedule()` itself (fuse/schedule/buffers.hpp), to freeze every
/// `BufferDecl::elements` into the `ExecutionPlan` -- which happens
/// BEFORE the `Runner` that will eventually read those buffers is even
/// constructed (`Runner`'s constructor takes the already-built
/// `ExecutionPlan`). Application code therefore does not call
/// `execute_remap_halo_instructions()` itself between loading the
/// backend and calling `schedule()`: this function does it, so a caller
/// cannot forget it or place it too late by hand. `make_region_sizes()`
/// is the one function EVERY caller passes `dag` to before `schedule()`
/// runs, so it is the one place the halo build and the sizing it feeds
/// cannot drift apart.
///
/// A graph with no cross-rank weighted channel at all makes this an
/// unconditional no-op (zero instructions, zero cells) -- exactly as
/// `execute_remap_halo_instructions()`'s own doc says -- so calling this
/// unconditionally, as every existing caller already does, changes
/// nothing for a graph that never needed a halo.
///
/// `backend` is a non-const reference: building a remap halo mutates the backend's own mesh
/// state (ghost entities adopted as ordinary non-owned local ones), and
/// that mutation must be visible to the `placed_region_size`/
/// `placed_view` calls the returned callback makes later.
///
/// MESH IDS ARE RESOLVED BY NAME, NOT BY POSITION. A `Region` carries the
/// GRAPH's `MeshId`, which indexes `g`'s meshes. The backend has its own,
/// independent, also-positional id space. Passing one to the other works
/// only while the two happen to line up. This adapter therefore reads
/// `g.mesh(r.mesh).name` and asks the backend for ITS id for that name,
/// so a backend that loaded the meshes in a different order (or loaded
/// extra ones) still sizes the right mesh entity set instead of a wrong-but-plausible
/// one. `MeshBackend::mesh_id` documents why that ordering cannot be
/// assumed.
///
/// LIFETIME. The returned callable stores references to `g` and `backend`.
/// Both must outlive it, and `g` must not have mesh entity sets added or removed while
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
///     the mesh entity set cannot be sized -- a mesh name the backend never loaded, an
///     ambiguous mesh name, an unknown entity set, a dimension the mesh does
///     not represent -- OR if the collective halo build above itself
///     refuses (the same exception type, carrying THAT error's own code
///     and message, thrown before the callback is even built).
///
/// Returning 0 instead, as a size-returning callback naturally tempts one to
/// do, would allocate an empty buffer for a real mesh entity set and let the run
/// proceed to produce silently empty exchanges. `plan_buffers` returns a plain
/// value and is documented to throw, so an exception propagates to the caller
/// intact.
///
/// NO TWO-ARGUMENT OVERLOAD. Every caller in this tree already has `dag`
/// in hand at this call site (it is `schedule()`'s own next argument, a
/// few tokens away) -- `examples/*/main.cpp`, `src/abi/participant.cpp`,
/// `python/src/component_bindings.cpp` were all checked. A two-argument
/// form with no instruction stream to build the halo from would have to
/// either silently skip the halo or refuse
/// unconditionally for every graph regardless of whether it actually
/// has a qualifying channel, which is worse than not offering it: there
/// is no caller that needs it.
[[nodiscard]] RegionSizes make_region_sizes(const CouplingGraph& g,
                                            const OperatorDAG& dag,
                                            MeshBackend& backend);

}  // namespace fuse
