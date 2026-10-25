#pragma once

// PartitionedTestBackend / PartitionedTestMesh, moved verbatim out of
// test/parallel/runner_transport_test.cpp so examples/layout_driver can link
// against the same per-placement decomposition the parallel test suite
// verifies against, rather than a second, drifting copy. Everything below is
// unchanged from the test file except this file's own #pragma/#include block
// and namespace-scope placement; see runner_transport_test.cpp's git history
// for the class's own change log.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// MPI is optional HERE on purpose. Every backend below partitions by a
// declared RankSet, and at a single participant that partition is the whole
// mesh -- no MPI call is needed to know that. Compiling without MPI lets the
// serial tier drive the same Runner fixtures the MPI tests use, which is the
// only way a one-rank failure in an MPI-registered test can be debugged in a
// sandbox where MPI cannot initialise at all.
#include "fuse/base/config.hpp"

#if FUSE_HAVE_MPI
#include <mpi.h>
#endif

namespace fuse_test {
/// This participant's rank, or 0 on a build with no MPI.
[[nodiscard]] inline int this_rank() {
#if FUSE_HAVE_MPI
  int me = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &me);
  return me;
#else
  return 0;
#endif
}
}  // namespace fuse_test


#include <fuse/backend/mesh_view.hpp>
#include <fuse/model/component.hpp>

// ---------------------------------------------------------------------------
// A MeshView whose local entities are a contiguous, population-balanced slice
// of a shared global id space -- unlike SerialMeshBackend, genuinely different
// per rank. Point/centroid are unused: GlobalId matching never reads them.
// ---------------------------------------------------------------------------
class PartitionedTestMesh final : public fuse::MeshView {
 public:
  PartitionedTestMesh(int dim, std::vector<std::size_t> gids)
      : dim_(dim), gid_(std::move(gids)) {}
  [[nodiscard]] int dimension() const override { return dim_; }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    return dim == dim_ ? gid_.size() : 0;
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override {
    return entity_count(dim);
  }
  [[nodiscard]] bool is_owned(int, std::size_t) const override { return true; }
  [[nodiscard]] fuse::Point centroid(int, std::size_t) const override {
    return fuse::Point{};
  }
  [[nodiscard]] double measure(int, std::size_t) const override { return 1.0; }
  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    if (dim != dim_ || i >= gid_.size()) {
      throw std::out_of_range("PartitionedTestMesh: index out of range");
    }
    return gid_[i];
  }

  /// No geometry. This mesh exists to exercise partitioning and transport, and
  /// it has no coordinates at all -- `centroid` returns the origin and
  /// `measure` returns 1. Reporting zero simplices is therefore the honest
  /// answer, and it is the case MeshView's contract names: a weight builder
  /// must refuse against this view rather than build an empty supermesh and
  /// call it conservative.
  [[nodiscard]] std::size_t simplex_count(int, std::size_t) const override {
    return 0;
  }
  void simplex_points(int, std::size_t, std::size_t,
                      fuse::Point*) const override {
    throw std::logic_error(
        "PartitionedTestMesh::simplex_points: this view reports zero "
        "simplices, so nothing may ask it for one");
  }

 private:
  int                      dim_;
  std::vector<std::size_t> gid_;
};

/// One mesh, split into a contiguous block per rank within whichever of
/// `fluid_`/`solid_` contains the calling rank; empty on a rank in neither.
///
/// PER-PLACEMENT DECOMPOSITION. Given `set_placements`, the
/// backend also builds one decomposition PER COMPONENT, each laid out over that
/// component's own ranks, and answers `placed_view` / `placed_region_size`
/// from it. A rank in two partially overlapping placements then holds two
/// blocks -- one per component -- which is what lets each side of a
/// CommGraph cover the whole region. A block is a function of the RankSet
/// alone, so identical placements get identical decompositions, as the
/// MeshBackend contract requires. Without `set_placements` the placed queries
/// fall back to the default view, exactly as a single-decomposition backend's
/// do.
/// `entity_set` on the `Region` is what the region guard checks -- this backend
/// does not itself narrow by it, unlike SerialMeshBackend's real one
/// (src/mesh/serial_backend.cpp:438-457).
class PartitionedTestBackend final : public fuse::MeshBackend {
 public:
  PartitionedTestBackend(fuse::RankSet fluid, fuse::RankSet solid,
                         std::size_t n_entities, int dim)
      : fluid_(fluid), solid_(solid), n_(n_entities), dim_(dim) {}

  /// One RankSet per graph component, indexed by ComponentId. Must precede
  /// every `load`, because each load builds one decomposition per entry.
  void set_placements(std::vector<fuse::RankSet> placements) {
    placements_ = std::move(placements);
  }

  fuse::Result<void> load(const fuse::MeshDecl& decl) override {
    const int me = fuse_test::this_rank();
    std::vector<std::size_t> gids;
    if (fluid_.contains(me)) {
      gids = block(fluid_, me);
    } else if (solid_.contains(me)) {
      gids = block(solid_, me);
    }
    meshes_.push_back(
        std::make_unique<PartitionedTestMesh>(dim_, std::move(gids)));
    names_.push_back(decl.name);

    std::vector<std::unique_ptr<fuse::MeshView>> per;
    for (const fuse::RankSet& rs : placements_) {
      per.push_back(std::make_unique<PartitionedTestMesh>(
          dim_, rs.contains(me) ? block(rs, me) : std::vector<std::size_t>{}));
    }
    placed_.push_back(std::move(per));
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(
      fuse::MeshId id, fuse::ComponentId placement) const override {
    if (!placement.valid() || placements_.empty()) return view(id);
    if (!id.valid() || id.value() >= placed_.size()) {
      return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    }
    if (placement.value() >= placed_[id.value()].size()) {
      return fuse::Failure{"E-MESH-PLACEMENT",
                           "no decomposition for component " +
                               std::to_string(placement.value())};
    }
    return placed_[id.value()][placement.value()].get();
  }

  [[nodiscard]] fuse::Result<std::size_t> placed_region_size(
      const fuse::Region& r, fuse::ComponentId placement) const override {
    const auto v = placed_view(r.mesh, placement);
    if (!v.ok()) return v.error();
    return v.value()->entity_count(dim_);
  }

  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(
      std::string_view name) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) {
        return fuse::MeshId{static_cast<std::uint32_t>(i)};
      }
    }
    return fuse::Failure{"E-MESH-NAME",
                       "no mesh named '" + std::string(name) + "'"};
  }

  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(
      fuse::MeshId id) const override {
    if (!id.valid() || id.value() >= meshes_.size()) {
      return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    }
    return meshes_[id.value()].get();
  }

  [[nodiscard]] fuse::Result<std::size_t> region_size(
      const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    return v.value()->entity_count(dim_);
  }

  [[nodiscard]] std::string name() const override { return "partitioned-test"; }

 private:
  [[nodiscard]] std::vector<std::size_t> block(fuse::RankSet rs, int me) const {
    // Position IN the set, not an offset from its first rank: a RankSet may
    // be strided (docs/user/placement.md), and then the two differ.
    int pos = 0;
    for (int r = rs.min_rank(); r < me; ++r) pos += rs.contains(r) ? 1 : 0;
    const std::size_t base = n_ / static_cast<std::size_t>(static_cast<int>(rs.size()));
    const std::size_t rem  = n_ % static_cast<std::size_t>(static_cast<int>(rs.size()));
    const std::size_t begin =
        static_cast<std::size_t>(pos) * base + std::min<std::size_t>(pos, rem);
    const std::size_t count =
        base + (static_cast<std::size_t>(pos) < rem ? 1 : 0);
    std::vector<std::size_t> gids(count);
    // REVERSED, not ascending: CommGraph's send/recv lists are sorted by
    // (global_id, ...) -- comm_graph.cpp:1125-1133 -- so an ascending block
    // here would put each entity's SORTED-MATCH position at exactly its own
    // LOCAL STORAGE index, making source_indices()/dest_indices() (and so
    // Runner's p.gather/p.scatter) the identity permutation on every rank at
    // every rank count this suite tries. That made TheDeliveredFieldDoesNot-
    // DependOnTheRankCount's mutation checks (replacing
    // p.gather[k]/p.scatter[k] with plain k) pass when they should fail: a
    // dropped index map went undetected because k already equalled the right
    // answer. Reversing this rank's local order (local index 0 holds the
    // block's LARGEST global id, not its smallest) decouples "sorted-by-gid
    // position" from "local storage index" without changing which ids this
    // rank owns, which is all CommGraph's matching depends on.
    for (std::size_t k = 0; k < count; ++k) gids[k] = begin + (count - 1 - k);
    return gids;
  }

  fuse::RankSet fluid_, solid_;
  std::size_t   n_;
  int           dim_;
  std::vector<std::unique_ptr<fuse::MeshView>> meshes_;
  std::vector<std::string>                     names_;
  std::vector<fuse::RankSet>                   placements_;
  // placed_[mesh][component]
  std::vector<std::vector<std::unique_ptr<fuse::MeshView>>> placed_;
};

// ---------------------------------------------------------------------------
// SharedVertexLineMesh / SharedVertexLineBackend -- used by
// test/parallel/nodal_field_test.cpp, for a genuinely SHARED
// vertex across ranks: unlike PartitionedTestMesh's disjoint blocks (which
// cannot exercise ownership at all -- there is nothing to own jointly), a
// line of `n_cells` cells / `n_cells + 1` vertices split across `num_ranks`
// ranks with a ONE-VERTEX OVERLAP at every internal boundary. Rank r's local
// vertex block is the CLOSED interval [lo(r), hi(r)] of global vertex
// indices, and hi(r) == lo(r+1): that boundary vertex is held by BOTH r and
// r+1. docs/dev/native-mesh.md's rule -- the LOWEST rank holding a vertex
// owns it -- then makes rank r+1's copy of its OWN FIRST vertex the
// non-owned one (except rank 0, which owns its first vertex because there is
// no lower rank to hold it), and rank r's copy of its OWN LAST vertex always
// owned (no rank lower than r can hold a vertex past r's block). Cells are a
// plain, non-overlapping partition -- `n_cells` cells, no cell is shared,
// exactly PartitionedTestMesh's block() shape -- so `dim=1` queries behave
// exactly as they would without this class existing at all.
// ---------------------------------------------------------------------------
class SharedVertexLineMesh final : public fuse::MeshView {
 public:
  SharedVertexLineMesh(std::size_t n_cells, int num_ranks, int my_rank)
      : n_cells_(n_cells) {
    // Vertex block: rank r covers global vertices [lo, hi], overlapping its
    // neighbours by exactly one vertex on each shared boundary.
    const std::size_t n_verts = n_cells + 1;
    const std::size_t base = n_verts / static_cast<std::size_t>(num_ranks);
    const std::size_t rem = n_verts % static_cast<std::size_t>(num_ranks);
    // A non-overlapping partition of the n_verts "OWNED" vertices first (this
    // is exactly PartitionedTestMesh::block()'s formula), then each rank
    // (other than the last) extends its block by one vertex to the right --
    // the shared boundary -- to also hold its right neighbour's first vertex.
    auto owned_begin = [&](int r) {
      return static_cast<std::size_t>(r) * base +
             std::min<std::size_t>(static_cast<std::size_t>(r), rem);
    };
    const std::size_t my_owned_begin = owned_begin(my_rank);
    const std::size_t my_owned_count =
        base + (static_cast<std::size_t>(my_rank) < rem ? 1 : 0);
    lo_ = my_owned_begin;
    hi_ = my_owned_begin + my_owned_count - 1;
    if (my_rank + 1 < num_ranks) hi_ += 1;  // shared boundary with the right neighbour
    owner_of_first_ = (my_rank == 0) ? 0 : my_rank - 1;  // shared with left neighbour
    my_rank_ = my_rank;
    // Cells: plain partition of the n_cells cells, aligned with each rank's
    // OWNED vertex range (not the overlap-extended one) -- cell k spans
    // vertices [k, k+1], and rank r owns cell k for k in
    // [my_owned_begin, my_owned_begin + my_owned_count), except the last
    // rank, which has one fewer cell than vertex (n_cells vs n_verts).
    cell_begin_ = my_owned_begin;
    cell_count_ = (my_rank + 1 < num_ranks) ? my_owned_count : my_owned_count - 1;
  }

  [[nodiscard]] int dimension() const override { return 1; }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    if (dim == 0) return hi_ - lo_ + 1;
    if (dim == 1) return cell_count_;
    return 0;
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override {
    if (dim != 0) return entity_count(dim);
    return entity_count(0) - (my_rank_ > 0 ? 1 : 0);
  }
  [[nodiscard]] bool is_owned(int dim, std::size_t i) const override {
    if (dim != 0) return true;  // cells are never shared here
    // Local index 0 is this rank's first vertex, shared with the LEFT
    // neighbour when one exists -- owned by whichever of the two is lower,
    // i.e. by `owner_of_first_`. Every other local vertex (including the
    // last, shared with the RIGHT neighbour when one exists) is owned here:
    // this rank IS the lower of the two holders for its own last vertex.
    if (i == 0) return owner_of_first_ == my_rank_;
    return true;
  }
  [[nodiscard]] fuse::Point centroid(int, std::size_t) const override { return fuse::Point{}; }
  [[nodiscard]] double measure(int dim, std::size_t) const override {
    if (dim == 1) return 1.0 / static_cast<double>(n_cells_);
    throw std::domain_error("SharedVertexLineMesh: no measure at dimension 0 (a vertex)");
  }
  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    if (dim == 0) {
      if (i > hi_ - lo_) throw std::out_of_range("SharedVertexLineMesh: vertex index out of range");
      return lo_ + i;
    }
    if (dim == 1) {
      if (i >= cell_count_) throw std::out_of_range("SharedVertexLineMesh: cell index out of range");
      return cell_begin_ + i;
    }
    throw std::out_of_range("SharedVertexLineMesh: no entities at this dimension");
  }
  [[nodiscard]] std::size_t simplex_count(int, std::size_t) const override { return 0; }
  void simplex_points(int, std::size_t, std::size_t, fuse::Point*) const override {
    throw std::logic_error(
        "SharedVertexLineMesh::simplex_points: this view reports zero simplices");
  }

  /// This rank's global vertex block, [lo, hi] inclusive -- for tests to
  /// check against directly rather than re-deriving it.
  [[nodiscard]] std::size_t lo() const noexcept { return lo_; }
  [[nodiscard]] std::size_t hi() const noexcept { return hi_; }

 private:
  std::size_t n_cells_;
  std::size_t lo_ = 0, hi_ = 0;
  std::size_t cell_begin_ = 0, cell_count_ = 0;
  int owner_of_first_ = 0;
  int my_rank_ = 0;
};

/// One nodal region (`SharedVertexLineMesh`) over every rank of `ranks`. No
/// per-placement decomposition (`set_placements`) -- one mesh, one
/// component, exactly what a plain nodal Transport (`InterfaceTransfer.
/// NodalFieldPassesThroughAPlainChannelOnTheSameMesh`'s parallel analogue)
/// needs, without PartitionedTestBackend's two-sided-interface machinery.
class SharedVertexLineBackend final : public fuse::MeshBackend {
 public:
  SharedVertexLineBackend(std::size_t n_cells, fuse::RankSet ranks)
      : n_cells_(n_cells), ranks_(ranks) {}

  fuse::Result<void> load(const fuse::MeshDecl& decl) override {
    const int me = fuse_test::this_rank();
    int pos = 0;
    for (int r = ranks_.min_rank(); r < me; ++r) pos += ranks_.contains(r) ? 1 : 0;
    meshes_.push_back(std::make_unique<SharedVertexLineMesh>(
        n_cells_, static_cast<int>(ranks_.size()), pos));
    names_.push_back(decl.name);
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(fuse::MeshId id) const override {
    if (!id.valid() || id.value() >= meshes_.size()) {
      return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    }
    return meshes_[id.value()].get();
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(
      fuse::MeshId id, fuse::ComponentId) const override {
    return view(id);
  }
  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    const int dim = r.dimension >= 0 ? r.dimension : v.value()->dimension();
    return v.value()->entity_count(dim);
  }
  [[nodiscard]] fuse::Result<std::size_t> placed_region_size(
      const fuse::Region& r, fuse::ComponentId) const override {
    return region_size(r);
  }
  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(std::string_view name) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return fuse::MeshId{static_cast<std::uint32_t>(i)};
    }
    return fuse::Failure{"E-MESH-NAME", "no mesh named '" + std::string(name) + "'"};
  }
  [[nodiscard]] std::string name() const override { return "shared-vertex-line-test"; }

 private:
  std::size_t n_cells_;
  fuse::RankSet ranks_;
  std::vector<std::unique_ptr<fuse::MeshView>> meshes_;
  std::vector<std::string> names_;
};

// ---------------------------------------------------------------------------
// GeometricLineMesh / GeometricLineBackend -- used by
// test/parallel/nodal_remap_distributed_test.cpp, for a
// distributed 1D interface with REAL COORDINATES and REAL CONNECTIVITY:
// unlike PartitionedTestMesh (no geometry at all: centroid is the origin,
// measure is 1) and SharedVertexLineMesh (also no geometry), this is what
// `build_distributed_interp_rows` needs on BOTH sides of a genuinely
// non-matching interpolation -- a source and target line at different
// resolutions, each split over its own (possibly different) RankSet, with
// physical vertex positions and Edge cell-to-vertex connectivity
// (`cell_vertices`) a real interpolation can be checked against a closed
// form. Ownership follows SharedVertexLineMesh's own rule exactly (the
// lowest rank holding a shared vertex owns it; cells are a plain,
// non-overlapping partition) -- duplicated rather than reused because that
// class deliberately carries NO geometry and this one deliberately does; the
// two diverge in exactly the fields the distributed algorithm needs.
// ---------------------------------------------------------------------------
class GeometricLineMesh final : public fuse::MeshView {
 public:
  /// `n_cells` cells over `[x_lo, x_hi]`, this rank's slice being position
  /// `my_rank_in_set` of `num_ranks_in_set` (both relative to the OWNING
  /// RankSet, not to MPI_COMM_WORLD -- the same convention
  /// SharedVertexLineMesh uses).
  /// A genuinely empty mesh: zero vertices, zero cells -- what a rank
  /// OUTSIDE this component's RankSet must supply to a collective call, as
  /// opposed to a one-rank slice of a real (if tiny) mesh, which would
  /// falsely claim ownership of a vertex no other rank agrees it owns.
  static GeometricLineMesh empty(double x_lo, double x_hi) {
    GeometricLineMesh m;
    m.x_lo_ = x_lo;
    m.x_hi_ = x_hi;
    m.n_cells_ = 0;
    m.lo_ = 0;
    m.hi_ = 0;
    m.cell_begin_ = 0;
    m.cell_count_ = 0;
    m.empty_ = true;
    return m;
  }

  GeometricLineMesh(std::size_t n_cells, int num_ranks_in_set, int my_rank_in_set, double x_lo,
                    double x_hi)
      : n_cells_(n_cells), x_lo_(x_lo), x_hi_(x_hi) {
    const std::size_t n_verts = n_cells + 1;
    const std::size_t base = n_verts / static_cast<std::size_t>(num_ranks_in_set);
    const std::size_t rem = n_verts % static_cast<std::size_t>(num_ranks_in_set);
    const auto owned_begin = [&](int r) {
      return static_cast<std::size_t>(r) * base + std::min<std::size_t>(static_cast<std::size_t>(r), rem);
    };
    const std::size_t my_owned_begin = owned_begin(my_rank_in_set);
    const std::size_t my_owned_count =
        base + (static_cast<std::size_t>(my_rank_in_set) < rem ? 1 : 0);
    lo_ = my_owned_begin;
    hi_ = my_owned_begin + my_owned_count - 1;
    if (my_rank_in_set + 1 < num_ranks_in_set) hi_ += 1;
    owner_of_first_ = (my_rank_in_set == 0) ? 0 : my_rank_in_set - 1;
    my_rank_ = my_rank_in_set;
    cell_begin_ = my_owned_begin;
    cell_count_ = (my_rank_in_set + 1 < num_ranks_in_set) ? my_owned_count : my_owned_count - 1;
  }

  [[nodiscard]] int dimension() const override { return 1; }
  [[nodiscard]] std::size_t entity_count(int dim) const override {
    if (empty_) return 0;
    if (dim == 0) return hi_ - lo_ + 1;
    if (dim == 1) return cell_count_;
    return 0;
  }
  [[nodiscard]] std::size_t owned_count(int dim) const override {
    if (empty_) return 0;
    if (dim != 0) return entity_count(dim);
    return entity_count(0) - (my_rank_ > 0 ? 1 : 0);
  }
  [[nodiscard]] bool is_owned(int dim, std::size_t i) const override {
    // Lowest-rank-owns-a-shared-vertex rule (docs/dev/native-mesh.md), worded
    // deliberately differently from SharedVertexLineMesh's own identical rule
    // just above in this file: an IDENTICAL `old` string in two mutation
    // anchors makes tools/mutate.py's STALE-ANCHOR check flag both as
    // ambiguous (this exact hazard is called out in resolve_remap's own
    // comment on the same tool). Same semantics, distinct text.
    if (empty_) return false;
    const bool shares_left_boundary_vertex = (dim == 0 && i == 0);
    if (shares_left_boundary_vertex) return my_rank_ == owner_of_first_;
    return true;
  }
  [[nodiscard]] fuse::Point centroid(int dim, std::size_t i) const override {
    const double h = (x_hi_ - x_lo_) / static_cast<double>(n_cells_);
    if (dim == 0) return fuse::Point{x_lo_ + static_cast<double>(lo_ + i) * h, 0, 0};
    return fuse::Point{x_lo_ + (static_cast<double>(cell_begin_ + i) + 0.5) * h, 0, 0};
  }
  [[nodiscard]] double measure(int dim, std::size_t) const override {
    if (dim == 1) return (x_hi_ - x_lo_) / static_cast<double>(n_cells_);
    throw std::domain_error("GeometricLineMesh: no measure at dimension 0 (a vertex)");
  }
  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    if (dim == 0) {
      if (i > hi_ - lo_) throw std::out_of_range("GeometricLineMesh: vertex index out of range");
      return lo_ + i;
    }
    if (dim == 1) {
      if (i >= cell_count_) throw std::out_of_range("GeometricLineMesh: cell index out of range");
      return cell_begin_ + i;
    }
    throw std::out_of_range("GeometricLineMesh: no entities at this dimension");
  }
  [[nodiscard]] std::size_t simplex_count(int dim, std::size_t) const override {
    return dim == 1 ? 1 : 0;
  }
  void simplex_points(int dim, std::size_t i, std::size_t, fuse::Point* out) const override {
    if (dim != 1) throw std::logic_error("GeometricLineMesh::simplex_points: wrong dimension");
    out[0] = centroid(0, i);
    out[1] = centroid(0, i + 1);
  }
  [[nodiscard]] fuse::Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
    if (dim != 1 || i >= cell_count_) {
      return fuse::Failure{"E-MESH-NO-CELL-VERTICES", "GeometricLineMesh: wrong dimension"};
    }
    CellVertices out;
    out.shape = fuse::CellShape::Edge;
    out.vertices = {i, i + 1};  // local dim-0 indices: cell i spans [i, i+1]
    return out;
  }

  [[nodiscard]] std::size_t lo() const noexcept { return lo_; }
  [[nodiscard]] std::size_t hi() const noexcept { return hi_; }

 private:
  GeometricLineMesh() = default;  // only `empty()` uses this

  std::size_t n_cells_ = 0;
  double x_lo_ = 0.0, x_hi_ = 1.0;
  std::size_t lo_ = 0, hi_ = 0;
  std::size_t cell_begin_ = 0, cell_count_ = 0;
  bool empty_ = false;
  int owner_of_first_ = 0;
  int my_rank_ = 0;
};

/// One nodal region (`GeometricLineMesh`) over every rank of `ranks`, over
/// `[x_lo, x_hi]` -- the geometric analogue of `SharedVertexLineBackend`,
/// for tests that need real coordinates on both the source and target side
/// of a distributed nodal remap.
class GeometricLineBackend final : public fuse::MeshBackend {
 public:
  GeometricLineBackend(std::size_t n_cells, fuse::RankSet ranks, double x_lo = 0.0,
                       double x_hi = 1.0)
      : n_cells_(n_cells), ranks_(ranks), x_lo_(x_lo), x_hi_(x_hi) {}

  fuse::Result<void> load(const fuse::MeshDecl& decl) override {
    const int me = fuse_test::this_rank();
    if (!ranks_.contains(me)) {
      // Not in this component: an EMPTY mesh (zero cells, zero vertices),
      // not a slice computed as if it were -- a rank outside `ranks_` must
      // still supply a valid MeshView to a collective call, exactly like
      // PartitionedTestBackend's own `gids = {}` branch.
      meshes_.push_back(std::make_unique<GeometricLineMesh>(GeometricLineMesh::empty(x_lo_, x_hi_)));
      names_.push_back(decl.name);
      return fuse::Result<void>::success();
    }
    int pos = 0;
    for (int r = ranks_.min_rank(); r < me; ++r) pos += ranks_.contains(r) ? 1 : 0;
    meshes_.push_back(std::make_unique<GeometricLineMesh>(
        n_cells_, static_cast<int>(ranks_.size()), pos, x_lo_, x_hi_));
    names_.push_back(decl.name);
    return fuse::Result<void>::success();
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(fuse::MeshId id) const override {
    if (!id.valid() || id.value() >= meshes_.size()) return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    return meshes_[id.value()].get();
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(fuse::MeshId id,
                                                                fuse::ComponentId) const override {
    return view(id);
  }
  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    const int dim = r.dimension >= 0 ? r.dimension : v.value()->dimension();
    return v.value()->entity_count(dim);
  }
  [[nodiscard]] fuse::Result<std::size_t> placed_region_size(const fuse::Region& r,
                                                             fuse::ComponentId) const override {
    return region_size(r);
  }
  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(std::string_view name) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return fuse::MeshId{static_cast<std::uint32_t>(i)};
    }
    return fuse::Failure{"E-MESH-NAME", "no mesh named '" + std::string(name) + "'"};
  }
  [[nodiscard]] std::string name() const override { return "geometric-line-test"; }

 private:
  std::size_t n_cells_;
  fuse::RankSet ranks_;
  double x_lo_, x_hi_;
  std::vector<std::unique_ptr<fuse::MeshView>> meshes_;
  std::vector<std::string> names_;
};

// ---------------------------------------------------------------------------
// MultiMeshGeometricBackend -- used by
// test/parallel/nodal_remap_runner_test.cpp, for driving ONE real
// Runner/CouplingGraph pair through a beam/fluid channel pair like FSI
// uses, where the two meshes have DIFFERENT shapes (cell counts) and
// DIFFERENT, independently-declared RankSets. `Runner` takes a single
// `MeshBackend&` for every region in the graph, but `GeometricLineBackend`
// (above) bakes ONE (n_cells, RankSet, x_lo, x_hi) into the whole instance,
// so serving a beam mesh and a fluid mesh of different shapes needs a
// SECOND `GeometricLineBackend` instance -- exactly what a single `Runner`
// cannot be handed. This class is the small composite that closes that
// gap: it holds one (n_cells, RankSet, x_lo, x_hi) `Spec` PER MESH NAME
// (registered before `load()`, mirroring `PartitionedTestBackend::
// set_placements`'s own "declare before loading" convention) and reuses
// `GeometricLineMesh` UNCHANGED -- one instance per registered mesh name,
// each built from its OWN spec exactly as `GeometricLineBackend::load`
// already does for its single mesh.
//
// GLOBAL IDS ARE CONSISTENT ACROSS PARTITIONS BY CONSTRUCTION, NOT BY
// ANYTHING THIS CLASS ADDS: `GeometricLineMesh`'s global id for a given
// local index is a pure function of (n_cells, x_lo, x_hi, that index's
// position along the line) -- never of the calling rank or the rank
// COUNT -- so the SAME mesh built at 1 rank, 3 ranks or 5 ranks numbers
// the SAME physical vertex with the SAME global id every time. This is
// the property `apply_distributed_HT`'s bit-identical-across-rank-counts
// argument (nodal_remap_distributed.hpp) already rests on; this class
// does not need to re-derive it, only avoid breaking it, which reusing
// `GeometricLineMesh` verbatim guarantees.
//
// No per-placement decomposition (`set_placements`): each mesh has
// exactly one decomposition, driven by its own spec's RankSet, exactly
// like `GeometricLineBackend`'s own single-mesh design -- a test using
// this class puts each graph component on the SAME RankSet as its own
// mesh's spec, so `view()`/`placed_view()` never need to differ.
class MultiMeshGeometricBackend final : public fuse::MeshBackend {
 public:
  struct Spec {
    std::size_t n_cells = 1;
    fuse::RankSet ranks{0, 1};
    double x_lo = 0.0;
    double x_hi = 1.0;
  };

  /// Must precede the corresponding `load()` call. Registering the same
  /// name twice overwrites the earlier spec -- there is exactly one
  /// decomposition per mesh name, matching every other backend in this
  /// file.
  void add_mesh_spec(std::string name, Spec spec) {
    specs_[std::move(name)] = spec;
  }

  fuse::Result<void> load(const fuse::MeshDecl& decl) override {
    const auto it = specs_.find(decl.name);
    if (it == specs_.end()) {
      return fuse::Failure{"E-MESH-NAME",
                           "MultiMeshGeometricBackend: no spec registered for mesh '" +
                               decl.name + "' -- call add_mesh_spec() before load()"};
    }
    const Spec& s = it->second;
    const int me = fuse_test::this_rank();
    if (!s.ranks.contains(me)) {
      meshes_.push_back(std::make_unique<GeometricLineMesh>(GeometricLineMesh::empty(s.x_lo, s.x_hi)));
    } else {
      int pos = 0;
      for (int r = s.ranks.min_rank(); r < me; ++r) pos += s.ranks.contains(r) ? 1 : 0;
      meshes_.push_back(std::make_unique<GeometricLineMesh>(
          s.n_cells, static_cast<int>(s.ranks.size()), pos, s.x_lo, s.x_hi));
    }
    names_.push_back(decl.name);
    return fuse::Result<void>::success();
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(fuse::MeshId id) const override {
    if (!id.valid() || id.value() >= meshes_.size()) return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    return meshes_[id.value()].get();
  }
  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(fuse::MeshId id,
                                                                fuse::ComponentId) const override {
    return view(id);
  }
  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    const int dim = r.dimension >= 0 ? r.dimension : v.value()->dimension();
    return v.value()->entity_count(dim);
  }
  [[nodiscard]] fuse::Result<std::size_t> placed_region_size(const fuse::Region& r,
                                                             fuse::ComponentId) const override {
    return region_size(r);
  }
  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(std::string_view name) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return fuse::MeshId{static_cast<std::uint32_t>(i)};
    }
    return fuse::Failure{"E-MESH-NAME", "no mesh named '" + std::string(name) + "'"};
  }
  [[nodiscard]] std::string name() const override { return "multi-mesh-geometric-test"; }

 private:
  std::unordered_map<std::string, Spec> specs_;
  std::vector<std::unique_ptr<fuse::MeshView>> meshes_;
  std::vector<std::string> names_;
};

// ---------------------------------------------------------------------------
// GeometricPlacedLineBackend -- additive fixture for the
// "distributed-remap-reads-the-default-view" mutation
// (build-evidence/mutations/nodal-fields.toml): MultiMeshGeometricBackend's
// own header comment says plainly that its `placed_view()` forwards
// straight to `view()`, so it cannot tell `build_nodal_dist_remap`'s two
// resolution calls apart no matter what shape a test built from it uses.
// This class exists ONLY to close that specific gap.
//
// A mesh's DEFAULT decomposition (`view()`) is built from its own
// `add_mesh_spec` RankSet exactly like MultiMeshGeometricBackend. A
// component's PLACED decomposition (`placed_view(id, who)`) is, when
// `set_component_ranks(mesh_name, who, ranks)` has registered one for that
// (mesh, component) pair, a SEPARATE `GeometricLineMesh` decomposed over
// THAT component's own `ranks` instead -- reusing `GeometricLineMesh`
// UNCHANGED, exactly the "duplicated Spec, same underlying mesh class"
// pattern MultiMeshGeometricBackend's own header comment already
// establishes for per-mesh-name specs. `GeometricLineMesh`'s global ids are
// a pure function of (n_cells, x_lo, x_hi, position along the line) --
// never of rank or rank count (that class's own header comment) -- so the
// default and placed decompositions of the SAME mesh agree on which
// physical entity a given global id names, even though they disagree on
// which RANK holds it locally and how many entities that rank sees. That
// is exactly the divergence `build_nodal_dist_remap`'s
// `backend.view(bid.value())` (wrong) vs
// `backend.placed_view(bid.value(), who)` (right) must be told apart by,
// and it needs no collective to observe: for a mesh whose default ranks
// and a given component's own ranks differ, at least one rank's LOCAL
// slice size differs between the two decompositions, checkable from that
// rank alone.
class GeometricPlacedLineBackend final : public fuse::MeshBackend {
 public:
  struct Spec {
    std::size_t n_cells = 1;
    fuse::RankSet ranks{0, 1};
    double x_lo = 0.0;
    double x_hi = 1.0;
  };

  /// Must precede the corresponding `load()` call, mirroring every other
  /// backend in this file's "declare before loading" convention.
  void add_mesh_spec(std::string name, Spec spec) { specs_[std::move(name)] = spec; }

  /// Registers component `who`'s OWN placement of mesh `mesh_name`, decomposed
  /// over `ranks` (which may differ from the mesh's own default spec ranks
  /// above). Must precede `load()`: `load()` builds every registered placed
  /// decomposition once, alongside the default one.
  void set_component_ranks(const std::string& mesh_name, fuse::ComponentId who,
                           fuse::RankSet ranks) {
    placement_specs_[mesh_name][who.value()] = ranks;
  }

  fuse::Result<void> load(const fuse::MeshDecl& decl) override {
    const auto it = specs_.find(decl.name);
    if (it == specs_.end()) {
      return fuse::Failure{"E-MESH-NAME",
                           "GeometricPlacedLineBackend: no spec registered for mesh '" +
                               decl.name + "' -- call add_mesh_spec() before load()"};
    }
    const int me = fuse_test::this_rank();
    meshes_.push_back(std::make_unique<GeometricLineMesh>(build(it->second, me)));
    names_.push_back(decl.name);

    std::unordered_map<std::uint32_t, std::unique_ptr<fuse::MeshView>> per;
    const auto pit = placement_specs_.find(decl.name);
    if (pit != placement_specs_.end()) {
      for (const auto& [who, ranks] : pit->second) {
        Spec s = it->second;
        s.ranks = ranks;
        per.emplace(who, std::make_unique<GeometricLineMesh>(build(s, me)));
      }
    }
    placed_.push_back(std::move(per));
    return fuse::Result<void>::success();
  }

  [[nodiscard]] fuse::Result<const fuse::MeshView*> view(fuse::MeshId id) const override {
    if (!id.valid() || id.value() >= meshes_.size()) return fuse::Failure{"E-MESH-ID", "invalid mesh id"};
    return meshes_[id.value()].get();
  }

  /// Returns the PLACED decomposition registered via `set_component_ranks`
  /// for this (mesh, component) pair, or falls back to `view()` when none
  /// was registered -- the same "no override declared" fallback
  /// `PartitionedTestBackend::placed_view` uses for an unrecognised
  /// placement, so a test that only cares about ONE side's placed/default
  /// divergence need not register the other.
  [[nodiscard]] fuse::Result<const fuse::MeshView*> placed_view(
      fuse::MeshId id, fuse::ComponentId who) const override {
    if (!who.valid() || !id.valid() || id.value() >= placed_.size()) return view(id);
    const auto& per = placed_[id.value()];
    const auto it = per.find(who.value());
    if (it == per.end()) return view(id);
    return it->second.get();
  }

  [[nodiscard]] fuse::Result<std::size_t> region_size(const fuse::Region& r) const override {
    const auto v = view(r.mesh);
    if (!v.ok()) return v.error();
    const int dim = r.dimension >= 0 ? r.dimension : v.value()->dimension();
    return v.value()->entity_count(dim);
  }
  [[nodiscard]] fuse::Result<std::size_t> placed_region_size(const fuse::Region& r,
                                                             fuse::ComponentId who) const override {
    const auto v = placed_view(r.mesh, who);
    if (!v.ok()) return v.error();
    const int dim = r.dimension >= 0 ? r.dimension : v.value()->dimension();
    return v.value()->entity_count(dim);
  }
  [[nodiscard]] fuse::Result<fuse::MeshId> mesh_id(std::string_view name) const override {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) return fuse::MeshId{static_cast<std::uint32_t>(i)};
    }
    return fuse::Failure{"E-MESH-NAME", "no mesh named '" + std::string(name) + "'"};
  }
  [[nodiscard]] std::string name() const override { return "geometric-placed-line-test"; }

 private:
  static GeometricLineMesh build(const Spec& s, int me) {
    if (!s.ranks.contains(me)) return GeometricLineMesh::empty(s.x_lo, s.x_hi);
    int pos = 0;
    for (int r = s.ranks.min_rank(); r < me; ++r) pos += s.ranks.contains(r) ? 1 : 0;
    return GeometricLineMesh(s.n_cells, static_cast<int>(s.ranks.size()), pos, s.x_lo, s.x_hi);
  }

  std::unordered_map<std::string, Spec> specs_;
  std::unordered_map<std::string, std::unordered_map<std::uint32_t, fuse::RankSet>> placement_specs_;
  std::vector<std::unique_ptr<fuse::MeshView>> meshes_;
  std::vector<std::unordered_map<std::uint32_t, std::unique_ptr<fuse::MeshView>>> placed_;
  std::vector<std::string> names_;
};
