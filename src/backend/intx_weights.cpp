#include <fmt/format.h>

#include "fuse/base/config.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/backend/supermesh.hpp"

#include "high_order_backend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#if FUSE_HAVE_MOAB

// ONE OF THE TWO TRANSLATION UNITS THAT MAY SEE MOAB; the other is
// src/mesh/moab_backend.cpp. Nothing below leaks a MOAB type through a header.
#include <moab/CN.hpp>
#include <moab/Core.hpp>
#include <moab/IntxMesh/Intx2MeshInPlane.hpp>
#include <moab/IntxMesh/Intx2MeshOnSphere.hpp>
#include <moab/Range.hpp>

#include <MBTagConventions.hpp>

// MOABConfig.h arrives via Core.hpp, so MOAB_HAVE_MPI is only knowable here.
// Both halves are required, exactly as in src/mesh/moab_backend.cpp: MOAB must
// have been built with MPI, and FUSE must have been configured with it, or
// there is no mpi.h on the include path. This is the second of the two
// translation units allowed to see MOAB, so it is also allowed to see the MPI
// that comes with it; no FUSE header names either.
#if defined(MOAB_HAVE_MPI) && FUSE_HAVE_MPI  // FUSE_HAVE_MPI is #cmakedefine01: test its VALUE
#define FUSE_REMAP_PARALLEL 1
#include <moab/ParallelComm.hpp>
#include <mpi.h>
#endif

#include <array>
#include <cmath>
#include <exception>
#include <cstdio>
#include <numeric>
#include <unordered_map>

#include "fuse/backend/moab_backend.hpp"

#endif  // FUSE_HAVE_MOAB

namespace fuse {

// ===========================================================================
// The cache. Independent of MOAB, so it is compiled either way.
// ===========================================================================

Result<const RemapWeights*> RemapCache::get_or_build(
    const RemapPlan& plan, const CouplingGraph& graph,
    const MeshBackend& backend) {
  const std::string k = plan.key();
  const auto it = cache_.find(k);
  if (it != cache_.end()) {
    ++hits_;
    return it->second.get();
  }
  // Counted before the build, and a failure is NOT cached: a caller that
  // retries after fixing the mesh must get a real attempt, not the stale
  // error, and `misses()` must count the work actually attempted.
  ++misses_;
  Result<RemapWeights> built = build_weights(plan, graph, backend);
  if (!built.ok()) return built.error();
  auto owned = std::make_unique<RemapWeights>(std::move(built.value()));
  const RemapWeights* p = owned.get();
  // unique_ptr, so rehashing the map moves the pointer, never the weights:
  // `p` stays valid for the life of the cache.
  cache_.emplace(k, std::move(owned));
  return p;
}

#if !FUSE_HAVE_MOAB

/// Without MOAB, weights come from the dependency-free simplex supermesh.
///
/// This used to refuse, on the grounds that "the supermesh intersection has no
/// dependency-free implementation". That is no longer true: `supermesh.cpp`
/// clips the simplices every `MeshView` provides and computes analytic
/// moments, so lowest-order conservative remap on arbitrary geometry is
/// available in the DEFAULT build. What MOAB adds above this is its mesh
/// reader, its spatial search and its partitioning; what TempestRemap adds is
/// high order on the sphere.
Result<RemapWeights> build_weights(const RemapPlan& plan,
                                   const CouplingGraph& graph,
                                   const MeshBackend& backend) {
  // Only what this builder can honour. The checks are deliberately the same
  // shape as the MOAB path's, so a plan refused there is refused here for the
  // same stated reason rather than silently accepted by the other build.
  if (plan.src_disc != Discretization::FV ||
      plan.tgt_disc != Discretization::FV) {
    return Failure{"E-REMAP-UNSUPPORTED",
                   std::string("the supermesh builder makes FV<->FV weights "
                               "only; the plan asks for ") +
                       to_cstring(plan.src_disc) + " -> " +
                       to_cstring(plan.tgt_disc)};
  }
  const int asked = std::max(plan.src_order, plan.tgt_order);
  if (asked > 1 || has(plan.flags, RemapFlags::HighOrder)) {
    return Failure{
        "E-REMAP-NOHIGHORDER",
        "this build has no MOAB, so the only reconstruction available is "
        "piecewise constant (order 1), and order " + std::to_string(asked) +
            " was asked for. Order 1 is exact for constants and conservative; "
            "higher order on the sphere is what the TempestRemap tier adds. "
            "Drop RemapFlags::HighOrder, or configure with "
            "-DFUSE_ENABLE_MOAB=ON against a MOAB built with TempestRemap."};
  }

  // Region -> MeshView, with the same validation the MOAB path applies. An
  // entity-set region is refused for the reason given there: the weights index
  // whole-mesh local indices, and a sub-set region would be wrong by a
  // permutation that nothing here could detect.
  struct Side {
    const MeshView* view = nullptr;
    int             dim = 0;
  };
  const auto side = [&](RegionId id, const char* which) -> Result<Side> {
    if (!id.valid() || id.value() >= graph.region_count()) {
      return Failure{"E-REMAP-REGION",
                     std::string(which) + " region id " +
                         std::to_string(id.value()) + " is not one of the "
                         "graph's " + std::to_string(graph.region_count()) +
                         " regions"};
    }
    const Region& r = graph.region(id);
    if (!(r.entity_set.empty() || r.entity_set == "all")) {
      return Failure{"E-REMAP-REGION",
                     std::string(which) + " region '" + r.name +
                         "' names entity set '" + r.entity_set +
                         "'. Remap weights index MeshView local indices over "
                         "the whole mesh, so a sub-set region would give a "
                         "matrix wrong by a permutation"};
    }
    if (!r.mesh.valid() || r.mesh.value() >= graph.mesh_count()) {
      return Failure{"E-REMAP-REGION", std::string(which) + " region '" +
                                           r.name + "' names no valid mesh"};
    }
    const auto mid = backend.mesh_id(graph.mesh(r.mesh).name);
    if (!mid.ok()) return mid.error();
    const auto view = backend.view(mid.value());
    if (!view.ok()) return view.error();
    Side s;
    s.view = view.value();
    s.dim = r.dimension >= 0 ? r.dimension : s.view->dimension();
    return s;
  };

  auto s = side(plan.src, "source");
  if (!s.ok()) return s.error();
  auto d = side(plan.tgt, "target");
  if (!d.ok()) return d.error();
  if (s.value().dim != d.value().dim) {
    return Failure{"E-REMAP-DIM",
                   "the source cells are dimension " +
                       std::to_string(s.value().dim) +
                       " and the target cells dimension " +
                       std::to_string(d.value().dim) +
                       "; a conservative FV remap needs the same dimension on "
                       "both sides"};
  }

  return build_supermesh_weights(*s.value().view, *d.value().view,
                                 s.value().dim);
}

Result<std::size_t> build_remap_halo(const RemapPlan&, const CouplingGraph&,
                                     MeshBackend&) {
  // A successful no-op, not an error: without MOAB there is no partitioned
  // read to build a halo for, and a caller that makes the call
  // unconditionally (which the header invites) must not have to branch on
  // which backends were compiled in. `build_weights` refuses this build
  // anyway, with E-REMAP-NOMOAB.
  return std::size_t{0};
}

#else  // FUSE_HAVE_MOAB

namespace remap_detail {

using V3 = std::array<double, 3>;

[[nodiscard]] V3 cross(const V3& a, const V3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
[[nodiscard]] double dot(const V3& a, const V3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
[[nodiscard]] double norm(const V3& a) { return std::sqrt(dot(a, a)); }

// ---------------------------------------------------------------------------
// Tolerances. All relative; each is justified by a measurement recorded in the
// commit message and reproduced by test/unit/remap_test.cpp.
// ---------------------------------------------------------------------------

/// How far a cell's supermesh coverage may sit from 1 before the overlap is
/// treated as partial rather than as arithmetic. The worst per-cell coverage
/// defect measured on the hardest supported case -- MOAB's 3456-cell cubed
/// sphere against an equiangular 1536-cell partner, 7374 intersection polygons
/// including slivers -- is 7.7e-14, five decades below this.
constexpr double kCoverageTol = 1e-9;

/// Relative slack when deciding that a coordinate sits on a cell's bounding
/// box, that a vertex sits on a sphere, or that a mesh is planar.
constexpr double kGeomTol = 1e-9;

/// Length tolerance handed to MOAB's advancing front, relative to the mesh's
/// characteristic size. MOAB derives its area tolerance as eps*sqrt(eps).
constexpr double kIntxTol = 1e-8;

// ---------------------------------------------------------------------------
// Areas. Both functionals below are EXACTLY ADDITIVE over a partition of a
// cell, which is the property the whole construction rests on: the weights use
// only the ratio overlap/parent, so any error common to both cancels, but an
// error that fails to telescope over the pieces shows up as a coverage defect
// and, under RemapFlags::Conservative, as E-REMAP-COVERAGE.
// ---------------------------------------------------------------------------

/// Spherical excess of the triangle (a, b, c), by the tangent half-angle
/// formula of van Oosterom & Strackee, IEEE Trans. Biomed. Eng. BME-30 (1983)
/// 125.
///
/// WHY NOT `moab::IntxAreaUtils`. Measured on the cubed-sphere pair above,
/// MOAB's DEFAULT area method (`lHuiller`) leaves a per-cell coverage defect of
/// 5.1e-07 -- seven digits, not sixteen -- because l'Huilier's formula
/// cancels catastrophically on the sliver triangles an advancing front
/// produces. Its `GaussQuadrature` method reaches 6.7e-14 but is compiled into
/// MOAB only `#ifdef MOAB_HAVE_TEMPESTREMAP`, so selecting it would make the
/// accuracy of a conservation check depend on how MOAB was configured, and
/// would silently fall back on a build without it. This formula reaches
/// 7.7e-14 with no such dependency: the numerator is a triple product that
/// vanishes smoothly with the triangle, and `atan2` handles the limit without
/// subtracting nearly equal quantities.
[[nodiscard]] double spherical_excess(const V3& a, const V3& b, const V3& c) {
  const double num = std::fabs(dot(a, cross(b, c)));
  const double la = norm(a);
  const double lb = norm(b);
  const double lc = norm(c);
  const double den =
      la * lb * lc + dot(a, b) * lc + dot(a, c) * lb + dot(b, c) * la;
  return 2.0 * std::atan2(num, den);
}

/// Area of a spherical polygon of radius `r` whose vertices are `xyz`,
/// fanned from vertex 0. Exact for a convex polygon; MOAB's intersection
/// polygons are convex by construction (an intersection of convex cells).
[[nodiscard]] double spherical_polygon_area(const std::vector<double>& xyz,
                                            double r) {
  const std::size_t n = xyz.size() / 3;
  if (n < 3) return 0.0;
  const V3 v0{xyz[0], xyz[1], xyz[2]};
  double excess = 0.0;
  for (std::size_t i = 1; i + 1 < n; ++i) {
    const V3 vi{xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    const V3 vj{xyz[3 * i + 3], xyz[3 * i + 4], xyz[3 * i + 5]};
    excess += spherical_excess(v0, vi, vj);
  }
  return excess * r * r;
}

/// Signed area of a polygon projected on the x-y plane (the shoelace rule).
/// Exact, and additive over any planar subdivision.
[[nodiscard]] double planar_signed_area(const std::vector<double>& xyz) {
  const std::size_t n = xyz.size() / 3;
  double twice = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const std::size_t j = (i + 1) % n;
    twice += xyz[3 * i] * xyz[3 * j + 1] - xyz[3 * j] * xyz[3 * i + 1];
  }
  return 0.5 * twice;
}

// ---------------------------------------------------------------------------
// MOAB plumbing
// ---------------------------------------------------------------------------

[[nodiscard]] std::string moab_text(moab::Interface& mb, moab::ErrorCode rval) {
  std::string text =
      (rval >= moab::MB_SUCCESS && rval <= moab::MB_FAILURE)
          ? std::string{moab::ErrorCodeStr[static_cast<int>(rval)]}
          : ("ErrorCode " + std::to_string(static_cast<int>(rval)));
  std::string last;
  if (mb.get_last_error(last) == moab::MB_SUCCESS && !last.empty()) {
    text += " (" + last + ")";
  }
  return text;
}

[[nodiscard]] Failure intx_error(moab::Interface& mb, const std::string& what,
                               moab::ErrorCode rval) {
  return Failure{"E-REMAP-INTX", what + ": " + moab_text(mb, rval)};
}

/// Corner coordinates of one element, flattened. Every MOAB code is checked.
[[nodiscard]] Result<std::vector<double>> corner_coords(
    moab::Interface& mb, moab::EntityHandle h, std::vector<double>& scratch) {
  const moab::EntityHandle* conn = nullptr;
  int nconn = 0;
  std::vector<moab::EntityHandle> storage;
  const moab::ErrorCode rval =
      mb.get_connectivity(h, conn, nconn, /*corners_only=*/true, &storage);
  if (rval != moab::MB_SUCCESS || conn == nullptr || nconn <= 0) {
    return intx_error(mb,
                      "could not read the connectivity of entity " +
                          std::to_string(h),
                      rval);
  }
  scratch.assign(static_cast<std::size_t>(nconn) * 3, 0.0);
  const moab::ErrorCode crval = mb.get_coords(conn, nconn, scratch.data());
  if (crval != moab::MB_SUCCESS) {
    return intx_error(
        mb, "could not read the coordinates of entity " + std::to_string(h),
        crval);
  }
  return scratch;
}

// ---------------------------------------------------------------------------
// One side of the plan, resolved
// ---------------------------------------------------------------------------

struct Side {
  const MeshView* view = nullptr;
  const MoabMeshBackend* backend = nullptr;
  MeshId mid;  ///< the BACKEND's id, resolved by name; not the graph's
  moab::Interface* mb = nullptr;
  moab::EntityHandle file_set = 0;
  void* pcomm = nullptr;  ///< moab::ParallelComm*, null when serial
  int dim = 0;
  std::vector<moab::EntityHandle> cells;  ///< local index -> handle
  /// MeshView::measure, per local index.
  std::vector<double> measure;
  /// Sum of `measure` over the entities this rank OWNS. Halo cells are copies
  /// of another rank's and would be double counted in any global reduction;
  /// in serial every entity is owned, so this is the old number unchanged.
  double measure_total = 0.0;
  /// How many of `cells` this rank does not own. Zero unless a remap halo was
  /// built (or the read has ghosts).
  std::size_t not_owned = 0;
  std::string label;
};

[[nodiscard]] Result<Side> resolve(const CouplingGraph& graph,
                                   const MeshBackend& backend, RegionId id,
                                   const char* which) {
  if (!id.valid() || id.value() >= graph.region_count()) {
    return Failure{"E-REMAP-REGION",
                 std::string(which) + " region id " +
                     std::to_string(id.value()) +
                     " is not one of the graph's " +
                     std::to_string(graph.region_count()) + " regions"};
  }
  const Region& r = graph.region(id);

  if (!(r.entity_set.empty() || r.entity_set == "all")) {
    return Failure{
        "E-REMAP-REGION",
        std::string(which) + " region '" + r.name + "' names entity set '" +
            r.entity_set +
            "'. Remap weights index MeshView local indices over the whole "
            "mesh, and MeshBackend does not say whether a buffer for a "
            "sub-set region is indexed by set position or by mesh-local "
            "index; guessing would produce a matrix wrong by a permutation. "
            "Use a whole-mesh region, or wait for the sub-set index map."};
  }

  const auto* moab_backend = dynamic_cast<const MoabMeshBackend*>(&backend);
  if (moab_backend == nullptr) {
    return Failure{"E-REMAP-BACKEND",
                 "build_weights needs the MOAB backend for the supermesh "
                 "intersection; this is the '" +
                     backend.name() + "' backend"};
  }

  // NOTE ON THE PARTITIONED CASE. There is no refusal here any more. The
  // supermesh below intersects the cells this rank HOLDS, and on a partitioned
  // read that is only correct once the source mesh carries a remap halo -- but
  // once it does, this code needs no change at all, because a halo cell is an
  // ordinary non-owned local entity with an ordinary local index. The check
  // that a halo was actually built lives in `build_weights`, after both sides
  // are resolved, because it is a statement about the PAIR.

  // By NAME, never by position: the graph's MeshIds and the backend's are two
  // independent positional id spaces (see MeshBackend::mesh_id).
  if (!r.mesh.valid() || r.mesh.value() >= graph.mesh_count()) {
    return Failure{"E-REMAP-REGION", std::string(which) + " region '" + r.name +
                                       "' names mesh id " +
                                       std::to_string(r.mesh.value()) +
                                       ", of which the graph declares " +
                                       std::to_string(graph.mesh_count())};
  }
  const auto mid = moab_backend->mesh_id(graph.mesh(r.mesh).name);
  if (!mid.ok()) return mid.error();

  const auto view = moab_backend->view(mid.value());
  if (!view.ok()) return view.error();
  const auto native = moab_backend->native_mesh(mid.value());
  if (!native.ok()) return native.error();

  Side s;
  s.view = view.value();
  s.backend = moab_backend;
  s.mid = mid.value();
  s.mb = static_cast<moab::Interface*>(native.value().interface);
  s.file_set = static_cast<moab::EntityHandle>(native.value().file_set);
  s.pcomm = native.value().parallel_comm;
  s.dim = r.dimension >= 0 ? r.dimension : s.view->dimension();
  s.label = std::string(which) + " region '" + r.name + "'";

  if (s.dim < 1 || s.dim > 3) {
    return Failure{"E-REMAP-DIM",
                 s.label + " has dimension " + std::to_string(s.dim) +
                     "; a finite-volume remap needs cells of dimension 1, 2 or "
                     "3 (dimension 0 has no measure at all)"};
  }

  moab::Range range;
  const moab::ErrorCode rval = s.mb->get_entities_by_dimension(
      s.file_set, s.dim, range, /*recursive=*/true);
  if (rval != moab::MB_SUCCESS) {
    return intx_error(*s.mb,
                      "could not list the dimension-" + std::to_string(s.dim) +
                          " cells of " + s.label,
                      rval);
  }
  // A moab::Range iterates in ascending handle order, which is exactly how
  // MoabMeshBackend fixed the local indexing at load. Cross-checked rather
  // than trusted: if the two ever diverge every weight lands in the wrong row.
  if (range.size() != s.view->entity_count(s.dim)) {
    return Failure{"E-REMAP-REGION",
                 s.label + " has " + std::to_string(range.size()) +
                     " dimension-" + std::to_string(s.dim) +
                     " cells in MOAB but " +
                     std::to_string(s.view->entity_count(s.dim)) +
                     " in its MeshView; the local index spaces disagree"};
  }
  if (range.empty()) {
    return Failure{"E-REMAP-REGION", s.label + " holds no dimension-" +
                                       std::to_string(s.dim) + " cells"};
  }
  if (range.size() >
      static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return Failure{"E-REMAP-SIZE",
                 s.label + " holds " + std::to_string(range.size()) +
                     " cells, more than the int32 COO indices can address"};
  }
  s.cells.assign(range.begin(), range.end());

  s.measure.resize(s.cells.size());
  for (std::size_t i = 0; i < s.cells.size(); ++i) {
    // MeshView::measure THROWS by contract; build_weights returns a Result and
    // must not, so the contract is converted here rather than propagated.
    try {
      s.measure[i] = s.view->measure(s.dim, i);
    } catch (const std::exception& e) {
      return Failure{"E-REMAP-MEASURE", s.label + ": " + e.what()};
    }
    if (!(s.measure[i] > 0.0)) {
      return Failure{"E-REMAP-MEASURE",
                   s.label + " cell " + std::to_string(i) + " has measure " +
                       std::to_string(s.measure[i]) +
                       "; a zero or degenerate cell cannot carry a "
                       "conservative weight"};
    }
    // OWNED ONLY. A halo cell is a copy of another rank's, and adding its
    // measure here would make `src_measure_total` count it once per rank that
    // holds a copy. In serial every entity is owned and this is unchanged.
    if (s.view->is_owned(s.dim, i)) {
      s.measure_total += s.measure[i];
    } else {
      ++s.not_owned;
    }
  }
  return s;
}

// ---------------------------------------------------------------------------
// Raw overlaps, before normalisation
// ---------------------------------------------------------------------------

struct Overlap {
  std::int32_t row = 0;  ///< target local index
  std::int32_t col = 0;  ///< source local index
  double area = 0.0;     ///< in the supermesh's own measure
};

struct Supermesh {
  std::vector<Overlap> entries;
  std::vector<double> src_parent;  ///< parent measure in the SAME functional
  std::vector<double> tgt_parent;
  const char* method = "";
};

// ---------------------------------------------------------------------------
// Construction 1: axis-aligned cells
// ---------------------------------------------------------------------------

struct BoxMesh {
  std::vector<std::array<double, 3>> lo, hi;
  std::array<bool, 3> active{false, false, false};
  std::array<double, 3> flat{0.0, 0.0, 0.0};  ///< coordinate on inactive axes
  int active_count = 0;
};

/// Try to read `s` as a mesh of axis-aligned boxes. Returns an empty optional
/// (via `ok == false`) rather than an error when the mesh simply is not one --
/// that is a classification result, not a failure.
struct BoxTry {
  bool ok = false;
  BoxMesh mesh;
  Failure error{};   ///< set only when `failed`
  bool failed = false;
};

[[nodiscard]] BoxTry try_boxes(const Side& s) {
  BoxTry out;
  BoxMesh& m = out.mesh;
  m.lo.resize(s.cells.size());
  m.hi.resize(s.cells.size());

  std::array<double, 3> gmin{1e300, 1e300, 1e300};
  std::array<double, 3> gmax{-1e300, -1e300, -1e300};
  std::vector<double> scratch;

  for (std::size_t i = 0; i < s.cells.size(); ++i) {
    auto xyz = corner_coords(*s.mb, s.cells[i], scratch);
    if (!xyz.ok()) {
      out.failed = true;
      out.error = xyz.error();
      return out;
    }
    const std::vector<double>& c = xyz.value();
    const std::size_t n = c.size() / 3;
    std::array<double, 3> lo{1e300, 1e300, 1e300};
    std::array<double, 3> hi{-1e300, -1e300, -1e300};
    for (std::size_t v = 0; v < n; ++v) {
      for (std::size_t k = 0; k < 3; ++k) {
        lo[k] = std::min(lo[k], c[3 * v + k]);
        hi[k] = std::max(hi[k], c[3 * v + k]);
      }
    }
    m.lo[i] = lo;
    m.hi[i] = hi;
    for (std::size_t k = 0; k < 3; ++k) {
      gmin[k] = std::min(gmin[k], lo[k]);
      gmax[k] = std::max(gmax[k], hi[k]);
    }
  }

  double span = 0.0;
  for (std::size_t k = 0; k < 3; ++k) span = std::max(span, gmax[k] - gmin[k]);
  if (!(span > 0.0)) return out;  // a point; nothing sensible to do
  const double eps = kGeomTol * span;

  for (std::size_t k = 0; k < 3; ++k) {
    m.active[k] = (gmax[k] - gmin[k]) > eps;
    if (m.active[k]) {
      ++m.active_count;
    } else {
      m.flat[k] = 0.5 * (gmax[k] + gmin[k]);
    }
  }
  if (m.active_count != s.dim) return out;

  // Every cell must be a full box on the active axes: 2^dim corners, one at
  // each combination of (lo, hi), and flat on the inactive ones. Verified, not
  // assumed -- a mesh of skewed quads has the same bounding boxes and would
  // otherwise produce a plausible, wrong matrix.
  const std::size_t want_corners = std::size_t{1} << s.dim;
  for (std::size_t i = 0; i < s.cells.size(); ++i) {
    auto xyz = corner_coords(*s.mb, s.cells[i], scratch);
    if (!xyz.ok()) {
      out.failed = true;
      out.error = xyz.error();
      return out;
    }
    const std::vector<double>& c = xyz.value();
    if (c.size() / 3 != want_corners) return out;

    double volume = 1.0;
    for (std::size_t k = 0; k < 3; ++k) {
      if (!m.active[k]) continue;
      const double w = m.hi[i][k] - m.lo[i][k];
      if (!(w > eps)) return out;
      volume *= w;
    }

    std::vector<char> seen(want_corners, 0);
    for (std::size_t v = 0; v < want_corners; ++v) {
      std::size_t sig = 0;
      std::size_t bit = 0;
      bool on_box = true;
      for (std::size_t k = 0; k < 3; ++k) {
        const double x = c[3 * v + k];
        if (!m.active[k]) {
          if (std::fabs(x - m.flat[k]) > eps) on_box = false;
          continue;
        }
        if (std::fabs(x - m.lo[i][k]) <= eps) {
          // bit stays 0
        } else if (std::fabs(x - m.hi[i][k]) <= eps) {
          sig |= (std::size_t{1} << bit);
        } else {
          on_box = false;
        }
        ++bit;
      }
      if (!on_box) return out;
      if (seen[sig] != 0) return out;  // two corners at the same box vertex
      seen[sig] = 1;
    }

    // The whole point of this construction is that the overlap arithmetic and
    // MeshView::measure are the SAME number. Check it; if the backend's
    // integrator ever disagrees the weights would be conservative against a
    // measure nobody uses.
    if (std::fabs(volume - s.measure[i]) > kGeomTol * s.measure[i]) return out;
  }

  out.ok = true;
  return out;
}

[[nodiscard]] Result<Supermesh> intersect_boxes(const Side& src,
                                                const Side& tgt,
                                                const BoxMesh& a,
                                                const BoxMesh& b) {
  for (std::size_t k = 0; k < 3; ++k) {
    if (a.active[k] != b.active[k]) {
      return Failure{"E-REMAP-GEOMETRY",
                   "the two meshes are axis-aligned but extend along "
                   "different axes, so they do not discretise the same "
                   "domain"};
    }
    const double flat_tol =
        kGeomTol * std::max({std::fabs(a.flat[k]), std::fabs(b.flat[k]), 1.0});
    if (!a.active[k] && std::fabs(a.flat[k] - b.flat[k]) > flat_tol) {
      // Both are flat on this axis but at different coordinates: parallel,
      // disjoint slabs. Their intersection is empty, and pretending otherwise
      // by ignoring the axis is exactly the silent-wrong-answer case.
      return Failure{"E-REMAP-GEOMETRY",
                   "the two meshes are flat on axis " + std::to_string(k) +
                       " but at different coordinates (" +
                       std::to_string(a.flat[k]) + " and " +
                       std::to_string(b.flat[k]) + "); they never intersect"};
    }
  }

  int axis0 = 0;
  for (std::size_t k = 0; k < 3; ++k) {
    if (a.active[k]) {
      axis0 = static_cast<int>(k);
      break;
    }
  }
  const std::size_t ax = static_cast<std::size_t>(axis0);

  // Broad phase: source cells sorted by their low coordinate on `axis0`, so a
  // target cell only has to look at a contiguous window of them. Without it a
  // 3456 x 1536 pair is five million box tests; with it the cost is
  // O(n log n + overlaps).
  std::vector<std::uint32_t> order(a.lo.size());
  std::iota(order.begin(), order.end(), 0u);
  std::sort(order.begin(), order.end(),
            [&](std::uint32_t p, std::uint32_t q) {
              return a.lo[p][ax] < a.lo[q][ax];
            });
  std::vector<double> keys(order.size());
  double widest = 0.0;
  for (std::size_t i = 0; i < order.size(); ++i) {
    keys[i] = a.lo[order[i]][ax];
    widest = std::max(widest, a.hi[order[i]][ax] - a.lo[order[i]][ax]);
  }

  Supermesh out;
  out.method = "axis-aligned";
  out.src_parent = src.measure;  // identical functional, checked in try_boxes
  out.tgt_parent = tgt.measure;

  for (std::size_t j = 0; j < b.lo.size(); ++j) {
    const double tlo = b.lo[j][ax];
    const double thi = b.hi[j][ax];
    const auto begin =
        std::lower_bound(keys.begin(), keys.end(), tlo - widest);
    const auto end = std::upper_bound(keys.begin(), keys.end(), thi);
    for (auto it = begin; it != end; ++it) {
      const std::size_t i = order[static_cast<std::size_t>(it - keys.begin())];
      double area = 1.0;
      for (std::size_t k = 0; k < 3; ++k) {
        if (!a.active[k]) continue;
        const double o = std::min(a.hi[i][k], b.hi[j][k]) -
                         std::max(a.lo[i][k], b.lo[j][k]);
        if (!(o > 0.0)) {
          area = 0.0;
          break;
        }
        area *= o;
      }
      if (area > 0.0) {
        out.entries.push_back(Overlap{static_cast<std::int32_t>(j),
                                      static_cast<std::int32_t>(i), area});
      }
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// Constructions 2 and 3: MOAB's Intx2Mesh, on a sphere or in a plane
// ---------------------------------------------------------------------------

enum class Surface { Sphere, Plane };

struct SurfaceFit {
  bool ok = false;
  Surface kind = Surface::Plane;
  double radius = 0.0;  ///< sphere only
  double z = 0.0;       ///< plane only
  double span = 0.0;
};

[[nodiscard]] Result<SurfaceFit> fit_surface(const Side& s) {
  moab::Range verts;
  const moab::ErrorCode rval =
      s.mb->get_entities_by_dimension(s.file_set, 0, verts, /*recursive=*/true);
  if (rval != moab::MB_SUCCESS || verts.empty()) {
    return intx_error(*s.mb, "could not list the vertices of " + s.label, rval);
  }
  std::vector<moab::EntityHandle> vh(verts.begin(), verts.end());
  std::vector<double> xyz(vh.size() * 3, 0.0);
  const moab::ErrorCode crval =
      s.mb->get_coords(vh.data(), static_cast<int>(vh.size()), xyz.data());
  if (crval != moab::MB_SUCCESS) {
    return intx_error(*s.mb, "could not read the vertices of " + s.label,
                      crval);
  }

  double rmin = 1e300, rmax = 0.0, zmin = 1e300, zmax = -1e300, span = 0.0;
  for (std::size_t i = 0; i < vh.size(); ++i) {
    const double x = xyz[3 * i], y = xyz[3 * i + 1], z = xyz[3 * i + 2];
    const double r = std::sqrt(x * x + y * y + z * z);
    rmin = std::min(rmin, r);
    rmax = std::max(rmax, r);
    zmin = std::min(zmin, z);
    zmax = std::max(zmax, z);
    span = std::max(span, r);
  }

  SurfaceFit f;
  f.span = span;
  if (rmax > 0.0 && (rmax - rmin) <= kGeomTol * rmax) {
    f.ok = true;
    f.kind = Surface::Sphere;
    f.radius = 0.5 * (rmin + rmax);
    return f;
  }
  if ((zmax - zmin) <= kGeomTol * std::max(span, 1.0)) {
    f.ok = true;
    f.kind = Surface::Plane;
    f.z = 0.5 * (zmin + zmax);
    return f;
  }
  return f;  // ok == false
}

/// MOAB's advancing front assumes positively oriented cells: outward from the
/// origin on a sphere, counter-clockwise in the x-y plane. MOAB's own cubed
/// sphere (share/examples/fortran/spherecube.h5m) is NOT -- its signed
/// spherical area comes out at -4 pi R^2 -- so this is not a hypothetical.
///
/// Reversing a cell's connectivity is invisible through `MeshView`: the
/// measure is unsigned, the centroid is order-independent, the global id and
/// the local index are untouched.
[[nodiscard]] Result<void> orient(const Side& s, const SurfaceFit& f) {
  std::vector<double> scratch;
  std::vector<moab::EntityHandle> conn;
  for (const moab::EntityHandle h : s.cells) {
    auto xyz = corner_coords(*s.mb, h, scratch);
    if (!xyz.ok()) return xyz.error();
    const std::vector<double>& c = xyz.value();
    const std::size_t n = c.size() / 3;
    if (n < 3) {
      return Failure{"E-REMAP-GEOMETRY",
                   s.label + " contains a cell with " + std::to_string(n) +
                       " corners; a surface cell needs at least three"};
    }

    double signed_measure = 0.0;
    if (f.kind == Surface::Plane) {
      signed_measure = planar_signed_area(c);
    } else {
      // Newell normal dotted with the outward radial direction.
      V3 nrm{0.0, 0.0, 0.0};
      V3 mid{0.0, 0.0, 0.0};
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = (i + 1) % n;
        const V3 a{c[3 * i], c[3 * i + 1], c[3 * i + 2]};
        const V3 b{c[3 * j], c[3 * j + 1], c[3 * j + 2]};
        const V3 x = cross(a, b);
        for (std::size_t k = 0; k < 3; ++k) {
          nrm[k] += x[k];
          mid[k] += a[k];
        }
      }
      signed_measure = dot(nrm, mid);
    }
    if (signed_measure >= 0.0) continue;

    const moab::EntityHandle* raw = nullptr;
    int nconn = 0;
    std::vector<moab::EntityHandle> storage;
    const moab::ErrorCode grval =
        s.mb->get_connectivity(h, raw, nconn, /*corners_only=*/true, &storage);
    if (grval != moab::MB_SUCCESS || raw == nullptr || nconn <= 0) {
      return intx_error(*s.mb,
                        "could not read the connectivity of entity " +
                            std::to_string(h) + " to reorient it",
                        grval);
    }
    // Copied out BEFORE set_connectivity: `raw` may point into MOAB's own
    // sequence storage, which set_connectivity is about to overwrite.
    conn.assign(raw, raw + nconn);
    std::reverse(conn.begin(), conn.end());
    const moab::ErrorCode srval =
        s.mb->set_connectivity(h, conn.data(), nconn);
    if (srval != moab::MB_SUCCESS) {
      return intx_error(
          *s.mb, "could not reorient entity " + std::to_string(h), srval);
    }
  }
  return Result<void>::success();
}

/// Saves the GLOBAL_ID of a set of cells and puts it back, whatever happens.
/// MOAB's intersection identifies an intersection polygon's parents through
/// that tag, so it has to be rewritten; leaving it rewritten would change what
/// a later mesh write, or a later remap, sees.
class GidGuard {
 public:
  GidGuard(moab::Interface& mb, moab::Tag tag,
           std::vector<moab::EntityHandle> cells, std::vector<int> saved)
      : mb_(&mb),
        tag_(tag),
        cells_(std::move(cells)),
        saved_(std::move(saved)) {}
  GidGuard(const GidGuard&) = delete;
  GidGuard& operator=(const GidGuard&) = delete;
  ~GidGuard() {
    if (cells_.empty()) return;
    // Destructor: nothing useful to do with a failure, and throwing out of one
    // terminates. The values were read from MOAB moments ago with the same
    // handles, so a failure here means the instance is already broken.
    static_cast<void>(mb_->tag_set_data(tag_, cells_.data(),
                                        static_cast<int>(cells_.size()),
                                        saved_.data()));
  }

 private:
  moab::Interface* mb_;
  moab::Tag tag_;
  std::vector<moab::EntityHandle> cells_;
  std::vector<int> saved_;
};

/// Deletes the intersection polygons and the set that held them.
void discard_supermesh(moab::Interface& mb, moab::EntityHandle set) {
  moab::Range polys;
  if (mb.get_entities_by_dimension(set, 2, polys, /*recursive=*/true) ==
      moab::MB_SUCCESS) {
    static_cast<void>(mb.delete_entities(polys));
  }
  static_cast<void>(mb.delete_entities(&set, 1));
}

[[nodiscard]] Result<Supermesh> intersect_surfaces(const Side& src,
                                                   const Side& tgt,
                                                   const SurfaceFit& fs,
                                                   const SurfaceFit& ft) {
  moab::Interface& mb = *src.mb;
  const bool sphere = (fs.kind == Surface::Sphere);

  const std::size_t ns = src.cells.size();
  const std::size_t nt = tgt.cells.size();

  // --- parent identification -------------------------------------------------
  //
  // MOAB tags each intersection polygon with its two parents, but the two
  // workers disagree about what the number MEANS: measured against this MOAB
  // 5.6.0, Intx2MeshOnSphere writes the parent's GLOBAL_ID while
  // Intx2MeshInPlane writes its 0-based position in the set. Both were
  // observed directly -- with source ids deliberately set to 10000..13455 the
  // sphere worker reported 10000..13455 and the plane worker reported 0..1535.
  //
  // Rather than hard-code either convention, the ids are rewritten to a block
  // that CANNOT be confused with an index, and the convention is then read off
  // the values that come back. A third possibility is a hard error, not a
  // guess.
  moab::Tag gid = nullptr;
  {
    // Two steps, matching moab_backend.cpp's ensure_tag: asking for CREAT on a
    // tag that already exists with different storage is an error, and
    // GLOBAL_ID normally already exists.
    moab::ErrorCode trval =
        mb.tag_get_handle(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER, gid);
    if (trval == moab::MB_TAG_NOT_FOUND) {
      const int unset = -1;
      trval = mb.tag_get_handle(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER,
                                gid, moab::MB_TAG_DENSE | moab::MB_TAG_CREAT,
                                &unset);
    }
    if (trval != moab::MB_SUCCESS) {
      return intx_error(mb, "could not obtain the GLOBAL_ID tag", trval);
    }
  }
  if (ns + nt + ns + nt + 2 >
      static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Failure{"E-REMAP-SIZE",
                 "the two meshes together hold too many cells for a distinct "
                 "int32 global-id block"};
  }
  const int src_base = static_cast<int>(ns + nt + 1);
  const int tgt_base = src_base + static_cast<int>(ns);

  std::vector<int> src_saved(ns, 0), tgt_saved(nt, 0), tmp;
  if (mb.tag_get_data(gid, src.cells.data(), static_cast<int>(ns),
                      src_saved.data()) != moab::MB_SUCCESS ||
      mb.tag_get_data(gid, tgt.cells.data(), static_cast<int>(nt),
                      tgt_saved.data()) != moab::MB_SUCCESS) {
    return Failure{"E-REMAP-GID",
                 "could not read the GLOBAL_ID tag of the meshes being "
                 "intersected"};
  }
  GidGuard src_guard(mb, gid, src.cells, src_saved);
  GidGuard tgt_guard(mb, gid, tgt.cells, tgt_saved);

  tmp.resize(ns);
  for (std::size_t i = 0; i < ns; ++i) tmp[i] = src_base + static_cast<int>(i);
  if (mb.tag_set_data(gid, src.cells.data(), static_cast<int>(ns),
                      tmp.data()) != moab::MB_SUCCESS) {
    return Failure{"E-REMAP-GID", "could not stamp source global ids"};
  }
  tmp.resize(nt);
  for (std::size_t j = 0; j < nt; ++j) tmp[j] = tgt_base + static_cast<int>(j);
  if (mb.tag_set_data(gid, tgt.cells.data(), static_cast<int>(nt),
                      tmp.data()) != moab::MB_SUCCESS) {
    return Failure{"E-REMAP-GID", "could not stamp target global ids"};
  }

  // --- orientation -----------------------------------------------------------
  {
    const auto o1 = orient(src, fs);
    if (!o1.ok()) return o1.error();
    const auto o2 = orient(tgt, ft);
    if (!o2.ok()) return o2.error();
  }

  // --- the intersection ------------------------------------------------------
  moab::EntityHandle out_set = 0;
  const moab::ErrorCode srval = mb.create_meshset(moab::MESHSET_SET, out_set);
  if (srval != moab::MB_SUCCESS) {
    return intx_error(mb, "could not create the supermesh set", srval);
  }

  std::unique_ptr<moab::Intx2Mesh> worker;
  double radius = 0.0;
  if (sphere) {
    radius = 0.5 * (fs.radius + ft.radius);
    auto* w = new moab::Intx2MeshOnSphere(&mb);
    w->set_radius_source_mesh(radius);
    w->set_radius_destination_mesh(radius);
    worker.reset(w);
    worker->set_error_tolerance(radius * kIntxTol);
  } else {
    worker.reset(new moab::Intx2MeshInPlane(&mb));
    worker->set_error_tolerance(std::max(fs.span, 1.0) * kIntxTol);
  }

  moab::ErrorCode rval = worker->FindMaxEdges(src.file_set, tgt.file_set);
  if (rval != moab::MB_SUCCESS) {
    discard_supermesh(mb, out_set);
    return intx_error(mb, "Intx2Mesh::FindMaxEdges failed", rval);
  }
  rval = worker->intersect_meshes(src.file_set, tgt.file_set, out_set);
  if (rval != moab::MB_SUCCESS) {
    discard_supermesh(mb, out_set);
    return intx_error(mb, "Intx2Mesh::intersect_meshes failed", rval);
  }

  moab::Tag src_parent = nullptr;
  moab::Tag tgt_parent = nullptr;
  if (mb.tag_get_handle("SourceParent", src_parent) != moab::MB_SUCCESS ||
      mb.tag_get_handle("TargetParent", tgt_parent) != moab::MB_SUCCESS) {
    discard_supermesh(mb, out_set);
    return Failure{"E-REMAP-INTX",
                 "MOAB's intersection produced no SourceParent/TargetParent "
                 "tags, so the overlap polygons cannot be attributed"};
  }

  moab::Range polys;
  rval = mb.get_entities_by_dimension(out_set, 2, polys, /*recursive=*/true);
  if (rval != moab::MB_SUCCESS) {
    discard_supermesh(mb, out_set);
    return intx_error(mb, "could not list the supermesh polygons", rval);
  }

  Supermesh out;
  out.method = sphere ? "sphere" : "plane";
  out.entries.reserve(polys.size());

  auto decode = [](int raw, int base, std::size_t n,
                   std::int32_t& local) -> bool {
    if (raw >= base && raw < base + static_cast<int>(n)) {
      local = static_cast<std::int32_t>(raw - base);
      return true;
    }
    if (raw >= 0 && raw < static_cast<int>(n)) {  // 0-based position
      local = static_cast<std::int32_t>(raw);
      return true;
    }
    return false;
  };

  std::vector<double> scratch;
  for (const moab::EntityHandle p : polys) {
    int sraw = 0, traw = 0;
    if (mb.tag_get_data(src_parent, &p, 1, &sraw) != moab::MB_SUCCESS ||
        mb.tag_get_data(tgt_parent, &p, 1, &traw) != moab::MB_SUCCESS) {
      discard_supermesh(mb, out_set);
      return Failure{"E-REMAP-INTX",
                   "could not read the parent tags of supermesh polygon " +
                       std::to_string(p)};
    }
    std::int32_t si = 0, tj = 0;
    if (!decode(sraw, src_base, ns, si) || !decode(traw, tgt_base, nt, tj)) {
      discard_supermesh(mb, out_set);
      return Failure{"E-REMAP-INTX",
                   "MOAB attributed a supermesh polygon to parents (" +
                       std::to_string(sraw) + ", " + std::to_string(traw) +
                       "), which are neither the global ids this build "
                       "stamped (" + std::to_string(src_base) + "+, " +
                       std::to_string(tgt_base) +
                       "+) nor valid 0-based positions"};
    }
    auto xyz = corner_coords(mb, p, scratch);
    if (!xyz.ok()) {
      discard_supermesh(mb, out_set);
      return xyz.error();
    }
    const double area = sphere
                            ? spherical_polygon_area(xyz.value(), radius)
                            : std::fabs(planar_signed_area(xyz.value()));
    if (area > 0.0) out.entries.push_back(Overlap{tj, si, area});
  }

  // Parent measures in the SAME functional as the pieces. This is what makes
  // the coverage numbers below a pure geometric statement rather than a
  // restatement of the bilinear-versus-spherical discrepancy.
  auto parent_measures = [&](const Side& side, double r,
                             std::vector<double>& into) -> Result<void> {
    into.resize(side.cells.size());
    std::vector<double> local;
    for (std::size_t i = 0; i < side.cells.size(); ++i) {
      auto c = corner_coords(mb, side.cells[i], local);
      if (!c.ok()) return c.error();
      into[i] = sphere ? spherical_polygon_area(c.value(), r)
                       : std::fabs(planar_signed_area(c.value()));
      if (!(into[i] > 0.0)) {
        return Failure{"E-REMAP-GEOMETRY",
                     side.label + " cell " + std::to_string(i) +
                         " has zero area in the supermesh's own measure"};
      }
    }
    return Result<void>::success();
  };
  auto pm = parent_measures(src, radius, out.src_parent);
  if (!pm.ok()) {
    discard_supermesh(mb, out_set);
    return pm.error();
  }
  pm = parent_measures(tgt, radius, out.tgt_parent);
  if (!pm.ok()) {
    discard_supermesh(mb, out_set);
    return pm.error();
  }

  discard_supermesh(mb, out_set);
  return out;
}

// ---------------------------------------------------------------------------
// The audit, and making it a global statement
// ---------------------------------------------------------------------------

/// `%.17g`, because `std::to_string` gives six decimals and a coverage excess
/// of 1 + 1.7e-8 prints as "1.000000".
[[nodiscard]] std::string g17(double x) {
  // fmt, not snprintf: snprintf's %g reads the C locale, so a coverage figure
  // in a diagnostic could come out with a comma decimal point on one machine
  // and a period on another.
  return fmt::format("{:.17g}", x);
}

/// Every quantity the audit reports or decides on, held together so the
/// reduction below is one place rather than eight.
struct Audit {
  double min_src_coverage = 1.0;
  double min_tgt_coverage = 1.0;
  double max_row_defect = 0.0;
  double max_col_defect = 0.0;
  double min_weight = 0.0;
  bool over_src = false;
  bool over_tgt = false;
  double over_value = 0.0;
  std::size_t over_index = 0;
  std::size_t worst_src = 0;
};

/// Send each halo column's share of the coverage and of the column sum to the
/// rank that owns the cell, and add it there. A no-op without a halo, so the
/// serial path is bit-for-bit what it was.
[[nodiscard]] Result<void> fold_halo_contributions(const Side& src,
                                                   std::vector<double>& covered,
                                                   std::vector<double>& col_sum) {
  if (src.not_owned == 0 || src.backend == nullptr) {
    return Result<void>::success();
  }
  const Result<void> a =
      src.backend->accumulate_remap_halo(src.mid, src.dim, covered.data(), 1);
  if (!a.ok()) return a;
  return src.backend->accumulate_remap_halo(src.mid, src.dim, col_sum.data(),
                                            1);
}

/// Reduce the audit over the communicator, so every rank reports the same
/// numbers and, above all, reaches the same accept/refuse verdict.
[[nodiscard]] Result<void> reduce_audit([[maybe_unused]] const Side& src,
                                        [[maybe_unused]] Audit& audit) {
#ifdef FUSE_REMAP_PARALLEL
  if (src.pcomm == nullptr) return Result<void>::success();
  auto* pc = static_cast<moab::ParallelComm*>(src.pcomm);
  const MPI_Comm comm = pc->comm();
  int nproc = 1;
  MPI_Comm_size(comm, &nproc);
  if (nproc <= 1) return Result<void>::success();

  double mins[3] = {audit.min_src_coverage, audit.min_tgt_coverage,
                    audit.min_weight};
  double maxs[3] = {audit.max_row_defect, audit.max_col_defect,
                    audit.over_value};
  int flags[2] = {audit.over_src ? 1 : 0, audit.over_tgt ? 1 : 0};
  if (MPI_Allreduce(MPI_IN_PLACE, mins, 3, MPI_DOUBLE, MPI_MIN, comm) !=
          MPI_SUCCESS ||
      MPI_Allreduce(MPI_IN_PLACE, maxs, 3, MPI_DOUBLE, MPI_MAX, comm) !=
          MPI_SUCCESS ||
      MPI_Allreduce(MPI_IN_PLACE, flags, 2, MPI_INT, MPI_MAX, comm) !=
          MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "the collective reduction of the conservation audit failed, "
                 "so this rank cannot know whether the other ranks accepted "
                 "the map; refusing rather than proceeding out of step"};
  }
  audit.min_src_coverage = mins[0];
  audit.min_tgt_coverage = mins[1];
  audit.min_weight = mins[2];
  audit.max_row_defect = maxs[0];
  audit.max_col_defect = maxs[1];
  audit.over_value = maxs[2];
  audit.over_src = flags[0] != 0;
  audit.over_tgt = flags[1] != 0;
#endif
  return Result<void>::success();
}

/// What to tell someone whose conservative map was refused for coverage.
///
/// NOT "clear the Conservative flag" when the read is partitioned. On a
/// partitioned mesh that advice converts a loud, correct refusal into the
/// silent truncation this whole guard exists to prevent -- measured at two
/// thirds of the transferred quantity on two ranks. The honest advice there is
/// to look at the halo, which is what actually decides whether the source
/// cells are fully seen.
[[nodiscard]] std::string coverage_advice(const Side& src) {
  const int ranks = src.backend == nullptr ? 1 : src.backend->communicator_size();
  if (ranks <= 1) {
    return "Or clear the Conservative flag to accept a truncating map -- the "
           "shortfall is then reported in RemapWeights::min_src_coverage.";
  }
  return "This read is partitioned across " + std::to_string(ranks) +
         " ranks, so DO NOT reach for the Conservative flag here: clearing it "
         "on a partitioned mesh turns this refusal into a silent truncation, "
         "which is the failure the flag is currently the only backstop "
         "against. Check instead that build_remap_halo() ran for this pair "
         "and that it adopted the cells this rank's targets need (" +
         std::to_string(src.not_owned) +
         " halo cells are present on this rank).";
}

// ---------------------------------------------------------------------------
// Assembly
// ---------------------------------------------------------------------------

/// Turn raw overlaps into the sorted, merged, normalised COO matrix and fill
/// the conservation audit.
///
///     W_ji = (overlap_ij / parent_i) * (m^s_i / m^t_j)
///
/// The first factor is the dimensionless fraction of source cell `i` that lies
/// in target cell `j`; the second converts a source density to a target
/// density. Nothing is renormalised to force either sum: see the header for
/// why the tempting `overlap / sum_k overlap_ik` was rejected.
[[nodiscard]] Result<RemapWeights> assemble(const Side& src, const Side& tgt,
                                            Supermesh sm, bool conservative) {
  RemapWeights w;
  w.n_rows = tgt.cells.size();
  w.n_cols = src.cells.size();
  w.src_measure_total = src.measure_total;
  w.tgt_measure_total = tgt.measure_total;

  std::sort(sm.entries.begin(), sm.entries.end(),
            [](const Overlap& a, const Overlap& b) {
              return a.row != b.row ? a.row < b.row : a.col < b.col;
            });

  std::vector<double> covered_src(w.n_cols, 0.0);
  std::vector<double> covered_tgt(w.n_rows, 0.0);
  std::vector<double> row_sum(w.n_rows, 0.0);
  std::vector<double> col_sum(w.n_cols, 0.0);

  w.row.reserve(sm.entries.size());
  w.col.reserve(sm.entries.size());
  w.val.reserve(sm.entries.size());

  for (std::size_t k = 0; k < sm.entries.size();) {
    std::size_t e = k + 1;
    double area = sm.entries[k].area;
    while (e < sm.entries.size() && sm.entries[e].row == sm.entries[k].row &&
           sm.entries[e].col == sm.entries[k].col) {
      area += sm.entries[e].area;
      ++e;
    }
    const std::size_t i = static_cast<std::size_t>(sm.entries[k].col);
    const std::size_t j = static_cast<std::size_t>(sm.entries[k].row);
    const double weight =
        (area / sm.src_parent[i]) * (src.measure[i] / tgt.measure[j]);
    covered_src[i] += area;
    covered_tgt[j] += area;
    row_sum[j] += weight;
    col_sum[i] += weight * tgt.measure[j];
    w.row.push_back(sm.entries[k].row);
    w.col.push_back(sm.entries[k].col);
    w.val.push_back(weight);
    k = e;
  }

  // ==========================================================================
  // THE AUDIT IS A GLOBAL STATEMENT, SO IT IS COMPUTED GLOBALLY.
  //
  // Two things go wrong if the numbers below are left per rank once a halo
  // exists, and the second is worse than the first:
  //
  //   * A SOURCE CELL STRADDLING A TARGET PARTITION BOUNDARY is covered partly
  //     by the target cells this rank holds and partly by another rank's. Each
  //     rank sees a fraction, and a correct, fully conservative map is reported
  //     as a coverage failure. Only the sum over the ranks holding a copy is
  //     the cell's real coverage, which is what `accumulate_remap_halo` is for.
  //   * WORSE: that fraction differs from rank to rank, so with
  //     `RemapFlags::Conservative` set, `E-REMAP-COVERAGE` fires on SOME ranks
  //     and not others. The ranks that error unwind while the rest walk into
  //     the next collective, and the job hangs instead of failing. That is
  //     precisely the failure mode E-REMAP-PARALLEL was added to avoid, so
  //     reproducing it here would be self-defeating. Every quantity that
  //     decides an error below is reduced over the communicator first, so all
  //     ranks reach the same verdict.
  // ==========================================================================
  {
    const Result<void> f =
        fold_halo_contributions(src, covered_src, col_sum);
    if (!f.ok()) return f.error();
  }

  Audit audit;
  for (std::size_t i = 0; i < w.n_cols; ++i) {
    // Halo columns were folded into their owners and zeroed; counting them
    // here would report a coverage of 0 for a cell that is fully covered
    // somewhere else.
    if (!src.view->is_owned(src.dim, i)) continue;
    const double c = covered_src[i] / sm.src_parent[i];
    if (c < audit.min_src_coverage) {
      audit.min_src_coverage = c;
      audit.worst_src = i;
    }
    // Overlaps partition a cell: their total can never EXCEED it. If it does,
    // the supermesh is double counting or the parent attribution is wrong --
    // which is exactly what a mis-read parent tag looks like -- and no flag
    // makes that acceptable.
    if (c > 1.0 + kCoverageTol) {
      audit.over_src = true;
      audit.over_value = c;
      audit.over_index = i;
    }
    audit.max_col_defect =
        std::max(audit.max_col_defect,
                 std::fabs(col_sum[i] - src.measure[i]) / src.measure[i]);
  }
  for (std::size_t j = 0; j < w.n_rows; ++j) {
    if (!tgt.view->is_owned(tgt.dim, j)) continue;
    const double c = covered_tgt[j] / sm.tgt_parent[j];
    audit.min_tgt_coverage = std::min(audit.min_tgt_coverage, c);
    if (c > 1.0 + kCoverageTol) {
      audit.over_tgt = true;
      audit.over_value = c;
      audit.over_index = j;
    }
    audit.max_row_defect =
        std::max(audit.max_row_defect, std::fabs(row_sum[j] - 1.0));
  }
  audit.min_weight = w.val.empty()
                         ? 0.0
                         : *std::min_element(w.val.begin(), w.val.end());

  {
    const Result<void> r = reduce_audit(src, audit);
    if (!r.ok()) return r.error();
  }

  if (audit.over_src || audit.over_tgt) {
    return Failure{"E-REMAP-INTX",
                 "the supermesh assigns " + g17(audit.over_value) +
                     " times its own area to a cell of " +
                     (audit.over_src ? src.label : tgt.label) +
                     " (worst local index " +
                     std::to_string(audit.over_index) +
                     " on some rank); overlaps cannot exceed the cell, so the "
                     "intersection or its parent attribution is wrong"};
  }

  w.min_src_coverage = audit.min_src_coverage;
  w.min_tgt_coverage = audit.min_tgt_coverage;
  w.max_row_defect = audit.max_row_defect;
  w.max_col_defect = audit.max_col_defect;
  // This builder assembles the weights from MeshView::measure directly, so the
  // measure it is conservative in IS MeshView::measure and the two conservation
  // residuals coincide by construction. They do not for the TempestRemap path;
  // see RemapWeights::max_col_defect_native.
  w.max_col_defect_native = w.max_col_defect;
  w.order = 1;
  w.min_weight = audit.min_weight;

  if (conservative && w.min_src_coverage < 1.0 - kCoverageTol) {
    return Failure{
        "E-REMAP-COVERAGE",
        "RemapFlags::Conservative was requested, but the target domain covers "
        "only " +
            std::to_string(w.min_src_coverage * 100.0) + "% of the worst " +
            src.label + " cell (of " + std::to_string(w.n_cols) +
            " on this rank; the figure is reduced over every rank). The mass "
            "in the uncovered part would be dropped, not transferred, so the "
            "map would not be conservative. Make the two regions cover the "
            "same domain. " +
            coverage_advice(src)};
  }
  return w;
}

// ---------------------------------------------------------------------------
// The high-order path: geometry out of MOAB, matrix back in, audit on top.
//
// Everything MOAB-specific stops here. `remap_backend::build_high_order_weights`
// sees a polygon soup and a geometry classification, never a moab::Interface --
// see src/remap/high_order_backend.hpp for why the seam is drawn there.
// ---------------------------------------------------------------------------

/// Row sums must be this close to 1, and every weight non-negative, before
/// `RemapFlags::Monotone` is certified. Both conditions together are exactly
/// bound preservation over each row's support; neither alone is.
///
/// 1e-12 and not 1e-9: the two builders that can satisfy this reach 2e-16
/// (axis-aligned supermesh) and 4e-16 (TempestRemap order 1) respectively, and
/// the case this has to reject -- a first-order supermesh map between two
/// cubed spheres, whose rows sum to 1 - 1e-3 because the two meshes do not
/// report the same total measure -- misses by nine decades. Anything in between
/// is a defect worth hearing about.
constexpr double kMonotoneRowTol = 1e-12;

/// The coverage tolerance for the HIGH-ORDER path, three decades looser than
/// `kCoverageTol` (1e-9) which Task 5.2 justified for its own supermesh.
///
/// Not slack for its own sake: it is a property of the backend's areas, and it
/// was measured. On a 32x4 latitude-longitude source the sum of the overlap
/// polygons attributed to a POLAR TRIANGLE exceeds TempestRemap's own area for
/// that triangle by 1.7e-8 relative (1.2e-10 absolute) -- TempestRemap's own
/// area-correction step uses a 1e-10 ABSOLUTE threshold and declines to correct
/// it for the same reason. On the cubed-sphere pairs the same quantity is
/// 4e-13. Refusing a legitimate lat-lon mesh over 1.7e-8 would be a false
/// refusal; the thing this check exists to catch -- an overlap polygon
/// attributed to the wrong parent, or counted twice -- is wrong by O(1), not by
/// eight decimal places.
constexpr double kHighOrderCoverageTol = 1e-6;

/// Copy a side's cells out as a polygon soup indexed exactly like the
/// `MeshView`, so a matrix row/column index needs no translation.
///
/// Reads only; nothing is written back to the `moab::Core`. Node order is by
/// first appearance while walking cells in ascending handle order, which is
/// deterministic for a given mesh.
[[nodiscard]] Result<remap_backend::PolygonSurface> polygon_surface(
    const Side& s) {
  remap_backend::PolygonSurface out;
  out.label = s.label;
  out.face_start.reserve(s.cells.size() + 1);
  out.face_start.push_back(0);

  std::unordered_map<moab::EntityHandle, std::int32_t> index;
  std::vector<moab::EntityHandle> nodes;
  std::vector<moab::EntityHandle> storage;

  for (const moab::EntityHandle h : s.cells) {
    const moab::EntityHandle* conn = nullptr;
    int nconn = 0;
    storage.clear();
    const moab::ErrorCode rval =
        s.mb->get_connectivity(h, conn, nconn, /*corners_only=*/true, &storage);
    if (rval != moab::MB_SUCCESS || conn == nullptr || nconn <= 0) {
      return intx_error(*s.mb,
                        "could not read the connectivity of entity " +
                            std::to_string(h),
                        rval);
    }
    for (int k = 0; k < nconn; ++k) {
      const auto it = index.find(conn[k]);
      std::int32_t local = 0;
      if (it == index.end()) {
        local = static_cast<std::int32_t>(nodes.size());
        if (nodes.size() >=
            static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
          return Failure{"E-REMAP-SIZE",
                       s.label + " has more vertices than int32 can address"};
        }
        index.emplace(conn[k], local);
        nodes.push_back(conn[k]);
      } else {
        local = it->second;
      }
      out.face_node.push_back(local);
    }
    out.face_start.push_back(static_cast<std::int32_t>(out.face_node.size()));
  }

  out.node_xyz.assign(nodes.size() * 3, 0.0);
  if (!nodes.empty()) {
    const moab::ErrorCode crval = s.mb->get_coords(
        nodes.data(), static_cast<int>(nodes.size()), out.node_xyz.data());
    if (crval != moab::MB_SUCCESS) {
      return intx_error(*s.mb, "could not read the vertices of " + s.label,
                        crval);
    }
  }
  return out;
}

/// Turn the backend's triplets into the sorted COO matrix plus the full audit.
///
/// The weights are taken AS THEY COME. Nothing is renormalised, and in
/// particular they are not rescaled into `MeshView::measure` -- the rescaling
/// that would make `max_col_defect` small perturbs every weight by a smooth
/// O(h^2) factor and caps the observed order of accuracy at 2, which was
/// measured before this comment was written. `max_col_defect` therefore reports
/// the honest ~1e-3 mismatch on a curved surface and
/// `max_col_defect_native` reports the ~1e-15 the map actually achieves in the
/// backend's own areas.
[[nodiscard]] Result<RemapWeights> assemble_high_order(
    const Side& src, const Side& tgt, remap_backend::HighOrderResult hi,
    bool conservative) {
  RemapWeights w;
  w.n_rows = tgt.cells.size();
  w.n_cols = src.cells.size();
  w.src_measure_total = src.measure_total;
  w.tgt_measure_total = tgt.measure_total;
  w.order = hi.order;

  std::vector<std::size_t> order(hi.val.size());
  std::iota(order.begin(), order.end(), std::size_t{0});
  std::sort(order.begin(), order.end(),
            [&](std::size_t a, std::size_t b) {
              return hi.row[a] != hi.row[b] ? hi.row[a] < hi.row[b]
                                            : hi.col[a] < hi.col[b];
            });

  std::vector<double> row_sum(w.n_rows, 0.0);
  std::vector<double> col_sum(w.n_cols, 0.0);
  std::vector<double> col_sum_native(w.n_cols, 0.0);

  w.row.reserve(order.size());
  w.col.reserve(order.size());
  w.val.reserve(order.size());
  for (std::size_t k = 0; k < order.size();) {
    const std::int32_t r = hi.row[order[k]];
    const std::int32_t c = hi.col[order[k]];
    double v = 0.0;
    std::size_t e = k;
    while (e < order.size() && hi.row[order[e]] == r &&
           hi.col[order[e]] == c) {
      v += hi.val[order[e]];
      ++e;
    }
    const std::size_t i = static_cast<std::size_t>(c);
    const std::size_t j = static_cast<std::size_t>(r);
    row_sum[j] += v;
    col_sum[i] += v * tgt.measure[j];
    col_sum_native[i] += v * hi.tgt_area[j];
    w.row.push_back(r);
    w.col.push_back(c);
    w.val.push_back(v);
    k = e;
  }

  for (std::size_t j = 0; j < w.n_rows; ++j) {
    w.max_row_defect = std::max(w.max_row_defect, std::fabs(row_sum[j] - 1.0));
    if (!(hi.tgt_area[j] > 0.0)) {
      return Failure{"E-REMAP-GEOMETRY",
                   tgt.label + " cell " + std::to_string(j) +
                       " has area " + std::to_string(hi.tgt_area[j]) +
                       " in the high-order backend's own measure"};
    }
    const double cov = hi.tgt_covered[j] / hi.tgt_area[j];
    w.min_tgt_coverage = std::min(w.min_tgt_coverage, cov);
    if (cov > 1.0 + kHighOrderCoverageTol) {
      return Failure{"E-REMAP-INTX",
                   "the high-order backend's supermesh assigns " + g17(cov) +
                       " times its own area to " + tgt.label + " cell " +
                       std::to_string(j) +
                       "; overlaps partition a cell and cannot exceed it, so "
                       "the intersection or its parent attribution is wrong"};
    }
  }
  std::size_t worst_src = 0;
  for (std::size_t i = 0; i < w.n_cols; ++i) {
    if (!(hi.src_area[i] > 0.0)) {
      return Failure{"E-REMAP-GEOMETRY",
                   src.label + " cell " + std::to_string(i) +
                       " has area " + std::to_string(hi.src_area[i]) +
                       " in the high-order backend's own measure"};
    }
    w.max_col_defect =
        std::max(w.max_col_defect,
                 std::fabs(col_sum[i] - src.measure[i]) / src.measure[i]);
    w.max_col_defect_native =
        std::max(w.max_col_defect_native,
                 std::fabs(col_sum_native[i] - hi.src_area[i]) / hi.src_area[i]);
    const double cov = hi.src_covered[i] / hi.src_area[i];
    if (cov < w.min_src_coverage) {
      w.min_src_coverage = cov;
      worst_src = i;
    }
    if (cov > 1.0 + kHighOrderCoverageTol) {
      return Failure{"E-REMAP-INTX",
                   "the high-order backend's supermesh assigns " + g17(cov) +
                       " times its own area to " + src.label + " cell " +
                       std::to_string(i) +
                       "; overlaps partition a cell and cannot exceed it, so "
                       "the intersection or its parent attribution is wrong"};
    }
  }

  w.min_weight =
      w.val.empty() ? 0.0 : *std::min_element(w.val.begin(), w.val.end());

  if (conservative && w.min_src_coverage < 1.0 - kHighOrderCoverageTol) {
    return Failure{
        "E-REMAP-COVERAGE",
        "RemapFlags::Conservative was requested, but the target domain covers "
        "only " +
            std::to_string(w.min_src_coverage * 100.0) + "% of " + src.label +
            " cell " + std::to_string(worst_src) + " (worst of " +
            std::to_string(w.n_cols) +
            "). The mass in the uncovered part would be dropped, not "
            "transferred. Clear the Conservative flag to accept a truncating "
            "map, or make the two regions cover the same domain."};
  }
  return w;
}

/// Check, do not assume. `RemapFlags::Monotone` is a promise to the caller, and
/// the two conditions below are exactly what the promise needs; a builder that
/// merely believes its weights are non-negative is how a limiter that does
/// nothing ships.
[[nodiscard]] Result<void> certify_monotone(RemapWeights& w,
                                            const char* builder) {
  if (w.min_weight < 0.0) {
    return Failure{"E-REMAP-MONOTONE",
                 std::string("RemapFlags::Monotone was requested and the ") +
                     builder + " map is not bound preserving: its smallest "
                     "weight is " +
                     std::to_string(w.min_weight) +
                     ", so a target cell can fall below every source value it "
                     "draws from"};
  }
  if (w.max_row_defect > kMonotoneRowTol) {
    return Failure{
        "E-REMAP-MONOTONE",
        std::string("RemapFlags::Monotone was requested and the ") + builder +
            " map is not bound preserving: its worst row sum is off unity by " +
            std::to_string(w.max_row_defect) +
            ", so a constant field does not survive the transfer and a target "
            "value can leave the range of the source values it draws from. "
            "This is the measure mismatch documented in remap_plan.hpp: the two "
            "meshes do not report the same total MeshView::measure, and this "
            "builder resolves that in favour of conservation. Use method "
            "\"tempest\" with order 1, which is conservative in its own "
            "spherical areas and has exact row sums, or accept a "
            "non-bound-preserving map by clearing RemapFlags::Monotone."};
  }
  w.monotone = true;
  return Result<void>::success();
}

}  // namespace remap_detail

// ===========================================================================
// The remap halo (R3). See remap_plan.hpp for the contract and moab_backend.hpp
// for the mesh-layer half.
//
// WHAT THIS DOES, in one sentence: it asks MOAB for the source cells that
// overlap this rank's target cells, wherever they live, and hands them to the
// backend to adopt as ordinary non-owned local entities of the source mesh.
//
// WHY THAT IS ENOUGH. Once those cells are local, the supermesh construction
// below needs no parallel mode at all -- it intersects "the cells this rank
// holds", and the cells this rank holds are now the right ones. Measured on
// MOAB's 3456-cell cubed sphere (8 partition sets) against a 1536-cell partner
// (4 partition sets, deliberately misaligned): the ORDINARY serial
// intersection over the halo-extended source gives target coverage 1.0 to
// 1e-16 at 1, 2 and 4 ranks, and the conservation test in
// test/unit/remap_parallel_test.cpp pins the integral end to end.
// ===========================================================================

Result<std::size_t> build_remap_halo(const RemapPlan& plan,
                                     const CouplingGraph& graph,
                                     MeshBackend& backend) {
  using namespace remap_detail;  // NOLINT: file-local detail namespace

  auto* moab_backend = dynamic_cast<MoabMeshBackend*>(&backend);
  if (moab_backend == nullptr) {
    return Failure{"E-REMAP-BACKEND",
                 "build_remap_halo needs the MOAB backend; this is the '" +
                     backend.name() + "' backend"};
  }

  auto src = resolve(graph, backend, plan.src, "source");
  if (!src.ok()) return src.error();
  auto tgt = resolve(graph, backend, plan.tgt, "target");
  if (!tgt.ok()) return tgt.error();
  Side& a = src.value();
  Side& b = tgt.value();

  if (a.dim != b.dim) {
    return Failure{"E-REMAP-DIM",
                 "the source cells are dimension " + std::to_string(a.dim) +
                     " and the target cells dimension " +
                     std::to_string(b.dim) + "; a halo pairs one with the "
                     "other and needs the same dimension on both sides"};
  }
  if (a.mb != b.mb) {
    return Failure{"E-REMAP-BACKEND",
                 "the two regions come from different MOAB instances"};
  }

  // Serial, a one-rank communicator, or the identity (both regions on one
  // mesh) all need no halo. Recorded as BUILT all the same, so a caller may
  // make the call unconditionally and `build_weights` can key its refusal on
  // "was this asked for" rather than on "did it find anything".
  const int ranks = moab_backend->communicator_size();
  if (ranks <= 1 || a.pcomm == nullptr || a.file_set == b.file_set) {
    return moab_backend->adopt_remap_halo(a.mid, b.mid, a.dim, nullptr, 0);
  }

#ifndef FUSE_REMAP_PARALLEL
  return Failure{"E-MOAB-NOMPI",
               "this build has no MPI, yet the backend reports a communicator "
               "of " + std::to_string(ranks) + " ranks"};
#else
  // GEOMETRY. The covering-set construction in this MOAB is declared on the
  // spherical intersection worker only. That is a limit of the backend doing
  // the construction, and the message names it as such -- a planar or
  // volumetric covering construction would slot in right here.
  if (a.dim != 2) {
    return Failure{"E-MESH-HALO-GEOMETRY",
                 "a partitioned remap needs a covering set, and the only "
                 "construction available in this build works on dimension-2 "
                 "SURFACES; these cells are dimension " +
                     std::to_string(a.dim) +
                     ". Volume covering sets are UNIMPLEMENTED here, not "
                     "impossible. Read the meshes without a communicator, or "
                     "add a volumetric covering construction."};
  }
  auto fa = fit_surface(a);
  if (!fa.ok()) return fa.error();
  auto fb = fit_surface(b);
  if (!fb.ok()) return fb.error();
  if (!fa.value().ok || !fb.value().ok ||
      fa.value().kind != Surface::Sphere || fb.value().kind != Surface::Sphere) {
    return Failure{
        "E-MESH-HALO-GEOMETRY",
        "a partitioned remap needs a covering set, and the only construction "
        "available in this build supports SPHERICAL surfaces only -- every "
        "vertex of both meshes at a common radius about the origin. This pair "
        "is not one. A planar covering construction is UNIMPLEMENTED here, "
        "not impossible: it would slot in behind build_remap_halo() and "
        "nothing above it would change. Until then, read the meshes without a "
        "communicator and remap redundantly on every rank."};
  }
  const double ra = fa.value().radius;
  const double rb = fb.value().radius;
  if (std::fabs(ra - rb) > kGeomTol * std::max(ra, rb)) {
    return Failure{"E-REMAP-GEOMETRY",
                 "the two surfaces are spheres of different radius (" +
                     std::to_string(ra) + " and " + std::to_string(rb) +
                     "); they are not the same domain"};
  }
  const double radius = 0.5 * (ra + rb);

  // ORIENTATION FIRST, AND THIS IS NOT OPTIONAL. Measured: with MOAB's own
  // cubed sphere left exactly as read -- its cells are negatively oriented,
  // see orient() -- the covering construction and the intersection that
  // follows reach only 0.80 of the target area. Oriented first, 1.0 to 1e-16.
  {
    const auto o1 = orient(a, fa.value());
    if (!o1.ok()) return o1.error();
    const auto o2 = orient(b, fb.value());
    if (!o2.ok()) return o2.error();
  }

  moab::Interface& mb = *a.mb;
  auto* pc = static_cast<moab::ParallelComm*>(a.pcomm);

  moab::Intx2MeshOnSphere worker(&mb);
  worker.set_radius_source_mesh(radius);
  worker.set_radius_destination_mesh(radius);
  worker.set_error_tolerance(radius * kIntxTol);
  worker.set_parallel_comm(pc);

  moab::ErrorCode rval = worker.FindMaxEdges(a.file_set, b.file_set);
  if (rval != moab::MB_SUCCESS) {
    return intx_error(mb, "Intx2Mesh::FindMaxEdges failed while building the "
                          "remap halo", rval);
  }
  // The bounding boxes of every rank's TARGET cells, which is what decides
  // which source cells have to travel.
  moab::Range local_verts;
  rval = worker.build_processor_euler_boxes(b.file_set, local_verts);
  if (rval != moab::MB_SUCCESS) {
    return intx_error(mb, "Intx2Mesh::build_processor_euler_boxes failed",
                      rval);
  }

  moab::EntityHandle covering = 0;
  rval = mb.create_meshset(moab::MESHSET_SET, covering);
  if (rval != moab::MB_SUCCESS) {
    return intx_error(mb, "could not create the covering set", rval);
  }
  // NOT create_departure_mesh_3rd_alg, which is the CSLAM entry point and
  // assumes the two sets are one distribution: driven on an independently
  // partitioned pair it produced 1824 covering cells on one rank and ZERO on
  // the other, and an intersection missing 15% of its polygons.
  rval = worker.construct_covering_set(a.file_set, covering, /*gnomonic=*/true,
                                       /*nb_ghost_layers=*/0,
                                       /*include_edges=*/false);
  if (rval != moab::MB_SUCCESS) {
    static_cast<void>(mb.delete_entities(&covering, 1));
    return intx_error(mb, "Intx2MeshOnSphere::construct_covering_set failed",
                      rval);
  }

  moab::Range cov;
  rval = mb.get_entities_by_dimension(covering, a.dim, cov, /*recursive=*/true);
  if (rval != moab::MB_SUCCESS) {
    static_cast<void>(mb.delete_entities(&covering, 1));
    return intx_error(mb, "could not list the covering set's cells", rval);
  }
  std::vector<std::uint64_t> handles;
  handles.reserve(cov.size());
  for (const moab::EntityHandle h : cov) {
    handles.push_back(static_cast<std::uint64_t>(h));
  }
  // The SET goes; the cells do not. The backend keeps the ones it adopts, and
  // the rest -- copies of cells this rank already holds -- stay in the
  // moab::Core as unreferenced geometry, the same bounded leak Task 5.2's
  // intersection already has (R27). Bounded because this is idempotent per
  // (source, target, dimension).
  static_cast<void>(mb.delete_entities(&covering, 1));

  return moab_backend->adopt_remap_halo(a.mid, b.mid, a.dim, handles.data(),
                                        handles.size());
#endif
}

Result<RemapWeights> build_weights(const RemapPlan& plan,
                                   const CouplingGraph& graph,
                                   const MeshBackend& backend) {
  using namespace remap_detail;  // NOLINT: file-local detail namespace

  // =========================================================================
  // Plan-level dispatch. Nothing here touches a mesh: these are statements
  // about what was asked for.
  // =========================================================================

  if (plan.src_disc != Discretization::FV ||
      plan.tgt_disc != Discretization::FV) {
    // NOT "TempestRemap cannot do this" -- it can, `GenerateOfflineMapWithMeshes`
    // takes "cgll" and "dgll". The blocker is on FUSE's side and is named as
    // such, because that is what a reader has to fix.
    return Failure{
        "E-REMAP-UNSUPPORTED",
        std::string("both builders here make FV<->FV weights only; the plan "
                    "asks for ") +
            to_cstring(plan.src_disc) + " -> " + to_cstring(plan.tgt_disc) +
            ". This is a FUSE limitation, not a backend one: RemapWeights "
            "indexes rows and columns by MeshView CELL local index, and "
            "MeshBackend exposes no numbering for spectral-element "
            "degrees of freedom or for point-cloud nodes, so there is nothing "
            "to address the rows and columns of such a matrix with. Closing "
            "this needs a DOF map on MeshBackend first."};
  }

  // Order. `src_order`/`tgt_order` default to 0, meaning "unspecified".
  if (plan.src_order < 0 || plan.tgt_order < 0) {
    return Failure{"E-REMAP-UNSUPPORTED",
                 "negative reconstruction order (" +
                     std::to_string(plan.src_order) + ", " +
                     std::to_string(plan.tgt_order) + ")"};
  }
  if (plan.src_order > 0 && plan.tgt_order > 0 &&
      plan.src_order != plan.tgt_order) {
    return Failure{
        "E-REMAP-UNSUPPORTED",
        "the plan asks for source order " + std::to_string(plan.src_order) +
            " and target order " + std::to_string(plan.tgt_order) +
            ". For an FV->FV map the reconstruction lives on the SOURCE and "
            "there is nothing for a different target order to mean, so one of "
            "the two would have to be ignored. Set them equal, or leave one at "
            "0."};
  }
  const bool want_high = has(plan.flags, RemapFlags::HighOrder);
  const bool want_monotone = has(plan.flags, RemapFlags::Monotone);
  const int asked_order = std::max(plan.src_order, plan.tgt_order);
  const int order = asked_order > 0 ? asked_order : (want_high ? 2 : 1);
  if (want_high && order < 2) {
    return Failure{"E-REMAP-UNSUPPORTED",
                 "RemapFlags::HighOrder was set together with reconstruction "
                 "order 1, which is the piecewise-constant map. Drop the flag, "
                 "or ask for an order above 1."};
  }
  if (want_monotone && order > 1) {
    // The same refusal the backend makes, made earlier so it also applies to
    // geometries the backend would reject first and so the reason quoted is the
    // accuracy/monotonicity conflict rather than "not a sphere".
    return Failure{
        "E-REMAP-UNSUPPORTED",
        "RemapFlags::Monotone was requested together with reconstruction order " +
            std::to_string(order) +
            ". Bound preservation for every input field is equivalent to `all "
            "weights >= 0 and all row sums 1`, and a reconstruction that is "
            "exact for polynomials of degree >= 1 needs a stencil wider than "
            "the immediate neighbours, whose least-squares fit puts negative "
            "entries in it (measured minimum weight -0.049 at order 2, -0.075 "
            "at order 4 on a cubed-sphere pair). TempestRemap resolves this by "
            "ignoring the order -- it calls its first-order kernel whenever "
            "fMonotone is set -- and this library will not return that map "
            "under the high-order name. Ask for order 1 with Monotone, or for "
            "order " + std::to_string(order) + " without it."};
  }

  // Builder. `method` is the explicit lever; an order above 1 selects the
  // high-order builder when no method is named.
  enum class Builder { Supermesh, HighOrder };
  Builder builder = Builder::Supermesh;
  if (plan.method.empty()) {
    builder = order > 1 ? Builder::HighOrder : Builder::Supermesh;
  } else if (plan.method == "intx" || plan.method == "conservative") {
    builder = Builder::Supermesh;
    if (order > 1) {
      return Failure{
          "E-REMAP-UNSUPPORTED",
          "method '" + plan.method +
              "' is the first-order conservative supermesh builder and the "
              "plan asks for order " + std::to_string(order) +
              ". It builds the piecewise-constant map and will not return it "
              "under another name. Leave `method` empty, or set it to "
              "'tempest', to get the high-order builder."};
    }
  } else if (plan.method == "tempest") {
    builder = Builder::HighOrder;
  } else {
    return Failure{"E-REMAP-UNSUPPORTED",
                 "unknown remap method '" + plan.method +
                     "'. Known methods: '' (default: supermesh below order 2, "
                     "TempestRemap at order 2 and above), 'intx' and "
                     "'conservative' (the first-order supermesh builder), and "
                     "'tempest' (the high-order backend, which in this build "
                     "is " +
                     std::string(remap_backend::high_order_backend_name()) +
                     ")."};
  }

  auto src = resolve(graph, backend, plan.src, "source");
  if (!src.ok()) return src.error();
  auto tgt = resolve(graph, backend, plan.tgt, "target");
  if (!tgt.ok()) return tgt.error();
  Side& a = src.value();
  Side& b = tgt.value();

  if (a.dim != b.dim) {
    return Failure{"E-REMAP-DIM",
                 "the source cells are dimension " + std::to_string(a.dim) +
                     " and the target cells dimension " +
                     std::to_string(b.dim) +
                     "; a conservative FV remap needs the same dimension on "
                     "both sides"};
  }
  if (a.mb != b.mb) {
    return Failure{"E-REMAP-BACKEND",
                 "the two regions come from different MOAB instances"};
  }

  // ==========================================================================
  // THE PARTITIONED CASE (R3). The supermesh below intersects the cells this
  // rank HOLDS. On a partitioned read that is the right answer exactly when
  // the source mesh carries a halo covering this rank's targets, and a silent
  // truncation otherwise -- measured at two thirds of the transferred quantity
  // on two ranks and seven eighths on four, returned as success. So the
  // question asked here is not "is this parallel" but "was the halo built".
  //
  // `> 1`, not "a communicator was set". A communicator of one rank is an
  // ordinary MPI job whose mesh happens to be undivided; every cell is local
  // and the serial construction is exactly right. Refusing it would take
  // remap away from every single-rank MPI run, and no serial test can see the
  // difference -- which is why test/unit/remap_parallel_test.cpp runs a
  // one-rank leg as a POSITIVE control.
  // ==========================================================================
  if (a.backend != nullptr && a.backend->communicator_size() > 1 &&
      a.file_set != b.file_set &&
      !a.backend->has_remap_halo(a.mid, b.mid, a.dim)) {
    return Failure{
        "E-REMAP-PARALLEL",
        "the mesh backend was given a communicator of " +
            std::to_string(a.backend->communicator_size()) +
            " ranks, so its meshes are partitioned, and no remap halo has "
            "been built for this source/target pair. The supermesh is built "
            "from the cells this rank holds; without a halo the source cells "
            "that overlap this rank's target cells but live on another rank "
            "are simply absent, and every overlap crossing a partition "
            "boundary is dropped -- measured at two thirds of the transferred "
            "quantity on two ranks and seven eighths on four, returned as "
            "success unless RemapFlags::Conservative happens to be set. Call "
            "build_remap_halo(plan, graph, backend) after loading the meshes "
            "and BEFORE anything sizes a buffer on them (it changes "
            "entity_count, and so region_size), or read the meshes without a "
            "communicator and remap redundantly on every rank."};
  }

  const bool conservative = has(plan.flags, RemapFlags::Conservative);

  // Same mesh on both sides: the supermesh is the mesh, and W is the identity.
  // Short-circuited because it is exactly right and because handing MOAB a set
  // to intersect with itself is not something Intx2Mesh is specified to do.
  //
  // Ahead of the builder split, and for every order: the identity reproduces
  // any field exactly, so it satisfies an order-N request as well as an order-1
  // one, and it is bound preserving. Both facts are recorded rather than
  // assumed -- `monotone` is set because a permutation of ones has non-negative
  // weights and unit row sums, which is the same criterion `certify_monotone`
  // applies everywhere else.
  if (a.file_set == b.file_set) {
    RemapWeights w;
    w.n_rows = b.cells.size();
    w.n_cols = a.cells.size();
    w.src_measure_total = a.measure_total;
    w.tgt_measure_total = b.measure_total;
    w.row.resize(w.n_rows);
    w.col.resize(w.n_rows);
    w.val.assign(w.n_rows, 1.0);
    for (std::size_t i = 0; i < w.n_rows; ++i) {
      w.row[i] = static_cast<std::int32_t>(i);
      w.col[i] = static_cast<std::int32_t>(i);
    }
    w.order = order;
    w.min_weight = w.val.empty() ? 0.0 : 1.0;
    w.monotone = true;
    return w;
  }

  // =========================================================================
  // The high-order builder.
  // =========================================================================
  if (builder == Builder::HighOrder) {
    // NOT ON A PARTITIONED READ. The halo makes the FIRST-ORDER supermesh path
    // correct in parallel and that is what is tested; the high-order backend
    // is handed a polygon soup that, on a partitioned read, is a piece of a
    // surface with a halo fringe rather than a closed one.
    //
    // This was CHECKED, not assumed. With this refusal removed and TempestRemap
    // present, the halo-extended source at 2 ranks gives
    // "E-REMAP-INTX: TempestRemap could not build the overlap mesh ...
    // (return code 0, 1024 overlap faces)" -- it does not merely lose accuracy
    // near the fringe, it fails outright, and it fails by printing while
    // returning success (which is why Task 5.3 reads its Announce stream at
    // all). Refusing here turns that into one clear sentence at the right
    // level instead of a backend diagnostic about overlap faces.
    if (a.backend != nullptr && a.backend->communicator_size() > 1 &&
        a.file_set != b.file_set) {
      return Failure{
          "E-REMAP-PARALLEL",
          "the high-order builder has not been verified on a partitioned "
          "read and will not guess. The mesh is partitioned across " +
              std::to_string(a.backend->communicator_size()) +
              " ranks; with a remap halo the source surface this backend sees "
              "is a piece of the domain plus a fringe, not a closed surface, "
              "and its reconstruction stencils near that fringe are "
              "unmeasured. The FIRST-ORDER conservative supermesh builder "
              "(method 'intx') IS supported on a partitioned read once "
              "build_remap_halo() has run, and is conservation-tested there. "
              "Use it, or read the meshes without a communicator."};
    }
    // Geometry classification is FUSE's job, not the backend's: the backend is
    // told what the surface IS and decides whether it can work on it. A mesh
    // this library cannot classify as a surface at all is refused here, so the
    // message is the same whether or not a backend happens to be linked in.
    remap_backend::HighOrderRequest req;
    req.order = order;
    req.monotone = want_monotone;
    req.kind = remap_backend::SurfaceKind::Other;
    req.radius = 1.0;

    if (a.dim == 2) {
      auto fa = fit_surface(a);
      if (!fa.ok()) return fa.error();
      auto fb = fit_surface(b);
      if (!fb.ok()) return fb.error();
      if (fa.value().ok && fb.value().ok &&
          fa.value().kind == fb.value().kind) {
        if (fa.value().kind == Surface::Sphere) {
          const double ra = fa.value().radius;
          const double rb = fb.value().radius;
          if (std::fabs(ra - rb) > kGeomTol * std::max(ra, rb)) {
            return Failure{"E-REMAP-GEOMETRY",
                         "the two surfaces are spheres of different radius (" +
                             std::to_string(ra) + " and " + std::to_string(rb) +
                             "); they are not the same domain"};
          }
          req.kind = remap_backend::SurfaceKind::Sphere;
          req.radius = 0.5 * (ra + rb);
        } else {
          req.kind = remap_backend::SurfaceKind::Plane;
        }
      }
    }
    if (req.kind == remap_backend::SurfaceKind::Other) {
      return Failure{
          "E-REMAP-UNSUPPORTED",
          "high-order and monotone remap are provided by a surface backend, "
          "and this pair is not a surface this library can classify: the cells "
          "are dimension " +
              std::to_string(a.dim) +
              " and, at dimension 2, every vertex of both meshes must lie "
              "either at a common radius about the origin (a sphere) or in a "
              "common plane z = const. Volume meshes and other curved surfaces "
              "are UNIMPLEMENTED for high-order remap -- the first-order "
              "conservative supermesh builder (method 'intx') does handle "
              "axis-aligned volumes."};
    }

    auto psrc = polygon_surface(a);
    if (!psrc.ok()) return psrc.error();
    auto ptgt = polygon_surface(b);
    if (!ptgt.ok()) return ptgt.error();

    auto hi = remap_backend::build_high_order_weights(req, psrc.value(),
                                                      ptgt.value());
    if (!hi.ok()) return hi.error();
    if (hi.value().order != order) {
      // The backend contract says it must refuse rather than under-deliver.
      // Enforced, because "the backend would never do that" is how a silently
      // degraded map gets shipped.
      return Failure{"E-REMAP-DEGRADED",
                   std::string(remap_backend::high_order_backend_name()) +
                       " returned an order-" + std::to_string(hi.value().order) +
                       " map for an order-" + std::to_string(order) +
                       " request"};
    }
    auto w = assemble_high_order(a, b, std::move(hi.value()), conservative);
    if (!w.ok()) return w.error();
    if (want_monotone) {
      auto ok = certify_monotone(w.value(),
                                 remap_backend::high_order_backend_name());
      if (!ok.ok()) return ok.error();
    }
    return w;
  }

  // =========================================================================
  // The first-order supermesh builder, unchanged from Task 5.2.
  // =========================================================================
  const BoxTry ba = try_boxes(a);
  if (ba.failed) return ba.error;
  const BoxTry bb = try_boxes(b);
  if (bb.failed) return bb.error;

  // `RemapFlags::Monotone` on this builder is a CHECK, not a different
  // construction: the map it makes is a convex combination whenever the two
  // meshes report the same total measure, and is not one when they do not. The
  // check is the same one the high-order builder's order-1 map goes through.
  const auto finish = [&](Result<RemapWeights> r) -> Result<RemapWeights> {
    if (!r.ok() || !want_monotone) return r;
    auto ok = certify_monotone(r.value(), "first-order supermesh");
    if (!ok.ok()) return ok.error();
    return r;
  };

  if (ba.ok && bb.ok) {
    auto sm = intersect_boxes(a, b, ba.mesh, bb.mesh);
    if (!sm.ok()) return sm.error();
    return finish(assemble(a, b, std::move(sm.value()), conservative));
  }

  if (a.dim == 2) {
    auto fa = fit_surface(a);
    if (!fa.ok()) return fa.error();
    auto fb = fit_surface(b);
    if (!fb.ok()) return fb.error();
    if (fa.value().ok && fb.value().ok && fa.value().kind == fb.value().kind) {
      if (fa.value().kind == Surface::Sphere) {
        const double ra = fa.value().radius;
        const double rb = fb.value().radius;
        if (std::fabs(ra - rb) > kGeomTol * std::max(ra, rb)) {
          return Failure{"E-REMAP-GEOMETRY",
                       "the two surfaces are spheres of different radius (" +
                           std::to_string(ra) + " and " + std::to_string(rb) +
                           "); they are not the same domain, and remapping "
                           "between them would conserve a density, not a "
                           "quantity"};
        }
      } else if (std::fabs(fa.value().z - fb.value().z) >
                 kGeomTol * std::max({std::fabs(fa.value().z),
                                      std::fabs(fb.value().z), 1.0})) {
        return Failure{"E-REMAP-GEOMETRY",
                     "the two surfaces are planes at different z (" +
                         std::to_string(fa.value().z) + " and " +
                         std::to_string(fb.value().z) + ")"};
      }
      auto sm = intersect_surfaces(a, b, fa.value(), fb.value());
      if (!sm.ok()) return sm.error();
      return finish(assemble(a, b, std::move(sm.value()), conservative));
    }
  }

  return Failure{
      "E-REMAP-UNSUPPORTED",
      "no supermesh construction fits this pair. Supported: cells that are "
      "axis-aligned boxes in any dimension; dimension-2 surfaces that are both "
      "spheres about the origin of the same radius; dimension-2 surfaces that "
      "are both planes z = const. General unstructured VOLUME supermeshing is "
      "not implemented, and neither is a curved surface that is neither sphere "
      "nor plane. The high-order backend does not help with either: it remaps "
      "spherical surfaces only. Both would need a new supermesh construction "
      "here (a tetrahedral clipper for volumes; a surface parameterisation for "
      "the general curved case)."};
}

#endif  // FUSE_HAVE_MOAB

}  // namespace fuse
