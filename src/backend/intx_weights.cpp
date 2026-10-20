#include <fmt/format.h>

#include "fuse/base/config.hpp"
#include "fuse/backend/consistent_interp.hpp"
#include "fuse/backend/mls_weights.hpp"
#include "fuse/backend/native_halo.hpp"
#include "fuse/backend/remap_plan.hpp"
#include "fuse/backend/remap_weight_normalize.hpp"
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
// src/backend/moab_backend.cpp. Nothing below leaks a MOAB type through a
// header.
#include <moab/CN.hpp>
#include <moab/Core.hpp>
#include <moab/IntxMesh/Intx2MeshInPlane.hpp>
#include <moab/IntxMesh/Intx2MeshOnSphere.hpp>
#include <moab/Range.hpp>

#include <MBTagConventions.hpp>

// MOABConfig.h arrives via Core.hpp, so MOAB_HAVE_MPI is only knowable here.
// Both halves are required, exactly as in src/backend/moab_backend.cpp: MOAB must
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
#include <cstdlib>
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

namespace {

/// A mesh entity set resolved to a view and a dimension, using only the
/// backend-agnostic `MeshBackend` interface -- no MOAB dependency, so this
/// compiles and is used identically whether or not `FUSE_HAVE_MOAB` is set.
struct SupermeshSide {
  const MeshView* view = nullptr;
  int dim = 0;
  // The backend's own MeshId for this side, needed by the native remap
  // halo's E-REMAP-PARALLEL check below (and by `build_remap_halo`'s
  // native branch) to ask the backend-specific questions
  // `fuse::native::NativeMeshBackend` answers beyond the generic
  // `MeshBackend` interface -- the same reason `resolve()`'s own `Side`
  // (MOAB's equivalent, above) carries a `mid` field.
  MeshId mid;
};

/// How far a cell's supermesh coverage may sit from 1 before the overlap is
/// treated as partial rather than as arithmetic -- a property of the
/// backend-agnostic coverage audit (`audit_coverage`, used by both the MOAB
/// and the dependency-free path, since the clipping it measures comes from
/// `MeshView`, not from MOAB itself; shared here so the two paths cannot
/// drift onto two different numbers for the same check). The worst per-cell
/// coverage defect measured on the hardest supported case -- MOAB's
/// 3456-cell cubed sphere against an equiangular 1536-cell partner, 7374
/// intersection polygons including slivers -- is 7.7e-14, five decades below
/// this.
constexpr double kCoverageTol = 1e-9;

[[nodiscard]] Result<SupermeshSide> supermesh_side(const CouplingGraph& graph,
                                                    const MeshBackend& backend,
                                                    RegionId id, ComponentId placement,
                                                    const char* which) {
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
  // The processor set's OWN decomposition (RemapPlan::placement); the default
  // view when the backend does not distinguish processor sets.
  const auto view = backend.placed_view(mid.value(), placement);
  if (!view.ok()) return view.error();
  SupermeshSide s;
  s.view = view.value();
  s.mid = mid.value();
  s.dim = r.dimension >= 0 ? r.dimension : s.view->dimension();
  return s;
}

/// The dependency-free FV<->FV path: exact-simplex supermesh weights, order 1
/// only. This is the entirety of `build_weights()` when the library is
/// compiled without MOAB, and it is ALSO what a MOAB-enabled build must fall
/// back to when the backend handed to `build_weights()` is not a
/// `MoabMeshBackend` (REVIEW.md N-BACKEND-4: before this fix, a MOAB build's
/// `build_weights()` always took the MOAB-only path -- via `resolve()`'s
/// `dynamic_cast<const MoabMeshBackend*>` -- and rejected E-REMAP-BACKEND for
/// a serial mesh pair that a base build configuration would have served, a capability a
/// MOAB-enabled build configuration must not lose just because it CAN also do more).
[[nodiscard]] Result<RemapWeights> build_weights_via_supermesh(
    const RemapPlan& plan, const CouplingGraph& graph,
    const MeshBackend& backend) {
  const int asked = std::max(plan.src_order, plan.tgt_order);
  if (asked > 1 || has(plan.flags, RemapFlags::HighOrder)) {
    return Failure{
        "E-REMAP-NOHIGHORDER",
        "this backend has no MOAB/TempestRemap high-order path, so the only "
        "reconstruction available is piecewise constant (order 1), and "
        "order " + std::to_string(asked) +
            " was asked for. Order 1 is exact for constants and "
            "conservative; higher order on the sphere is what the "
            "TempestRemap tier adds. Drop RemapFlags::HighOrder, or ask for "
            "it against a MOAB mesh pair in a build configured with "
            "-DFUSE_ENABLE_MOAB=ON against a MOAB built with TempestRemap."};
  }
  auto s = supermesh_side(graph, backend, plan.src, plan.placement, "source");
  if (!s.ok()) return s.error();
  auto d = supermesh_side(graph, backend, plan.tgt, plan.placement, "target");
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
  // ==========================================================================
  // THE PARTITIONED CASE, for `fuse::native::NativeMeshBackend` (docs/dev/
  // native-halo.md). This builder intersects the cells this rank HOLDS --
  // correct on a partitioned read only once the source mesh carries a remap
  // halo built by `fuse::native::build_native_remap_halo`, exactly the MOAB
  // path's own E-REMAP-PARALLEL reasoning (below, in the FUSE_HAVE_MOAB half
  // of this file) applied to the one other backend that can report more
  // than one rank. `s.mid.value() == d.mid.value()` is the identity map (the
  // same region on both sides): every one of the source's cells IS a target
  // cell already, local by construction, so no halo is ever needed for it.
  // ==========================================================================
  // Hoisted out of the `if` below (round 7, RemapHalo track): the
  // conservative-coverage fold further down needs this SAME pointer, not a
  // second `dynamic_cast`, and `nullptr` on a non-native backend is exactly
  // the "this check does not apply" guard that code needs too.
  const auto* native_backend = dynamic_cast<const native::NativeMeshBackend*>(&backend);
  if (native_backend != nullptr) {
    // KNOWN_ISSUES.md D73 (RemapHalo track round 6, host measurement 2026-10-19): `plan.placement`
    // is the ONLY argument missing here -- `has_remap_halo`'s own 4th parameter defaults to
    // `ComponentId{}` (the "unplaced decomposition", that default parameter's own doc comment,
    // native_backend.hpp), but `fuse::native::build_native_remap_halo`'s own `commit_remap_halo`
    // call (native_halo.cpp) records the triple under `placement` -- the TARGET's own, real
    // ComponentId, round 3's own fix, never the default. A halo genuinely built and committed under
    // a real placement is therefore invisible to this exact check, which asked under a DIFFERENT
    // key (the default) that was never written to -- measured on the owner's host: `initialize()`
    // refused E-REMAP-PARALLEL on BOTH ranks of a GREEN row whose own trace already showed the halo
    // had built cleanly (adopted_source_cells printed correctly, no exception) moments earlier.
    // Passing `plan.placement` here is the SAME placement `supermesh_side()` already resolved both
    // `s`/`d` through, two lines above -- the query now asks under the identical key the commit
    // wrote.
    if (native_backend->communicator_size() > 1 && s.value().mid.value() != d.value().mid.value() &&
        !native_backend->has_remap_halo(s.value().mid, d.value().mid, s.value().dim, plan.placement)) {
      return Failure{
          "E-REMAP-PARALLEL",
          "the native mesh backend reports a communicator of " +
              std::to_string(native_backend->communicator_size()) +
              " ranks, so its meshes are partitioned, and no remap halo has been "
              "built for this source/target pair. The supermesh is built from the "
              "cells this rank holds; without a halo the source cells that overlap "
              "this rank's target cells but live on another rank are simply "
              "absent, and every overlap crossing a partition boundary is dropped. "
              "Call build_remap_halo(plan, graph, backend) after loading the "
              "meshes and BEFORE anything sizes a buffer on them (it changes "
              "entity_count, and so region_size), or read the meshes without a "
              "communicator and remap redundantly on every rank."};
    }
  }
  auto built = build_supermesh_weights(*s.value().view, *d.value().view,
                                       s.value().dim, plan.conservation);
  if (!built.ok()) return built.error();
  // Soma-import finding (docs/dev/soma-import.md, RemapPartial.* rows), same
  // defect as REVIEW.md TF-5: this path computed min_src_coverage all along
  // (build_supermesh_weights's own audit) but never read it against
  // RemapFlags::Conservative -- only the FUSE_HAVE_MOAB half of this file
  // enforced the refusal, so a tier-0 (no-MOAB) build silently dropped
  // uncovered source mass on a channel explicitly marked Conservative, and a
  // MOAB-capable build whose backend happens not to be MoabMeshBackend
  // (N-BACKEND-4's own dispatch just above routes here) silently accepted
  // the flag and then ignored it: the exact I4 shape (a documented
  // guarantee quietly not enforced). Same check, same tolerance, same error
  // code as the MOAB path below, so a plan rejected there is rejected here
  // for the identical reason.
  //
  // KNOWN_ISSUES.md D73 (RemapHalo track round 7, host measurement
  // 2026-10-19): `min_src_coverage`, as `build_supermesh_weights` just
  // returned it, is the coverage of the intersections THIS rank's own
  // merged views produced -- correct on a single rank, but on a partitioned
  // read a source cell this rank OWNS may have every one of its
  // intersections live on a PEER, which adopted it through the halo (the
  // "rank 0 (4 source cells, NO target cells)" shape: every one of its
  // owned cells shows 0% local coverage, even though the SAME cells are
  // fully covered once the peer's own halo-copy intersection is counted).
  // `fuse::native::fold_and_reduce_source_coverage` folds each halo copy's
  // own share back to its owner (the reverse of the forward fill
  // `build_native_remap_halo` itself already performs) and reduces the
  // result to one number over the WHOLE base communicator, so every rank's
  // Conservative check reads the SAME, now-correct value -- "agree first",
  // never a refusal one rank reaches alone from a locally computed 0%.
  if (has(plan.flags, RemapFlags::Conservative) && native_backend != nullptr &&
      native_backend->communicator_size() > 1 && s.value().mid.value() != d.value().mid.value()) {
    // KNOWN_ISSUES.md D82 (owner decision 2026-10-19): recovers the RAW
    // intersection measure from the (already divided) weight -- undo
    // whichever division build_supermesh_weights just applied: Extensive
    // divided by the SOURCE cell's own measure, Intensive by the target's.
    const bool extensive = plan.conservation == Conservation::Extensive;
    std::vector<double> src_covered(built.value().n_cols, 0.0);
    for (std::size_t k = 0; k < built.value().val.size(); ++k) {
      const auto i = static_cast<std::size_t>(built.value().col[k]);
      const auto j = static_cast<std::size_t>(built.value().row[k]);
      src_covered[i] += extensive ? built.value().val[k] * s.value().view->measure(s.value().dim, i)
                                  : built.value().val[k] * d.value().view->measure(d.value().dim, j);
    }
    const auto folded = native::fold_and_reduce_source_coverage(
        *native_backend, s.value().mid, d.value().mid, s.value().dim, plan.placement,
        *s.value().view, src_covered);
    if (!folded.ok()) return folded.error();
    built.value().min_src_coverage = folded.value();
  }
  if (has(plan.flags, RemapFlags::Conservative) &&
      built.value().min_src_coverage < 1.0 - kCoverageTol) {
    return Failure{
        "E-REMAP-COVERAGE",
        "RemapFlags::Conservative was requested, but the target domain "
        "covers only " +
            std::to_string(built.value().min_src_coverage * 100.0) +
            "% of the worst source cell (of " +
            std::to_string(built.value().n_cols) +
            "). The mass in the uncovered part would be dropped, not "
            "transferred, so the map would not be conservative. Make the "
            "two regions cover the same domain, or clear the Conservative "
            "flag to accept a truncating map -- the shortfall is then "
            "reported in RemapWeights::min_src_coverage."};
  }
  return built;
}

}  // namespace

#if !FUSE_HAVE_MOAB

/// Without MOAB, weights come from the dependency-free simplex supermesh.
///
/// This used to reject, on the grounds that "the supermesh intersection has no
/// dependency-free implementation". That is no longer true: `supermesh.cpp`
/// clips the simplices every `MeshView` provides and computes analytic
/// moments, so lowest-order conservative remap on arbitrary geometry is
/// available in the DEFAULT build. What MOAB adds above this is its mesh
/// reader, its spatial search and its partitioning; what TempestRemap adds is
/// high order on the sphere.
Result<RemapWeights> build_weights(const RemapPlan& plan,
                                   const CouplingGraph& graph,
                                   const MeshBackend& backend) {
  // The consistent-interpolation H / H^T D pair
  // (fuse/backend/consistent_interp.hpp, header-only) is dependency-free -- it needs only
  // MeshView::centroid at dimension 0, which every backend already provides
  // -- so it dispatches BEFORE the FV<->FV-only check below, in both this
  // translation unit's build_weights() and the MOAB one. Neither the P0
  // supermesh path nor these two methods need each other, so there is no
  // ordering hazard in checking the method string first.
  if (plan.method == "consistent" || plan.method == "consistent-transpose") {
    return build_consistent_weights_for_plan(plan, graph, backend);
  }
  // Moving least squares (fuse/backend/mls_weights.hpp): dependency-free, so
  // dispatched identically with or without MOAB. Its options are part of the
  // method string and are never defaulted (E-MLS-OPTIONS).
  if (plan.method == "mls" || plan.method.rfind("mls:", 0) == 0) {
    return build_mls_weights(plan, graph, backend);
  }

  // Only what this builder can honour. The checks are deliberately the same
  // shape as the MOAB path's, so a plan rejected there is rejected here for the
  // same stated reason rather than silently accepted by the other build.
  if (plan.src_disc != Discretization::FV ||
      plan.tgt_disc != Discretization::FV) {
    return Failure{"E-REMAP-UNSUPPORTED",
                   std::string("the supermesh builder makes FV<->FV weights "
                               "only; the plan asks for ") +
                       to_cstring(plan.src_disc) + " -> " +
                       to_cstring(plan.tgt_disc)};
  }
  return build_weights_via_supermesh(plan, graph, backend);
}

Result<std::size_t> build_remap_halo(const RemapPlan& plan, const CouplingGraph& graph,
                                     MeshBackend& backend) {
  // `fuse::native::NativeMeshBackend` can report more than one rank with no
  // MOAB at all (docs/dev/native-halo.md), so its own planar/volumetric
  // covering-set construction must dispatch here too, not only in the
  // FUSE_HAVE_MOAB half of this file below -- the same dynamic_cast split
  // `build_weights_via_supermesh` above already uses for its own
  // E-REMAP-PARALLEL check.
  if (auto* native_backend = dynamic_cast<native::NativeMeshBackend*>(&backend)) {
    auto src = supermesh_side(graph, backend, plan.src, plan.placement, "source");
    if (!src.ok()) return src.error();
    auto tgt = supermesh_side(graph, backend, plan.tgt, plan.placement, "target");
    if (!tgt.ok()) return tgt.error();
    if (src.value().dim != tgt.value().dim) {
      return Failure{"E-REMAP-DIM",
                   "the source cells are dimension " + std::to_string(src.value().dim) +
                       " and the target cells dimension " +
                       std::to_string(tgt.value().dim) +
                       "; a halo pairs one with the other and needs the same dimension "
                       "on both"};
    }
    return native::build_native_remap_halo(*native_backend, src.value().mid, tgt.value().mid,
                                          src.value().dim, plan.placement);
  }
  // A successful no-op, not an error: without MOAB there is no OTHER
  // partitioned read to build a halo for (the native backend's own case is
  // handled above), and a caller that makes the call unconditionally
  // (which the header invites) must not have to branch on which backends
  // were compiled in.
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
// Tolerances. All relative; each is justified by a measurement recorded in
// the commit that introduced it. REVIEW.md R-11: no test in this tree reads
// any of them directly -- `test/unit/remap_test.cpp`, cited here previously
// as reproducing that measurement, does not exist under `test/backend/`.
// Implemented, untested (docs/dev/testing.md state).
// ---------------------------------------------------------------------------

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

/// Reconcile a per-rank outcome across the communicator before any rank acts
/// on it -- the mechanism docs/dev/rank-agreement.md names and invariant I2
/// requires. `resolve()`, `fit_surface()` and `orient()` each compute from
/// this rank's own partition (its local cell count, its local vertices, its
/// local cell orientation) and can therefore fail on one rank without
/// failing on a peer; without this, the ranks that did not fail walk into
/// the next MOAB collective (`FindMaxEdges`, `reduce_audit`'s
/// `MPI_Allreduce`, ...) and hang waiting for the one that did (REVIEW.md
/// R-01). This is the same shape as `reduce_audit` below, applied earlier,
/// to the checks that run before it.
///
/// `pcomm_raw` is a `moab::ParallelComm*`, or null (serial, or this rank's
/// own resolve never got far enough to obtain one) -- then there is nothing
/// to reconcile against and `local` is returned unchanged, matching
/// `reduce_audit`'s own no-op path. A communicator of one rank is also a
/// no-op: there is no peer to disagree with.
template <class T>
[[nodiscard]] Result<T> reconcile(void* pcomm_raw, Result<T> local) {
#ifdef FUSE_REMAP_PARALLEL
  if (pcomm_raw == nullptr) return local;
  auto* pc = static_cast<moab::ParallelComm*>(pcomm_raw);
  const MPI_Comm comm = pc->comm();
  int nproc = 1;
  if (MPI_Comm_size(comm, &nproc) != MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "could not size the communicator to reconcile a per-rank "
                 "outcome"};
  }
  if (nproc <= 1) return local;
  int rank = 0;
  static_cast<void>(MPI_Comm_rank(comm, &rank));

  const int local_failed = local.ok() ? 0 : 1;
  int any_failed = 0;
  if (MPI_Allreduce(&local_failed, &any_failed, 1, MPI_INT, MPI_MAX, comm) !=
          MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "the collective reduction of a per-rank resolve/fit/orient "
                 "outcome failed, so this rank cannot know whether its peers "
                 "agree; refusing rather than proceeding out of step"};
  }
  if (any_failed == 0) return local;

  // At least one rank failed. Every rank must return the IDENTICAL Failure
  // -- "some ranks reject, some proceed" is the hang this exists to rule
  // out -- so broadcast the lowest failing rank's message rather than each
  // rank keeping its own.
  const int candidate = local_failed != 0 ? rank : nproc;
  int failing_rank = nproc;
  if (MPI_Allreduce(&candidate, &failing_rank, 1, MPI_INT, MPI_MIN, comm) !=
          MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "the collective reduction of a per-rank resolve/fit/orient "
                 "outcome failed, so this rank cannot know whether its peers "
                 "agree; refusing rather than proceeding out of step"};
  }
  std::string msg = (rank == failing_rank && !local.ok())
                        ? local.error().message
                        : std::string();
  int len = static_cast<int>(msg.size());
  if (MPI_Bcast(&len, 1, MPI_INT, failing_rank, comm) != MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "the broadcast of the failing rank's message failed"};
  }
  msg.resize(static_cast<std::size_t>(len));
  if (len > 0 && MPI_Bcast(msg.data(), len, MPI_CHAR, failing_rank, comm) !=
                     MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "the broadcast of the failing rank's message failed"};
  }
  return Failure{"E-REMAP-PARALLEL",
               "rank " + std::to_string(failing_rank) + " refused: " + msg};
#else
  return local;
#endif
}

/// REVIEW.md N-BACKEND-1: `reconcile<T>` above agrees whether a per-rank
/// computation SUCCEEDED; it says nothing about whether the ranks that all
/// succeeded computed the SAME value. `Side::dim` is exactly such a value --
/// `moab_backend.hpp`'s own documented `MeshView::dimension()` limitation
/// says a rank whose local partition is empty, or holds only a
/// lower-dimensional boundary, reports a smaller dimension than its peers,
/// and `resolve()` only ever checks that ITS OWN `dim` is in `[1,3]` and
/// non-empty -- two ranks can each pass that check while disagreeing with
/// each other. Every comparison downstream (`a.dim != b.dim`, `a.dim != 2`)
/// then risks evaluating to a different answer on different ranks, racing
/// into a collective exactly like the outcome `reconcile<T>` already guards.
/// Agreed the same way: min and max across the communicator must coincide.
[[nodiscard]] Result<int> reconcile_dim(void* pcomm_raw, int dim,
                                        const std::string& label) {
#ifdef FUSE_REMAP_PARALLEL
  if (pcomm_raw == nullptr) return dim;
  auto* pc = static_cast<moab::ParallelComm*>(pcomm_raw);
  const MPI_Comm comm = pc->comm();
  int nproc = 1;
  if (MPI_Comm_size(comm, &nproc) != MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "could not size the communicator to reconcile " + label +
                     "'s dimension"};
  }
  if (nproc <= 1) return dim;
  int max_dim = dim;
  int min_dim = dim;
  if (MPI_Allreduce(&dim, &max_dim, 1, MPI_INT, MPI_MAX, comm) !=
          MPI_SUCCESS ||
      MPI_Allreduce(&dim, &min_dim, 1, MPI_INT, MPI_MIN, comm) !=
          MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "the collective reduction of " + label +
                     "'s dimension failed"};
  }
  if (max_dim != min_dim) {
    return Failure{
        "E-REMAP-DIM",
        label + "'s MeshView::dimension() disagrees across ranks (" +
            std::to_string(min_dim) + " to " + std::to_string(max_dim) +
            "); a rank whose local partition is empty at the region's usual "
            "dimension, or holds only a lower-dimensional boundary, reports "
            "a different dimension than its peers -- moab_backend.hpp's own "
            "documented MeshView::dimension() limitation. Set Region::dimension "
            "explicitly rather than relying on the mesh-wide default so every "
            "rank agrees without a reduction."};
  }
  return dim;
#else
  return dim;
#endif
}

[[nodiscard]] Result<Side> resolve(const CouplingGraph& graph,
                                   const MeshBackend& backend, RegionId id,
                                   ComponentId placement, const char* which,
                                   bool allow_empty = false) {
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

  // NOTE ON THE PARTITIONED CASE. There is no rejection here any more. The
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

  // D5, the fix this track's round 2 found necessary: the VIEW (what
  // entities exist, local-index addressed) and the Core/file_set/
  // communicator TRIPLE it is read through must come from the SAME
  // placement -- `native_mesh(mid, placement)` (moab_backend.hpp), not the
  // placement-oblivious overload, which always answered the backend's
  // DEFAULT Core regardless of which processor set's view was resolved.
  const auto view = moab_backend->placed_view(mid.value(), placement);
  if (!view.ok()) return view.error();

  // STRUCTURAL emptiness (this rank is outside `placement`'s own processor
  // set, mesh_view.hpp's documented "empty view, never the default"
  // contract) is derived from the plan's placement itself
  // (`placement_participates`), never from a local size -- exactly what
  // this track's round-2 brief requires, so a genuinely empty mesh (a data
  // problem) is not silently accepted just because its local count also
  // happens to be zero. There is no Core to ask for a native handle when
  // this rank does not participate at all, so `native_mesh` is skipped in
  // that case; `allow_empty` callers get a Side with no MOAB handles,
  // which is correct -- nothing downstream of `resolve()` may dereference
  // `s.mb`/`s.file_set` for a Side this function returns with empty
  // `cells`.
  const bool structurally_outside_placement =
      placement.valid() && !moab_backend->placement_participates(placement);
  Result<MoabNativeMesh> native =
      structurally_outside_placement
          ? Result<MoabNativeMesh>(MoabNativeMesh{})
          : moab_backend->native_mesh(mid.value(), placement);
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

  // STRUCTURALLY outside `placement` (this rank's own copy of `native` is
  // the default-constructed MoabNativeMesh{} above, deliberately: there is
  // no Core here to query at all) -- a VALID, empty contribution for an
  // `allow_empty` caller, derived from `placement_participates` above, not
  // from `range.empty()` below. `s.dim` falls back to the REGION's own
  // declared dimension (never 0, the empty view's own answer) when one was
  // given, because a caller resolving the cross-placement pair needs SOME
  // dimension to agree `a.dim == b.dim` against; an unset Region::dimension
  // here is `E-REMAP-DIM` below, the same "cells of dimension 1, 2 or 3"
  // floor every Side must clear regardless of how it got there.
  if (structurally_outside_placement) {
    if (!allow_empty) {
      return Failure{"E-MESH-PLACEMENT",
                   s.label + " is outside its own placement's processor "
                             "set, and this call did not permit an empty "
                             "contribution"};
    }
    if (r.dimension < 1 || r.dimension > 3) {
      return Failure{"E-REMAP-DIM",
                   s.label + " is outside its own placement's processor set, "
                             "so its dimension cannot be read off a view that "
                             "holds nothing; Region::dimension must be set "
                             "explicitly (1, 2 or 3) for a cross-placement "
                             "resolve"};
    }
    s.dim = r.dimension;
    return s;  // cells/measure empty, measure_total 0, not_owned 0 -- all
               // default-constructed, and correctly so: nothing here.
  }

  // REVIEW.md N-BACKEND-1: agree on the DIMENSION ITSELF before it feeds any
  // further local computation or a downstream a.dim/b.dim comparison in
  // build_remap_halo/build_weights -- reconcile<T> below only agrees whether
  // resolve() succeeded, not what value a rank that succeeded computed.
  {
    auto reconciled_dim = reconcile_dim(s.pcomm, s.dim, s.label);
    if (!reconciled_dim.ok()) return reconciled_dim.error();
    s.dim = reconciled_dim.value();
  }

  // REVIEW.md R-01: everything from here on reads data that is local to
  // THIS RANK's partition -- its cell count at this dimension, its own copy
  // of each cell's measure -- and, per moab_backend.hpp's own documented
  // limitation on MeshView::dimension(), can legitimately disagree with a
  // peer's. Computed as an ordinary local Result and reconciled once, at the
  // return below, rather than at each individual check, so a rank that fails
  // here still reaches whatever collective its peers are about to call next
  // with the SAME outcome instead of racing ahead of it (rank-agreement.md).
  Result<Side> local = [&]() -> Result<Side> {
    if (s.dim < 1 || s.dim > 3) {
      return Failure{"E-REMAP-DIM",
                   s.label + " has dimension " + std::to_string(s.dim) +
                       "; a finite-volume remap needs cells of dimension 1, "
                       "2 or 3 (dimension 0 has no measure at all)"};
    }

    moab::Range range;
    const moab::ErrorCode rval = s.mb->get_entities_by_dimension(
        s.file_set, s.dim, range, /*recursive=*/true);
    if (rval != moab::MB_SUCCESS) {
      return intx_error(*s.mb,
                        "could not list the dimension-" +
                            std::to_string(s.dim) + " cells of " + s.label,
                        rval);
    }
    // A moab::Range iterates in ascending handle order, which is exactly how
    // MoabMeshBackend fixed the local indexing at load. Cross-checked rather
    // than trusted: if the two ever diverge every weight lands in the wrong
    // row.
    if (range.size() != s.view->entity_count(s.dim)) {
      return Failure{"E-REMAP-REGION",
                   s.label + " has " + std::to_string(range.size()) +
                       " dimension-" + std::to_string(s.dim) +
                       " cells in MOAB but " +
                       std::to_string(s.view->entity_count(s.dim)) +
                       " in its MeshView; the local index spaces disagree"};
    }
    if (range.empty()) {
      // `allow_empty`: this rank DOES participate in `placement` (the
      // branch above already returned for the case it does not), but its
      // own local chunk -- owned cells plus whatever a remap halo may have
      // adopted for this pair -- happens to hold none of this mesh at this
      // dimension. Legitimate for a genuinely disjoint source/target
      // placement pair (this target rank's own region simply does not
      // overlap any source cell), so a caller that already knows it is
      // resolving one half of such a pair may ask for a valid empty
      // contribution rather than a hard refusal here. NOT keyed on
      // `has_remap_halo` (this function resolves ONE region at a time and
      // does not have the (source, target) PAIR id to ask it with) --
      // `allow_empty` is this track's own, narrower approximation of "the
      // plan's placement and the halo's presence" the design calls for;
      // every EXISTING caller still passes the default `false` and is
      // completely unaffected.
      if (allow_empty) return s;
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
      // MeshView::measure THROWS by contract; build_weights returns a Result
      // and must not, so the contract is converted here rather than
      // propagated.
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
      // measure here would make `src_measure_total` count it once per rank
      // that holds a copy. In serial every entity is owned and this is
      // unchanged.
      if (s.view->is_owned(s.dim, i)) {
        s.measure_total += s.measure[i];
      } else {
        ++s.not_owned;
      }
    }
    return s;
  }();
  return reconcile(s.pcomm, std::move(local));
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

/// `BoxTry::failed` is `try_boxes`' own Result-shaped outcome (its OTHER
/// field, `ok`, is a classification verdict -- "this mesh is not boxes" --
/// not a failure); reconstituting it as a `Result<BoxTry>` is what lets
/// `reconcile<T>` reach it below, the same mechanism `build_remap_halo`
/// already applies to `fit_surface`'s own Result.
[[nodiscard]] Result<BoxTry> to_result(BoxTry bt) {
  if (bt.failed) return bt.error;
  return bt;
}

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
      // disjoint common-refinement intervals. Their intersection is empty, and pretending otherwise
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
  //
  // REVIEW.md TF-2: `orient()` reorients THIS rank's own local cells only
  // (the exact R-01 hazard `build_remap_halo`'s own orient calls already
  // reconcile), and `intersect_surfaces` is reached from `build_weights`'
  // first-order path, ahead of `assemble()`'s collectives below in that
  // caller -- so a per-cell connectivity read/write failure here must be
  // agreed across the communicator before any rank proceeds, not returned
  // straight from this rank alone.
  {
    const auto o1 = reconcile(src.pcomm, orient(src, fs));
    if (!o1.ok()) return o1.error();
    const auto o2 = reconcile(tgt.pcomm, orient(tgt, ft));
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
///
/// KNOWN_ISSUES D82 round 12 (HOST-MEASURED, RemapHaloMpiSphere.Extensive...,
/// 2/3 ranks): this USED to early-return on `src.not_owned == 0` -- "this
/// rank holds no halo copies, so nothing of mine needs folding" -- which is
/// true for EVERY rank only in the serial/no-halo case this comment's own
/// intent names, but is a LOCALLY evaluated condition in a genuinely
/// partitioned halo, where it is entirely possible for rank A to hold zero
/// halo copies (so it skipped the call) while rank B holds some that belong
/// to cells A OWNS -- B's own accumulate_remap_halo call still runs (it is
/// the collective's one and only call on B), but A, having never called it
/// at all, never receives the contribution MOAB routes to its owner. The
/// owned slot A reads afterwards is still whatever LOCAL intersection A's
/// own rank computed -- zero, for a source cell whose only intersection
/// happened on B's adopted copy -- which is exactly `min_src_coverage=0`
/// "a source cell on this rank whose covering target cells all live on
/// another rank" the host measured. This is the same class of bug D27
/// (`accumulate_remap_halo`'s own since-fixed `ghost_n==0` skip) and round
/// 7 (`adopt_remap_halo`'s collective-agreement branch) were about: a
/// per-rank, not collectively-agreed, condition deciding whether to enter a
/// collective. Fixed by removing it: `accumulate_remap_halo` (via
/// `run_halo_exchange`) already has its OWN collectively-consistent
/// short-circuits (`halo() == nullptr`, `ex.req_counts.empty()`,
/// `!pr.participates`), each evaluated from state every rank in the same
/// communicator necessarily agrees on (the halo was built, or not, by the
/// SAME collective `build_remap_halo` call) -- calling it unconditionally
/// whenever a backend exists is the serial-path no-op this function's own
/// header already promises, now made safe for the general partitioned case.
[[nodiscard]] Result<void> fold_halo_contributions(const Side& src,
                                                   std::vector<double>& covered,
                                                   std::vector<double>& col_sum) {
  if (src.backend == nullptr) {
    return Result<void>::success();
  }
  const Result<void> a =
      src.backend->accumulate_remap_halo(src.mid, src.dim, covered.data(), 1);
  if (!a.ok()) return a;
  return src.backend->accumulate_remap_halo(src.mid, src.dim, col_sum.data(),
                                            1);
}

// ---------------------------------------------------------------------------
// KNOWN_ISSUES D82 round 13: instrumentation only, behind FUSE_REMAP_HALO_
// TRACE=1 (the SAME switch native_halo.cpp's own trace uses, off by
// default; cached once per process, exactly as remap_halo_trace_enabled()
// there does, so getenv is paid at most once). Nothing in this section
// changes a single computed value -- it reads state `assemble()` already
// has in hand and, for the ONE genuinely cross-rank piece (which rank's
// target region a given centroid falls in), performs its own small
// Allgather. The round-12 lesson applies here too: that Allgather is
// EVERY rank's own call whenever tracing is on and the communicator has
// more than one rank, never conditioned on whether THIS rank personally
// has anything to report.
// ---------------------------------------------------------------------------

[[nodiscard]] bool remap_halo_trace_enabled() {
  static const bool enabled = std::getenv("FUSE_REMAP_HALO_TRACE") != nullptr;
  return enabled;
}

/// Everything round 13's own dispatch asked for, in one place because all
/// four pieces read the SAME per-call state (`covered_before_fold`,
/// `covered_after_fold`, `intersected_src`, `sm`, `src`, `tgt`) and the
/// owner's own three hypotheses are only separable by reading them
/// together:
///
///   (A) the covering set misses boundary cells -- look at whether an
///       uncovered cell's centroid falls inside ANOTHER rank's target box
///       (printed per cell below) while that cell was never adopted there
///       (not directly observable from here -- the adopting rank's own
///       trace line, `adopted_source_cells`, is the other half: if NO
///       rank's adopted_source_cells is large enough to plausibly include
///       it, or the centroid falls in NO rank's target box at all, (A) is
///       ruled out; if it falls in a rank's box that this trace shows had
///       `adopted_cells_never_intersected` include it, (A) is implicated).
///   (B) adopted cells are not intersected -- `adopted_cells_never_
///       intersected` on the rank that plausibly adopted it (per (2)'s own
///       centroid-vs-box check) being close to `adopted_source_cells`
///       (most or all of what it adopted never intersected anything) is
///       this hypothesis; a rescale-not-applied symptom would show as
///       EVERY adopted cell failing to intersect, not a scattered few.
///   (C) the fold does not reach the owner -- `owned_cells_receiving_a_
///       fold_contribution=0` on a rank whose `zero_coverage_before_fold`
///       is nonzero, while SOME other rank's own trace shows nonzero
///       intersections against cells it adopted (i.e. mass was computed
///       somewhere, but never arrived), is this hypothesis, distinct from
///       (A)/(b): the geometry worked, the exchange did not.
void trace_remap_halo_coverage(const Side& src, const Side& tgt,
                               const Supermesh& sm,
                               const std::vector<double>& covered_before_fold,
                               const std::vector<double>& covered_after_fold,
                               const std::vector<char>& intersected_src) {
  if (!remap_halo_trace_enabled()) return;
  try {
    int my_rank = 0;
    int nproc = 1;
    bool parallel = false;
    MPI_Comm comm = MPI_COMM_NULL;
#ifdef FUSE_REMAP_PARALLEL
    if (src.pcomm != nullptr) {
      auto* pc = static_cast<moab::ParallelComm*>(src.pcomm);
      comm = pc->comm();
      if (MPI_Comm_size(comm, &nproc) == MPI_SUCCESS && nproc > 1) {
        parallel = true;
        static_cast<void>(MPI_Comm_rank(comm, &my_rank));
      }
    }
#endif
    const std::size_t n_src = src.cells.size();

    // (1): before/after zero-coverage counts over OWNED cells, and how many
    // owned cells' coverage CHANGED due to the fold (a nonzero contribution
    // arrived from a peer).
    std::size_t owned_n = 0, zero_before = 0, zero_after = 0, received = 0;
    for (std::size_t i = 0; i < n_src; ++i) {
      if (!src.view->is_owned(src.dim, i)) continue;
      ++owned_n;
      if (!(covered_before_fold[i] > 0.0)) ++zero_before;
      if (!(covered_after_fold[i] > 0.0)) ++zero_after;
      if (covered_after_fold[i] != covered_before_fold[i]) ++received;
    }

    // (4): per-owned-cell coverage-ratio histogram, AFTER the fold.
    std::size_t hist_0 = 0, hist_low = 0, hist_mid = 0, hist_full = 0;
    for (std::size_t i = 0; i < n_src; ++i) {
      if (!src.view->is_owned(src.dim, i)) continue;
      const double c = covered_after_fold[i] / sm.src_parent[i];
      if (c <= 0.0) ++hist_0;
      else if (c < 0.5) ++hist_low;
      else if (c < 0.99) ++hist_mid;
      else ++hist_full;
    }

    // (3): adopted (non-owned, i.e. halo copy) cells that were NEVER
    // intersected with any local target cell on THIS, the adopting, rank.
    std::size_t adopted_n = 0, adopted_zero_intersections = 0;
    for (std::size_t i = 0; i < n_src; ++i) {
      if (src.view->is_owned(src.dim, i)) continue;
      ++adopted_n;
      if (!intersected_src[i]) ++adopted_zero_intersections;
    }

    fmt::print(
        stderr,
        "FUSE_REMAP_HALO_TRACE rank={} source_mesh={} target_mesh={} dim={} "
        "owned_source_cells={} zero_coverage_before_fold={} "
        "zero_coverage_after_fold={} owned_cells_receiving_a_fold_"
        "contribution={} adopted_source_cells={} "
        "adopted_cells_never_intersected={} coverage_histogram[0]={} "
        "coverage_histogram(0,0.5)={} coverage_histogram[0.5,0.99)={} "
        "coverage_histogram[0.99,+)={}\n",
        my_rank, src.mid.value(), tgt.mid.value(), src.dim, owned_n,
        zero_before, zero_after, received, adopted_n,
        adopted_zero_intersections, hist_0, hist_low, hist_mid, hist_full);

    // (2): up to 5 uncovered owned cells -- global id, centroid, and which
    // rank's own target bounding box contains it. COLLECTIVE (the Allgather
    // below): every rank participates whenever tracing is on and the
    // communicator has more than one rank, not just ranks with something
    // to report.
    if (!parallel) return;

    // This rank's own LOCAL target-cell AABB (owned + halo copies it holds)
    // -- a proxy for Intx2MeshOnSphere's own internal "processor euler box"
    // (not claimed identical; MOAB does not expose that box to read back).
    double lo[3] = {1e300, 1e300, 1e300};
    double hi[3] = {-1e300, -1e300, -1e300};
    std::vector<double> scratch;
    for (std::size_t j = 0; j < tgt.cells.size(); ++j) {
      auto xyz = corner_coords(*tgt.mb, tgt.cells[j], scratch);
      if (!xyz.ok()) continue;
      const std::size_t nv = xyz.value().size() / 3;
      for (std::size_t v = 0; v < nv; ++v) {
        for (int k = 0; k < 3; ++k) {
          const double c =
              xyz.value()[3 * v + static_cast<std::size_t>(k)];
          lo[k] = std::min(lo[k], c);
          hi[k] = std::max(hi[k], c);
        }
      }
    }
    double my_box[6] = {lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]};
    std::vector<double> all_boxes(static_cast<std::size_t>(nproc) * 6, 0.0);
    if (MPI_Allgather(my_box, 6, MPI_DOUBLE, all_boxes.data(), 6, MPI_DOUBLE,
                      comm) != MPI_SUCCESS) {
      fmt::print(stderr,
                "FUSE_REMAP_HALO_TRACE rank={} Allgather of target boxes "
                "failed\n",
                my_rank);
      return;
    }

    std::size_t printed = 0;
    for (std::size_t i = 0; i < n_src && printed < 5; ++i) {
      if (!src.view->is_owned(src.dim, i)) continue;
      const double c = covered_after_fold[i] / sm.src_parent[i];
      if (c > 0.0) continue;  // only the UNCOVERED ones
      ++printed;
      const Point p = src.view->centroid(src.dim, i);
      const double r = std::sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
      const double lon = std::atan2(p.y, p.x) * 180.0 / M_PI;
      const double lat =
          r > 0.0 ? std::asin(std::clamp(p.z / r, -1.0, 1.0)) * 180.0 / M_PI
                  : 0.0;
      std::size_t gid = 0;
      try {
        gid = src.view->global_id(src.dim, i);
      } catch (const std::exception&) {
        gid = 0;
      }
      std::string containing;
      for (int rr = 0; rr < nproc; ++rr) {
        const double* b = &all_boxes[static_cast<std::size_t>(rr) * 6];
        if (p.x >= b[0] && p.x <= b[3] && p.y >= b[1] && p.y <= b[4] &&
            p.z >= b[2] && p.z <= b[5]) {
          if (!containing.empty()) containing += ",";
          containing += std::to_string(rr);
        }
      }
      if (containing.empty()) containing = "NONE";
      fmt::print(
          stderr,
          "FUSE_REMAP_HALO_TRACE rank={} uncovered_owned_source_cell "
          "local_index={} global_id={} centroid_xyz=({:.6g},{:.6g},{:.6g}) "
          "lon={:.4g} lat={:.4g} contained_by_target_box_of_rank={}\n",
          my_rank, i, gid, p.x, p.y, p.z, lon, lat, containing);
    }
  } catch (const std::exception& e) {
    fmt::print(stderr,
              "FUSE_REMAP_HALO_TRACE: trace itself threw: {} (diagnostic "
              "only -- the real result is unaffected)\n",
              e.what());
  }
}

/// KNOWN_ISSUES D82 round 14: a point is INSIDE a CONVEX spherical polygon
/// (CCW as seen from outside the sphere -- the orientation `orient()`
/// already established on both sides before any covering-set construction
/// ran, not re-derived here) when it sits on the inner side of every one
/// of the polygon's own great-circle edges: for edge (v_i, v_i+1), the
/// great circle through them has normal `cross(v_i, v_i+1)`, and a
/// correctly-oriented convex polygon has its own interior on the side
/// where `dot(cross(v_i, v_i+1), p) >= 0` for EVERY edge simultaneously.
/// `kInsideTol` is generous (not the usual `kGeomTol`/`kIntxTol` scale)
/// because this is a diagnostic-only classification, not a quantity any
/// production result depends on -- a point within this margin of an edge
/// reads as "inside" rather than risking a false "outside" from rounding.
[[nodiscard]] bool point_in_convex_spherical_polygon(
    const std::vector<double>& xyz, const V3& p) {
  const std::size_t n = xyz.size() / 3;
  if (n < 3) return false;
  constexpr double kInsideTol = -1e-6;
  for (std::size_t i = 0; i < n; ++i) {
    const V3 vi{xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
    const std::size_t jn = (i + 1) % n;
    const V3 vj{xyz[3 * jn], xyz[3 * jn + 1], xyz[3 * jn + 2]};
    if (dot(cross(vi, vj), p) < kInsideTol) return false;
  }
  return true;
}

/// KNOWN_ISSUES D82 round 14 (owner's own per-cell probe request,
/// 2026-10-19): behind the SAME FUSE_REMAP_HALO_TRACE=1 switch, separates
/// hypotheses (A)/(B)/(C) for each of round 13's own uncovered/partially-
/// covered owned cells. Every rank answers, for every probed cell (gid +
/// owner rank + centroid, exchanged via one Allgatherv every rank in the
/// communicator joins UNCONDITIONALLY, round 12's own lesson applied
/// rather than relearned a third time):
///
///   present_locally          a local cell (owned OR an adopted halo copy)
///                             carries this gid at all.
///   intersected_area_here    the sum of THIS rank's own intersection
///                             entries (`sm.entries`) for that local
///                             index, if present -- 0 if absent or if
///                             present but never intersected.
///   independent_overlap_here a point-in-convex-spherical-polygon test of
///                             the probed cell's own centroid against
///                             EVERY local (owned+halo) target cell,
///                             computed OUTSIDE the covering-set machinery
///                             entirely -- no Intx2MeshOnSphere call, no
///                             shared state with the production path
///                             except the already-established orientation
///                             and the (read-only) target cell geometry.
///
/// Verdict, per the owner's own framing: `independent_overlap_here` true
/// and `present_locally` false on a rank is hypothesis (A) (the covering
/// set should have sent this cell here and did not); `present_locally`
/// true (and not the owner) with `intersected_area_here == 0` while
/// `independent_overlap_here` is true is (B) (the adopted copy sits here
/// but never intersected anything, despite genuinely overlapping);
/// `independent_overlap_here` false on EVERY rank that answers is the
/// "genuinely uncovered by the geometry itself" case the owner named as
/// impossible given the 1-rank pass.
void trace_remap_halo_probe(const Side& src, const Side& tgt,
                            const Supermesh& sm,
                            const std::vector<double>& covered_after_fold) {
  if (!remap_halo_trace_enabled()) return;
  try {
    int my_rank = 0;
    int nproc = 1;
    bool parallel = false;
    MPI_Comm comm = MPI_COMM_NULL;
#ifdef FUSE_REMAP_PARALLEL
    if (src.pcomm != nullptr) {
      auto* pc = static_cast<moab::ParallelComm*>(src.pcomm);
      comm = pc->comm();
      if (MPI_Comm_size(comm, &nproc) == MPI_SUCCESS && nproc > 1) {
        parallel = true;
        static_cast<void>(MPI_Comm_rank(comm, &my_rank));
      }
    }
#endif
    if (!parallel) return;  // nothing cross-rank to probe at one rank

    const std::size_t n_src = src.cells.size();

    // gid -> local index, every local cell (owned or halo copy), built
    // once per call -- O(n_src), acceptable for a capped diagnostic probe.
    std::unordered_map<std::size_t, std::size_t> gid_to_local;
    gid_to_local.reserve(n_src);
    for (std::size_t i = 0; i < n_src; ++i) {
      try {
        gid_to_local.emplace(src.view->global_id(src.dim, i), i);
      } catch (const std::exception&) {
        // no usable global numbering at this index -- it cannot be probed
        // by gid, same tolerance the rest of this file already has for a
        // local index with no global id.
      }
    }

    // This rank's own candidates: owned cells uncovered or partially
    // covered after the fold, capped at 20 (owner's own instruction).
    struct Candidate {
      std::size_t gid;
      double x, y, z;
    };
    std::vector<Candidate> mine;
    for (std::size_t i = 0; i < n_src && mine.size() < 20; ++i) {
      if (!src.view->is_owned(src.dim, i)) continue;
      const double c = covered_after_fold[i] / sm.src_parent[i];
      if (c >= 0.99) continue;
      std::size_t gid = 0;
      try {
        gid = src.view->global_id(src.dim, i);
      } catch (const std::exception&) {
        continue;  // cannot be named/probed without one
      }
      const Point p = src.view->centroid(src.dim, i);
      mine.push_back({gid, p.x, p.y, p.z});
    }

    // Packed as 5 doubles per candidate -- gid and owner rank as doubles
    // are exact for any value this project's own meshes use (well under
    // 2^53), and MPI_Allgatherv needs one element type.
    std::vector<double> my_packed(mine.size() * 5);
    for (std::size_t k = 0; k < mine.size(); ++k) {
      my_packed[5 * k + 0] = static_cast<double>(mine[k].gid);
      my_packed[5 * k + 1] = static_cast<double>(my_rank);
      my_packed[5 * k + 2] = mine[k].x;
      my_packed[5 * k + 3] = mine[k].y;
      my_packed[5 * k + 4] = mine[k].z;
    }

    std::vector<int> counts(static_cast<std::size_t>(nproc), 0);
    const int my_count = static_cast<int>(my_packed.size());
    if (MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT,
                      comm) != MPI_SUCCESS) {
      fmt::print(stderr,
                "FUSE_REMAP_HALO_TRACE rank={} probe: Allgather of counts "
                "failed\n",
                my_rank);
      return;
    }
    std::vector<int> displs(static_cast<std::size_t>(nproc) + 1, 0);
    for (int r = 0; r < nproc; ++r) {
      displs[static_cast<std::size_t>(r) + 1] =
          displs[static_cast<std::size_t>(r)] + counts[static_cast<std::size_t>(r)];
    }
    std::vector<double> all_packed(
        static_cast<std::size_t>(displs[static_cast<std::size_t>(nproc)]), 0.0);
    if (MPI_Allgatherv(my_packed.data(), my_count, MPI_DOUBLE,
                       all_packed.data(), counts.data(), displs.data(),
                       MPI_DOUBLE, comm) != MPI_SUCCESS) {
      fmt::print(stderr,
                "FUSE_REMAP_HALO_TRACE rank={} probe: Allgatherv of "
                "candidates failed\n",
                my_rank);
      return;
    }
    const std::size_t n_probe = all_packed.size() / 5;

    for (std::size_t k = 0; k < n_probe; ++k) {
      const std::size_t gid =
          static_cast<std::size_t>(all_packed[5 * k + 0]);
      const int owner_rank = static_cast<int>(all_packed[5 * k + 1]);
      const V3 p{all_packed[5 * k + 2], all_packed[5 * k + 3],
                all_packed[5 * k + 4]};

      bool present_locally = false;
      double intersected_area_here = 0.0;
      const auto it = gid_to_local.find(gid);
      if (it != gid_to_local.end()) {
        present_locally = true;
        const std::size_t li = it->second;
        for (const Overlap& ov : sm.entries) {
          if (static_cast<std::size_t>(ov.col) == li) {
            intersected_area_here += ov.area;
          }
        }
      }

      bool independent_overlap_here = false;
      std::vector<double> scratch;
      for (std::size_t j = 0;
          j < tgt.cells.size() && !independent_overlap_here; ++j) {
        auto xyz = corner_coords(*tgt.mb, tgt.cells[j], scratch);
        if (!xyz.ok()) continue;
        if (point_in_convex_spherical_polygon(xyz.value(), p)) {
          independent_overlap_here = true;
        }
      }

      fmt::print(stderr,
                "FUSE_REMAP_HALO_TRACE rank={} probe gid={} owner_rank={} "
                "present_locally={} intersected_area_here={:.17g} "
                "independent_overlap_here={}\n",
                my_rank, gid, owner_rank, present_locally,
                intersected_area_here, independent_overlap_here);
    }
  } catch (const std::exception& e) {
    fmt::print(stderr,
              "FUSE_REMAP_HALO_TRACE: probe itself threw: {} (diagnostic "
              "only)\n",
              e.what());
  }
}

/// Reduce the audit over the communicator, so every rank reports the same
/// numbers and, above all, reaches the same accept/reject outcome.
[[nodiscard]] Result<void> reduce_audit([[maybe_unused]] const Side& src,
                                        [[maybe_unused]] Audit& audit) {
#ifdef FUSE_REMAP_PARALLEL
  if (src.pcomm == nullptr) return Result<void>::success();
  auto* pc = static_cast<moab::ParallelComm*>(src.pcomm);
  const MPI_Comm comm = pc->comm();
  int nproc = 1;
  // REVIEW.md TF-8 (N-BACKEND-3 class): an unchecked MPI_Comm_size leaves
  // `nproc` at its initialiser value of 1 on failure, which this function
  // would then read as "no peers to reconcile with" and return success
  // having skipped every Allreduce below -- exactly the silent fallback
  // reconcile<T> and reconcile_halo already refuse rather than risk (both
  // reject outright when they cannot size their own communicator).
  if (MPI_Comm_size(comm, &nproc) != MPI_SUCCESS) {
    return Failure{"E-REMAP-PARALLEL",
                 "reduce_audit: could not size the communicator to reduce "
                 "the per-rank audit; refusing rather than treating it as "
                 "having no peers to reconcile with"};
  }
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

/// What to tell someone whose conservative map was rejected for coverage.
///
/// NOT "clear the Conservative flag" when the read is partitioned. On a
/// partitioned mesh that advice converts a loud, correct rejection into the
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
///     Conservation::Intensive (and None), UNCHANGED:
///         W_ji = (overlap_ij / parent_i) * (m^s_i / m^t_j)
///     Conservation::Extensive, KNOWN_ISSUES D82 (round 11):
///         W_ji = overlap_ij / m^s_i
///
/// `normalize_remap_weight` (fuse/backend/remap_weight_normalize.hpp) is the
/// one place either formula is written; see its own doc for why Extensive
/// drops the density-to-density factor entirely rather than keeping it with
/// a different denominator, and for the two real per-cell measurements
/// (wholeATM_T.h5m, recMeshOcn.h5m) behind the test tolerance that formula's
/// own residual needs. Nothing is renormalised to force either sum to
/// exactly 1: see the header for why the tempting `overlap / sum_k
/// overlap_ik` was rejected.
[[nodiscard]] Result<RemapWeights> assemble(const Side& src, const Side& tgt,
                                            Supermesh sm, bool conservative,
                                            Conservation conservation) {
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
  // KNOWN_ISSUES D82 round 13 instrumentation (hypothesis (B), "adopted
  // cells are not intersected"): every source local index that appears as
  // the `col` of at least one entry in `sm.entries` -- i.e. Intx2MeshOnSphere
  // found SOME overlap for it, regardless of which rank's target cell. Cost
  // is one bool per source cell regardless of whether tracing is on; kept
  // unconditional rather than gated, so it cannot itself be the thing that
  // silently diverges between a traced and an untraced run.
  std::vector<char> intersected_src(w.n_cols, 0);
  const bool extensive = conservation == Conservation::Extensive;

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
    const double weight = normalize_remap_weight(
        area, sm.src_parent[i], src.measure[i], tgt.measure[j], conservation);
    covered_src[i] += area;
    covered_tgt[j] += area;
    intersected_src[i] = 1;
    row_sum[j] += weight;
    // Extensive: `weight` is already the fraction of source cell i's AMOUNT
    // this entry carries, so the column total that must read 1 (round 11's
    // own "source total == target total" test) is the raw weight sum, not
    // a measure-weighted one -- multiplying by tgt.measure[j] here would
    // re-introduce exactly the density conversion Extensive exists to drop.
    // Intensive/None: unchanged (the amount leaving source cell i, checked
    // against its own measure below).
    col_sum[i] += extensive ? weight : weight * tgt.measure[j];
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
  //     ranks reach the same outcome.
  // ==========================================================================
  // KNOWN_ISSUES D82 round 13 instrumentation: snapshot BEFORE the fold --
  // hypothesis (C), "the fold does not reach the owner", is read off the
  // difference between this and covered_src AFTER the call below.
  const std::vector<double> covered_src_before_fold = covered_src;

  {
    const Result<void> f =
        fold_halo_contributions(src, covered_src, col_sum);
    if (!f.ok()) return f.error();
  }

  trace_remap_halo_coverage(src, tgt, sm, covered_src_before_fold, covered_src,
                            intersected_src);
  trace_remap_halo_probe(src, tgt, sm, covered_src);

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
    // Extensive: col_sum[i] is already the raw weight-sum (should read 1,
    // the fraction of source cell i's amount that was fully distributed);
    // Intensive/None: col_sum[i] is tgt.measure-weighted (should read
    // src.measure[i], unchanged from before round 11).
    audit.max_col_defect =
        std::max(audit.max_col_defect,
                 extensive ? std::fabs(col_sum[i] - 1.0)
                           : std::fabs(col_sum[i] - src.measure[i]) /
                                 src.measure[i]);
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
// see src/backend/high_order_backend.hpp for why the seam is drawn there.
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
/// 4e-13. Rejecting a legitimate lat-lon mesh over 1.7e-8 would be a false
/// rejection; the thing this check exists to catch -- an overlap polygon
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

namespace {

/// KNOWN_ISSUES D81 round 8: the number of ranks a covering-set
/// construction over `pcomm` will ACTUALLY run with -- `pcomm`'s own
/// communicator, never `MoabMeshBackend::communicator_size()`'s answer
/// (always the BACKEND's base communicator, `impl_->pcomm`'s own size,
/// moab_backend.cpp). The two coincide for the single-Core/no-placements
/// case, where `a.pcomm` IS `impl_->pcomm` by construction -- but a
/// DECLARED placement's own sub-communicator (`set_placements()`,
/// moab_backend.hpp) can be smaller than the base one it was split from: a
/// single-rank placement inside a multi-rank job, the exact shape
/// `moab_placed_weights_mpi_test.cpp`'s own `DisjointRankSets` fixture
/// gives placement A at a 2-rank run (A = {0} alone). Shared between the
/// single-placement `build_remap_halo` below and anywhere else that needs
/// the SAME answer, the same way `measure_sphere_radius`/`project_sphere`
/// (moab_backend.cpp) are shared between `sphere_radius`'s default and
/// placement-aware overloads -- so the default-Core and a placement's own
/// path can never read this differently.
int active_comm_size(void* pcomm) {
  if (pcomm == nullptr) return 1;
#ifdef FUSE_REMAP_PARALLEL
  const int n = static_cast<int>(
      static_cast<moab::ParallelComm*>(pcomm)->proc_config().proc_size());
  return n > 0 ? n : 1;
#else
  // No MPI in this build: `pcomm` cannot be anything but null here (every
  // ParallelComm this file ever constructs needs FUSE_REMAP_PARALLEL), so
  // this branch exists only so the function type-checks in that
  // configuration too.
  return 1;
#endif
}

}  // namespace

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
// 1e-16 at 1, 2 and 4 ranks, from the task report that measured this.
// REVIEW.md TF-7: this used to say no test in the tree pins any of this --
// half-stale since `test/parallel/remap_halo_mpi_test.cpp` ported
// (host-verified per that file's own header): the halo's own forward/
// reverse round trip is checked there by EXACT global id and bookkeeping
// identity (not a tolerance), and E-REMAP-PARALLEL's rejection boundary
// fires by name at every registered rank count. Only the specific 1.0-to-
// 1e-16 coverage FIGURE quoted above remains diagnostic-only: that file
// prints, but does not gate on, the corresponding number on its own real
// spherical pair, since this exact cubed-sphere/partner pair's true
// overlap has not been independently measured to gate against.
// ===========================================================================

Result<std::size_t> build_remap_halo(const RemapPlan& plan,
                                     const CouplingGraph& graph,
                                     MeshBackend& backend) {
  using namespace remap_detail;  // NOLINT: file-local detail namespace

  // `fuse::native::NativeMeshBackend` dispatches here too (docs/dev/
  // native-halo.md), BEFORE the MOAB-only rejection below: in a build with
  // both MOAB and the native backend compiled in, a native mesh pair must
  // not lose the planar/volumetric halo capability just because this
  // translation unit also knows how to build a spherical one for MOAB.
  if (auto* native_backend = dynamic_cast<native::NativeMeshBackend*>(&backend)) {
    auto src = supermesh_side(graph, backend, plan.src, plan.placement, "source");
    if (!src.ok()) return src.error();
    auto tgt = supermesh_side(graph, backend, plan.tgt, plan.placement, "target");
    if (!tgt.ok()) return tgt.error();
    if (src.value().dim != tgt.value().dim) {
      return Failure{"E-REMAP-DIM",
                   "the source cells are dimension " + std::to_string(src.value().dim) +
                       " and the target cells dimension " +
                       std::to_string(tgt.value().dim) +
                       "; a halo pairs one with the other and needs the same dimension "
                       "on both"};
    }
    return native::build_native_remap_halo(*native_backend, src.value().mid, tgt.value().mid,
                                          src.value().dim, plan.placement);
  }

  auto* moab_backend = dynamic_cast<MoabMeshBackend*>(&backend);
  if (moab_backend == nullptr) {
    return Failure{"E-REMAP-BACKEND",
                 "build_remap_halo needs the MOAB backend; this is the '" +
                     backend.name() + "' backend"};
  }

  auto src = resolve(graph, backend, plan.src, plan.placement, "source");
  if (!src.ok()) return src.error();
  auto tgt = resolve(graph, backend, plan.tgt, plan.placement, "target");
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

  // Serial, a one-rank communicator, or the identity (both mesh entity sets on one
  // mesh) all need no halo. Recorded as BUILT all the same, so a caller may
  // make the call unconditionally and `build_weights` can key its rejection on
  // "was this asked for" rather than on "did it find anything".
  //
  // KNOWN_ISSUES D81 round 8 (MEASURED 2026-10-19, host): this guard used
  // to read `moab_backend->communicator_size()` -- the BACKEND's base
  // communicator, always correct for the single-Core/no-placements case
  // (where `a.pcomm` IS `impl_->pcomm`), but WRONG for a placement whose
  // own sub-communicator is smaller than the base one:
  // `SamePlacementOnBothSidesDelegatesToTheSinglePlacementPath`'s own
  // placement A is a genuine 1-rank communicator at a 2-rank run, so
  // `communicator_size()` read 2 (the BASE communicator) and this guard
  // did not shortcut -- sending a genuinely 1-rank `ParallelComm` into
  // `Intx2MeshOnSphere`'s real, cross-rank covering-set construction, a
  // configuration no PASSING test in this tree had exercised before
  // (`remap_halo_mpi_test.cpp` always runs the single-Core path over the
  // FULL base communicator; a 1-rank run THERE takes this same shortcut
  // before ever reaching the construction, never a 1-rank PLACEMENT inside
  // a larger job). The resulting covering cell carried no positive
  // GLOBAL_ID -- `adopt_remap_halo`'s own `E-MESH-HALO` refusal, doing
  // exactly its job once asked to adopt a cell it cannot trust; kept
  // unchanged. Fixed by reading `active_comm_size(a.pcomm)` -- the size of
  // the communicator THIS CONSTRUCTION will actually run over -- instead
  // of the backend's unrelated base communicator size; the two can only
  // ever disagree on a placement path, which is exactly the case this
  // guard's own "one-rank communicator... need no halo" comment already
  // meant to cover.
  const int ranks = active_comm_size(a.pcomm);
  if (ranks <= 1 || a.pcomm == nullptr || a.file_set == b.file_set) {
    // KNOWN_ISSUES D83: threaded through `plan.placement` so this
    // degenerate no-op is filed under the SAME (mesh, mesh, dim, placement)
    // cache key the real covering-set build below uses for the identical
    // pair -- the default Core and every declared placement each get
    // their own entry, never aliasing one another.
    return moab_backend->adopt_remap_halo(a.mid, b.mid, a.dim, nullptr, 0,
                                          plan.placement);
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
  // REVIEW.md R-01: fit_surface() reads only this rank's local vertices, so
  // a rank whose local partition happens to be non-spherical (or empty)
  // cannot agree with a peer's answer without reconciling first -- otherwise
  // that rank rejects here while its peers walk into FindMaxEdges below and
  // hang waiting for it.
  auto fa = reconcile(a.pcomm, fit_surface(a));
  if (!fa.ok()) return fa.error();
  auto fb = reconcile(b.pcomm, fit_surface(b));
  if (!fb.ok()) return fb.error();
  // REVIEW.md R-01: the classification below is a value computed from each
  // rank's own local vertices (the SAME hazard fit_surface() itself has, one
  // level up in the SurfaceFit::ok/kind/radius payload rather than in
  // Result's error channel), so it is reconciled too before its radius is
  // used by any collective below.
  const Result<double> radius_result =
      reconcile(a.pcomm, [&]() -> Result<double> {
        if (!fa.value().ok || !fb.value().ok ||
            fa.value().kind != Surface::Sphere ||
            fb.value().kind != Surface::Sphere) {
          return Failure{
              "E-MESH-HALO-GEOMETRY",
              "a partitioned remap needs a covering set, and the only "
              "construction available in this build supports SPHERICAL "
              "surfaces only -- every vertex of both meshes at a common "
              "radius about the origin. This pair is not one. A planar "
              "covering construction is UNIMPLEMENTED here, not impossible: "
              "it would slot in behind build_remap_halo() and nothing above "
              "it would change. Until then, read the meshes without a "
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
        return 0.5 * (ra + rb);
      }());
  if (!radius_result.ok()) return radius_result.error();
  const double radius = radius_result.value();

  // ORIENTATION FIRST, AND THIS IS NOT OPTIONAL. Measured: with MOAB's own
  // cubed sphere left exactly as read -- its cells are negatively oriented,
  // see orient() -- the covering construction and the intersection that
  // follows reach only 0.80 of the target area. Oriented first, 1.0 to 1e-16.
  {
    // REVIEW.md R-01: orient() reorients this rank's own local cells only,
    // and a per-cell connectivity read/write failure is exactly as local as
    // the checks in resolve() -- reconciled the same way, before
    // FindMaxEdges below.
    const auto o1 = reconcile(a.pcomm, orient(a, fa.value()));
    if (!o1.ok()) return o1.error();
    const auto o2 = reconcile(b.pcomm, orient(b, fb.value()));
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

  // KNOWN_ISSUES D83 (owner decision 2026-10-19: make the adoption
  // placement-aware): `handles` are `moab::EntityHandle`s in `mb` (= `a.mb`,
  // resolved above through `plan.placement`) -- `plan.placement` is passed
  // through so `adopt_remap_halo` reads them back through the SAME Core
  // that allocated them, never the backend's default one unconditionally.
  return moab_backend->adopt_remap_halo(a.mid, b.mid, a.dim, handles.data(),
                                        handles.size(), plan.placement);
#endif
}

// ---------------------------------------------------------------------------
// KNOWN_ISSUES D5's last half, round 2: HALO BRIDGING across two
// independently decomposed placements (the lead's ruling; see this
// function's own declaration in remap_plan.hpp for the full design). The
// target's processor set is `plan.placement`; the source's is
// `source_placement`, a DIFFERENT processor set in general.
// ---------------------------------------------------------------------------
Result<std::size_t> build_remap_halo(const RemapPlan& plan, const CouplingGraph& graph,
                                     MeshBackend& backend, ComponentId source_placement) {
  using namespace remap_detail;  // NOLINT: file-local detail namespace

  auto* moab_backend = dynamic_cast<MoabMeshBackend*>(&backend);
  if (moab_backend == nullptr) {
    return Failure{"E-REMAP-BACKEND",
                 "build_remap_halo (cross-placement overload) needs the MOAB "
                 "backend; this is the '" + backend.name() + "' backend"};
  }

  // KNOWN_ISSUES D81 round 7 (owner, 2026-10-19): "a collective path never
  // returns on a locally computed condition: agree first, then every rank
  // refuses with the same code." The ORIGINAL version of this function
  // decided "delegate to the single-placement path" vs "refuse, the two
  // placements genuinely differ" by comparing `a.mb == b.mb` -- the TWO
  // SIDES' OWN moab::Core pointers, resolved per-rank. That comparison is
  // NOT collectively consistent: for a rank outside `plan.placement`,
  // `b.mb` reads nullptr regardless of what `source_placement` resolves
  // to, so the SAME call can read as "same Core" on one rank and
  // "different Core" on another, depending only on which of the two
  // placements that particular rank happens to belong to -- READ AGAINST
  // this file's own test (1), ARankOutsideTheTargetPlacementIsASuccess
  // fulNoOp: there, a rank outside B (the target) that IS inside A (the
  // source) would misread as "same Core" under the pointer test even
  // though source_placement (A) and plan.placement (B) genuinely differ.
  // The question this branch actually needs answered is GLOBAL to the
  // CALL, not local to a rank: do `source_placement` and `plan.placement`
  // name the SAME declared processor set? That is a plain equality of two
  // `ComponentId` values (the same way `fuse::native::build_native_remap_
  // halo` decides its own analogous question once, over the backend's
  // single shared communicator, before any rank branches on its own
  // local state -- read 2026-10-19, native_halo.cpp, to confirm the
  // pattern before reusing it here) -- identical, deterministically, on
  // every rank, with no MOAB/MPI call at all.
  const bool same_named_placement = (source_placement == plan.placement);

  // The target: resolved through plan.placement, exactly as the
  // single-placement path does. `allow_empty`: a rank outside
  // plan.placement is expected not to be called here at all (the Runner's
  // own structural, collectively agreed condition), but this function does
  // not assume its caller enforced that -- it tolerates the case rather
  // than crashing or hanging on it.
  auto tgt = resolve(graph, backend, plan.tgt, plan.placement, "target", /*allow_empty=*/true);
  if (!tgt.ok()) return tgt.error();
  Side& b = tgt.value();

  // The source: resolved through source_placement, NOT plan.placement --
  // this is the whole point of the overload. `allow_empty` for the
  // identical reason: this target rank may legitimately own none of the
  // source placement's own processor set's local chunk overlapping it
  // (checked geometrically below, not assumed). Resolved HERE,
  // unconditionally, rather than only on a rank inside plan.placement:
  // `a.mid` (a name lookup, never placement-dependent) is needed below
  // even by a rank outside plan.placement, when `same_named_placement`
  // sends it into `adopt_remap_halo` too (see the next comment).
  auto src = resolve(graph, backend, plan.src, source_placement, "source", /*allow_empty=*/true);
  if (!src.ok()) return src.error();
  Side& a = src.value();

  if (same_named_placement) {
    // SAME declared placement on both sides (the degenerate case -- no
    // placements declared at all, or both name the identical processor
    // set).
    //
    // KNOWN_ISSUES D83 round 10 (owner decision 2026-10-19: make the
    // adoption placement-aware, not a refusal) SUPERSEDES round 7's own
    // fix here. Round 7 made an off-placement rank call
    // `adopt_remap_halo(..., nullptr, 0)` to join a collective inside it
    // that, at the time, ALWAYS ran over `impl_->pcomm->comm()` -- the
    // backend's BASE communicator -- regardless of placement, because
    // `adopt_remap_halo` had no way to resolve a placement's own Core at
    // all. D83's own fix changes WHICH communicator that collective runs
    // over: `adopt_remap_halo(..., placement)` now resolves
    // `reconcile_halo`'s own communicator THROUGH `placement` (its own
    // `PlacementContext::pcomm`, `moab_backend.cpp`), so the agreement is
    // scoped to `plan.placement`'s OWN members -- never the base
    // communicator, and never a rank outside the placement. A rank
    // outside `plan.placement` therefore has NOTHING to agree on with
    // placement A's own members any more, and returns a bare, LOCALLY
    // decided `std::size_t{0}` again -- round 6's original shape, now
    // correct again because the collective it used to need to join no
    // longer includes it. (If the agreement had instead stayed on the
    // base communicator, this rank would still need to join it here --
    // stated explicitly per the round-10 brief's own instruction to say
    // which and why.)
    if (!moab_backend->placement_participates(plan.placement)) {
      return std::size_t{0};
    }
    if (a.dim != b.dim) {
      return Failure{"E-REMAP-DIM",
                   "the source cells are dimension " + std::to_string(a.dim) +
                       " and the target cells dimension " + std::to_string(b.dim) +
                       "; a halo pairs one with the other and needs the same dimension on "
                       "both sides"};
    }
    // The EXISTING single-placement machinery already handles this
    // correctly, and calling it is simpler and more tested than
    // duplicating its geometry here.
    return build_remap_halo(plan, graph, backend);
  }

  // DIFFERENT declared placements: a rank outside the target's own
  // placement has nothing to build -- a genuine, LOCAL-ONLY no-op (no
  // MPI collective either here or in the refusal below), so ranks may
  // disagree about their own participation here without agreeing first --
  // unlike the `same_named_placement` branch above, nothing downstream of
  // either arm of this `if` ever calls `adopt_remap_halo` or any other
  // collective.
  if (!moab_backend->placement_participates(plan.placement)) {
    return std::size_t{0};
  }
  if (a.dim != b.dim) {
    return Failure{"E-REMAP-DIM",
                 "the source cells are dimension " + std::to_string(a.dim) +
                     " and the target cells dimension " + std::to_string(b.dim) +
                     "; a halo pairs one with the other and needs the same dimension on "
                     "both sides"};
  }

  // The genuinely cross-placement case this track's round
  // 2 measured and could not complete in the time available (KNOWN_ISSUES
  // D5). MOAB's own covering-set construction (Intx2MeshOnSphere,
  // FindMaxEdges, construct_covering_set, all used by the single-placement
  // path above) takes ONE moab::Interface and intersects two file sets
  // WITHIN it; it has no notion of a file set living in a DIFFERENT Core,
  // and there is no MOAB entry point that accepts one. Bridging this pair
  // needs a MIGRATE-BY-GLOBAL-ID exchange -- box overlap over the base
  // communicator (the SAME technique `fuse::native::build_native_remap_
  // halo`/native_halo.cpp already implements and this track's own D27 work
  // reused for its reverse exchange), packing each needed source cell's
  // vertices/connectivity/GLOBAL_ID on its owning rank (in
  // source_placement) and creating fresh MOAB entities for it inside the
  // TARGET placement's own Core (`create_vertices`/the element-creation
  // calls `adopt_remap_halo`'s own file-set `add_entities` step assumes
  // already exist, which they do not here) -- not yet built. Refused
  // explicitly, naming the exact gap, rather than silently returning an
  // incomplete or wrong halo.
  return Failure{"E-MESH-HALO-CROSS-PLACEMENT-UNIMPLEMENTED",
               "build_remap_halo: source placement and target placement resolve to "
               "DIFFERENT moab::Core instances (MoabMeshBackend::set_placements's own "
               "one-Core-per-processor-set contract). MOAB's covering-set construction "
               "needs both meshes in the SAME moab::Interface; bridging two independent "
               "Cores needs a migrate-by-global-id exchange (the MOAB analogue of "
               "fuse::native::build_native_remap_halo's box-overlap/pack/merge "
               "architecture), which is not yet implemented. See KNOWN_ISSUES.md D5."};
}

Result<RemapWeights> build_weights(const RemapPlan& plan,
                                   const CouplingGraph& graph,
                                   const MeshBackend& backend) {
  using namespace remap_detail;  // NOLINT: file-local detail namespace

  // See the identical early dispatch in the
  // !FUSE_HAVE_MOAB build_weights() above for why this check comes first.
  if (plan.method == "consistent" || plan.method == "consistent-transpose") {
    return build_consistent_weights_for_plan(plan, graph, backend);
  }
  // Moving least squares (fuse/backend/mls_weights.hpp): dependency-free, so
  // dispatched identically with or without MOAB. Its options are part of the
  // method string and are never defaulted (E-MLS-OPTIONS).
  if (plan.method == "mls" || plan.method.rfind("mls:", 0) == 0) {
    return build_mls_weights(plan, graph, backend);
  }

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

  // REVIEW.md N-BACKEND-4: dispatch on the ACTUAL backend the caller handed
  // us, not on which builders this translation unit was compiled with.
  // Every check below this point assumes the MOAB backend (`resolve()`'s own
  // `dynamic_cast` rejects otherwise with E-REMAP-BACKEND) -- so a plan whose
  // meshes are NOT MoabMeshBackend must take the always-available r3d
  // supermesh path here, exactly as it would in a build with no MOAB at
  // all. Without this, compiling MOAB in silently took away the capability
  // a base build configuration has for the identical plan and the identical serial mesh
  // pair.
  if (dynamic_cast<const MoabMeshBackend*>(&backend) == nullptr) {
    return build_weights_via_supermesh(plan, graph, backend);
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
    // The same rejection the backend makes, made earlier so it also applies to
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

  auto src = resolve(graph, backend, plan.src, plan.placement, "source");
  if (!src.ok()) return src.error();
  auto tgt = resolve(graph, backend, plan.tgt, plan.placement, "target");
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
  // and the serial construction is exactly right. Rejecting it would take
  // remap away from every single-rank MPI run, and no serial test can see
  // the difference. REVIEW.md TF-7: this used to say no test runs the one-
  // rank leg to check that -- stale since
  // `test/parallel/remap_halo_mpi_test.cpp`'s
  // `ERemapParallelRefusesAboveOneRankAndNotAtOne` ported (host-verified
  // per that file's own header): `== 1` must succeed and `> 1` must refuse
  // `E-REMAP-PARALLEL`, checked at every registered rank count 1..5, both
  // sides of the boundary this paragraph names.
  // ==========================================================================
  // KNOWN_ISSUES D83: `plan.placement` passed through -- `build_weights`
  // always resolves both sides through `plan.placement` alone (this
  // function's own design, remap_plan.hpp), so the halo this call checks
  // for is the one filed under THAT placement's own cache entry, not the
  // default Core's.
  if (a.backend != nullptr && a.backend->communicator_size() > 1 &&
      a.file_set != b.file_set &&
      !a.backend->has_remap_halo(a.mid, b.mid, a.dim, plan.placement)) {
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
    // This was CHECKED, not assumed. With this rejection removed and TempestRemap
    // present, the halo-extended source at 2 ranks gives
    // "E-REMAP-INTX: TempestRemap could not build the overlap mesh ...
    // (return code 0, 1024 overlap faces)" -- it does not merely lose accuracy
    // near the fringe, it fails outright, and it fails by printing while
    // returning success (which is why Task 5.3 reads its Announce stream at
    // all). Rejecting here turns that into one clear sentence at the right
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
    // this library cannot classify as a surface at all is rejected here, so the
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
      // The backend contract says it must reject rather than under-deliver.
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
  //
  // REVIEW.md TF-2: `try_boxes`, `intersect_boxes`, `fit_surface` and the
  // kind/radius/plane-z match below each run a computation (reading this
  // rank's own local cell corners, tagging GIDs) that can FAIL outright on
  // one rank -- a MOAB connectivity read error, an out-of-range tag --
  // while a peer's identical call succeeds, ahead of `assemble()`'s own
  // collectives (`accumulate_remap_halo`, `reduce_audit`'s
  // `MPI_Allreduce`). `reconcile<T>` is exactly `resolve()`'s and
  // `build_remap_halo`'s own fit_surface/orient fix (R-01): it agrees
  // whether every rank's call SUCCEEDED, so a one-rank error returns the
  // SAME Failure everywhere instead of that rank returning alone while its
  // peers walk into those collectives and hang. It does NOT, and cannot,
  // agree whether ranks that all succeeded classified their OWN local
  // partition into the SAME box/surface verdict (`BoxTry::ok`,
  // `SurfaceFit::kind`) -- that is a different, value-level hazard
  // `reconcile_dim` exists for elsewhere (Side::dim) and TF-2 does not
  // claim to close it here.
  // =========================================================================
  auto ba_r = reconcile(a.pcomm, to_result(try_boxes(a)));
  if (!ba_r.ok()) return ba_r.error();
  const BoxTry ba = ba_r.value();
  auto bb_r = reconcile(b.pcomm, to_result(try_boxes(b)));
  if (!bb_r.ok()) return bb_r.error();
  const BoxTry bb = bb_r.value();

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
    auto sm = reconcile(a.pcomm, intersect_boxes(a, b, ba.mesh, bb.mesh));
    if (!sm.ok()) return sm.error();
    return finish(assemble(a, b, std::move(sm.value()), conservative, plan.conservation));
  }

  if (a.dim == 2) {
    auto fa = reconcile(a.pcomm, fit_surface(a));
    if (!fa.ok()) return fa.error();
    auto fb = reconcile(b.pcomm, fit_surface(b));
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
      auto sm = reconcile(a.pcomm, intersect_surfaces(a, b, fa.value(), fb.value()));
      if (!sm.ok()) return sm.error();
      return finish(assemble(a, b, std::move(sm.value()), conservative, plan.conservation));
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
