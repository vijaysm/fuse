// Before the FUSE_HAVE_MPI test below: config.hpp defines the feature
// macros, and an undefined one reads as 0 -- which had compiled this
// file's whole parallel half out of every build, MPI or not.
#include "fuse/base/config.hpp"
#include "fuse/base/error.hpp"

#include "fuse/backend/moab_backend.hpp"
#include "fuse/backend/cell_simplices.hpp"

// ONE OF THE TWO TRANSLATION UNITS THAT MAY SEE MOAB (Global Constraints
// dependency rule); the other is src/backend/intx_weights.cpp, which needs the
// actual polygons to build a supermesh and reaches them through
// MoabMeshBackend::native_mesh below. Nothing here or there leaks a MOAB type
// through a header.
#include <moab/CN.hpp>
#include <moab/Core.hpp>
#include <moab/Range.hpp>

#include <MBTagConventions.hpp>

// MOABConfig.h arrives via Core.hpp, so MOAB_HAVE_MPI is only knowable here.
// Both halves are required: MOAB must have been built with MPI, and FUSE must
// have been configured with it, or there is no mpi.h on the include path.
#if defined(MOAB_HAVE_MPI) && FUSE_HAVE_MPI
#define FUSE_MOAB_PARALLEL 1
#include <MBParallelConventions.h>
#include <moab/ParallelComm.hpp>
#include <mpi.h>
#endif

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fuse {
/// Implementation detail of this one translation unit. A NAMED namespace
/// rather than an anonymous one only because `MoabMeshBackend::Impl` has
/// external linkage and holds a `MoabMeshView`; a member whose type came from
/// the anonymous namespace is ill-formed-no-diagnostic-required territory and
/// GCC warns about it (-Wsubobject-linkage).
namespace moab_detail {

// ===========================================================================
// Small vector algebra. std::array rather than moab::CartVect so the geometry
// below can be read -- and, in the tests, re-derived -- without MOAB.
// ===========================================================================

using V3 = std::array<double, 3>;

[[nodiscard]] V3 cross(const V3& a, const V3& b) {
  return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
          a[0] * b[1] - a[1] * b[0]};
}
[[nodiscard]] double dot(const V3& a, const V3& b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
[[nodiscard]] V3 sub(const V3& a, const V3& b) {
  return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
[[nodiscard]] double norm(const V3& a) { return std::sqrt(dot(a, a)); }

// ===========================================================================
// Quadrature
// ===========================================================================

/// 8-point Gauss-Legendre on [-1, 1]: the four positive abscissae and their
/// weights, the other four being mirror images. Exact through degree 15.
constexpr std::array<double, 4> kGL8Abscissa{
    0.18343464249564980, 0.52553240991632899, 0.79666647741362674,
    0.96028985649753623};
constexpr std::array<double, 4> kGL8Weight{
    0.36268378337836198, 0.31370664587788729, 0.22238103445337447,
    0.10122853629037626};

struct Rule1D {
  std::array<double, 8> x{};  ///< nodes on [0, 1]
  std::array<double, 8> w{};  ///< weights, summing to 1
};

[[nodiscard]] const Rule1D& gauss8() {
  static const Rule1D r = [] {
    Rule1D t{};
    for (std::size_t i = 0; i < 4; ++i) {
      t.x[3 - i] = 0.5 - 0.5 * kGL8Abscissa[i];
      t.x[4 + i] = 0.5 + 0.5 * kGL8Abscissa[i];
      t.w[3 - i] = 0.5 * kGL8Weight[i];
      t.w[4 + i] = 0.5 * kGL8Weight[i];
    }
    return t;
  }();
  return r;
}

/// Corner signs of the trilinear hexahedron in MOAB/Exodus order: bottom face
/// 0-1-2-3, then the top face 4-5-6-7 directly above it.
constexpr int kHexCorner[8][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0},
                                  {0, 0, 1}, {1, 0, 1}, {1, 1, 1}, {0, 1, 1}};
constexpr int kQuadCorner[4][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};

/// The integral of det(J) over the reference cube for the trilinear map with
/// these eight corners. SIGNED; the caller takes the absolute value.
///
/// EXACT, not approximate. Each coordinate of the map is trilinear, so each
/// first derivative is bilinear in the other two parameters, and det(J), a
/// sum of triple products of them, has degree at most 2 in each of u, v, w.
/// Two-point Gauss-Legendre is exact through degree 3 per variable, so 2x2x2
/// evaluates the integral exactly, to rounding; refining it changes nothing.
///
/// That is why this is not a tetrahedral decomposition. Splitting a
/// hexahedron into five tetrahedra is exact only when the element is affine.
/// On the trilinear hexahedron whose base is the unit square and whose top
/// face is the bilinear surface z = 1 + xy -- volume exactly 1.25 -- the
/// five-tetrahedron sum gives 1.3333..., 6.7% high, and the answer depends on
/// which diagonal the decomposition picks.
[[nodiscard]] double hex_signed_volume(const std::array<V3, 8>& p) {
  const double lo = 0.5 - 0.5 / std::sqrt(3.0);
  const double hi = 0.5 + 0.5 / std::sqrt(3.0);
  const double g[2] = {lo, hi};
  double total = 0.0;
  for (const double u : g) {
    for (const double v : g) {
      for (const double w : g) {
        V3 du{0.0, 0.0, 0.0};
        V3 dv{0.0, 0.0, 0.0};
        V3 dw{0.0, 0.0, 0.0};
        for (std::size_t i = 0; i < 8; ++i) {
          const double fu = kHexCorner[i][0] ? u : 1.0 - u;
          const double fv = kHexCorner[i][1] ? v : 1.0 - v;
          const double fw = kHexCorner[i][2] ? w : 1.0 - w;
          const double su = kHexCorner[i][0] ? 1.0 : -1.0;
          const double sv = kHexCorner[i][1] ? 1.0 : -1.0;
          const double sw = kHexCorner[i][2] ? 1.0 : -1.0;
          for (std::size_t k = 0; k < 3; ++k) {
            du[k] += su * fv * fw * p[i][k];
            dv[k] += fu * sv * fw * p[i][k];
            dw[k] += fu * fv * sw * p[i][k];
          }
        }
        total += 0.125 * dot(cross(du, dv), dw);  // (1/2)^3 for the weights
      }
    }
  }
  return total;
}

/// Area of the bilinear quadrilateral patch with these four corners: the
/// integral of |dX/du x dX/dv|, by 8x8 Gauss-Legendre.
///
///   * For a PLANAR quadrilateral the integrand is bilinear in (u, v), so
///     every Gauss rule -- even one point -- is exact, and the result is the
///     ordinary planar area to rounding.
///   * For a WARPED one the integrand is the square root of a polynomial and
///     has no elementary antiderivative. Gauss-Legendre converges
///     geometrically: against a converged reference, 8 points give a relative
///     error near 3e-13 when the warp is comparable to the edge length, and
///     that degrades for extreme warp.
///
/// The cheap alternative -- half the magnitude of the cross product of the
/// diagonals -- is NOT this quantity. It is exactly the one-point rule
/// applied to the VECTOR area, |integral of Xu x Xv|, which by the triangle
/// inequality never exceeds the scalar area and is strictly smaller whenever
/// the quadrilateral is non-planar. On MOAB's 3456-element cubed sphere it
/// understates the total by 9.7e3 out of 8.0e8; on a unit square with one
/// corner lifted by 1 it understates by 4.4%.
[[nodiscard]] double quad_area(const std::array<V3, 4>& p) {
  const Rule1D& g = gauss8();
  double total = 0.0;
  for (std::size_t iu = 0; iu < 8; ++iu) {
    for (std::size_t iv = 0; iv < 8; ++iv) {
      const double u = g.x[iu];
      const double v = g.x[iv];
      V3 du{0.0, 0.0, 0.0};
      V3 dv{0.0, 0.0, 0.0};
      for (std::size_t i = 0; i < 4; ++i) {
        const double fu = kQuadCorner[i][0] ? u : 1.0 - u;
        const double fv = kQuadCorner[i][1] ? v : 1.0 - v;
        const double su = kQuadCorner[i][0] ? 1.0 : -1.0;
        const double sv = kQuadCorner[i][1] ? 1.0 : -1.0;
        for (std::size_t k = 0; k < 3; ++k) {
          du[k] += su * fv * p[i][k];
          dv[k] += fu * sv * p[i][k];
        }
      }
      total += g.w[iu] * g.w[iv] * norm(cross(du, dv));
    }
  }
  return total;
}

/// Newell's formula, exact for any simple PLANAR polygon, convex or not. A
/// non-planar polygon has no canonical area -- it depends on how one chooses
/// to fill it -- and this reports the magnitude of its vector area, a lower
/// bound.
[[nodiscard]] double polygon_area(const std::vector<V3>& p) {
  V3 n{0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < p.size(); ++i) {
    const V3 c = cross(p[i], p[(i + 1) % p.size()]);
    for (std::size_t k = 0; k < 3; ++k) n[k] += c[k];
  }
  return 0.5 * norm(n);
}

// ===========================================================================
// Error plumbing. Every MOAB ErrorCode is checked and converted; none is
// discarded, and no MB_CHK_* macro is used -- they return or abort behind the
// caller's back, which would bypass Result.
// ===========================================================================

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

// ===========================================================================
// Element geometry
// ===========================================================================

[[nodiscard]] std::vector<V3> node_coords(moab::Interface& mb,
                                          const moab::EntityHandle* conn,
                                          int n, const char* what) {
  std::vector<double> xyz(static_cast<std::size_t>(n) * 3, 0.0);
  const moab::ErrorCode rval = mb.get_coords(conn, n, xyz.data());
  if (rval != moab::MB_SUCCESS) {
    FUSE_THROW(InvariantViolation, "{}", std::string(std::string("MoabMeshView::") + what +
                             ": MOAB could not read vertex coordinates: " +
                             moab_text(mb, rval)));
  }
  std::vector<V3> p(static_cast<std::size_t>(n));
  for (std::size_t i = 0; i < p.size(); ++i) {
    p[i] = {xyz[3 * i], xyz[3 * i + 1], xyz[3 * i + 2]};
  }
  return p;
}

/// Corner permutations that express a lower solid as a DEGENERATE trilinear
/// hexahedron with the same image, so one exact integrator serves all of
/// them. Pinned against closed forms in
/// `test/backend/moab_mesh_test.cpp:MixedElementMeasuresMatchClosedForms`: the
/// reference tetrahedron comes out at 1/6, a unit-triangle wedge of height 1
/// at 1/2, and a unit-square pyramid of height 1 at 1/3.
constexpr int kHexAsHex[8] = {0, 1, 2, 3, 4, 5, 6, 7};
constexpr int kPyramidAsHex[8] = {0, 1, 2, 3, 4, 4, 4, 4};
constexpr int kPrismAsHex[8] = {0, 1, 2, 2, 3, 4, 5, 5};

[[nodiscard]] std::array<V3, 8> as_hex(const std::vector<V3>& p,
                                       const int (&map)[8]) {
  std::array<V3, 8> h{};
  for (std::size_t i = 0; i < 8; ++i) {
    h[i] = p[static_cast<std::size_t>(map[i])];
  }
  return h;
}

/// The unsigned physical measure of one element.
///
/// Throws rather than returning a number for an element type it cannot
/// integrate exactly, and in particular for a HIGHER-ORDER one: a quadratic
/// hexahedron has 20 or 27 nodes describing a curved image, and quietly using
/// its eight corners would report the measure of a different, straight-sided
/// solid. `MeshView::measure` promises the true integral of the Jacobian
/// determinant, so the only truthful options are to compute it or to refuse,
/// and refusing loudly beats a quiet few-percent lie inside a conservation
/// sum.
[[nodiscard]] double element_measure(moab::Interface& mb,
                                     moab::EntityHandle h) {
  const moab::EntityType type = mb.type_from_handle(h);
  const moab::EntityHandle* conn = nullptr;
  int nconn = 0;
  std::vector<moab::EntityHandle> storage;
  const moab::ErrorCode rval =
      mb.get_connectivity(h, conn, nconn, /*corners_only=*/false, &storage);
  if (rval != moab::MB_SUCCESS) {
    FUSE_THROW(InvariantViolation, "{}", std::string(
        "MoabMeshView::measure: MOAB could not read the connectivity of "
        "entity " +
        std::to_string(h) + ": " + moab_text(mb, rval)));
  }
  if (conn == nullptr || nconn <= 0) {
    FUSE_THROW(InvariantViolation, "{}", std::string("MoabMeshView::measure: entity " +
                             std::to_string(h) +
                             " has no connectivity, so no geometry to measure"));
  }

  const std::string type_name = moab::CN::EntityTypeName(type);
  if (type != moab::MBPOLYGON && type != moab::MBPOLYHEDRON) {
    const int corners = moab::CN::VerticesPerEntity(type);
    if (nconn != corners) {
      FUSE_THROW(InvariantViolation, "{}", std::string(
          "MoabMeshView::measure: entity " + std::to_string(h) + " is a " +
          type_name + " with " + std::to_string(nconn) + " nodes, not " +
          std::to_string(corners) +
          ". This backend integrates straight-sided elements exactly and "
          "refuses to report the straight-sided measure of a curved one, "
          "which would misstate a conservation sum with no diagnostic."));
    }
  }

  const std::vector<V3> p = node_coords(mb, conn, nconn, "measure");

  switch (type) {
    case moab::MBEDGE:
      return norm(sub(p[1], p[0]));
    case moab::MBTRI:
      return 0.5 * norm(cross(sub(p[1], p[0]), sub(p[2], p[0])));
    case moab::MBQUAD:
      return quad_area({p[0], p[1], p[2], p[3]});
    case moab::MBPOLYGON:
      if (nconn < 3) {
        FUSE_THROW(InvariantViolation, "{}", std::string(
            "MoabMeshView::measure: polygon " + std::to_string(h) + " has " +
            std::to_string(nconn) + " nodes; a polygon needs at least three"));
      }
      return polygon_area(p);
    case moab::MBTET:
      // Affine, so det(J) is constant and this closed form is exact.
      return std::fabs(dot(cross(sub(p[1], p[0]), sub(p[2], p[0])),
                           sub(p[3], p[0]))) /
             6.0;
    case moab::MBPYRAMID:
      return std::fabs(hex_signed_volume(as_hex(p, kPyramidAsHex)));
    case moab::MBPRISM:
      return std::fabs(hex_signed_volume(as_hex(p, kPrismAsHex)));
    case moab::MBHEX:
      return std::fabs(hex_signed_volume(as_hex(p, kHexAsHex)));
    default:
      FUSE_THROW(InvariantViolation, "{}", std::string(
          "MoabMeshView::measure: no exact measure is implemented for a " +
          type_name + " (entity " + std::to_string(h) +
          "); this backend refuses to guess one"));
  }
}

// ===========================================================================
// The view
// ===========================================================================

/// Everything the view caches for one topological dimension. Built once at
/// load and never resized afterwards, so no index or iterator into it is
/// invalidated by a later `load`.
struct DimData {
  std::vector<moab::EntityHandle> handles;  ///< local index -> MOAB handle
  std::vector<std::size_t> global_ids;      ///< parallel to `handles`
  std::vector<char> owned;                  ///< parallel to `handles`, 1/0
  std::size_t owned_count = 0;
  /// Why this dimension has no usable global numbering, empty if it has one.
  ///
  /// Diagnosed at load but reported from `global_id`, not from `load`. A
  /// partitioned MOAB read routinely manufactures auxiliary entities that no
  /// one numbered -- PARALLEL_RESOLVE_SHARED_ENTS creates interface edges
  /// with GLOBAL_ID left at MOAB's -1 default -- and refusing the whole mesh
  /// because a dimension nobody couples on lacks ids would make real files
  /// unloadable. Refusing at the point where a wrong id would actually be
  /// used keeps the diagnostic and drops the collateral damage.
  std::string gid_error;
};

class MoabMeshView final : public MeshView {
 public:
  MoabMeshView(moab::Interface& mb, moab::EntityHandle file_set)
      : mb_(&mb), file_set_(file_set) {}

  [[nodiscard]] int dimension() const override { return top_dim_; }

  [[nodiscard]] std::size_t entity_count(int dim) const override {
    if (dim < 0 || dim > 3) return 0;
    return dims_[static_cast<std::size_t>(dim)].handles.size();
  }

  [[nodiscard]] std::size_t owned_count(int dim) const override {
    if (dim < 0 || dim > 3) return 0;
    return dims_[static_cast<std::size_t>(dim)].owned_count;
  }

  [[nodiscard]] bool is_owned(int dim, std::size_t i) const override {
    check(dim, i, "is_owned");
    return dims_[static_cast<std::size_t>(dim)].owned[i] != 0;
  }

  [[nodiscard]] Point centroid(int dim, std::size_t i) const override {
    check(dim, i, "centroid");
    const moab::EntityHandle h = dims_[static_cast<std::size_t>(dim)].handles[i];
    if (dim == 0) {
      double c[3] = {0.0, 0.0, 0.0};
      const moab::ErrorCode rval = mb_->get_coords(&h, 1, c);
      if (rval != moab::MB_SUCCESS) {
        FUSE_THROW(InvariantViolation, "{}", std::string(
            "MoabMeshView::centroid: MOAB could not read vertex " +
            std::to_string(h) + ": " + moab_text(*mb_, rval)));
      }
      return Point{c[0], c[1], c[2]};
    }
    const moab::EntityHandle* conn = nullptr;
    int nconn = 0;
    std::vector<moab::EntityHandle> storage;
    const moab::ErrorCode rval =
        mb_->get_connectivity(h, conn, nconn, /*corners_only=*/true, &storage);
    if (rval != moab::MB_SUCCESS || conn == nullptr || nconn <= 0) {
      FUSE_THROW(InvariantViolation, "{}", std::string(
          "MoabMeshView::centroid: MOAB could not read the connectivity of "
          "entity " +
          std::to_string(h) + ": " + moab_text(*mb_, rval)));
    }
    const std::vector<V3> p = node_coords(*mb_, conn, nconn, "centroid");
    V3 c{0.0, 0.0, 0.0};
    for (const V3& v : p) {
      for (std::size_t k = 0; k < 3; ++k) c[k] += v[k];
    }
    const double inv = 1.0 / static_cast<double>(p.size());
    return Point{c[0] * inv, c[1] * inv, c[2] * inv};
  }

  [[nodiscard]] double measure(int dim, std::size_t i) const override {
    check(dim, i, "measure");
    if (dim == 0) {
      FUSE_THROW(InvalidArgument, "{}", std::string(
          "MoabMeshView::measure: this backend defines no measure for "
          "vertices (dim 0); it has no dual or control volume to report, and "
          "a zero weight would silently drop them from a conservation sum"));
    }
    return element_measure(*mb_, dims_[static_cast<std::size_t>(dim)].handles[i]);
  }

  [[nodiscard]] std::size_t global_id(int dim, std::size_t i) const override {
    check(dim, i, "global_id");
    const DimData& dd = dims_[static_cast<std::size_t>(dim)];
    if (!dd.gid_error.empty()) {
      // Not `out_of_range`: the entity exists and the index is fine, it is
      // the numbering that is unusable. Returning the raw value, or 0, would
      // let two entities match that are not the same entity.
      FUSE_THROW(InvariantViolation, "{}", std::string("MoabMeshView::global_id: " + dd.gid_error));
    }
    return dd.global_ids[i];
  }

  // --- geometry as simplices (MeshView's clipping contract) ----------------
  //
  // The interface requires every backend to present cells as simplices. The
  // decomposition itself is NOT written here: it is the shared index table in
  // fuse/backend/cell_simplices.hpp, so this backend and any other decompose
  // the same cell the same way. That matters beyond tidiness -- a hexahedron
  // with non-planar faces has a different volume under each diagonal, so two
  // backends using different conventions would disagree about cell volumes and
  // a conservation check between them would report a defect that is really a
  // convention mismatch.

  [[nodiscard]] std::size_t simplex_count(int dim,
                                          std::size_t i) const override {
    check(dim, i, "simplex_count");
    if (dim == 0) return 0;  // a vertex carries no geometry to clip
    const auto shape = shape_of(dim, i);
    if (!shape) return 0;
    return cell_simplex_count(shape->first, shape->second);
  }

  void simplex_points(int dim, std::size_t i, std::size_t s,
                      Point* out) const override {
    check(dim, i, "simplex_points");
    if (out == nullptr) {
      FUSE_THROW(InvalidArgument, "{}", std::string(
          "MoabMeshView::simplex_points: null output pointer"));
    }
    const auto shape = shape_of(dim, i);
    if (!shape) {
      FUSE_THROW(InvariantViolation, "{}", std::string(
          "MoabMeshView::simplex_points: entity " + std::to_string(i) +
          " of dimension " + std::to_string(dim) +
          " has a cell type this backend cannot decompose; see "
          "fuse/backend/cell_simplices.hpp for the shapes it supports"));
    }
    int corners[4] = {0, 0, 0, 0};
    if (!simplex_corners(shape->first, shape->second, s, corners)) {
      FUSE_THROW(InvalidArgument, "{}", std::string(
          "MoabMeshView::simplex_points: simplex " + std::to_string(s) +
          " of entity " + std::to_string(i) + " does not exist (it has " +
          std::to_string(cell_simplex_count(shape->first, shape->second)) + ")"));
    }
    const moab::EntityHandle h = dims_[static_cast<std::size_t>(dim)].handles[i];
    const moab::EntityHandle* conn = nullptr;
    int nconn = 0;
    std::vector<moab::EntityHandle> storage;
    const moab::ErrorCode rval =
        mb_->get_connectivity(h, conn, nconn, /*corners_only=*/true, &storage);
    if (rval != moab::MB_SUCCESS || conn == nullptr || nconn <= 0) {
      FUSE_THROW(InvariantViolation, "{}", std::string(
          "MoabMeshView::simplex_points: MOAB could not read the connectivity "
          "of entity " + std::to_string(h) + ": " + moab_text(*mb_, rval)));
    }
    const std::vector<V3> p = node_coords(*mb_, conn, nconn, "simplex_points");
    const int per = shape_dimension(shape->first) + 1;
    for (int k = 0; k < per; ++k) {
      const V3& v = p[static_cast<std::size_t>(corners[k])];
      out[k] = Point{v[0], v[1], v[2]};
    }
  }

  // --- mesh_view.hpp's cell-to-vertex connectivity query (nodal-remap
  // track) -- straightforward here: `shape_of` already maps MOAB's own
  // entity type onto `CellShape` (used above for the simplex decomposition),
  // and `get_connectivity(..., corners_only=true, ...)` gives the corner
  // HANDLES in MOAB's own (identical to cell_simplices.hpp's) corner order;
  // `index_of` translates each to this view's own dimension-0 local index.
  [[nodiscard]] Result<CellVertices> cell_vertices(int dim, std::size_t i) const override {
    check(dim, i, "cell_vertices");
    const auto shape = shape_of(dim, i);
    if (!shape) {
      return Failure{"E-MESH-NO-CELL-VERTICES",
                     "MoabMeshView::cell_vertices: entity " + std::to_string(i) +
                         " of dimension " + std::to_string(dim) +
                         " has a cell type with no fuse::CellShape equivalent "
                         "(MBPOLYHEDRON or a higher-order type; see "
                         "cell_simplices.hpp for the shapes this backend supports)"};
    }
    const moab::EntityHandle h = dims_[static_cast<std::size_t>(dim)].handles[i];
    const moab::EntityHandle* conn = nullptr;
    int nconn = 0;
    std::vector<moab::EntityHandle> storage;
    const moab::ErrorCode rval =
        mb_->get_connectivity(h, conn, nconn, /*corners_only=*/true, &storage);
    if (rval != moab::MB_SUCCESS || conn == nullptr || nconn <= 0) {
      return Failure{"E-MESH-NO-CELL-VERTICES",
                     "MoabMeshView::cell_vertices: MOAB could not read the corner "
                     "connectivity of entity " + std::to_string(h) + ": " +
                         moab_text(*mb_, rval)};
    }
    CellVertices out;
    out.shape = shape->first;
    out.vertices.reserve(static_cast<std::size_t>(nconn));
    for (int k = 0; k < nconn; ++k) {
      const std::size_t vidx = index_of(0, conn[k]);
      if (vidx == kNoIndex) {
        return Failure{
            "E-MESH-NO-CELL-VERTICES",
            "MoabMeshView::cell_vertices: corner " + std::to_string(k) + " of entity " +
                std::to_string(h) +
                " is not among this rank's own dimension-0 entities (a vertex this "
                "view was not given, e.g. outside a partitioned read's halo)"};
      }
      out.vertices.push_back(vidx);
    }
    return out;
  }

  // --- used by the backend, not part of MeshView ---------------------------

  [[nodiscard]] moab::EntityHandle file_set() const { return file_set_; }
  [[nodiscard]] const DimData& dim_data(int dim) const {
    return dims_[static_cast<std::size_t>(dim)];
  }
  [[nodiscard]] DimData& dim_data_mut(int dim) {
    return dims_[static_cast<std::size_t>(dim)];
  }
  void set_top_dim(int d) { top_dim_ = d; }

  /// MOAB's entity type and corner count for a cell, mapped onto the shared
  /// decomposition's vocabulary. Returns nothing for a type this library does
  /// not decompose -- reported by the callers above rather than guessed at,
  /// because treating an unknown 8-corner cell as a hexahedron would produce a
  /// plausible volume for the wrong solid.
  [[nodiscard]] std::optional<std::pair<CellShape, std::size_t>> shape_of(
      int dim, std::size_t i) const {
    const moab::EntityHandle h = dims_[static_cast<std::size_t>(dim)].handles[i];
    const moab::EntityType type = mb_->type_from_handle(h);
    const moab::EntityHandle* conn = nullptr;
    int nconn = 0;
    std::vector<moab::EntityHandle> storage;
    if (mb_->get_connectivity(h, conn, nconn, /*corners_only=*/true, &storage) !=
            moab::MB_SUCCESS ||
        conn == nullptr || nconn <= 0) {
      return std::nullopt;
    }
    const auto n = static_cast<std::size_t>(nconn);
    switch (type) {
      case moab::MBEDGE:    return std::pair{CellShape::Edge, n};
      case moab::MBTRI:     return std::pair{CellShape::Triangle, n};
      case moab::MBQUAD:    return std::pair{CellShape::Quad, n};
      case moab::MBPOLYGON: return std::pair{CellShape::Polygon, n};
      case moab::MBTET:     return std::pair{CellShape::Tet, n};
      case moab::MBPYRAMID: return std::pair{CellShape::Pyramid, n};
      case moab::MBPRISM:   return std::pair{CellShape::Prism, n};
      case moab::MBHEX:     return std::pair{CellShape::Hex, n};
      // MBKNIFE, MBPOLYHEDRON and the higher-order types are deliberately
      // absent: each needs a decomposition of its own, and an approximate one
      // would be wrong in conservation rather than visibly unsupported.
      default: return std::nullopt;
    }
  }

  /// Local index of `h` among the dimension-`dim` entities, or npos. The
  /// handle vector is ascending -- a moab::Range iterates in handle order --
  /// so a binary search is valid.
  [[nodiscard]] std::size_t index_of(int dim, moab::EntityHandle h) const {
    const auto& v = dims_[static_cast<std::size_t>(dim)].handles;
    const auto it = std::lower_bound(v.begin(), v.end(), h);
    if (it == v.end() || *it != h) return kNoIndex;
    return static_cast<std::size_t>(it - v.begin());
  }

  static constexpr std::size_t kNoIndex =
      std::numeric_limits<std::size_t>::max();

 private:
  /// Checked unconditionally, never with assert(): `-O3 -DNDEBUG` compiles an
  /// assert out, and every accessor above would then index a vector out of
  /// bounds -- undefined behaviour that in a Release build usually reads a
  /// plausible number rather than crashing.
  void check(int dim, std::size_t i, const char* what) const {
    if (dim < 0 || dim > 3) {
      FUSE_THROW(InvalidArgument, "{}", std::string(std::string("MoabMeshView::") + what +
                              ": dimension " + std::to_string(dim) +
                              " is outside [0, 3]"));
    }
    const std::size_t n = dims_[static_cast<std::size_t>(dim)].handles.size();
    if (i >= n) {
      FUSE_THROW(InvalidArgument, "{}", std::string(std::string("MoabMeshView::") + what +
                              ": entity index " + std::to_string(i) +
                              " is outside [0, " + std::to_string(n) +
                              ") for dimension " + std::to_string(dim) +
                              " on this rank"));
    }
  }

  moab::Interface* mb_;
  moab::EntityHandle file_set_;
  int top_dim_ = 0;
  std::array<DimData, 4> dims_{};
};

// ===========================================================================
// Source-string parsing. Deliberately the same grammar SerialMeshBackend
// accepts (src/backend/serial_backend.cpp), so "generated:..." means the same
// thing whichever backend answers; MoabMesh.RejectsTheSameMalformedSources
// pins the two against each other.
// ===========================================================================

/// A plain positive decimal integer, and nothing else: no sign, no
/// whitespace, no other base. `strtol` would accept all three and saturate
/// instead of failing on overflow.
[[nodiscard]] bool parse_size(std::string_view s, std::size_t& out) {
  if (s.empty()) return false;
  for (const char c : s) {
    if (c < '0' || c > '9') return false;
  }
  std::size_t v = 0;
  const char* first = s.data();
  const char* last = first + s.size();
  const auto res = std::from_chars(first, last, v);
  if (res.ec != std::errc{} || res.ptr != last) return false;  // incl. ERANGE
  if (v == 0) return false;
  out = v;
  return true;
}

/// `a * b`, or false on overflow. `std::size_t` is unsigned, so `a * b` WRAPS
/// silently: defined behaviour, invisible to UBSan, and the wrapped value is
/// an ordinary plausible-looking count.
[[nodiscard]] bool checked_mul(std::size_t a, std::size_t b, std::size_t& out) {
  if (a != 0 && b > std::numeric_limits<std::size_t>::max() / a) return false;
  out = a * b;
  return true;
}

/// The trailing `_<n>` of a name like "sideset_3", if the prefix matches.
[[nodiscard]] bool match_tagged_set_name(std::string_view name,
                                         std::string_view prefix, int& value) {
  if (name.size() <= prefix.size() + 1) return false;
  if (name.substr(0, prefix.size()) != prefix) return false;
  if (name[prefix.size()] != '_') return false;
  const std::string_view digits = name.substr(prefix.size() + 1);
  std::size_t v = 0;
  if (!parse_size(digits, v)) return false;
  if (v > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  value = static_cast<int>(v);
  return true;
}

struct TaggedSetPrefix {
  std::string_view prefix;
  const char* tag_name;
};

/// The brief's rule -- "a NEUMANN_SET / DIRICHLET_SET / MATERIAL_SET tag
/// whose value matches a trailing integer" -- does not say WHICH of the three
/// to look in, and trying all three makes "sideset_3" ambiguous the moment a
/// mesh has both a sideset 3 and a block 3, which Exodus meshes routinely do.
/// The prefix picks the tag; nothing is guessed.
constexpr std::array<TaggedSetPrefix, 6> kTaggedSetPrefixes{{
    {"sideset", NEUMANN_SET_TAG_NAME},
    {"neumann", NEUMANN_SET_TAG_NAME},
    {"nodeset", DIRICHLET_SET_TAG_NAME},
    {"dirichlet", DIRICHLET_SET_TAG_NAME},
    {"block", MATERIAL_SET_TAG_NAME},
    {"material", MATERIAL_SET_TAG_NAME},
}};

/// The six named boundary sets a generated box defines, in the order the
/// serial backend blocks its boundary faces. The NEUMANN_SET ids are 1-based
/// positions in this table, so "sideset_1" is another spelling of "left".
struct BoxSide {
  const char* name;
  int axis;      ///< 0 = x, 1 = y, 2 = z
  bool at_max;   ///< false = the min face, true = the max face
};
constexpr std::array<BoxSide, 6> kBoxSides{{
    {"left", 0, false},
    {"right", 0, true},
    {"front", 1, false},
    {"back", 1, true},
    {"bottom", 2, false},
    {"top", 2, true},
}};

}  // namespace moab_detail

using namespace moab_detail;  // NOLINT: one translation unit, no headers

// ===========================================================================
// Impl
// ===========================================================================

struct MoabMeshBackend::Impl {
  // Declaration order is destruction order reversed: `mb` is declared first
  // so it outlives the views that hold a pointer to it and the ParallelComm
  // that holds an Interface*.
  /// `mutable` because MOAB declares even pure queries -- get_coords,
  /// get_entities_by_dimension, tag_get_data -- as non-const members of
  /// Interface. The alternative is a const_cast at every call site in a const
  /// method, which is the same thing said less honestly.
  mutable moab::Core mb;
#ifdef FUSE_MOAB_PARALLEL
  /// Our own duplicate of the caller's communicator, with MPI_ERRORS_RETURN
  /// set on it. Declared BEFORE `pcomm` so it is destroyed after the
  /// ParallelComm that holds it. Freed in the destructor unless MPI has
  /// already been finalised, exactly as the MPI transport does.
  struct OwnedComm {
    MPI_Comm comm = MPI_COMM_NULL;
    ~OwnedComm() {
      if (comm == MPI_COMM_NULL) return;
      int finalized = 0;
      if (MPI_Finalized(&finalized) == MPI_SUCCESS && finalized == 0) {
        MPI_Comm_free(&comm);
      }
      comm = MPI_COMM_NULL;
    }
  } owned_comm;
  std::unique_ptr<moab::ParallelComm> pcomm;
#endif
  std::vector<std::unique_ptr<MoabMeshView>> meshes;
  std::vector<std::string> names;  ///< parallel to `meshes`

  // --- the remap halo (R3) --------------------------------------------------
  //
  // Kept on the backend and not on the view: a halo is a property of a PAIR of
  // meshes, and the exchange plan needs the communicator, which the view has
  // no business knowing about.

  /// The gather/scatter plan between a mesh's halo slots and their owners.
  /// Rebuilt from scratch whenever cells are adopted, because adopting
  /// renumbers the local index space.
  struct HaloExchange {
    /// What this rank ASKS for, grouped by the rank that owns it.
    std::vector<int> req_counts, req_displs;
    /// Per request, in the same order, the local index of the halo slot it
    /// fills.
    std::vector<std::size_t> fill_index;
    /// What this rank ANSWERS, grouped by the rank that asked.
    std::vector<int> ans_counts, ans_displs;
    /// Per incoming request, the local index of the owned entity to read.
    std::vector<std::size_t> send_index;
    [[nodiscard]] std::size_t requested() const { return fill_index.size(); }
    [[nodiscard]] std::size_t answered() const { return send_index.size(); }
  };

  struct HaloState {
    /// Adopted handles, so `gather()`'s ownership pass can be corrected and a
    /// later adoption can tell what is already here.
    std::set<moab::EntityHandle> cells;
    HaloExchange ex;
  };

  /// (mesh index, dimension) -> halo. `std::map`, not `unordered_map`: a
  /// handful of entries, and a stable iteration order keeps diagnostics
  /// reproducible.
  std::map<std::pair<std::size_t, int>, HaloState> halos;
  /// (source mesh, target mesh, dimension) -> cells adopted for that pair.
  /// Presence, not the count, is what `has_remap_halo` answers: zero adopted
  /// cells is a legitimate outcome and must not read as "never built".
  std::map<std::tuple<std::size_t, std::size_t, int>, std::size_t> halo_pairs;

  [[nodiscard]] const HaloState* halo(std::size_t mesh, int dim) const {
    const auto it = halos.find({mesh, dim});
    return it == halos.end() ? nullptr : &it->second;
  }

  /// Re-derive `dim`'s local index space after cells were added to the file
  /// set, and put the halo cells back to NOT OWNED -- `gather()` asks
  /// `ParallelComm::filter_pstatus`, and a halo cell was created locally by
  /// the covering-set construction, so MOAB rightly considers it this rank's
  /// own. It is not: its values come from elsewhere and it must be excluded
  /// from every reduction.
  [[nodiscard]] Result<void> regather_with_halo(std::size_t mesh, int dim);

  /// Build `HaloExchange` for `dim` of `mesh` from scratch.
  [[nodiscard]] Result<void> plan_halo_exchange(std::size_t mesh, int dim);

  /// The body of both public exchange directions. `gather == true` moves
  /// owner -> halo and overwrites; `gather == false` moves halo -> owner,
  /// adding, and then zeroes the halo slots.
  [[nodiscard]] Result<void> run_halo_exchange(std::size_t mesh, int dim,
                                               double* buffer, int components,
                                               bool gather) const;

  [[nodiscard]] bool parallel() const {
#ifdef FUSE_MOAB_PARALLEL
    return pcomm != nullptr;
#else
    return false;
#endif
  }

  [[nodiscard]] Failure moab_error(std::string code, const std::string& what,
                                 moab::ErrorCode rval) {
    return Failure{std::move(code), what + ": " + moab_text(mb, rval)};
  }

  /// Look a conventional tag up, creating it only if it is genuinely absent.
  ///
  /// `tag_get_handle(..., MB_TAG_CREAT, &default)` is NOT idempotent: if the
  /// tag already exists with a different default it returns
  /// MB_ALREADY_ALLOCATED rather than the handle. moab::Core creates
  /// GLOBAL_ID itself, with default -1, so asking for it with default 0 fails
  /// on a freshly constructed Core -- which is exactly the case a generated
  /// mesh hits.
  [[nodiscard]] Result<moab::Tag> ensure_tag(const char* tag_name, int size,
                                             moab::DataType type,
                                             unsigned create_flags,
                                             const void* default_value);

  Result<void> read_file(const std::string& path, moab::EntityHandle file_set);
  Result<void> generate(const std::string& spec, moab::EntityHandle file_set);
  Result<void> generate_line(std::size_t n, moab::EntityHandle file_set);
  Result<void> generate_box(const std::array<std::size_t, 3>& n,
                            moab::EntityHandle file_set);
  Result<void> tag_sequential_global_ids(moab::EntityHandle file_set);
  Result<void> name_set(moab::EntityHandle set, std::string_view name);
  Result<void> gather(MoabMeshView& view);
  void discard(moab::EntityHandle file_set);

  /// A region resolved against a loaded mesh.
  struct Resolved {
    const MoabMeshView* view = nullptr;
    moab::EntityHandle set = 0;  ///< the file set for "" / "all"
    int dim = 0;
  };
  [[nodiscard]] Result<Resolved> resolve(const Region& r) const;
  [[nodiscard]] Result<moab::EntityHandle> find_named_set(
      const MoabMeshView& view, const std::string& set_name) const;
};

// --- construction -----------------------------------------------------------

MoabMeshBackend::MoabMeshBackend() : impl_(std::make_unique<Impl>()) {}
MoabMeshBackend::~MoabMeshBackend() = default;
MoabMeshBackend::MoabMeshBackend(MoabMeshBackend&&) noexcept = default;
MoabMeshBackend& MoabMeshBackend::operator=(MoabMeshBackend&&) noexcept =
    default;

// --- helpers ----------------------------------------------------------------

Result<moab::Tag> MoabMeshBackend::Impl::ensure_tag(const char* tag_name,
                                                    int size,
                                                    moab::DataType type,
                                                    unsigned create_flags,
                                                    const void* default_value) {
  moab::Tag tag = nullptr;
  moab::ErrorCode rval = mb.tag_get_handle(tag_name, size, type, tag);
  if (rval == moab::MB_SUCCESS) return tag;
  if (rval != moab::MB_TAG_NOT_FOUND) {
    return Failure{"E-MOAB-TAG", std::string("could not look up the ") +
                                   tag_name + " tag: " + moab_text(mb, rval)};
  }
  rval = mb.tag_get_handle(tag_name, size, type, tag,
                           create_flags | moab::MB_TAG_CREAT, default_value);
  if (rval != moab::MB_SUCCESS) {
    return Failure{"E-MOAB-TAG", std::string("could not create the ") +
                                   tag_name + " tag: " + moab_text(mb, rval)};
  }
  return tag;
}

Result<void> MoabMeshBackend::Impl::name_set(moab::EntityHandle set,
                                             std::string_view name) {
  // NAME is char[NAME_TAG_SIZE], zero padded. A longer name cannot be stored,
  // and truncating it would create a set that answers to a name nobody asked
  // for, so it is an error.
  if (name.size() >= static_cast<std::size_t>(NAME_TAG_SIZE)) {
    return Failure{"E-MOAB-SET-NAME",
                 "entity set name '" + std::string{name} + "' is " +
                     std::to_string(name.size()) +
                     " characters; MOAB's NAME tag holds at most " +
                     std::to_string(NAME_TAG_SIZE - 1)};
  }
  char buf[NAME_TAG_SIZE] = {};
  std::memcpy(buf, name.data(), name.size());
  const auto tag_r = ensure_tag(NAME_TAG_NAME, NAME_TAG_SIZE,
                                moab::MB_TYPE_OPAQUE, moab::MB_TAG_SPARSE,
                                nullptr);
  if (!tag_r.ok()) return tag_r.error();
  const moab::Tag tag = tag_r.value();
  const moab::ErrorCode rval = mb.tag_set_data(tag, &set, 1, buf);
  if (rval != moab::MB_SUCCESS) {
    return moab_error("E-MOAB-TAG",
                      "could not name entity set '" + std::string{name} + "'",
                      rval);
  }
  return Result<void>::success();
}

/// Number the entities of each dimension 1..n. Used only for meshes this
/// backend generated, which have no numbering of their own.
Result<void> MoabMeshBackend::Impl::tag_sequential_global_ids(
    moab::EntityHandle file_set) {
  const int unset = -1;  // MOAB's own convention for GLOBAL_ID
  const auto gid_r = ensure_tag(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER,
                                moab::MB_TAG_DENSE, &unset);
  if (!gid_r.ok()) return gid_r.error();
  const moab::Tag gid = gid_r.value();
  moab::ErrorCode rval = moab::MB_SUCCESS;
  for (int d = 0; d <= 3; ++d) {
    moab::Range r;
    rval = mb.get_entities_by_dimension(file_set, d, r, /*recursive=*/true);
    if (rval != moab::MB_SUCCESS) {
      return moab_error("E-MOAB-QUERY",
                        "could not list dimension-" + std::to_string(d) +
                            " entities of a generated mesh",
                        rval);
    }
    if (r.empty()) continue;
    if (r.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      return Failure{"E-MESH-SIZE",
                   "a generated mesh has " + std::to_string(r.size()) +
                       " dimension-" + std::to_string(d) +
                       " entities, more than MOAB's 32-bit GLOBAL_ID can "
                       "number"};
    }
    std::vector<int> ids(r.size());
    for (std::size_t i = 0; i < ids.size(); ++i) {
      ids[i] = static_cast<int>(i) + 1;
    }
    rval = mb.tag_set_data(gid, r, ids.data());
    if (rval != moab::MB_SUCCESS) {
      return moab_error("E-MOAB-TAG", "could not assign GLOBAL_IDs", rval);
    }
  }
  return Result<void>::success();
}

void MoabMeshBackend::Impl::discard(moab::EntityHandle file_set) {
  // Best effort: we are already returning an error, and leaving the partial
  // mesh behind would only leak. Nothing else references these entities --
  // every load creates its own -- so deleting them cannot disturb a mesh that
  // loaded successfully.
  for (int d = 3; d >= 0; --d) {
    moab::Range r;
    if (mb.get_entities_by_dimension(file_set, d, r, true) == moab::MB_SUCCESS &&
        !r.empty()) {
      (void)mb.delete_entities(r);
    }
  }
  moab::Range sets;
  if (mb.get_entities_by_type(file_set, moab::MBENTITYSET, sets, true) ==
      moab::MB_SUCCESS) {
    for (const moab::EntityHandle s : sets) (void)mb.delete_entities(&s, 1);
  }
  (void)mb.delete_entities(&file_set, 1);
}

// --- reading ----------------------------------------------------------------

Result<void> MoabMeshBackend::Impl::read_file(const std::string& path,
                                              moab::EntityHandle file_set) {
  std::string opts;
#ifdef FUSE_MOAB_PARALLEL
  if (pcomm != nullptr) {
    opts =
        "PARALLEL=READ_PART;PARTITION=" PARALLEL_PARTITION_TAG_NAME
        ";PARALLEL_RESOLVE_SHARED_ENTS;PARALLEL_COMM=" +
        std::to_string(pcomm->get_id());
  }
#endif
  const moab::ErrorCode rval =
      mb.load_file(path.c_str(), &file_set, opts.empty() ? nullptr : opts.c_str());
  if (rval != moab::MB_SUCCESS) {
    return Failure{"E-MOAB-LOAD", "MOAB could not read '" + path + "'" +
                                    (opts.empty() ? "" : " with options '" +
                                                             opts + "'") +
                                    ": " + moab_text(mb, rval)};
  }
  return Result<void>::success();
}

Result<void> MoabMeshBackend::Impl::generate_line(std::size_t n,
                                                  moab::EntityHandle file_set) {
  const std::size_t nv = n + 1;  // n >= 1, so this cannot wrap into 0
  if (nv > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Failure{"E-MESH-SIZE",
                 "a line of " + std::to_string(n) +
                     " cells needs more vertices than MOAB's create_vertices "
                     "can accept in one call"};
  }
  std::vector<double> xyz(3 * nv, 0.0);
  for (std::size_t i = 0; i < nv; ++i) {
    xyz[3 * i] = static_cast<double>(i) / static_cast<double>(n);
  }
  moab::Range verts;
  moab::ErrorCode rval =
      mb.create_vertices(xyz.data(), static_cast<int>(nv), verts);
  if (rval != moab::MB_SUCCESS) {
    return moab_error("E-MOAB-CREATE", "could not create line vertices", rval);
  }
  if (verts.size() != nv) {
    return Failure{"E-MOAB-CREATE",
                 "MOAB returned " + std::to_string(verts.size()) +
                     " vertices for " + std::to_string(nv) + " requested"};
  }
  const std::vector<moab::EntityHandle> vh(verts.begin(), verts.end());
  moab::Range cells;
  for (std::size_t i = 0; i < n; ++i) {
    const moab::EntityHandle conn[2] = {vh[i], vh[i + 1]};
    moab::EntityHandle e = 0;
    rval = mb.create_element(moab::MBEDGE, conn, 2, e);
    if (rval != moab::MB_SUCCESS) {
      return moab_error("E-MOAB-CREATE", "could not create a line cell", rval);
    }
    cells.insert(e);
  }
  rval = mb.add_entities(file_set, verts);
  if (rval == moab::MB_SUCCESS) rval = mb.add_entities(file_set, cells);
  if (rval != moab::MB_SUCCESS) {
    return moab_error("E-MOAB-CREATE",
                      "could not put the generated line in its file set", rval);
  }
  return Result<void>::success();
}

Result<void> MoabMeshBackend::Impl::generate_box(
    const std::array<std::size_t, 3>& n, moab::EntityHandle file_set) {
  const std::size_t sx = n[0] + 1;
  const std::size_t sy = n[1] + 1;
  const std::size_t sz = n[2] + 1;
  std::size_t nv = 0;
  std::size_t ncell = 0;
  if (!checked_mul(sx, sy, nv) || !checked_mul(nv, sz, nv) ||
      !checked_mul(n[0], n[1], ncell) || !checked_mul(ncell, n[2], ncell)) {
    return Failure{"E-MESH-SIZE",
                 "the entity counts for this box do not fit std::size_t; they "
                 "would wrap silently"};
  }
  // create_vertices takes an int count: a larger request would be truncated,
  // and MOAB would happily build a mesh out of the wrong number of vertices.
  if (nv > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    return Failure{"E-MESH-SIZE",
                 "this box needs " + std::to_string(nv) +
                     " vertices, more than MOAB's create_vertices accepts in "
                     "one call"};
  }

  std::vector<double> xyz(3 * nv, 0.0);
  const double hx = 1.0 / static_cast<double>(n[0]);
  const double hy = 1.0 / static_cast<double>(n[1]);
  const double hz = 1.0 / static_cast<double>(n[2]);
  for (std::size_t k = 0; k < sz; ++k) {
    for (std::size_t j = 0; j < sy; ++j) {
      for (std::size_t i = 0; i < sx; ++i) {
        const std::size_t idx = i + sx * (j + sy * k);
        xyz[3 * idx + 0] = static_cast<double>(i) * hx;
        xyz[3 * idx + 1] = static_cast<double>(j) * hy;
        xyz[3 * idx + 2] = static_cast<double>(k) * hz;
      }
    }
  }
  moab::Range verts;
  moab::ErrorCode rval =
      mb.create_vertices(xyz.data(), static_cast<int>(nv), verts);
  if (rval != moab::MB_SUCCESS) {
    return moab_error("E-MOAB-CREATE", "could not create box vertices", rval);
  }
  if (verts.size() != nv) {
    return Failure{"E-MOAB-CREATE",
                 "MOAB returned " + std::to_string(verts.size()) +
                     " vertices for " + std::to_string(nv) + " requested"};
  }
  const std::vector<moab::EntityHandle> vh(verts.begin(), verts.end());

  // create_vertices is documented to return the handles in input order, but
  // the whole structured layout below is built on that, so check it rather
  // than trust it: a mismatch would produce a mesh whose cells are tangled
  // and whose volumes still sum to something plausible.
  {
    std::vector<double> back(3 * nv, 0.0);
    rval = mb.get_coords(vh.data(), static_cast<int>(nv), back.data());
    if (rval != moab::MB_SUCCESS) {
      return moab_error("E-MOAB-CREATE", "could not read back box vertices",
                        rval);
    }
    if (back != xyz) {
      return Failure{"E-MOAB-CREATE",
                   "MOAB did not return the generated vertices in input "
                   "order; the structured connectivity below would be wrong"};
    }
  }

  const auto v = [&](std::size_t i, std::size_t j, std::size_t k) {
    return vh[i + sx * (j + sy * k)];
  };

  std::vector<moab::EntityHandle> cell_handles;
  cell_handles.reserve(ncell);
  for (std::size_t k = 0; k < n[2]; ++k) {
    for (std::size_t j = 0; j < n[1]; ++j) {
      for (std::size_t i = 0; i < n[0]; ++i) {
        const moab::EntityHandle conn[8] = {
            v(i, j, k),         v(i + 1, j, k),         v(i + 1, j + 1, k),
            v(i, j + 1, k),     v(i, j, k + 1),         v(i + 1, j, k + 1),
            v(i + 1, j + 1, k + 1), v(i, j + 1, k + 1)};
        moab::EntityHandle hex = 0;
        rval = mb.create_element(moab::MBHEX, conn, 8, hex);
        if (rval != moab::MB_SUCCESS) {
          return moab_error("E-MOAB-CREATE", "could not create a box cell",
                            rval);
        }
        cell_handles.push_back(hex);
      }
    }
  }
  moab::Range cells;
  std::copy(cell_handles.begin(), cell_handles.end(),
            moab::range_inserter(cells));

  rval = mb.add_entities(file_set, verts);
  if (rval == moab::MB_SUCCESS) rval = mb.add_entities(file_set, cells);
  if (rval != moab::MB_SUCCESS) {
    return moab_error("E-MOAB-CREATE",
                      "could not put the generated box in its file set", rval);
  }

  // The boundary skin, created explicitly rather than with moab::Skinner so
  // that each face's SIDE is known by construction instead of being
  // recovered from its centroid. Interior faces are deliberately not
  // enumerated -- the serial backend does not either, and a field on "the
  // faces of a box" means the boundary in every use FUSE has.
  moab::Range wall;
  std::array<moab::Range, 6> side_faces;
  for (std::size_t s = 0; s < kBoxSides.size(); ++s) {
    const BoxSide& side = kBoxSides[s];
    const std::size_t a = static_cast<std::size_t>(side.axis);
    const std::size_t u = (a + 1) % 3;  // in-plane axes, u varying fastest
    const std::size_t w = (a + 2) % 3;
    const std::size_t fixed = side.at_max ? n[a] : 0;
    std::vector<moab::EntityHandle> faces;
    faces.reserve(n[u] * n[w]);
    for (std::size_t iw = 0; iw < n[w]; ++iw) {
      for (std::size_t iu = 0; iu < n[u]; ++iu) {
        // The four corners in (u, w) order, then reversed on a min face so
        // the quad's normal points out of the box.
        const std::size_t du[4] = {0, 1, 1, 0};
        const std::size_t dw[4] = {0, 0, 1, 1};
        moab::EntityHandle conn[4] = {0, 0, 0, 0};
        for (std::size_t c = 0; c < 4; ++c) {
          const std::size_t src = side.at_max ? c : 3 - c;
          std::size_t ijk[3] = {0, 0, 0};
          ijk[a] = fixed;
          ijk[u] = iu + du[src];
          ijk[w] = iw + dw[src];
          conn[c] = v(ijk[0], ijk[1], ijk[2]);
        }
        moab::EntityHandle q = 0;
        rval = mb.create_element(moab::MBQUAD, conn, 4, q);
        if (rval != moab::MB_SUCCESS) {
          return moab_error("E-MOAB-CREATE",
                            "could not create a boundary face", rval);
        }
        faces.push_back(q);
      }
    }
    std::copy(faces.begin(), faces.end(), moab::range_inserter(side_faces[s]));
    wall.merge(side_faces[s]);
  }
  rval = mb.add_entities(file_set, wall);
  if (rval != moab::MB_SUCCESS) {
    return moab_error("E-MOAB-CREATE",
                      "could not put the boundary faces in the file set", rval);
  }

  // Named entity sets: the six sides and their union, matching the names
  // SerialMeshBackend answers to, plus NEUMANN_SET ids 1..6 so the
  // "sideset_<n>" spelling resolves as well.
  const int minus_one = -1;
  const auto neumann_r = ensure_tag(NEUMANN_SET_TAG_NAME, 1,
                                    moab::MB_TYPE_INTEGER,
                                    moab::MB_TAG_SPARSE, &minus_one);
  if (!neumann_r.ok()) return neumann_r.error();
  const moab::Tag neumann = neumann_r.value();
  for (std::size_t s = 0; s < kBoxSides.size(); ++s) {
    moab::EntityHandle set = 0;
    rval = mb.create_meshset(moab::MESHSET_SET, set);
    if (rval == moab::MB_SUCCESS) rval = mb.add_entities(set, side_faces[s]);
    if (rval == moab::MB_SUCCESS) rval = mb.add_entities(file_set, &set, 1);
    if (rval != moab::MB_SUCCESS) {
      return moab_error("E-MOAB-CREATE",
                        std::string("could not create the '") +
                            kBoxSides[s].name + "' entity set",
                        rval);
    }
    const auto named = name_set(set, kBoxSides[s].name);
    if (!named.ok()) return named;
    const int id = static_cast<int>(s) + 1;
    rval = mb.tag_set_data(neumann, &set, 1, &id);
    if (rval != moab::MB_SUCCESS) {
      return moab_error("E-MOAB-TAG", "could not tag a box side as a sideset",
                        rval);
    }
  }
  {
    moab::EntityHandle set = 0;
    rval = mb.create_meshset(moab::MESHSET_SET, set);
    if (rval == moab::MB_SUCCESS) rval = mb.add_entities(set, wall);
    if (rval == moab::MB_SUCCESS) rval = mb.add_entities(file_set, &set, 1);
    if (rval != moab::MB_SUCCESS) {
      return moab_error("E-MOAB-CREATE", "could not create the 'wall' set",
                        rval);
    }
    const auto named = name_set(set, "wall");
    if (!named.ok()) return named;
  }
  return Result<void>::success();
}

Result<void> MoabMeshBackend::Impl::generate(const std::string& spec,
                                             moab::EntityHandle file_set) {
  const auto colon = spec.find(':');
  if (colon == std::string::npos) {
    return Failure{"E-MESH-SOURCE", "malformed generated spec 'generated:" +
                                      spec + "'"};
  }
  const std::string kind = spec.substr(0, colon);
  const std::string dims = spec.substr(colon + 1);

  if (kind == "line") {
    std::size_t n = 0;
    if (!parse_size(dims, n)) {
      return Failure{"E-MESH-SOURCE",
                   "line needs one positive decimal cell count, got '" + dims +
                       "' in 'generated:" + spec + "'"};
    }
    const auto built = generate_line(n, file_set);
    if (!built.ok()) return built;
    return tag_sequential_global_ids(file_set);
  }
  if (kind == "box") {
    // Split on 'x' by hand: a `k < 3` loop over an istringstream stops after
    // three tokens and silently accepts "2x2x2x2".
    std::array<std::size_t, 3> n{1, 1, 1};
    std::size_t k = 0;
    std::size_t pos = 0;
    while (true) {
      const auto sep = dims.find('x', pos);
      const std::string_view tok = std::string_view{dims}.substr(
          pos, sep == std::string::npos ? std::string::npos : sep - pos);
      if (k >= 3 || !parse_size(tok, n[k])) {
        return Failure{"E-MESH-SOURCE",
                     "box needs exactly three positive decimal counts "
                     "'NxMxP', got '" +
                         dims + "' in 'generated:" + spec + "'"};
      }
      ++k;
      if (sep == std::string::npos) break;
      pos = sep + 1;
    }
    if (k != 3) {
      return Failure{"E-MESH-SOURCE",
                   "box needs exactly three positive decimal counts 'NxMxP', "
                   "got '" +
                       dims + "' in 'generated:" + spec + "'"};
    }
    const auto built = generate_box(n, file_set);
    if (!built.ok()) return built;
    // A generated mesh has no numbering of its own, and MOAB's GLOBAL_ID
    // default is -1, not 0 -- reading it back unassigned would surface as a
    // negative id rather than an absent one.
    return tag_sequential_global_ids(file_set);
  }
  return Failure{"E-MESH-SOURCE", "unknown generated kind '" + kind + "'"};
}

// --- gathering --------------------------------------------------------------

Result<void> MoabMeshBackend::Impl::gather(MoabMeshView& view) {
  const moab::EntityHandle file_set = view.file_set();
  moab::ErrorCode rval = moab::MB_SUCCESS;

  moab::Tag gid_tag = nullptr;
  const moab::ErrorCode gid_rval =
      mb.tag_get_handle(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER, gid_tag);
  const bool have_gid_tag = (gid_rval == moab::MB_SUCCESS);
  if (!have_gid_tag && gid_rval != moab::MB_TAG_NOT_FOUND) {
    return moab_error("E-MOAB-TAG", "could not look up the GLOBAL_ID tag",
                      gid_rval);
  }

  int top = 0;
  for (int d = 0; d <= 3; ++d) {
    DimData& dd = view.dim_data_mut(d);
    moab::Range r;
    // recursive: an Exodus block or an h5m partition arrives as a child set
    // of the file set, and its elements would otherwise be invisible.
    rval = mb.get_entities_by_dimension(file_set, d, r, /*recursive=*/true);
    if (rval != moab::MB_SUCCESS) {
      return moab_error("E-MOAB-QUERY",
                        "could not list the dimension-" + std::to_string(d) +
                            " entities of this mesh",
                        rval);
    }
    if (r.empty()) continue;
    if (d > top) top = d;

    // A moab::Range iterates in increasing handle order, and copying it into
    // a vector fixes that order as the local indexing for this dimension.
    dd.handles.assign(r.begin(), r.end());
    const std::size_t n = dd.handles.size();

    // --- ownership ---
    dd.owned.assign(n, 1);
    dd.owned_count = n;
#ifdef FUSE_MOAB_PARALLEL
    if (pcomm != nullptr) {
      moab::Range owned = r;
      rval = pcomm->filter_pstatus(owned, PSTATUS_NOT_OWNED, PSTATUS_NOT);
      if (rval != moab::MB_SUCCESS) {
        return moab_error("E-MOAB-PSTATUS",
                          "could not filter dimension-" + std::to_string(d) +
                              " entities by ownership",
                          rval);
      }
      dd.owned_count = 0;
      auto it = owned.begin();
      for (std::size_t i = 0; i < n; ++i) {
        while (it != owned.end() && *it < dd.handles[i]) ++it;
        const bool mine = (it != owned.end() && *it == dd.handles[i]);
        dd.owned[i] = mine ? 1 : 0;
        if (mine) ++dd.owned_count;
      }
    }
#endif

    // --- global ids ---
    //
    // MOAB's own default for GLOBAL_ID is -1 (moab::Core creates the tag that
    // way at construction), while a file written by another tool may use 0.
    // Both mean "not numbered", so both are treated as unset; only a strictly
    // positive value is a real id.
    std::vector<int> raw(n, 0);
    if (have_gid_tag) {
      if (n > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return Failure{"E-MESH-SIZE",
                     "this rank holds " + std::to_string(n) + " dimension-" +
                         std::to_string(d) +
                         " entities, more than MOAB's tag_get_data can be "
                         "asked for in one call"};
      }
      rval = mb.tag_get_data(gid_tag, dd.handles.data(), static_cast<int>(n),
                             raw.data());
      if (rval != moab::MB_SUCCESS) {
        return moab_error("E-MOAB-GID",
                          "could not read GLOBAL_ID for dimension-" +
                              std::to_string(d) + " entities",
                          rval);
      }
    }
    std::size_t unset = 0;
    for (const int id : raw) {
      if (id <= 0) ++unset;
    }
    if (unset == n) {
      if (parallel()) {
        // Assigning them needs collective communication, and inventing a
        // per-rank numbering would make an owned entity and its ghost copies
        // disagree, which is precisely what a global id exists to prevent.
        dd.gid_error =
            "this mesh has no GLOBAL_ID for its " + std::to_string(n) +
            " dimension-" + std::to_string(d) +
            " entities and the read is partitioned, so no correct id can be "
            "invented locally; assign them collectively before loading "
            "(mbconvert, or ParallelComm::assign_global_ids)";
      } else {
        // One rank, so the local order is a valid global numbering.
        for (std::size_t i = 0; i < n; ++i) raw[i] = static_cast<int>(i) + 1;
      }
    } else if (unset != 0) {
      dd.gid_error =
          "this mesh numbers only some of its dimension-" + std::to_string(d) +
          " entities: " + std::to_string(n - unset) + " of " +
          std::to_string(n) +
          " carry a positive GLOBAL_ID. A partially populated numbering "
          "cannot be completed without discarding the part that is right";
    }
    if (dd.gid_error.empty()) {
      std::vector<int> sorted = raw;
      std::sort(sorted.begin(), sorted.end());
      const auto dup = std::adjacent_find(sorted.begin(), sorted.end());
      if (dup != sorted.end()) {
        dd.gid_error = "two dimension-" + std::to_string(d) +
                       " entities on this rank share GLOBAL_ID " +
                       std::to_string(*dup) +
                       "; a global id must identify an entity uniquely";
      }
    }
    if (dd.gid_error.empty()) {
      dd.global_ids.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        dd.global_ids[i] = static_cast<std::size_t>(raw[i]);
      }
    }
  }
  view.set_top_dim(top);
  return Result<void>::success();
}

// --- public surface ---------------------------------------------------------

Result<void> MoabMeshBackend::load(const MeshDecl& decl) {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED", "load() on a moved-from MoabMeshBackend"};
  }
  moab::EntityHandle file_set = 0;
  const moab::ErrorCode rval =
      impl_->mb.create_meshset(moab::MESHSET_SET, file_set);
  if (rval != moab::MB_SUCCESS) {
    return impl_->moab_error("E-MOAB-CREATE",
                             "could not create a file set for mesh '" +
                                 decl.name + "'",
                             rval);
  }

  constexpr std::string_view kPrefix = "generated:";
  Result<void> built =
      decl.source.rfind(kPrefix, 0) == 0
          ? impl_->generate(decl.source.substr(kPrefix.size()), file_set)
          : impl_->read_file(decl.source, file_set);
  if (!built.ok()) {
    impl_->discard(file_set);
    return built;
  }

  auto view = std::make_unique<MoabMeshView>(impl_->mb, file_set);
  const auto gathered = impl_->gather(*view);
  if (!gathered.ok()) {
    impl_->discard(file_set);
    return gathered;
  }

  // Both pushes happen only on the success path and in this order, so
  // meshes[i] is always the mesh named names[i] and a failed load consumes
  // no MeshId.
  impl_->meshes.push_back(std::move(view));
  impl_->names.push_back(decl.name);
  return Result<void>::success();
}

Result<MeshId> MoabMeshBackend::mesh_id(std::string_view name) const {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED", "mesh_id() on a moved-from MoabMeshBackend"};
  }
  if (name.empty()) {
    return Failure{"E-MESH-NAME",
                 "the empty mesh name matches nothing; name the mesh in its "
                 "MeshDecl"};
  }
  std::size_t found = 0;
  std::size_t first = 0;
  for (std::size_t i = 0; i < impl_->names.size(); ++i) {
    if (impl_->names[i] == name) {
      if (found == 0) first = i;
      ++found;
    }
  }
  if (found == 0) {
    return Failure{"E-MESH-NAME", "no loaded mesh is named '" +
                                    std::string{name} + "' (" +
                                    std::to_string(impl_->names.size()) +
                                    " loaded)"};
  }
  if (found > 1) {
    return Failure{"E-MESH-NAME-AMBIGUOUS",
                 std::to_string(found) + " loaded meshes are named '" +
                     std::string{name} +
                     "'; the name cannot identify one of them"};
  }
  return MeshId{static_cast<std::uint32_t>(first)};
}

Result<const MeshView*> MoabMeshBackend::view(MeshId id) const {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED", "view() on a moved-from MoabMeshBackend"};
  }
  if (!id.valid() || id.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID", "no mesh with id " + std::to_string(id.value()) +
                                  " (" + std::to_string(impl_->meshes.size()) +
                                  " loaded)"};
  }
  return static_cast<const MeshView*>(impl_->meshes[id.value()].get());
}

Result<MoabNativeMesh> MoabMeshBackend::native_mesh(MeshId id) const {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED",
                 "native_mesh() on a moved-from MoabMeshBackend"};
  }
  if (!id.valid() || id.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID", "no mesh with id " + std::to_string(id.value()) +
                                  " (" + std::to_string(impl_->meshes.size()) +
                                  " loaded)"};
  }
  // moab::EntityHandle is an unsigned integer type; static_assert rather than
  // trust it, because a narrowing conversion here would hand the remap builder
  // a truncated set handle and it would silently intersect the wrong thing.
  static_assert(sizeof(moab::EntityHandle) <= sizeof(std::uint64_t),
                "moab::EntityHandle does not fit in std::uint64_t");
  void* pc = nullptr;
#ifdef FUSE_MOAB_PARALLEL
  pc = static_cast<void*>(impl_->pcomm.get());
#endif
  return MoabNativeMesh{
      static_cast<void*>(&impl_->mb),
      static_cast<std::uint64_t>(impl_->meshes[id.value()]->file_set()), pc};
}

Result<moab::EntityHandle> MoabMeshBackend::Impl::find_named_set(
    const MoabMeshView& view, const std::string& set_name) const {
  moab::Interface& m = mb;
  moab::Range sets;
  moab::ErrorCode rval = m.get_entities_by_type(view.file_set(),
                                                moab::MBENTITYSET, sets,
                                                /*recursive=*/true);
  if (rval != moab::MB_SUCCESS) {
    return Failure{"E-MOAB-QUERY",
                 "could not list this mesh's entity sets: " +
                     moab_text(m, rval)};
  }

  // 1. An exact NAME tag match.
  moab::Tag name_tag = nullptr;
  rval = m.tag_get_handle(NAME_TAG_NAME, NAME_TAG_SIZE, moab::MB_TYPE_OPAQUE,
                          name_tag);
  if (rval == moab::MB_SUCCESS) {
    std::vector<moab::EntityHandle> hits;
    char buf[NAME_TAG_SIZE + 1] = {};
    for (const moab::EntityHandle s : sets) {
      std::memset(buf, 0, sizeof(buf));
      const moab::ErrorCode got = m.tag_get_data(name_tag, &s, 1, buf);
      if (got == moab::MB_TAG_NOT_FOUND) continue;
      if (got != moab::MB_SUCCESS) {
        return Failure{"E-MOAB-TAG",
                     "could not read a NAME tag: " + moab_text(m, got)};
      }
      // The stored value is zero-padded char[NAME_TAG_SIZE], not necessarily
      // NUL-terminated when the name fills it, hence the extra byte above.
      if (set_name == std::string{buf}) hits.push_back(s);
    }
    if (hits.size() > 1) {
      return Failure{"E-MESH-SET",
                   std::to_string(hits.size()) + " entity sets are named '" +
                       set_name + "'; the name cannot identify one of them"};
    }
    if (hits.size() == 1) return hits.front();
  } else if (rval != moab::MB_TAG_NOT_FOUND) {
    return Failure{"E-MOAB-TAG",
                 "could not look up the NAME tag: " + moab_text(m, rval)};
  }

  // 2. "<prefix>_<n>" against the conventional integer-valued set tags.
  for (const TaggedSetPrefix& p : kTaggedSetPrefixes) {
    int want = 0;
    if (!match_tagged_set_name(set_name, p.prefix, want)) continue;
    moab::Tag tag = nullptr;
    const moab::ErrorCode got =
        m.tag_get_handle(p.tag_name, 1, moab::MB_TYPE_INTEGER, tag);
    if (got == moab::MB_TAG_NOT_FOUND) {
      return Failure{"E-MESH-SET", "this mesh defines no " +
                                     std::string{p.tag_name} +
                                     ", so '" + set_name + "' names nothing"};
    }
    if (got != moab::MB_SUCCESS) {
      return Failure{"E-MOAB-TAG", "could not look up " +
                                     std::string{p.tag_name} + ": " +
                                     moab_text(m, got)};
    }
    std::vector<moab::EntityHandle> hits;
    for (const moab::EntityHandle s : sets) {
      int value = 0;
      const moab::ErrorCode read = m.tag_get_data(tag, &s, 1, &value);
      // A sparse tag with a default reports MB_SUCCESS and the DEFAULT for a
      // set that does not carry it, so a match on the default value would
      // sweep in every untagged set. Only an explicitly tagged set counts.
      if (read == moab::MB_TAG_NOT_FOUND) continue;
      if (read != moab::MB_SUCCESS) {
        return Failure{"E-MOAB-TAG",
                     "could not read " + std::string{p.tag_name} + ": " +
                         moab_text(m, read)};
      }
      if (value == want) hits.push_back(s);
    }
    if (hits.size() > 1) {
      return Failure{"E-MESH-SET",
                   std::to_string(hits.size()) + " entity sets carry " +
                       std::string{p.tag_name} + " == " +
                       std::to_string(want)};
    }
    if (hits.size() == 1) return hits.front();
    return Failure{"E-MESH-SET", "no entity set carries " +
                                   std::string{p.tag_name} + " == " +
                                   std::to_string(want) + " (from '" +
                                   set_name + "')"};
  }

  return Failure{"E-MESH-SET",
               "unknown entity set '" + set_name +
                   "': no set carries that NAME, and it is not of the form "
                   "sideset_<n>, neumann_<n>, nodeset_<n>, dirichlet_<n>, "
                   "block_<n> or material_<n>"};
}

Result<MoabMeshBackend::Impl::Resolved> MoabMeshBackend::Impl::resolve(
    const Region& r) const {
  if (!r.mesh.valid() || r.mesh.value() >= meshes.size()) {
    return Failure{"E-MESH-ID",
                 "region '" + r.name + "' names mesh id " +
                     std::to_string(r.mesh.value()) + ", of which " +
                     std::to_string(meshes.size()) + " are loaded"};
  }
  const MoabMeshView& view = *meshes[r.mesh.value()];
  if (r.dimension < -1) {
    return Failure{"E-MESH-DIM",
                 "region '" + r.name + "' has dimension " +
                     std::to_string(r.dimension) +
                     "; only -1 (meaning the mesh dimension) or a "
                     "non-negative dimension is meaningful"};
  }
  if (r.dimension > 3) {
    return Failure{"E-MESH-DIM",
                 "region '" + r.name + "' asks for dimension " +
                     std::to_string(r.dimension) +
                     "; MOAB represents dimensions 0 through 3"};
  }

  if (r.entity_set.empty() || r.entity_set == "all") {
    const int dim = r.dimension >= 0 ? r.dimension : view.dimension();
    if (dim > view.dimension()) {
      return Failure{"E-MESH-DIM",
                   "region '" + r.name + "' asks for dimension " +
                       std::to_string(dim) +
                       ", above this mesh's dimension " +
                       std::to_string(view.dimension())};
    }
    return Resolved{&view, view.file_set(), dim};
  }

  const auto set = find_named_set(view, r.entity_set);
  if (!set.ok()) return set.error();

  moab::Interface& m = mb;
  std::array<int, 4> counts{0, 0, 0, 0};
  for (int d = 0; d <= 3; ++d) {
    const moab::ErrorCode rval = m.get_number_entities_by_dimension(
        set.value(), d, counts[static_cast<std::size_t>(d)], /*recursive=*/true);
    if (rval != moab::MB_SUCCESS) {
      return Failure{"E-MOAB-QUERY",
                   "could not count entity set '" + r.entity_set +
                       "': " + moab_text(m, rval)};
    }
  }
  int highest = -1;
  for (int d = 3; d >= 0; --d) {
    if (counts[static_cast<std::size_t>(d)] > 0) {
      highest = d;
      break;
    }
  }
  if (r.dimension < 0) {
    // A locally empty set (a partition where this rank holds none of it) has
    // no dimension of its own; fall back to the mesh's so the answer is a
    // well-defined 0 rather than a guess.
    return Resolved{&view, set.value(),
                    highest >= 0 ? highest : view.dimension()};
  }
  if (highest >= 0 && counts[static_cast<std::size_t>(r.dimension)] == 0) {
    // Silently answering 0 would hand back an empty buffer for a region the
    // author believes is populated.
    return Failure{"E-MESH-DIM",
                 "entity set '" + r.entity_set + "' on region '" + r.name +
                     "' holds dimension-" + std::to_string(highest) +
                     " entities, but the region states dimension " +
                     std::to_string(r.dimension)};
  }
  return Resolved{&view, set.value(), r.dimension};
}

Result<std::size_t> MoabMeshBackend::region_size(const Region& r) const {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED", "region_size() on a moved-from "
                                 "MoabMeshBackend"};
  }
  const auto res = impl_->resolve(r);
  if (!res.ok()) return res.error();
  if (res.value().set == res.value().view->file_set()) {
    return res.value().view->entity_count(res.value().dim);
  }
  moab::Interface& m = impl_->mb;
  int n = 0;
  const moab::ErrorCode rval = m.get_number_entities_by_dimension(
      res.value().set, res.value().dim, n, /*recursive=*/true);
  if (rval != moab::MB_SUCCESS) {
    return Failure{"E-MOAB-QUERY", "could not size region '" + r.name +
                                     "': " + moab_text(m, rval)};
  }
  return static_cast<std::size_t>(n);
}

Result<std::vector<std::size_t>> MoabMeshBackend::owned_entities(
    const Region& r) const {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED",
                 "owned_entities() on a moved-from MoabMeshBackend"};
  }
  const auto res = impl_->resolve(r);
  if (!res.ok()) return res.error();
  const MoabMeshView& view = *res.value().view;
  const int dim = res.value().dim;
  const DimData& dd = view.dim_data(dim);

  std::vector<std::size_t> out;
  if (res.value().set == view.file_set()) {
    out.reserve(dd.owned_count);
    for (std::size_t i = 0; i < dd.handles.size(); ++i) {
      if (dd.owned[i] != 0) out.push_back(i);
    }
    return out;
  }

  moab::Interface& m = impl_->mb;
  moab::Range ents;
  const moab::ErrorCode rval = m.get_entities_by_dimension(
      res.value().set, dim, ents, /*recursive=*/true);
  if (rval != moab::MB_SUCCESS) {
    return Failure{"E-MOAB-QUERY", "could not list region '" + r.name +
                                     "': " + moab_text(m, rval)};
  }
  out.reserve(ents.size());
  for (const moab::EntityHandle h : ents) {
    const std::size_t i = view.index_of(dim, h);
    if (i == MoabMeshView::kNoIndex) {
      // The region would silently be short by one entity. That is a broken
      // mesh, not a smaller region.
      return Failure{"E-MESH-SET",
                   "entity set '" + r.entity_set + "' contains entity " +
                       std::to_string(h) +
                       ", which is not among this mesh's dimension-" +
                       std::to_string(dim) + " entities"};
    }
    if (dd.owned[i] != 0) out.push_back(i);
  }
  return out;  // ascending: a moab::Range iterates in handle order
}

Result<int> MoabMeshBackend::partition_rank_of(const Region& r,
                                               std::size_t local_index) const {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED",
                 "partition_rank_of() on a moved-from MoabMeshBackend"};
  }
  const auto res = impl_->resolve(r);
  if (!res.ok()) return res.error();
  const MoabMeshView& view = *res.value().view;
  const int dim = res.value().dim;
  const DimData& dd = view.dim_data(dim);
  if (local_index >= dd.handles.size()) {
    return Failure{"E-MESH-INDEX",
                 "local index " + std::to_string(local_index) +
                     " is outside [0, " + std::to_string(dd.handles.size()) +
                     ") for dimension " + std::to_string(dim) +
                     " of region '" + r.name + "' on this rank"};
  }
#ifdef FUSE_MOAB_PARALLEL
  if (impl_->pcomm != nullptr) {
    int owner = -1;
    const moab::ErrorCode rval =
        impl_->pcomm->get_owner(dd.handles[local_index], owner);
    if (rval != moab::MB_SUCCESS) {
      return Failure{"E-MOAB-PSTATUS",
                   "could not determine the owner of entity " +
                       std::to_string(dd.handles[local_index]) + ": " +
                       moab_text(impl_->mb, rval)};
    }
    return owner;
  }
#endif
  // No communicator: the mesh was read serially, so the effective
  // communicator is MPI_COMM_SELF and its only rank is 0.
  return 0;
}

int MoabMeshBackend::communicator_size() const noexcept {
  if (impl_ == nullptr) return 1;
#ifdef FUSE_MOAB_PARALLEL
  if (impl_->pcomm == nullptr) return 1;
  const int n = static_cast<int>(impl_->pcomm->proc_config().proc_size());
  return n > 0 ? n : 1;
#else
  // No MPI in this build, so nothing could have partitioned the read.
  return 1;
#endif
}

// ===========================================================================
// The remap halo (R3). See moab_backend.hpp for the contract.
// ===========================================================================

Result<void> MoabMeshBackend::Impl::regather_with_halo(std::size_t mesh,
                                                       int dim) {
  MoabMeshView& view = *meshes[mesh];
  const Result<void> g = gather(view);
  if (!g.ok()) return g;

  const HaloState* h = halo(mesh, dim);
  if (h == nullptr || h->cells.empty()) return Result<void>::success();

  DimData& dd = view.dim_data_mut(dim);
  dd.owned_count = 0;
  for (std::size_t i = 0; i < dd.handles.size(); ++i) {
    if (h->cells.count(dd.handles[i]) != 0) {
      dd.owned[i] = 0;
    }
    if (dd.owned[i] != 0) ++dd.owned_count;
  }
  return Result<void>::success();
}

#ifdef FUSE_MOAB_PARALLEL
namespace {

/// Reconcile a per-rank yes/no outcome across the communicator before the
/// next collective, so a data-dependent local refusal -- an inconsistent
/// global id, an owner contradiction (REVIEW.md R-02) -- produces the SAME
/// Failure on every rank instead of letting the ranks that did not hit it
/// walk into the next `MPI_Alltoall(v)` and hang waiting for the one that
/// did. `local_failed`/`local_message` describe only this rank's outcome;
/// a rank that did not fail passes an empty message. Mirrors the
/// reconciliation `intx_weights.cpp` performs for
/// `resolve()`/`fit_surface()`/`orient()` -- same finding, same fix, kept as
/// a separate copy here rather than a shared header because no MOAB/MPI type
/// may cross into a header outside its own backend translation unit (I7).
[[nodiscard]] Result<void> reconcile_halo(MPI_Comm comm, bool local_failed,
                                          const std::string& local_message) {
  int nproc = 1, rank = 0;
  // REVIEW.md N-BACKEND-3: the sibling reconcile<T> (intx_weights.cpp)
  // refuses rather than silently falling through when it cannot size its
  // communicator; this one used to ignore both return values, so a
  // corrupted/mismatched communicator -- exactly the class E-TRANSPORT-
  // PEERMATCH exists elsewhere to catch -- would leave nproc at its
  // initialiser value of 1 and take the "no peers to disagree with" path on
  // the one rank most likely to need reconciling (I4: no silent fallback).
  if (MPI_Comm_size(comm, &nproc) != MPI_SUCCESS ||
      MPI_Comm_rank(comm, &rank) != MPI_SUCCESS) {
    return Failure{"E-MESH-HALO",
                 "could not size or rank the communicator to reconcile a "
                 "per-rank halo-plan outcome; refusing rather than treating "
                 "it as having no peers to disagree with"};
  }
  if (nproc <= 1) {
    return local_failed ? Result<void>(Failure{"E-MESH-HALO", local_message})
                        : Result<void>::success();
  }
  const int lf = local_failed ? 1 : 0;
  int any = 0;
  if (MPI_Allreduce(&lf, &any, 1, MPI_INT, MPI_MAX, comm) != MPI_SUCCESS) {
    return Failure{"E-MESH-HALO",
                 "the collective reduction of a per-rank halo-plan outcome "
                 "failed, so this rank cannot know whether its peers agree; "
                 "refusing rather than proceeding out of step"};
  }
  if (any == 0) return Result<void>::success();

  const int candidate = local_failed ? rank : nproc;
  int failing_rank = nproc;
  if (MPI_Allreduce(&candidate, &failing_rank, 1, MPI_INT, MPI_MIN, comm) !=
          MPI_SUCCESS) {
    return Failure{"E-MESH-HALO",
                 "the collective reduction of a per-rank halo-plan outcome "
                 "failed, so this rank cannot know whether its peers agree; "
                 "refusing rather than proceeding out of step"};
  }
  std::string msg = (rank == failing_rank) ? local_message : std::string();
  int len = static_cast<int>(msg.size());
  if (MPI_Bcast(&len, 1, MPI_INT, failing_rank, comm) != MPI_SUCCESS) {
    return Failure{"E-MESH-HALO",
                 "the broadcast of the failing rank's message failed"};
  }
  msg.resize(static_cast<std::size_t>(len));
  if (len > 0 && MPI_Bcast(msg.data(), len, MPI_CHAR, failing_rank, comm) !=
                     MPI_SUCCESS) {
    return Failure{"E-MESH-HALO",
                 "the broadcast of the failing rank's message failed"};
  }
  return Failure{"E-MESH-HALO",
               "rank " + std::to_string(failing_rank) + " refused: " + msg};
}

}  // namespace
#endif  // FUSE_MOAB_PARALLEL

Result<void> MoabMeshBackend::Impl::plan_halo_exchange(std::size_t mesh,
                                                       int dim) {
#ifndef FUSE_MOAB_PARALLEL
  (void)mesh;
  (void)dim;
  return Result<void>::success();
#else
  HaloState& h = halos[{mesh, dim}];
  h.ex = HaloExchange{};
  if (pcomm == nullptr) return Result<void>::success();

  const MPI_Comm comm = pcomm->comm();
  int nproc = 1, me = 0;
  MPI_Comm_size(comm, &nproc);
  MPI_Comm_rank(comm, &me);
  const auto np = static_cast<std::size_t>(nproc);

  const MoabMeshView& view = *meshes[mesh];
  const DimData& dd = view.dim_data(dim);
  // REVIEW.md R-02: reconciled BEFORE the first collective below (the
  // MPI_Allgather a few lines down), not after -- this check needs no
  // per-rank data beyond what is already local at function entry, so there
  // is no reason to let a divergent refusal race a peer into that
  // collective.
  {
    const Result<void> r = reconcile_halo(
        comm, !dd.gid_error.empty(),
        "this mesh cannot carry a remap halo at dimension " +
            std::to_string(dim) +
            " because a halo cell is matched to its owner by global id and "
            "by nothing else: " + dd.gid_error);
    if (!r.ok()) return r.error();
  }

  // Owned entities, by global id, so an incoming request can be answered.
  std::unordered_map<std::size_t, std::size_t> owned_by_gid;
  owned_by_gid.reserve(dd.handles.size());
  std::vector<std::size_t> owned_gids;
  owned_gids.reserve(dd.handles.size());
  for (std::size_t i = 0; i < dd.handles.size(); ++i) {
    if (dd.owned[i] == 0) continue;
    owned_by_gid.emplace(dd.global_ids[i], i);
    owned_gids.push_back(dd.global_ids[i]);
  }

  // Who owns what. An allgather of every rank's owned ids -- O(total cells)
  // per rank, the same ceiling MpiTransport::build_plan has, and the reason
  // the header says so out loud. Setup only.
  std::vector<int> counts(np, 0);
  int mine = static_cast<int>(owned_gids.size());
  if (MPI_Allgather(&mine, 1, MPI_INT, counts.data(), 1, MPI_INT, comm) !=
      MPI_SUCCESS) {
    return Failure{"E-MESH-HALO", "MPI_Allgather of the owned-id counts failed"};
  }
  std::vector<int> displs(np + 1, 0);
  for (std::size_t r = 0; r < np; ++r) displs[r + 1] = displs[r] + counts[r];
  std::vector<long long> all(static_cast<std::size_t>(displs[np]), 0);
  std::vector<long long> mine_ll(owned_gids.begin(), owned_gids.end());
  if (MPI_Allgatherv(mine_ll.data(), mine, MPI_LONG_LONG, all.data(),
                     counts.data(), displs.data(), MPI_LONG_LONG,
                     comm) != MPI_SUCCESS) {
    return Failure{"E-MESH-HALO", "MPI_Allgatherv of the owned ids failed"};
  }
  std::unordered_map<long long, int> owner_of;
  owner_of.reserve(all.size());
  for (std::size_t r = 0; r < np; ++r) {
    for (int k = displs[r]; k < displs[r + 1]; ++k) {
      owner_of.emplace(all[static_cast<std::size_t>(k)], static_cast<int>(r));
    }
  }

  // Group this rank's halo slots by the rank that owns them.
  // REVIEW.md R-02: both refusals below depend on THIS RANK's own global-id
  // scan, and can therefore fire on one rank without firing on its peers.
  // The loop no longer returns from inside itself -- it records the first
  // local problem and reconciles it, once, before the MPI_Alltoall further
  // down -- so a rank hitting either condition still reaches the same
  // verdict as every other rank instead of racing ahead of it.
  std::vector<std::vector<std::size_t>> want(np);
  std::string local_error;
  for (std::size_t i = 0; i < dd.handles.size(); ++i) {
    if (dd.owned[i] != 0) continue;
    const auto it = owner_of.find(static_cast<long long>(dd.global_ids[i]));
    if (it == owner_of.end()) {
      local_error = "local entity " + std::to_string(i) + " of dimension " +
                    std::to_string(dim) + " has global id " +
                    std::to_string(dd.global_ids[i]) +
                    ", which no rank claims to own; the mesh's global "
                    "numbering is inconsistent across the communicator";
      break;
    }
    if (it->second == me) {
      // A non-owned entity whose owner is this rank is a contradiction: it
      // would be double counted in every reduction. Refuse rather than
      // quietly fetch a value from ourselves.
      local_error = "local entity " + std::to_string(i) +
                    " is marked not owned yet its global id " +
                    std::to_string(dd.global_ids[i]) + " is owned by this rank";
      break;
    }
    want[static_cast<std::size_t>(it->second)].push_back(i);
  }
  {
    const Result<void> r = reconcile_halo(comm, !local_error.empty(),
                                          local_error);
    if (!r.ok()) return r.error();
  }

  HaloExchange& ex = h.ex;
  ex.req_counts.assign(np, 0);
  ex.req_displs.assign(np + 1, 0);
  std::vector<long long> req_gids;
  for (std::size_t r = 0; r < np; ++r) {
    ex.req_counts[r] = static_cast<int>(want[r].size());
    for (const std::size_t i : want[r]) {
      ex.fill_index.push_back(i);
      req_gids.push_back(static_cast<long long>(dd.global_ids[i]));
    }
  }
  for (std::size_t r = 0; r < np; ++r) {
    ex.req_displs[r + 1] = ex.req_displs[r] + ex.req_counts[r];
  }

  // Tell each owner what we want, and learn what we are asked for.
  ex.ans_counts.assign(np, 0);
  ex.ans_displs.assign(np + 1, 0);
  if (MPI_Alltoall(ex.req_counts.data(), 1, MPI_INT, ex.ans_counts.data(), 1,
                   MPI_INT, comm) != MPI_SUCCESS) {
    return Failure{"E-MESH-HALO", "MPI_Alltoall of the request counts failed"};
  }
  for (std::size_t r = 0; r < np; ++r) {
    ex.ans_displs[r + 1] = ex.ans_displs[r] + ex.ans_counts[r];
  }
  std::vector<long long> ans_gids(
      static_cast<std::size_t>(ex.ans_displs[np]), 0);
  if (MPI_Alltoallv(req_gids.data(), ex.req_counts.data(),
                    ex.req_displs.data(), MPI_LONG_LONG, ans_gids.data(),
                    ex.ans_counts.data(), ex.ans_displs.data(), MPI_LONG_LONG,
                    comm) != MPI_SUCCESS) {
    return Failure{"E-MESH-HALO", "MPI_Alltoallv of the requested ids failed"};
  }
  // REVIEW.md N-BACKEND-2: this is a fourth per-rank-local refusal drawn
  // from this rank's own `owned_by_gid` map, the same shape R-02 fixed for
  // the two `want`-loop refusals above -- but it sits AFTER both collective
  // pairs in this function, so it does not itself race into a further
  // collective here. It still broke the function's own "every rank returns
  // the IDENTICAL Failure" contract that R-01/R-02 established everywhere
  // else in this file, which a caller upstream may rely on. Recorded and
  // reconciled the same way rather than returned from inside the loop.
  ex.send_index.resize(ans_gids.size());
  std::string send_error;
  for (std::size_t k = 0; k < ans_gids.size(); ++k) {
    const auto it =
        owned_by_gid.find(static_cast<std::size_t>(ans_gids[k]));
    if (it == owned_by_gid.end()) {
      send_error = "another rank asked this one for global id " +
                   std::to_string(ans_gids[k]) +
                   " at dimension " + std::to_string(dim) +
                   ", which this rank does not own";
      break;
    }
    ex.send_index[k] = it->second;
  }
  {
    const Result<void> r = reconcile_halo(comm, !send_error.empty(),
                                          send_error);
    if (!r.ok()) return r.error();
  }
  return Result<void>::success();
#endif
}

Result<void> MoabMeshBackend::Impl::run_halo_exchange(std::size_t mesh,
                                                      int dim, double* buffer,
                                                      int components,
                                                      bool gather_dir) const {
#ifndef FUSE_MOAB_PARALLEL
  (void)mesh;
  (void)dim;
  (void)buffer;
  (void)components;
  (void)gather_dir;
  return Result<void>::success();
#else
  const HaloState* h = halo(mesh, dim);
  if (h == nullptr || pcomm == nullptr) return Result<void>::success();
  const HaloExchange& ex = h->ex;
  if (ex.req_counts.empty()) return Result<void>::success();

  const MPI_Comm comm = pcomm->comm();
  const auto nc = static_cast<std::size_t>(components);
  const auto np = ex.req_counts.size();

  // Counts in doubles rather than in entities.
  std::vector<int> sc(np), sd(np + 1, 0), rc(np), rd(np + 1, 0);
  const std::vector<int>& from = gather_dir ? ex.ans_counts : ex.req_counts;
  const std::vector<int>& to = gather_dir ? ex.req_counts : ex.ans_counts;
  for (std::size_t r = 0; r < np; ++r) {
    sc[r] = static_cast<int>(from[r] * components);
    rc[r] = static_cast<int>(to[r] * components);
  }
  for (std::size_t r = 0; r < np; ++r) {
    sd[r + 1] = sd[r] + sc[r];
    rd[r + 1] = rd[r] + rc[r];
  }

  const std::vector<std::size_t>& pack_from =
      gather_dir ? ex.send_index : ex.fill_index;
  const std::vector<std::size_t>& unpack_to =
      gather_dir ? ex.fill_index : ex.send_index;

  std::vector<double> out(pack_from.size() * nc, 0.0);
  for (std::size_t k = 0; k < pack_from.size(); ++k) {
    const double* s = buffer + pack_from[k] * nc;
    for (std::size_t c = 0; c < nc; ++c) out[k * nc + c] = s[c];
  }
  std::vector<double> in(unpack_to.size() * nc, 0.0);
  if (MPI_Alltoallv(out.data(), sc.data(), sd.data(), MPI_DOUBLE, in.data(),
                    rc.data(), rd.data(), MPI_DOUBLE, comm) != MPI_SUCCESS) {
    return Failure{"E-MESH-HALO", "MPI_Alltoallv of the halo values failed"};
  }
  for (std::size_t k = 0; k < unpack_to.size(); ++k) {
    double* d = buffer + unpack_to[k] * nc;
    for (std::size_t c = 0; c < nc; ++c) {
      if (gather_dir) {
        d[c] = in[k * nc + c];
      } else {
        d[c] += in[k * nc + c];
      }
    }
  }
  if (!gather_dir) {
    // The halo copies have been handed to their owners; leaving them in place
    // would let a second call add them again.
    for (const std::size_t i : ex.fill_index) {
      double* d = buffer + i * nc;
      for (std::size_t c = 0; c < nc; ++c) d[c] = 0.0;
    }
  }
  return Result<void>::success();
#endif
}

Result<std::size_t> MoabMeshBackend::adopt_remap_halo(MeshId source,
                                                      MeshId target, int dim,
                                                      const std::uint64_t* cells,
                                                      std::size_t count) {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED", "adopt_remap_halo() on a moved-from backend"};
  }
  if (!source.valid() || source.value() >= impl_->meshes.size() ||
      !target.valid() || target.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID", "adopt_remap_halo(): unknown mesh id"};
  }
  if (dim < 0 || dim > 3) {
    return Failure{"E-MESH-DIM", "adopt_remap_halo(): dimension " +
                                   std::to_string(dim) + " is not 0..3"};
  }
  const std::size_t s = source.value();
  const std::size_t t = target.value();
  const auto key = std::make_tuple(s, t, dim);
  const auto seen = impl_->halo_pairs.find(key);
  if (seen != impl_->halo_pairs.end()) return seen->second;

  if (!impl_->parallel() || count == 0) {
    // Nothing to adopt, but the PAIR is still recorded as built: zero halo
    // cells is a real answer and must not read as "never asked for".
    impl_->halo_pairs.emplace(key, std::size_t{0});
    const Result<void> p = impl_->plan_halo_exchange(s, dim);
    if (!p.ok()) return p.error();
    return std::size_t{0};
  }

  MoabMeshView& view = *impl_->meshes[s];
  const DimData& dd = view.dim_data(dim);
  if (!dd.gid_error.empty()) {
    return Failure{"E-MESH-HALO-GID",
                 "this mesh cannot carry a remap halo at dimension " +
                     std::to_string(dim) + ": " + dd.gid_error};
  }
  std::set<std::size_t> present(dd.global_ids.begin(), dd.global_ids.end());

  moab::Tag gid = nullptr;
  if (impl_->mb.tag_get_handle(GLOBAL_ID_TAG_NAME, 1, moab::MB_TYPE_INTEGER,
                               gid) != moab::MB_SUCCESS) {
    return Failure{"E-MESH-HALO",
                 "the mesh has no GLOBAL_ID tag, so a halo cell cannot be "
                 "matched to the cell it copies"};
  }

  Impl::HaloState& state = impl_->halos[{s, dim}];
  std::vector<moab::EntityHandle> keep;
  keep.reserve(count);
  for (std::size_t k = 0; k < count; ++k) {
    const auto h = static_cast<moab::EntityHandle>(cells[k]);
    int g = 0;
    if (impl_->mb.tag_get_data(gid, &h, 1, &g) != moab::MB_SUCCESS || g <= 0) {
      return Failure{"E-MESH-HALO",
                   "a covering cell carries no positive GLOBAL_ID, so it "
                   "cannot be matched to the source cell it copies"};
    }
    const auto gg = static_cast<std::size_t>(g);
    if (present.count(gg) != 0) continue;  // this rank already holds it
    present.insert(gg);                    // and never twice
    keep.push_back(h);
  }
  if (!keep.empty()) {
    if (impl_->mb.add_entities(view.file_set(), keep.data(),
                               static_cast<int>(keep.size())) !=
        moab::MB_SUCCESS) {
      return Failure{"E-MESH-HALO",
                   "MOAB refused to add " + std::to_string(keep.size()) +
                       " halo cells to the mesh's file set"};
    }
    state.cells.insert(keep.begin(), keep.end());
  }

  const Result<void> rg = impl_->regather_with_halo(s, dim);
  if (!rg.ok()) return rg.error();
  const Result<void> p = impl_->plan_halo_exchange(s, dim);
  if (!p.ok()) return p.error();

  impl_->halo_pairs.emplace(key, keep.size());
  return keep.size();
}

bool MoabMeshBackend::has_remap_halo(MeshId source, MeshId target,
                                     int dim) const noexcept {
  if (impl_ == nullptr) return false;
  return impl_->halo_pairs.find(std::make_tuple(
             source.value(), target.value(), dim)) != impl_->halo_pairs.end();
}

std::size_t MoabMeshBackend::remap_halo_size(MeshId source,
                                             int dim) const noexcept {
  if (impl_ == nullptr) return 0;
  const Impl::HaloState* h = impl_->halo(source.value(), dim);
  return h == nullptr ? 0 : h->cells.size();
}

namespace {

[[nodiscard]] Failure halo_arg_error(const double* buffer, int components) {
  if (buffer == nullptr) {
    return Failure{"E-MESH-ARG", "remap halo exchange: null buffer"};
  }
  return Failure{"E-MESH-ARG",
               "remap halo exchange: components must be >= 1, not " +
                   std::to_string(components)};
}

}  // namespace

Result<void> MoabMeshBackend::exchange_remap_halo(MeshId source, int dim,
                                                  double* buffer,
                                                  int components) const {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED",
                 "exchange_remap_halo() on a moved-from backend"};
  }
  if (!source.valid() || source.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID", "exchange_remap_halo(): unknown mesh id"};
  }
  if (dim < 0 || dim > 3) {
    return Failure{"E-MESH-DIM", "exchange_remap_halo(): dimension " +
                                   std::to_string(dim) + " is not 0..3"};
  }
  if (buffer == nullptr || components < 1) {
    return halo_arg_error(buffer, components);
  }
  return impl_->run_halo_exchange(source.value(), dim, buffer, components,
                                  /*gather_dir=*/true);
}

Result<void> MoabMeshBackend::accumulate_remap_halo(MeshId source, int dim,
                                                    double* buffer,
                                                    int components) const {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED",
                 "accumulate_remap_halo() on a moved-from backend"};
  }
  if (!source.valid() || source.value() >= impl_->meshes.size()) {
    return Failure{"E-MESH-ID", "accumulate_remap_halo(): unknown mesh id"};
  }
  if (dim < 0 || dim > 3) {
    return Failure{"E-MESH-DIM", "accumulate_remap_halo(): dimension " +
                                   std::to_string(dim) + " is not 0..3"};
  }
  if (buffer == nullptr || components < 1) {
    return halo_arg_error(buffer, components);
  }
  return impl_->run_halo_exchange(source.value(), dim, buffer, components,
                                  /*gather_dir=*/false);
}

Result<void> MoabMeshBackend::set_communicator(int fortran_comm_handle) {
  if (impl_ == nullptr) {
    return Failure{"E-MESH-MOVED",
                 "set_communicator() on a moved-from MoabMeshBackend"};
  }
#ifndef FUSE_MOAB_PARALLEL
  (void)fortran_comm_handle;
  return Failure{"E-MOAB-NOMPI",
               "this build has no MPI (FUSE_ENABLE_MPI was off, or MOAB was "
               "built without MPI), so a communicator cannot be adopted"};
#else
  if (!impl_->meshes.empty()) {
    return Failure{"E-MOAB-COMM",
                 "set_communicator() must be called before the first load(): " +
                     std::to_string(impl_->meshes.size()) +
                     " mesh(es) have already been read serially, and adopting "
                     "a communicator now would not repartition them"};
  }
  if (impl_->pcomm != nullptr) {
    return Failure{"E-MOAB-COMM",
                 "a communicator has already been set on this backend"};
  }
  int initialized = 0;
  if (MPI_Initialized(&initialized) != MPI_SUCCESS || initialized == 0) {
    return Failure{"E-MOAB-MPIINIT",
                 "MPI is not initialised; MPI_Comm_f2c on a handle from an "
                 "uninitialised MPI is undefined"};
  }
  const MPI_Comm given =
      MPI_Comm_f2c(static_cast<MPI_Fint>(fortran_comm_handle));
  if (given == MPI_COMM_NULL) {
    return Failure{"E-MOAB-COMM",
                 "the Fortran communicator handle " +
                     std::to_string(fortran_comm_handle) +
                     " converts to MPI_COMM_NULL"};
  }
  // Every reconciliation helper in this backend checks the return code of
  // its MPI calls and refuses on failure (invariant I4). Those checks are
  // only reachable if the communicator returns errors instead of aborting
  // inside the call -- and the caller's communicator carries whatever
  // handler the application set, MPI_ERRORS_ARE_FATAL by default. So the
  // backend works on its OWN duplicate with MPI_ERRORS_RETURN, the same
  // arrangement the MPI transport uses (mpi_transport.cpp). Until 2026-09-20
  // the given communicator was handed to MOAB directly, which made every
  // `!= MPI_SUCCESS` branch in this file dead in practice (pass-3 review,
  // N-BACKEND-1). MPI_Comm_dup is collective: every rank that constructs the
  // backend calls set_communicator, in the same order, which is the contract
  // documented on the declaration.
  MPI_Comm dup = MPI_COMM_NULL;
  int rc = MPI_Comm_dup(given, &dup);
  if (rc != MPI_SUCCESS) {
    return Failure{"E-MOAB-COMM", "MPI_Comm_dup failed with code " + std::to_string(rc)};
  }
  rc = MPI_Comm_set_errhandler(dup, MPI_ERRORS_RETURN);
  if (rc != MPI_SUCCESS) {
    MPI_Comm_free(&dup);
    return Failure{"E-MOAB-COMM",
                 "MPI_Comm_set_errhandler failed with code " + std::to_string(rc)};
  }
  impl_->owned_comm.comm = dup;
  impl_->pcomm = std::make_unique<moab::ParallelComm>(&impl_->mb, dup);
  return Result<void>::success();
#endif
}

}  // namespace fuse
