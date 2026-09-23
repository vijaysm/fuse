#include "fuse/backend/native/geometry.hpp"

#include <algorithm>
#include <cstddef>

#include "fuse/base/error.hpp"

namespace fuse::native {

double triangle_area(const Vec3& a, const Vec3& b, const Vec3& c) {
  return 0.5 * norm(cross(b - a, c - a));
}

double tetrahedron_volume(const Tetrahedron& t) {
  return std::abs(dot(t.a - t.d, cross(t.b - t.d, t.c - t.d))) / 6.0;
}

Vec3 newell_vector_area(const std::vector<Vec3>& pts) {
  Vec3 sum{0.0, 0.0, 0.0};
  const std::size_t n = pts.size();
  for (std::size_t i = 0; i < n; ++i) {
    const Vec3& p = pts[i];
    const Vec3& q = pts[(i + 1) % n];
    const Vec3 c = cross(p, q);
    sum.x += c.x;
    sum.y += c.y;
    sum.z += c.z;
  }
  return {0.5 * sum.x, 0.5 * sum.y, 0.5 * sum.z};
}

namespace {

/// A length scale for `points`, used to turn the absolute area/volume
/// tolerance in the star-shapedness checks into something that scales with
/// the geometry rather than firing on a large cell's rounding noise or
/// missing a defect on a tiny one.
double bounding_scale(const std::vector<Vec3>& points) {
  double lo[3] = {points[0].x, points[0].y, points[0].z};
  double hi[3] = {points[0].x, points[0].y, points[0].z};
  for (const auto& p : points) {
    lo[0] = std::min(lo[0], p.x); hi[0] = std::max(hi[0], p.x);
    lo[1] = std::min(lo[1], p.y); hi[1] = std::max(hi[1], p.y);
    lo[2] = std::min(lo[2], p.z); hi[2] = std::max(hi[2], p.z);
  }
  const double dx = hi[0] - lo[0], dy = hi[1] - lo[1], dz = hi[2] - lo[2];
  double s = std::max({dx, dy, dz});
  return s > 0.0 ? s : 1.0;
}

}  // namespace

Result<std::vector<Triangle>> canonical_fan_triangulation(
    const std::vector<std::int64_t>& global_ids, const std::vector<Vec3>& points) {
  const std::size_t n = global_ids.size();
  if (points.size() != n) {
    return Failure{"E-MESH-POLYGON-DEGENERATE",
                   fmt::format("global_ids ({}) and points ({}) lengths disagree", n,
                               points.size())};
  }
  if (n < 3) {
    return Failure{"E-MESH-POLYGON-DEGENERATE",
                   fmt::format("a polygon face needs at least 3 corners, got {}", n)};
  }
  {
    std::vector<std::int64_t> sorted = global_ids;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      return Failure{"E-MESH-POLYGON-DEGENERATE",
                     "a polygon face repeats a vertex global id"};
    }
  }

  // The apex: the corner with the minimum global id. Rotate the loop so it
  // is at index 0 -- the SET of fan triangles this produces does not depend
  // on which of the two cyclic directions the loop is walked in afterwards
  // (see the header: the diagonal pattern from a fixed apex is
  // reversal-symmetric), so no direction convention needs to be agreed.
  std::size_t apex = 0;
  for (std::size_t i = 1; i < n; ++i) {
    if (global_ids[i] < global_ids[apex]) apex = i;
  }

  std::vector<int> rotated(n);
  for (std::size_t i = 0; i < n; ++i) rotated[i] = static_cast<int>((apex + i) % n);

  std::vector<Triangle> tris;
  tris.reserve(n - 2);
  double unsigned_sum = 0.0;
  for (std::size_t k = 1; k + 1 < n; ++k) {
    const int i0 = rotated[0], i1 = rotated[k], i2 = rotated[k + 1];
    tris.push_back({i0, i1, i2});
    unsigned_sum += triangle_area(points[static_cast<std::size_t>(i0)],
                                  points[static_cast<std::size_t>(i1)],
                                  points[static_cast<std::size_t>(i2)]);
  }

  const double true_area = norm(newell_vector_area(points));
  const double scale = bounding_scale(points);
  const double tol = 1e-9 * scale * scale;  // area scale ~ length^2
  if (unsigned_sum - true_area > tol) {
    return Failure{
        "E-MESH-POLYGON-NOT-STAR-SHAPED",
        fmt::format("the fan from the minimum-global-id corner (gid {}) is not a valid "
                   "decomposition of this {}-gon: its unsigned triangle-area sum {:.17g} "
                   "exceeds the polygon's true area {:.17g} by more than tolerance -- the "
                   "polygon is not star-shaped about that corner",
                   global_ids[apex], n, unsigned_sum, true_area)};
  }
  return tris;
}

Result<std::vector<Tetrahedron>> canonical_cell_tetrahedra(
    const std::vector<OrientedFace>& faces, const Vec3& centroid) {
  std::vector<Tetrahedron> tets;
  double unsigned_sum = 0.0;
  double signed_sum_from_origin = 0.0;
  double scale = 1.0;
  bool have_scale = false;

  for (const auto& face : faces) {
    auto tri_result = canonical_fan_triangulation(face.global_ids, face.points);
    if (!tri_result.ok()) return tri_result.error();
    const auto& tris = tri_result.value();

    if (!have_scale) {
      scale = bounding_scale(face.points);
      have_scale = true;
    }

    for (const auto& tri : tris) {
      const Vec3& a = face.points[static_cast<std::size_t>(tri[0])];
      const Vec3& b = face.points[static_cast<std::size_t>(tri[1])];
      const Vec3& c = face.points[static_cast<std::size_t>(tri[2])];

      // Divergence-theorem contribution of this outward-oriented triangle,
      // as a signed tetrahedron from the coordinate origin -- an identity
      // for ANY origin, so using (0,0,0) rather than `centroid` costs
      // nothing and lets this sum be computed before the star-shapedness
      // check it feeds.
      signed_sum_from_origin += dot(a, cross(b, c)) / 6.0;

      Tetrahedron t{centroid, a, b, c};
      tets.push_back(t);
      unsigned_sum += tetrahedron_volume(t);
    }
  }

  const double true_volume = std::abs(signed_sum_from_origin);
  const double tol = 1e-9 * scale * scale * scale;  // volume scale ~ length^3
  if (unsigned_sum - true_volume > tol) {
    return Failure{
        "E-MESH-CELL-NOT-STAR-SHAPED",
        fmt::format("fanning every face to the cell centroid is not a valid tetrahedral "
                   "decomposition: the unsigned tetrahedron-volume sum {:.17g} exceeds the "
                   "cell's true volume {:.17g} (from the closed-surface divergence "
                   "identity) by more than tolerance -- the cell is not star-shaped about "
                   "its own centroid",
                   unsigned_sum, true_volume)};
  }
  return tets;
}

}  // namespace fuse::native
