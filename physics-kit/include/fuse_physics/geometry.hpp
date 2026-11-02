#pragma once

/// Small, fixed-size spatial vectors and 3x3 matrices for finite-element
/// reference-cell geometry (physics-kit/fe/, physics-kit/spatial/).
///
/// WHY A NEW TYPE HERE (docs/dev/physics-kit-disposition.md, owner decision
/// D-A). KARMA's numerics/fe and numerics/spatial use Eigen fixed-size types
/// for this (upstream/karma/include/base/karma_geometry.h: Point = Vector3 =
/// Eigen::Matrix<Real,3,1>, Matrix3 = Eigen::Matrix<Real,3,3>). fuse has no
/// counterpart: fuse::linalg's public header names nothing third-party at
/// all (tools/check-layering.py's ALLOWED_PUBLIC_THIRD_PARTY["linalg"] is
/// empty, by design -- the facade exists so a backend is swappable), and its
/// DistVector/DistSparseMatrix/least_squares facade is sized for the
/// DISTRIBUTED coupled-iteration problem (thousands of rows), not a
/// per-element 1x1..3x3 local array evaluated inside a quadrature loop.
/// fuse::backend::native::Vec3 (include/fuse/backend/native/geometry.hpp) is
/// the closest existing fuse type -- also a plain {x,y,z} struct, no Eigen --
/// but it belongs to the native MESH BACKEND's own simplex-decomposition
/// code, not to a generic small-vector vocabulary physics-kit can depend on
/// without taking on a backend-specific include. This header is therefore a
/// thin physics-kit-own type, named here per D-A's instruction to name the
/// gap: a plain-struct Point/Vector3 and a row-major-free Matrix3, with the
/// handful of dense operations (determinant, inverse, symmetric eigenvalue
/// bounds) the ported FE core's geometry maps need, implemented natively so
/// that physics-kit/include/fuse_physics/ never names "Eigen/" (tools/check-
/// layering.py's FORBIDDEN_PUBLIC regex bans it in any public header, fuse::
/// or physics-kit's own, since S8.2).
///
/// SCOPE. Just enough linear algebra for a reference-cell map's Jacobian,
/// metric tensor, and inverse map: construction, componentwise arithmetic,
/// dot/cross/norm, and determinant/inverse/symmetric-eigenvalue routines for
/// the 1x1, 2x2 and 3x3 leading blocks a 1-D/2-D/3-D reference cell needs
/// (mirroring Eigen's topLeftCorner(dim,dim) pattern in the KARMA source,
/// since a Matrix3 here always represents a dim x dim Jacobian zero-padded
/// to 3x3, exactly as KARMA's own Matrix3::Zero() + topLeftCorner usage
/// does). This is not a general dense linear-algebra library and is not
/// meant to become one; a physics module that needs more belongs on
/// fuse::linalg's facade instead.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

namespace fuse::physics {

/// A point in 3-space, or (by the same representation) a displacement
/// vector. KARMA's own karma_geometry.h makes the identical choice and
/// states the reason: a point and a displacement have the same
/// representation and the same arithmetic, and distinguishing them in the
/// type system buys nothing here while forcing conversions at every
/// boundary.
struct Point {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;

  constexpr Point() = default;
  constexpr Point(double px, double py = 0.0, double pz = 0.0) : x(px), y(py), z(pz) {}

  /// Component access by index, matching the Eigen-ported source's p(0)/p(1)/p(2) spelling.
  [[nodiscard]] constexpr double operator()(unsigned i) const noexcept {
    return i == 0 ? x : i == 1 ? y : z;
  }
  [[nodiscard]] double& operator()(unsigned i) noexcept {
    return i == 0 ? x : i == 1 ? y : z;
  }

  [[nodiscard]] static constexpr Point zero() noexcept { return Point(0.0, 0.0, 0.0); }

  [[nodiscard]] constexpr bool operator==(const Point&) const = default;

  [[nodiscard]] bool all_finite() const noexcept {
    return std::isfinite(x) && std::isfinite(y) && std::isfinite(z);
  }
};

using Vector3 = Point;

[[nodiscard]] constexpr Point operator+(const Point& a, const Point& b) noexcept {
  return Point(a.x + b.x, a.y + b.y, a.z + b.z);
}
[[nodiscard]] constexpr Point operator-(const Point& a, const Point& b) noexcept {
  return Point(a.x - b.x, a.y - b.y, a.z - b.z);
}
[[nodiscard]] constexpr Point operator-(const Point& a) noexcept { return Point(-a.x, -a.y, -a.z); }
[[nodiscard]] constexpr Point operator*(double scale, const Point& a) noexcept {
  return Point(scale * a.x, scale * a.y, scale * a.z);
}
[[nodiscard]] constexpr Point operator*(const Point& a, double scale) noexcept { return scale * a; }
[[nodiscard]] constexpr Point operator/(const Point& a, double scale) noexcept {
  return Point(a.x / scale, a.y / scale, a.z / scale);
}
inline Point& operator+=(Point& a, const Point& b) noexcept { a = a + b; return a; }
inline Point& operator-=(Point& a, const Point& b) noexcept { a = a - b; return a; }

/// libMesh/Eigen spell this "a * b" for two vectors (RealGradient::operator*
/// or, in the KARMA port, a.dot(b)); spelled out here for the same reason
/// karma_geometry.h gives for naming it explicitly rather than overloading
/// operator*: a*b for two column vectors should not silently compile.
[[nodiscard]] constexpr double dot(const Point& a, const Point& b) noexcept {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
[[nodiscard]] constexpr Point cross(const Point& a, const Point& b) noexcept {
  return Point(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
[[nodiscard]] inline double norm(const Point& a) noexcept { return std::sqrt(dot(a, a)); }
[[nodiscard]] inline double distance(const Point& a, const Point& b) noexcept { return norm(a - b); }
[[nodiscard]] inline Point normalized(const Point& a) noexcept {
  const double n = norm(a);
  return n > 0.0 ? a / n : a;
}

/// A 3x3 matrix. col(j) is the j-th basis image, so matrix-vector and
/// matrix-matrix products follow Jacobian conventions (column j = image of
/// reference axis j). A 1-D or 2-D reference cell's Jacobian occupies the
/// leading dim columns/rows; the rest is zero, exactly as KARMA's own
/// Matrix3::Zero() + topLeftCorner(dim,dim) usage leaves it.
struct Matrix3 {
  std::array<std::array<double, 3>, 3> a{};  // a[row][col]

  [[nodiscard]] static constexpr Matrix3 zero() noexcept { return Matrix3{}; }

  [[nodiscard]] constexpr double operator()(unsigned row, unsigned col) const noexcept { return a[row][col]; }
  [[nodiscard]] double& operator()(unsigned row, unsigned col) noexcept { return a[row][col]; }

  [[nodiscard]] Point col(unsigned j) const noexcept { return Point(a[0][j], a[1][j], a[2][j]); }
  void set_col(unsigned j, const Point& value) noexcept {
    a[0][j] = value.x;
    a[1][j] = value.y;
    a[2][j] = value.z;
  }
};

[[nodiscard]] inline Matrix3 transpose(const Matrix3& m) noexcept {
  Matrix3 result;
  for (unsigned r = 0; r < 3; ++r)
    for (unsigned c = 0; c < 3; ++c) result(c, r) = m(r, c);
  return result;
}

[[nodiscard]] inline Matrix3 matmul(const Matrix3& lhs, const Matrix3& rhs) noexcept {
  Matrix3 result;
  for (unsigned r = 0; r < 3; ++r)
    for (unsigned c = 0; c < 3; ++c) {
      double sum = 0.0;
      for (unsigned k = 0; k < 3; ++k) sum += lhs(r, k) * rhs(k, c);
      result(r, c) = sum;
    }
  return result;
}

[[nodiscard]] inline Point matvec(const Matrix3& m, const Point& v) noexcept {
  return Point(m(0, 0) * v.x + m(0, 1) * v.y + m(0, 2) * v.z, m(1, 0) * v.x + m(1, 1) * v.y + m(1, 2) * v.z,
               m(2, 0) * v.x + m(2, 1) * v.y + m(2, 2) * v.z);
}

/// Determinant of the leading dim x dim block (dim in {1,2,3}), the rest of
/// the matrix ignored -- the native equivalent of Eigen's
/// m.topLeftCorner(dim,dim).determinant() call sites in the ported source.
[[nodiscard]] inline double block_determinant(const Matrix3& m, unsigned dim) noexcept {
  if (dim == 1) return m(0, 0);
  if (dim == 2) return m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);
  return m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0)) +
         m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0));
}

/// Inverse of the leading dim x dim block, zero-padded to 3x3 -- the native
/// equivalent of Eigen's m.topLeftCorner(dim,dim).inverse() call sites.
/// Caller is responsible for checking the determinant is not (numerically)
/// zero first, exactly as the ported source does before every call site.
[[nodiscard]] inline Matrix3 block_inverse(const Matrix3& m, unsigned dim) noexcept {
  Matrix3 result;
  const double det = block_determinant(m, dim);
  const double inv_det = 1.0 / det;
  if (dim == 1) {
    result(0, 0) = inv_det;
    return result;
  }
  if (dim == 2) {
    result(0, 0) = m(1, 1) * inv_det;
    result(0, 1) = -m(0, 1) * inv_det;
    result(1, 0) = -m(1, 0) * inv_det;
    result(1, 1) = m(0, 0) * inv_det;
    return result;
  }
  // 3x3 adjugate (cofactor transpose) over the determinant.
  result(0, 0) = (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1)) * inv_det;
  result(0, 1) = (m(0, 2) * m(2, 1) - m(0, 1) * m(2, 2)) * inv_det;
  result(0, 2) = (m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1)) * inv_det;
  result(1, 0) = (m(1, 2) * m(2, 0) - m(1, 0) * m(2, 2)) * inv_det;
  result(1, 1) = (m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0)) * inv_det;
  result(1, 2) = (m(0, 2) * m(1, 0) - m(0, 0) * m(1, 2)) * inv_det;
  result(2, 0) = (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0)) * inv_det;
  result(2, 1) = (m(0, 1) * m(2, 0) - m(0, 0) * m(2, 1)) * inv_det;
  result(2, 2) = (m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0)) * inv_det;
  return result;
}

/// Smallest and largest eigenvalues of a SYMMETRIC leading dim x dim block
/// (dim in {1,2,3}). Used only for a condition-number estimate in geometry
/// validation (ratio of largest to smallest), so only the bounds are
/// reported, not the eigenvectors.
///
/// dim == 2: the standard closed form from the trace and discriminant of a
/// symmetric 2x2 matrix (eigenvalues = (trace +/- sqrt(discriminant)) / 2,
/// discriminant = (a-d)^2 + 4bc for [[a,b],[b,d]]) -- the exact formula
/// KARMA's own update_affine_quad4_cache validation already uses inline for
/// its quad4 fast path (karma_geometry_map.C's "largest"/"smallest" local
/// variables).
///
/// dim == 3: the closed-form trigonometric solution for the eigenvalues of
/// a symmetric 3x3 matrix (Smith, O.K., "Eigenvalues of a symmetric 3x3
/// matrix", Communications of the ACM 4(4), 1961, p. 168; see also Kopp, J.,
/// "Efficient numerical diagonalization of hermitian 3x3 matrices",
/// Int. J. Mod. Phys. C 19 (2008), 523-548, section 2). Exact in exact
/// arithmetic; well-conditioned for the near-isotropic metrics a valid FE
/// Jacobian produces, which is the only regime geometry validation needs it
/// in.
[[nodiscard]] inline std::pair<double, double> symmetric_block_eigenvalue_bounds(const Matrix3& m,
                                                                                  unsigned dim) noexcept {
  if (dim == 1) return {m(0, 0), m(0, 0)};
  if (dim == 2) {
    const double trace = m(0, 0) + m(1, 1);
    const double discriminant = std::sqrt((m(0, 0) - m(1, 1)) * (m(0, 0) - m(1, 1)) + 4.0 * m(0, 1) * m(0, 1));
    const double largest = 0.5 * (trace + discriminant);
    const double smallest = 0.5 * (trace - discriminant);
    return {smallest, largest};
  }
  // Smith (1961) / Kopp (2008) closed-form symmetric 3x3 eigenvalues.
  const double p1 = m(0, 1) * m(0, 1) + m(0, 2) * m(0, 2) + m(1, 2) * m(1, 2);
  const double trace = m(0, 0) + m(1, 1) + m(2, 2);
  if (p1 == 0.0) {
    // Already diagonal.
    const double lo = std::min({m(0, 0), m(1, 1), m(2, 2)});
    const double hi = std::max({m(0, 0), m(1, 1), m(2, 2)});
    return {lo, hi};
  }
  const double q = trace / 3.0;
  const double p2 = (m(0, 0) - q) * (m(0, 0) - q) + (m(1, 1) - q) * (m(1, 1) - q) + (m(2, 2) - q) * (m(2, 2) - q) +
                     2.0 * p1;
  const double p = std::sqrt(p2 / 6.0);
  Matrix3 b;
  for (unsigned r = 0; r < 3; ++r)
    for (unsigned c = 0; c < 3; ++c) b(r, c) = (m(r, c) - (r == c ? q : 0.0)) / p;
  double detb = block_determinant(b, 3) / 2.0;
  detb = std::clamp(detb, -1.0, 1.0);
  const double phi = std::acos(detb) / 3.0;
  const double eig0 = q + 2.0 * p * std::cos(phi);
  const double eig2 = q + 2.0 * p * std::cos(phi + 2.0 * M_PI / 3.0);
  const double eig1 = trace - eig0 - eig2;  // trace is invariant.
  return {std::min({eig0, eig1, eig2}), std::max({eig0, eig1, eig2})};
}

}  // namespace fuse::physics
