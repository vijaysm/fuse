// test/partition/fixtures.hpp — synthetic centroid/weight/gid generators
// shared by the oracle-parity and mesh-adapter tests.
//
// RCB partitions on (centroid, weight, global id) alone -- nothing in
// fuse::partition's API ever names a cell type or connectivity. So a
// fixture "representing" triangles, hexahedra, or a polyhedral point cloud
// is, correctly, just a point distribution shaped like where such cells'
// centroids would fall: a structured grid for the axis-aligned cell
// families (line/quad/hex), an unstructured jittered/random cloud for the
// families whose centroids do not fall on a lattice (mixed/polyhedral/
// tet/prism/pyramid). This is not a simplification made to dodge writing a
// real mesh generator -- it is what "general over any cell type" MEANS for
// a partitioner whose whole interface is three doubles per element: two
// fixtures with the same centroid distribution are indistinguishable to
// this algorithm regardless of what cell type produced the centroids, and
// that is the property under test.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include "rcb_oracle.hpp"

namespace fuse::partition::testing::fixtures {

using OraclePoint3 = fuse::partition::testing::OraclePoint3;

struct PointSet {
  std::vector<OraclePoint3> centroids;
  std::vector<std::int64_t> gids;  // 0, 1, 2, ... by construction order
};

inline PointSet make_gids(std::vector<OraclePoint3> centroids) {
  PointSet ps;
  ps.centroids = std::move(centroids);
  ps.gids.resize(ps.centroids.size());
  for (std::size_t i = 0; i < ps.gids.size(); ++i) ps.gids[i] = static_cast<std::int64_t>(i);
  return ps;
}

/// 1D line: n points evenly spaced on [0, 1], y = z = 0.
inline PointSet line(std::size_t n) {
  std::vector<OraclePoint3> c(n);
  for (std::size_t i = 0; i < n; ++i) {
    c[i] = {static_cast<double>(i) / static_cast<double>(n == 1 ? 1 : n - 1), 0.0, 0.0};
  }
  return make_gids(std::move(c));
}

/// 2D planar, "mixed tri/quad": a structured n x n grid perturbed by a
/// deterministic (seed-free, index-based) jitter, so no two rows/columns of
/// centroids are collinear the way a pure quad grid's would be -- the
/// closest a centroid-only fixture can come to distinguishing "some cells
/// are triangles, some are quads" without an actual mesh.
inline PointSet planar_mixed(std::size_t n) {
  std::vector<OraclePoint3> c;
  c.reserve(n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      const double jitter_x = 0.15 * std::sin(3.1 * static_cast<double>(i + 1));
      const double jitter_y = 0.15 * std::cos(2.7 * static_cast<double>(j + 1));
      const double x = (static_cast<double>(i) + 0.5) / static_cast<double>(n) + jitter_x / n;
      const double y = (static_cast<double>(j) + 0.5) / static_cast<double>(n) + jitter_y / n;
      c.push_back({x, y, 0.0});
    }
  }
  return make_gids(std::move(c));
}

/// 2D manifold in 3D: a cubed-sphere-like point set on the unit sphere (6
/// faces, n x n points per face, equiangular within the face before
/// projection). Feed RAW (pre-gnomonic) to `rcb_partition` with
/// `CoordTransform::Gnomonic`; the oracle comparison must apply the exact
/// same projection first (rcb_oracle.hpp does not know about the
/// transform -- it is a pre-step, not part of the ported recursion).
inline PointSet cubed_sphere(std::size_t n) {
  std::vector<OraclePoint3> c;
  c.reserve(6 * n * n);
  for (int face = 0; face < 6; ++face) {
    for (std::size_t i = 0; i < n; ++i) {
      for (std::size_t j = 0; j < n; ++j) {
        const double xi = -1.0 + 2.0 * (static_cast<double>(i) + 0.5) / static_cast<double>(n);
        const double eta = -1.0 + 2.0 * (static_cast<double>(j) + 0.5) / static_cast<double>(n);
        double x, y, z;
        switch (face) {
          case 0: x = 1.0; y = xi; z = eta; break;
          case 1: x = -1.0; y = xi; z = eta; break;
          case 2: x = xi; y = 1.0; z = eta; break;
          case 3: x = xi; y = -1.0; z = eta; break;
          case 4: x = xi; y = eta; z = 1.0; break;
          default: x = xi; y = eta; z = -1.0; break;
        }
        const double r = std::sqrt(x * x + y * y + z * z);
        c.push_back({x / r, y / r, z / r});
      }
    }
  }
  return make_gids(std::move(c));
}

/// 3D structured hex box: n x n x n centroids in the unit cube.
inline PointSet hex_box(std::size_t n) {
  std::vector<OraclePoint3> c;
  c.reserve(n * n * n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      for (std::size_t k = 0; k < n; ++k) {
        c.push_back({(static_cast<double>(i) + 0.5) / static_cast<double>(n),
                    (static_cast<double>(j) + 0.5) / static_cast<double>(n),
                    (static_cast<double>(k) + 0.5) / static_cast<double>(n)});
      }
    }
  }
  return make_gids(std::move(c));
}

/// 3D unstructured point cloud in the unit cube -- stands in for a mixed
/// tet/prism/pyramid/polyhedral mesh's centroids: no two points share a
/// lattice, which a real unstructured mesh's cell centers would not either.
/// Deterministic (fixed LCG seed), so the test is reproducible without
/// `<random>`'s implementation-defined output.
inline PointSet unstructured_cloud(std::size_t n) {
  std::vector<OraclePoint3> c;
  c.reserve(n);
  std::uint64_t state = 0x9E3779B97F4A7C15ull;
  auto next = [&state]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return static_cast<double>(state % 1000000ull) / 1000000.0;
  };
  for (std::size_t i = 0; i < n; ++i) {
    c.push_back({next(), next(), next()});
  }
  return make_gids(std::move(c));
}

/// Uniform weight 1 for every element.
inline std::vector<OracleWeight> uniform_weights(std::size_t n) {
  return std::vector<OracleWeight>(n, 1);
}

/// Skewed weight: element i has weight (i % 7) + 1 -- deterministic, non-
/// monotonic-in-position skew so weight and geometric position are
/// decorrelated.
inline std::vector<OracleWeight> skewed_weights(std::size_t n) {
  std::vector<OracleWeight> w(n);
  for (std::size_t i = 0; i < n; ++i) w[i] = static_cast<OracleWeight>((i % 7) + 1);
  return w;
}

/// Every element weight 0 -- the "zero total weight" degenerate case.
inline std::vector<OracleWeight> zero_weights(std::size_t n) {
  return std::vector<OracleWeight>(n, 0);
}

/// Collapses every centroid onto the first -- the "all points identical"
/// degenerate case.
inline PointSet all_identical(std::size_t n) {
  PointSet ps = line(n);
  const OraclePoint3 p0 = ps.centroids.empty() ? OraclePoint3{0, 0, 0} : ps.centroids[0];
  for (auto& p : ps.centroids) p = p0;
  return ps;
}

}  // namespace fuse::partition::testing::fixtures
