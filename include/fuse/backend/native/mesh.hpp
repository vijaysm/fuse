#pragma once

/// FUSE's native geometry: an unstructured simplicial mesh on Kokkos views.
///
/// The representation follows the model described in NVIDIA's PhysicsNeMo mesh
/// documentation, with three deliberate departures noted below. Two dimensions
/// characterise a mesh:
///
///   * the **manifold dimension** `M` -- the intrinsic dimension of a cell. A
///     triangle is 2, a tetrahedron 3, an edge 1, a point 0.
///   * the **spatial dimension** `S` -- the dimension of the space the vertex
///     coordinates live in.
///
/// Their difference is the **codimension**, and it decides which geometric
/// operations are even well defined. A codimension-0 mesh (tetrahedra in 3D,
/// triangles in 2D) fills a region of space and has no unique normal. A
/// codimension-1 mesh (triangles in 3D, edges in 2D) has exactly one normal
/// per cell. A codimension-2 mesh (edges in 3D) has infinitely many, so asking
/// for "the" normal is a category error rather than a hard case. Here that
/// rule is a `static_assert`: the codimension-0 mesh that asks for a normal
/// does not compile.
///
/// Common configurations, with the same naming as the reference:
///
///   | Type             | `Mesh<M,S>` | Codim | Use                          |
///   |------------------|-------------|-------|------------------------------|
///   | point cloud      | `Mesh<0,3>` | 3     | particles, query points      |
///   | curve in 3D      | `Mesh<1,3>` | 2     | fibres, 1D networks          |
///   | 2D triangulation | `Mesh<2,2>` | 0     | planar FEM/FV                |
///   | surface mesh     | `Mesh<2,3>` | 1     | CAD surfaces, wall fluxes    |
///   | volume mesh      | `Mesh<3,3>` | 0     | 3D FEM/FV                    |
///
/// WHY THE DIMENSIONS ARE TEMPLATE PARAMETERS. The trailing extent of every
/// view is then a compile-time constant, so a coordinate read is direct
/// indexing rather than a runtime stride multiply -- and coordinate reads are
/// the innermost operation of the supermesh clipping kernel. The cost is a
/// closed set of instantiations, which `native_view.hpp` provides for the
/// pairs that physically occur.
///
/// THREE DEPARTURES FROM THE REFERENCE MODEL.
///
///  1. **Cells are simplices, and they carry a parent.** The reference stores
///     cells as `(n_cells, M+1)`, which is a simplex, and needs nothing more.
///     FUSE does: conservation is accounted per *user* cell, and a hexahedral
///     mesh stored as tetrahedra would otherwise lose the identity that remap
///     weights are reported against. Every simplex therefore names the cell it
///     came from, and `parent_cell_count()` is the number a conservation sum
///     has to balance over -- not `simplex_count()`.
///
///  2. **Fields live outside the mesh.** The reference bundles point, cell and
///     global data into the mesh, which suits a model that consumes a mesh as
///     one object. FUSE already has a field layer that carries units, intent
///     and semantic shape (`fuse/field/`, `fuse/model/semantics.hpp`), and a
///     second home for a field would mean two places to look for one and two
///     places for units to disagree. The mesh carries geometry only.
///
///  3. **Derived quantities are computed once, not lazily cached.** The
///     reference computes centroids, measures and normals on demand and caches
///     them. A `const` method that fills a cache is a data race the moment two
///     Kokkos threads call it, and the whole point of this layer is to be used
///     from parallel kernels. They are computed in one pass by `finalize()`
///     and are read-only thereafter.

#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <type_traits>

#include "fuse/base/error.hpp"
#include "fuse/base/result.hpp"

namespace fuse::native {

/// Vertex coordinates, cell connectivity, and the quantities derived from
/// them. Construct, fill the views, call `finalize()`, then read.
template <int ManifoldDims, int SpatialDims>
class Mesh {
 public:
  static constexpr int kManifoldDims = ManifoldDims;
  static constexpr int kSpatialDims  = SpatialDims;
  static constexpr int kCodimension  = SpatialDims - ManifoldDims;
  /// A simplex of intrinsic dimension M has exactly M+1 corners.
  static constexpr int kVertsPerCell = ManifoldDims + 1;

  static_assert(ManifoldDims >= 0 && ManifoldDims <= 3,
                "a manifold dimension outside 0..3 has no cell type here");
  static_assert(SpatialDims >= 1 && SpatialDims <= 3,
                "coordinates live in 1, 2 or 3 dimensions");
  static_assert(kCodimension >= 0,
                "codimension is negative: a manifold cannot have higher "
                "intrinsic dimension than the space it is embedded in");

  /// True when each cell has exactly one normal direction. Codimension 0 fills
  /// space and has none; codimension 2 or more has infinitely many.
  static constexpr bool kHasUniqueNormal = (kCodimension == 1);

  using Points   = Kokkos::View<double* [SpatialDims]>;
  using Cells    = Kokkos::View<int* [kVertsPerCell]>;
  using Parents  = Kokkos::View<std::size_t*>;
  using Scalars  = Kokkos::View<double*>;
  using Vectors  = Kokkos::View<double* [SpatialDims]>;

  Mesh() = default;

  /// Allocates for `n_points` vertices and `n_simplices` simplices belonging
  /// to `n_parent_cells` user cells. Nothing is computed yet.
  Mesh(const std::string& name, std::size_t n_points, std::size_t n_simplices,
       std::size_t n_parent_cells)
      : name_(name),
        points_(Kokkos::view_alloc(name + ".points", Kokkos::WithoutInitializing),
                n_points),
        cells_(Kokkos::view_alloc(name + ".cells", Kokkos::WithoutInitializing),
               n_simplices),
        parent_(Kokkos::view_alloc(name + ".parent", Kokkos::WithoutInitializing),
                n_simplices),
        n_parent_cells_(n_parent_cells) {}

  [[nodiscard]] const std::string& name() const noexcept { return name_; }
  [[nodiscard]] std::size_t point_count() const noexcept {
    return points_.extent(0);
  }
  [[nodiscard]] std::size_t simplex_count() const noexcept {
    return cells_.extent(0);
  }
  /// The number of USER cells, which is what a conservation sum balances over.
  /// Equal to `simplex_count()` only for an already-simplicial mesh.
  [[nodiscard]] std::size_t parent_cell_count() const noexcept {
    return n_parent_cells_;
  }

  [[nodiscard]] Points&  points()  noexcept { return points_; }
  [[nodiscard]] Cells&   cells()   noexcept { return cells_; }
  [[nodiscard]] Parents& parents() noexcept { return parent_; }
  [[nodiscard]] const Points&  points()  const noexcept { return points_; }
  [[nodiscard]] const Cells&   cells()   const noexcept { return cells_; }
  [[nodiscard]] const Parents& parents() const noexcept { return parent_; }

  /// Validates the connectivity and computes every derived quantity in one
  /// pass. Refuses rather than computing from a bad index: an out-of-range
  /// vertex would otherwise read arbitrary memory and produce a measure that
  /// looks like a number.
  [[nodiscard]] Result<void> finalize() {
    const std::size_t np = point_count();
    for (std::size_t c = 0; c < simplex_count(); ++c) {
      for (int v = 0; v < kVertsPerCell; ++v) {
        const int idx = cells_(c, v);
        if (idx < 0 || static_cast<std::size_t>(idx) >= np) {
          return Failure{"E-MESH-CONNECTIVITY",
                         "simplex " + std::to_string(c) + " corner " +
                             std::to_string(v) + " refers to vertex " +
                             std::to_string(idx) + ", but the mesh has " +
                             std::to_string(np) + " vertices"};
        }
      }
      if (parent_(c) >= n_parent_cells_) {
        return Failure{"E-MESH-PARENT",
                       "simplex " + std::to_string(c) + " names parent cell " +
                           std::to_string(parent_(c)) + " of " +
                           std::to_string(n_parent_cells_) +
                           "; a conservation sum would silently omit it"};
      }
    }

    centroid_ = Vectors(Kokkos::view_alloc(name_ + ".centroid",
                                           Kokkos::WithoutInitializing),
                        simplex_count());
    measure_  = Scalars(Kokkos::view_alloc(name_ + ".measure",
                                           Kokkos::WithoutInitializing),
                        simplex_count());
    for (std::size_t c = 0; c < simplex_count(); ++c) {
      for (int k = 0; k < kSpatialDims; ++k) {
        double sum = 0.0;
        for (int v = 0; v < kVertsPerCell; ++v) sum += points_(cells_(c, v), k);
        centroid_(c, k) = sum / static_cast<double>(kVertsPerCell);
      }
      measure_(c) = simplex_measure(c);
    }
    finalized_ = true;
    return Result<void>::success();
  }

  [[nodiscard]] bool finalized() const noexcept { return finalized_; }

  /// The m-dimensional measure of simplex `c`: length, area or volume
  /// according to the MANIFOLD dimension, never the spatial one. A triangle in
  /// 3D has an area, and that is what this returns.
  [[nodiscard]] double measure(std::size_t c) const {
    FUSE_ASSERT(finalized_, "measure() before finalize()");
    return measure_(c);
  }

  /// Component `k` of the centroid of simplex `c`.
  [[nodiscard]] double centroid(std::size_t c, int k) const {
    FUSE_ASSERT(finalized_, "centroid() before finalize()");
    return centroid_(c, k);
  }

  /// The unit normal of simplex `c`, written into `out`.
  ///
  /// Only declared when the codimension is exactly 1. On a volume mesh this
  /// does not compile, which is the intent: "the normal of a tetrahedron" has
  /// no answer, and a runtime refusal would be a worse way to learn that.
  void normal(std::size_t c, double* out) const
    requires(kCodimension == 1)
  {
    FUSE_ASSERT(finalized_, "normal() before finalize()");
    if constexpr (kManifoldDims == 2 && kSpatialDims == 3) {
      const int a = cells_(c, 0), b = cells_(c, 1), d = cells_(c, 2);
      const double u[3] = {points_(b, 0) - points_(a, 0),
                           points_(b, 1) - points_(a, 1),
                           points_(b, 2) - points_(a, 2)};
      const double w[3] = {points_(d, 0) - points_(a, 0),
                           points_(d, 1) - points_(a, 1),
                           points_(d, 2) - points_(a, 2)};
      double n[3] = {u[1] * w[2] - u[2] * w[1], u[2] * w[0] - u[0] * w[2],
                     u[0] * w[1] - u[1] * w[0]};
      const double len = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
      FUSE_ASSERT(len > 0.0, "simplex {} is degenerate: it has no normal", c);
      for (int k = 0; k < 3; ++k) out[k] = n[k] / len;
    } else {
      // Codimension 1 with M = 1, S = 2: an edge in the plane. The normal is
      // the in-plane perpendicular of the edge direction.
      const int a = cells_(c, 0), b = cells_(c, 1);
      const double t[2] = {points_(b, 0) - points_(a, 0),
                           points_(b, 1) - points_(a, 1)};
      const double len = std::sqrt(t[0] * t[0] + t[1] * t[1]);
      FUSE_ASSERT(len > 0.0, "simplex {} is degenerate: it has no normal", c);
      out[0] = -t[1] / len;
      out[1] = t[0] / len;
    }
  }

  /// Corner `v` of simplex `c`, component `k`. The accessor the clipping
  /// kernel uses, which is why it is inline and does no checking in a release
  /// build.
  [[nodiscard]] KOKKOS_INLINE_FUNCTION double vertex(std::size_t c, int v,
                                                     int k) const {
    return points_(cells_(c, v), k);
  }

 private:
  /// Unsigned m-measure via the Gram determinant of the edge matrix, which is
  /// the one formula that covers every (M, S) pair: for codimension 0 it
  /// reduces to |det|/M!, and for a surface or curve it gives the area or
  /// length rather than zero, which a naive determinant would.
  [[nodiscard]] double simplex_measure(std::size_t c) const {
    if constexpr (kManifoldDims == 0) {
      return 0.0;  // a point has no extent, and that is not a defect
    } else {
      constexpr int M = kManifoldDims;
      double e[M][kSpatialDims];
      const int a = cells_(c, 0);
      for (int i = 0; i < M; ++i) {
        const int b = cells_(c, i + 1);
        for (int k = 0; k < kSpatialDims; ++k) {
          e[i][k] = points_(b, k) - points_(a, k);
        }
      }
      // G = E E^T, an M x M Gram matrix; sqrt(det G) is the M-volume of the
      // parallelepiped, and the simplex is that over M!.
      double g[M][M];
      for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
          double s = 0.0;
          for (int k = 0; k < kSpatialDims; ++k) s += e[i][k] * e[j][k];
          g[i][j] = s;
        }
      }
      double det = 0.0;
      if constexpr (M == 1) {
        det = g[0][0];
      } else if constexpr (M == 2) {
        det = g[0][0] * g[1][1] - g[0][1] * g[1][0];
      } else {
        det = g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1]) -
              g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0]) +
              g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]);
      }
      // Rounding can make a degenerate simplex's Gram determinant slightly
      // negative; clamp rather than return a NaN from the square root.
      if (det <= 0.0) return 0.0;
      double fact = 1.0;
      for (int i = 2; i <= M; ++i) fact *= static_cast<double>(i);
      return std::sqrt(det) / fact;
    }
  }

  std::string name_;
  Points      points_;
  Cells       cells_;
  Parents     parent_;
  std::size_t n_parent_cells_ = 0;
  Vectors     centroid_;
  Scalars     measure_;
  bool        finalized_ = false;
};

/// The pairs that physically occur, named so a reader can see the closed set
/// the template parameterisation costs us.
using PointCloud3    = Mesh<0, 3>;
using Curve3         = Mesh<1, 3>;
using Triangulation2 = Mesh<2, 2>;
using Surface3       = Mesh<2, 3>;
using Volume3        = Mesh<3, 3>;
using Interval1      = Mesh<1, 1>;

}  // namespace fuse::native
