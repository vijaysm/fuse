#pragma once

/// Shared grid, mesh-index bookkeeping, and periodic-diffusion/backward-Euler
/// pieces for the multirate Brusselator example (Time windows track), ported
/// from the independently-written multirate 2D Brusselator ("the Toy":
/// /Users/mahadevan/Code/AI/Toy, its NUMERICS.md/README.md are the model and
/// algorithm reference).
///
/// MODEL (unchanged from the Toy). Periodic unit square:
///   U_t = a*Laplacian(U) + B - U + q + f
///   V_t = a*Laplacian(V) - q
///   q   = U^2*V - A*U            (A = 3.4, B = 1, a = 10 by default)
/// U0 = 22*[y(1-y)]^1.5, V0 = 27*[x(1-x)]^1.5, sampled at cell midpoints.
/// f = 5 inside the radius-0.1 circle at (0.3, 0.6) for t >= 1.1, else 0.
///
/// MESH. Unlike the Toy's hand-rolled `Grid` (include/brusselator.hpp),
/// cell geometry here comes from FUSE's native mesh backend
/// (fuse::native::NativeMeshBackend, "generated:quad:NxM" --
/// include/fuse/backend/native_backend.hpp): a genuinely different quad
/// mesh generator, not a copy of the Toy's index arithmetic. Its cell
/// global id IS `ix + iy*nx` (src/backend/native_generators.cpp's
/// `Grid::cell_gid`) and its vertices sit at `(ix/nx, iy/ny)`, so a native
/// cell's centroid is `((ix+0.5)/nx, (iy+0.5)/ny)` -- the same point the
/// Toy's `Grid::x(i)`/`y(j)` name -- but this header does not ASSUME that
/// coincidence: `NativeGrid::make` recovers the (i, j) <-> native local
/// cell index correspondence by reading back each cell's own
/// `MeshView::centroid`, so it is correct even if a future native-mesh
/// change reorders cells (e.g. under partitioning; this example runs at
/// one rank only -- MPI does not initialise in this sandbox -- but the
/// recovery does not depend on that).
///
/// FIELD STORAGE. Every field this example solves for is kept as a plain
/// `Eigen::VectorXd` in CANONICAL order (`index(i,j) = j*nx+i`, matching
/// the Toy exactly) for the periodic-Laplacian backward-Euler solve, which
/// reuses Eigen::SparseCholesky exactly as the Toy does. The reaction rate
/// `q = U^2*V - A*U` -- the one genuinely elementwise, embarrassingly
/// parallel kernel in this model -- is instead computed on a
/// `Kokkos::View<double*>` with `Kokkos::parallel_for` over the canonical
/// cell index space (`reaction_kokkos`, below): this is the "port to
/// Kokkos" the task asks for, and it is what makes `ctest -L omp` load-
/// bearing for this example (examples/CMakeLists.txt registers an
/// `omp_identical` leg exactly as mms_pair/dn_heat do). FUSE's remap
/// weights (fuse::RemapWeights, built over the NATIVE mesh's own local
/// cell numbering) are applied through `apply_weights` after permuting a
/// canonical-order field into native order with `NativeGrid::to_native`/
/// `to_canonical` -- plain gather loops, not a Kokkos kernel; see
/// docs/user/brusselator.md "What is and is not Kokkos here" for why that
/// split was chosen and what it costs.

#include <Eigen/Core>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>
#include <Kokkos_Core.hpp>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "fuse/backend/mesh_view.hpp"
#include "fuse/backend/native_backend.hpp"
#include "fuse/base/error.hpp"
#include "fuse/base/ids.hpp"
#include "fuse/base/result.hpp"

namespace brusselator {

using Vector = Eigen::VectorXd;
using Sparse = Eigen::SparseMatrix<double>;

/// A cell-centred nx*ny periodic grid backed by FUSE's native mesh, plus
/// the canonical (Toy-matching) index that lets the rest of this example's
/// Eigen solves stay a straight port of the Toy's own linear algebra.
class NativeGrid {
 public:
  NativeGrid(const NativeGrid&) = delete;
  NativeGrid& operator=(const NativeGrid&) = delete;
  NativeGrid(NativeGrid&&) = default;
  NativeGrid& operator=(NativeGrid&&) = default;

  /// Loads a "generated:quad:nxXny" native mesh named `name` onto `backend`
  /// (a SHARED backend: U and V's meshes must live on the SAME
  /// `fuse::MeshBackend` instance for `fuse::build_weights` to resolve
  /// both sides of a remap plan through it -- see main.cpp) and recovers
  /// the canonical index correspondence. Errors:
  ///   E-BRUSS-MESH-LOAD   the native backend refused the generated spec
  ///   E-BRUSS-MESH-COUNT  entity_count(2) != nx*ny
  ///   E-BRUSS-MESH-INDEX  a cell's centroid does not land on an (i, j)
  ///                       of this grid to within half a cell width, or
  ///                       two cells mapped to the same (i, j)
  [[nodiscard]] static fuse::Result<NativeGrid> make(fuse::native::NativeMeshBackend& backend,
                                                     const std::string& name, int nx, int ny);

  [[nodiscard]] int nx() const noexcept { return nx_; }
  [[nodiscard]] int ny() const noexcept { return ny_; }
  [[nodiscard]] int size() const noexcept { return nx_ * ny_; }
  [[nodiscard]] double x(int i) const noexcept { return (i + 0.5) / static_cast<double>(nx_); }
  [[nodiscard]] double y(int j) const noexcept { return (j + 0.5) / static_cast<double>(ny_); }
  [[nodiscard]] int index(int i, int j) const noexcept { return j * nx_ + i; }

  [[nodiscard]] fuse::MeshId mesh_id() const noexcept { return mesh_id_; }

  /// The native-mesh local cell index corresponding to canonical index
  /// `c = j*nx+i`. Exposed for verification code (entry-by-entry weight
  /// comparisons against an oracle built in canonical order) -- ordinary
  /// field transfer should use `to_native`/`to_canonical` below instead.
  [[nodiscard]] int native_index_of(int c) const noexcept { return native_of_canonical_[static_cast<std::size_t>(c)]; }

  /// Gathers a canonical-order field into native-mesh local-index order
  /// (what fuse::RemapWeights addresses).
  void to_native(const Vector& canonical, std::vector<double>* native) const {
    native->resize(static_cast<std::size_t>(size()));
    for (int c = 0; c < size(); ++c) (*native)[static_cast<std::size_t>(native_of_canonical_[static_cast<std::size_t>(c)])] = canonical[c];
  }
  /// Scatters a native-mesh-ordered field back into canonical order.
  void to_canonical(const std::vector<double>& native, Vector* canonical) const {
    canonical->resize(size());
    for (int c = 0; c < size(); ++c) (*canonical)[c] = native[static_cast<std::size_t>(native_of_canonical_[static_cast<std::size_t>(c)])];
  }

 private:
  NativeGrid(int nx, int ny) : nx_(nx), ny_(ny) {}

  int nx_ = 0, ny_ = 0;
  fuse::MeshId mesh_id_{};
  std::vector<int> native_of_canonical_;  ///< native_of_canonical_[j*nx+i] = native local index
};

inline fuse::Result<NativeGrid> NativeGrid::make(fuse::native::NativeMeshBackend& backend,
                                                 const std::string& name, int nx, int ny) {
  NativeGrid g(nx, ny);
  const std::string source =
      "generated:quad:" + std::to_string(nx) + "x" + std::to_string(ny);
  auto loaded = backend.load(fuse::MeshDecl{.name = name, .source = source});
  if (!loaded.ok()) {
    return fuse::Failure{"E-BRUSS-MESH-LOAD",
                        "NativeGrid::make('" + name + "', " + source +
                            "): backend.load failed: " + loaded.error().code + ": " +
                            loaded.error().message};
  }
  auto mid = backend.mesh_id(name);
  if (!mid.ok()) return mid.error();
  g.mesh_id_ = mid.value();
  auto view_r = backend.view(g.mesh_id_);
  if (!view_r.ok()) return view_r.error();
  const fuse::MeshView* view = view_r.value();
  const std::size_t n2 = view->entity_count(2);
  if (n2 != static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny)) {
    return fuse::Failure{"E-BRUSS-MESH-COUNT",
                        "NativeGrid::make('" + name + "'): native mesh reports " +
                            std::to_string(n2) + " dimension-2 entities, expected " +
                            std::to_string(nx * ny)};
  }
  g.native_of_canonical_.assign(static_cast<std::size_t>(nx * ny), -1);
  std::vector<int> seen(static_cast<std::size_t>(nx * ny), -1);
  for (std::size_t native_idx = 0; native_idx < n2; ++native_idx) {
    const fuse::Point c = view->centroid(2, native_idx);
    const int i = static_cast<int>(std::lround(c.x * nx - 0.5));
    const int j = static_cast<int>(std::lround(c.y * ny - 0.5));
    if (i < 0 || i >= nx || j < 0 || j >= ny ||
        std::abs(c.x - g.x(i)) > 0.25 / nx || std::abs(c.y - g.y(j)) > 0.25 / ny) {
      return fuse::Failure{"E-BRUSS-MESH-INDEX",
                          "NativeGrid::make('" + name + "'): native cell " +
                              std::to_string(native_idx) + " centroid (" +
                              std::to_string(c.x) + ", " + std::to_string(c.y) +
                              ") does not land on a unique (i,j) of a " + std::to_string(nx) +
                              "x" + std::to_string(ny) + " periodic grid"};
    }
    const int canonical = g.index(i, j);
    if (seen[static_cast<std::size_t>(canonical)] != -1) {
      return fuse::Failure{"E-BRUSS-MESH-INDEX",
                          "NativeGrid::make('" + name + "'): two native cells mapped to the "
                          "same canonical index " + std::to_string(canonical)};
    }
    seen[static_cast<std::size_t>(canonical)] = 1;
    g.native_of_canonical_[static_cast<std::size_t>(canonical)] = static_cast<int>(native_idx);
  }
  return g;
}

/// The periodic 5-point Laplacian on a canonical-order nx*ny grid --
/// identical stencil and identical sign convention to the Toy's
/// `bruss::laplacian` (include/brusselator.hpp / src/brusselator.cpp),
/// re-derived here rather than `#include`d from a read-only sibling repo.
[[nodiscard]] inline Sparse periodic_laplacian(int nx, int ny) {
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * 5);
  const double xx = double(nx) * nx, yy = double(ny) * ny;
  const auto index = [nx](int i, int j) { return j * nx + i; };
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const int row = index(i, j);
      entries.emplace_back(row, row, -2 * (xx + yy));
      entries.emplace_back(row, index((i + 1) % nx, j), xx);
      entries.emplace_back(row, index((i + nx - 1) % nx, j), xx);
      entries.emplace_back(row, index(i, (j + 1) % ny), yy);
      entries.emplace_back(row, index(i, (j + ny - 1) % ny), yy);
    }
  }
  Sparse result(nx * ny, nx * ny);
  result.setFromTriplets(entries.begin(), entries.end());
  return result;
}

/// U0 = 22*[y(1-y)]^1.5 (is_u=true), V0 = 27*[x(1-x)]^1.5 (is_u=false),
/// sampled at cell midpoints -- the Toy's `initial_condition`.
[[nodiscard]] inline Vector initial_condition(bool is_u, int nx, int ny) {
  Vector result(nx * ny);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double s = is_u ? (j + 0.5) / static_cast<double>(ny) : (i + 0.5) / static_cast<double>(nx);
      result[j * nx + i] = (is_u ? 22.0 : 27.0) * std::pow(s * (1 - s), 1.5);
    }
  }
  return result;
}

/// 5 inside the radius-0.1 circle centred at (0.3, 0.6), else 0 -- the
/// Toy's `forcing_mask`, evaluated on the U grid only (V has no forcing).
[[nodiscard]] inline Vector forcing_mask(int nx, int ny) {
  Vector result(nx * ny);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      const double x = (i + 0.5) / static_cast<double>(nx);
      const double y = (j + 0.5) / static_cast<double>(ny);
      result[j * nx + i] = (std::pow(x - 0.3, 2) + std::pow(y - 0.6, 2) <= 0.01) ? 5.0 : 0.0;
    }
  }
  return result;
}

/// Exact activated duration of the forcing over `[t_begin, t_end)`:
/// `max(0, t_end - max(t_begin, 1.1))`, the Toy's own closed form
/// (src/brusselator.cpp `Component::advance`), so a step that straddles
/// the t=1.1 switch integrates it exactly regardless of window alignment.
[[nodiscard]] inline double forcing_active_duration(double t_begin, double t_end) noexcept {
  return std::max(0.0, t_end - std::max(t_begin, 1.1));
}

/// q = U^2*V_on_u_grid - A*U, componentwise, computed on a
/// `Kokkos::View<double*>` with `Kokkos::parallel_for` over the canonical
/// cell index space -- the reaction kernel this port moves onto Kokkos
/// (see the file header). `A = 3.4`, matching the Toy exactly
/// (src/brusselator.cpp `reaction`, which hardcodes the same constant).
/// `u` and `v_on_u_grid` must both have the same size; `q_out` is resized
/// and overwritten.
inline void reaction_kokkos(const Vector& u, const Vector& v_on_u_grid, Vector* q_out) {
  const int n = static_cast<int>(u.size());
  if (v_on_u_grid.size() != n) {
    FUSE_THROW(fuse::InvalidArgument, "{}",
              std::string("reaction_kokkos: size mismatch, u has ") + std::to_string(n) +
                  " entries, v_on_u_grid has " + std::to_string(v_on_u_grid.size()));
  }
  q_out->resize(n);
  Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> u_view(
      u.data(), n);
  Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> v_view(
      v_on_u_grid.data(), n);
  Kokkos::View<double*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> q_view(
      q_out->data(), n);
  Kokkos::parallel_for(
      "brusselator::reaction",
      Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, n),
      KOKKOS_LAMBDA(const int m) {
        q_view(m) = u_view(m) * u_view(m) * v_view(m) - 3.4 * u_view(m);
      });
}

}  // namespace brusselator
