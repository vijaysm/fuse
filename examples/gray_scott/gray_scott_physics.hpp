#pragma once

/// Shared grid and reaction-kinetics pieces for the Gray-Scott example and
/// its verification tests: the periodic cell-centred P0 finite-volume grid,
/// the 5-point Laplacian it induces, and the Gray-Scott reaction term and its
/// derivatives. Kept separate from `species_component.hpp` (the
/// `fuse::Component` wrapper) and `monolithic_reference.hpp` (the fully
/// coupled reference solver) because both of those assemble a Newton system
/// out of exactly these pieces and must not drift against each other.
///
/// MODEL. On the unit square (or a domain of side `L`) with periodic
/// boundaries:
///   u_t = Du*Laplacian(u) - u*v^2 + F*(1 - u)
///   v_t = Dv*Laplacian(v) + u*v^2 - (F + k)*v
///
/// DISCRETISATION. Cell-centred P0 finite volumes on an N x N grid of square
/// cells, h = L / N, cell (i,j) centred at ((i+0.5)*h, (j+0.5)*h). The
/// 5-point Laplacian stencil is the standard second-order finite-volume
/// approximation for a uniform periodic grid: for cell m = i*N + j,
///   (Laplacian u)_m = (u_E + u_W + u_N + u_S - 4*u_m) / h^2
/// with periodic wraparound neighbours. This is exact to O(h^2) for a smooth
/// field, which is what the MMS test in gray_scott_mms_test.cpp measures.

#include <Eigen/Sparse>
#include <cstddef>
#include <vector>

namespace gray_scott {

/// Grid geometry shared by every piece: species components, the monolithic
/// reference, and the tests. `n` cells per side, domain side length `length`.
struct Grid {
  int    n = 1;
  double length = 1.0;

  [[nodiscard]] double h() const noexcept { return length / static_cast<double>(n); }
  [[nodiscard]] std::size_t size() const noexcept {
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  }
  [[nodiscard]] std::size_t index(int i, int j) const noexcept {
    return static_cast<std::size_t>(i) * static_cast<std::size_t>(n) + static_cast<std::size_t>(j);
  }
  /// Cell-centre coordinates of cell (i,j).
  [[nodiscard]] double x_center(int i) const noexcept { return (i + 0.5) * h(); }
  [[nodiscard]] double y_center(int j) const noexcept { return (j + 0.5) * h(); }
};

/// The periodic 5-point Laplacian operator matrix `L` (n^2 x n^2, symmetric,
/// negative semi-definite) such that `(L * u)` approximates `Laplacian(u)`
/// at every cell to O(h^2). Built once per grid; reused by every advance()
/// call and by the monolithic reference (scaled by each species' own
/// diffusion coefficient there).
[[nodiscard]] inline Eigen::SparseMatrix<double> periodic_laplacian(const Grid& g) {
  const int n = g.n;
  const double h2 = g.h() * g.h();
  const double off = 1.0 / h2;
  const double diag = -4.0 / h2;
  std::vector<Eigen::Triplet<double>> trips;
  trips.reserve(static_cast<std::size_t>(n) * static_cast<std::size_t>(n) * 5);
  for (int i = 0; i < n; ++i) {
    const int ip = (i + 1) % n;
    const int im = (i - 1 + n) % n;
    for (int j = 0; j < n; ++j) {
      const int jp = (j + 1) % n;
      const int jm = (j - 1 + n) % n;
      const auto m = static_cast<int>(g.index(i, j));
      trips.emplace_back(m, m, diag);
      trips.emplace_back(m, static_cast<int>(g.index(ip, j)), off);
      trips.emplace_back(m, static_cast<int>(g.index(im, j)), off);
      trips.emplace_back(m, static_cast<int>(g.index(i, jp)), off);
      trips.emplace_back(m, static_cast<int>(g.index(i, jm)), off);
    }
  }
  Eigen::SparseMatrix<double> lap(static_cast<int>(g.size()), static_cast<int>(g.size()));
  lap.setFromTriplets(trips.begin(), trips.end());
  lap.makeCompressed();
  return lap;
}

/// Gray-Scott reaction kinetics and their derivatives. `u` is the species
/// consumed quadratically by the autocatalytic term, `v` the species
/// produced by it -- i.e. exactly the roles in the model comment above.
/// Free functions rather than a per-species class: both
/// `GrayScottSpeciesComponent` (partner frozen, needs only the self
/// derivative) and the monolithic reference (both species unknown, needs
/// every cross derivative) call the same four numbers, so the physics
/// cannot read differently in the two solvers.
struct Kinetics {
  double F = 0.0;
  double k = 0.0;

  [[nodiscard]] double reaction_u(double u, double v) const noexcept {
    return -u * v * v + F * (1.0 - u);
  }
  [[nodiscard]] double reaction_v(double u, double v) const noexcept {
    return u * v * v - (F + k) * v;
  }
  [[nodiscard]] double d_reaction_u_du(double /*u*/, double v) const noexcept {
    return -v * v - F;
  }
  [[nodiscard]] double d_reaction_u_dv(double u, double v) const noexcept {
    return -2.0 * u * v;
  }
  [[nodiscard]] double d_reaction_v_du(double /*u*/, double v) const noexcept {
    return v * v;
  }
  [[nodiscard]] double d_reaction_v_dv(double u, double v) const noexcept {
    return 2.0 * u * v - (F + k);
  }
};

}  // namespace gray_scott
