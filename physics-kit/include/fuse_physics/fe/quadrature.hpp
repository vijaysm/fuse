#pragma once

/// Finite-element quadrature rules, adapted from KARMA's karma_quadrature.h
/// (docs/dev/physics-kit-disposition.md Appendix A, port-adapt).
///
/// NATIVE, NOT PETSc-BACKED (owner decision D-A + docs/dev/physics-kit.md's
/// tier-0 contract). KARMA's KARMAQuadrature generates its points and
/// weights by calling PETSc's PetscDT* functions (PetscDTGaussTensorQuadrature,
/// PetscDTSimplexQuadrature, PetscDTGaussJacobiQuadrature) unconditionally --
/// not behind a FUSE_HAVE_PETSC-style guard, because KARMA itself always
/// links PETSc. physics-kit is tier 0 (docs/dev/physics-kit.md: "needs
/// nothing beyond what fuse::runtime/backend/num/linalg already need" --
/// Kokkos and fmt; PETSc is OPTIONAL, read only through FUSE_HAVE_PETSC), so
/// a quadrature generator physics-kit's own tier-0 tests must build and run
/// without PETSc cannot wrap PetscDT at all -- this is the "KARMA class
/// wraps [a third party] with no fuse counterpart at this layer" case D-A
/// anticipates, except the fix is not a facade mapping (there is no
/// generic small-quadrature facade in fuse::linalg or fuse::petsc) but a
/// native reimplementation of the SAME textbook construction PETSc's own
/// PetscDT uses internally for this: Gauss-Legendre/Gauss-Jacobi nodes and
/// weights via the Golub-Welsch eigenvalue algorithm (Golub, G.H. and
/// Welsch, J.H., "Calculation of Gauss Quadrature Rules", Mathematics of
/// Computation 23 (106), 1969, pp. 221-230), and a collapsed-coordinate
/// (Duffy) conical product for the triangle and tetrahedron (Stroud, A.H.,
/// "Approximate Calculation of Multiple Integrals", 1971; Karniadakis,
/// G.E. and Sherwin, S.J., "Spectral/hp Element Methods for Computational
/// Fluid Dynamics", 2nd ed., 2005, section 2.3 -- "Quadrature on simplex
/// domains"). See quadrature.cpp's header comment for the recurrence-
/// coefficient formulas used.
///
/// WAVE 1 SCOPE. Edge, triangle, quadrilateral, tetrahedron and hexahedron
/// (CellFamily, reference_topology.hpp) at any requested polynomial degree.
/// Prism and pyramid are not implemented (reference_topology.hpp's own
/// scope note); a request for one refuses.

#include <cstddef>
#include <span>
#include <vector>

#include "fuse_physics/fe/reference_topology.hpp"
#include "fuse_physics/geometry.hpp"

namespace fuse::physics::fe {

/// A quadrature rule exact for polynomials up to `degree` on the reference
/// cell named by `family`. Points lie in the SAME reference coordinates
/// ReferenceTopology uses (unit simplex / unit tensor-product cube).
class Quadrature {
 public:
  Quadrature(CellFamily family, int degree);

  [[nodiscard]] CellFamily family() const noexcept { return family_; }
  [[nodiscard]] unsigned dimension() const noexcept { return cell_dimension(family_); }
  [[nodiscard]] int degree() const noexcept { return degree_; }
  [[nodiscard]] std::size_t n_points() const noexcept { return points_.size(); }
  [[nodiscard]] std::span<const Point> points() const noexcept { return points_; }
  [[nodiscard]] std::span<const double> weights() const noexcept { return weights_; }

  /// Jacobi sweeps the Golub-Welsch eigen-solve performed for the largest
  /// 1-D rule this rule was assembled from (quadrature.cpp,
  /// symmetric_tridiagonal_eigen; the stopping test is an off-diagonal
  /// Frobenius norm below n*eps). Enters round_off_bound().
  [[nodiscard]] int generation_sweeps() const noexcept { return sweeps_; }

  /// The round-off error bound this rule's exactness test (fe_quadrature_
  /// test.cpp) holds an integrated monomial of total degree
  /// `polynomial_degree` to, DERIVED (lead, 2026-10-31; the wave-1 draft
  /// bounded only the summation and needed a 1e-13 floor on top, which is a
  /// chosen constant and is gone):
  ///   |computed - exact| <= eps * [ (n_points + ops) * S  +  (sweeps+1) * n1d * (dim * S + p * V) ]
  /// with S = sum_q |w_q f(x_q)| (`sum_abs_terms`), p = `polynomial_degree`,
  /// V = `cell_measure` (the reference cell's measure, = sum of weights),
  /// n1d = points per axis, dim = dimension().
  /// First term: Higham, Accuracy and Stability of Numerical Algorithms, 2nd
  /// ed., (3.5)/(4.4): recursive summation of n_points terms, each formed
  /// from `ops` = 2*dim + p + 2 roundings (collapsed-coordinate map and
  /// weight products, p monomial multiplications, the product and the
  /// accumulation), has relative backward error at most (n_points + ops)*eps
  /// on S, to first order in eps. Second term: Golub and Van Loan, Matrix
  /// Computations, 4th ed., Section 8.4 (cyclic Jacobi on an n1d x n1d
  /// symmetric matrix of norm at most 1): the computed eigenvalues (nodes
  /// on [-1,1]) are exact for a matrix perturbed by at most c*sweeps*n1d*eps,
  /// plus the stopping test's own n1d*eps, so a node moves by at most
  /// (sweeps+1)*n1d*eps (c taken as 1 here; the test measures the margin)
  /// and a weight (first eigenvector component squared, times a constant)
  /// by the same relative amount per axis, dim axes. A node error dx moves
  /// the monomial x^p by at most p*dx on [0,1], so summed with the weights
  /// it contributes p*V*(sweeps+1)*n1d*eps; the weight errors contribute
  /// dim*S*(sweeps+1)*n1d*eps. The test records the largest observed
  /// |error|/bound ratio so the slack of this derivation is a measured
  /// number, not an assumption.
  [[nodiscard]] double round_off_bound(double sum_abs_terms, unsigned polynomial_degree,
                                       double cell_measure) const noexcept;

 private:
  CellFamily family_;
  int degree_;
  std::vector<Point> points_;
  std::vector<double> weights_;
  int sweeps_ = 0;
};

}  // namespace fuse::physics::fe
