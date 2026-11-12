#pragma once
// The convergence-order study of the steady nonlinear problem solved
// through fuse_physics::fe::DiffusionReactionSystem and fuse::num::solve on
// the unit cube [0,1]^d, shared by physics.diffusion_reaction_steady_order
// (bilinear and biquadratic quadrilaterals, trilinear hexahedra on meshes of
// 2, 4 and 8 cells per side) and physics.diffusion_reaction_steady_order_fine
// (trilinear hexahedra on 4, 8 and 16).
//
// Problem: -Laplacian(u) + alpha u^2 = f, u = 0 on the whole boundary,
// manufactured solution u = prod_k sin(pi x_k), alpha = 1/2.
//
// The L2 error of a degree-p Lagrange discretisation of a smooth solution
// is O(h^(p+1)), so the observed order r = log2(e_h / e_(h/2)) tends to
// p + 1. Three meshes give two observed orders r1, r2. An observed order
// differs from p + 1 by a term proportional to a power of h, so
// r2 - (p + 1) is at most r1 - r2 in magnitude to leading order; the
// assertion allows twice that, and requires r2 to be the closer of the two
// so that a wide band from a pre-asymptotic pair cannot pass.
//
// ALGEBRAIC ERROR. Newton's iteration stops at a state x with residual
// F(x) != 0, at a distance e = x - x* from the exact discrete solution.
// Its L2 norm is bounded here, and its effect on each observed order added
// to the band. With I the dofs not on the boundary:
//
//  (a) The Dirichlet values of x are exactly zero (asserted): the initial
//      guess is zero there, the Dirichlet rows of F are x_D - 0, and the
//      differenced Jacobian action of a vector that vanishes on those rows
//      vanishes on them exactly, so every Krylov vector and every Newton
//      update does. Hence e vanishes there and F_I(x) = Jbar e_I, with
//      Jbar the Jacobian averaged along the segment from x* to x.
//
//  (b) Jbar = K + M D. K is the stiffness block; for a conforming space
//      its Rayleigh quotient against the mass matrix M is at least the
//      smallest Dirichlet eigenvalue of the Laplacian on the unit cube,
//      d pi^2 (min-max principle), and the quadrature integrates both
//      forms exactly on these meshes (degree 2p per coordinate against a
//      rule of degree 2p + 2). The reaction enters the residual nodally,
//      as M applied to the vector of R(u_j), so its block is M D with D
//      the diagonal matrix of the derivatives R' at the nodes, averaged
//      along the segment: |D_jj| <= 2 alpha U, U a bound on the nodal
//      values along the segment. M D is not symmetric, and
//      |e^T M D e| <= ||e||_M ||D e||_M <= 2 alpha U sqrt(kappa(M)) e^T M e,
//      with kappa(M) the condition number of M_II. Together:
//      e^T Jbar e >= (d pi^2 - s) e^T M e, s = 2 alpha U sqrt(kappa(M)).
//
//  (c) e^T M e is the squared L2 norm of the error function, and
//      e^T Jbar e = e^T F <= ||e||_2 ||F||_2 <= ||e||_M ||F||_2 /
//      sqrt(lambda_min(M_II)). M_II is the d-fold tensor product of the
//      1-D mass matrix with its end dofs removed. Its 1-D factor has
//      smallest eigenvalue at least (1/2) of its smallest diagonal entry:
//      the element mass matrix scaled by its diagonal is
//      [[1, 1/2], [1/2, 1]] for degree 1 (eigenvalues 1/2, 3/2) and
//      [[1, 1/4, -1/4], [1/4, 1, 1/4], [-1/4, 1/4, 1]] for degree 2
//      (eigenvalues 1/2, 5/4, 5/4), and the smallest of these over the
//      elements bounds the assembled matrix (Wathen, "Realistic eigenvalue
//      bounds for the Galerkin mass matrix", IMA J. Numer. Anal. 7, 1987).
//      The smallest diagonal entry of the 1-D factor is 2h/3 for degree 1
//      and 4h/15 for degree 2 (a vertex shared by two elements), so
//      lambda_min(M_II) >= (c h)^d with c = 1/3 and 2/15. The largest
//      eigenvalue of the 1-D factor is at most the largest scaled element
//      eigenvalue (3/2, 5/4) times its largest diagonal entry (2h/3,
//      16h/30), that is h and 2h/3, so kappa(M_II) <= k^d with k = 3 for
//      degree 1 and 5 for degree 2.
//
//  Result: ||e||_L2 <= ||F||_2 / ((d pi^2 - s) (c h)^(d/2)).
//
//  U is taken as 2. The bound is consistent with that choice when
//  max|x| + ||e||_2 <= 2, where ||e||_2 <= ||e||_L2 / (c h)^(d/2); this is
//  asserted, so U is checked and not assumed.
//
// An error e_k known to within b_k changes log2(e_c / e_f) by at most
// log2((1 + a_c) / (1 - a_f)), a = b / e.

#include "fuse_physics/fe/diffusion_reaction_system.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "fuse/num/newton_krylov.hpp"
#include "fuse_physics/fe/basis.hpp"

namespace steady_order_study {

using fuse::physics::Point;
using fuse::physics::fe::BasisContinuity;
using fuse::physics::fe::BasisRegistry;
using fuse::physics::fe::BoundaryCondition;
using fuse::physics::fe::BoundaryId;
using fuse::physics::fe::CellFamily;
using fuse::physics::fe::DiffusionReactionCoefficients;
using fuse::physics::fe::DiffusionReactionSystem;
using fuse::physics::fe::DofMap;
using fuse::physics::fe::FieldBasisFamily;
using fuse::physics::fe::GeometryVariant;
using fuse::physics::fe::Variable;
using fuse::physics::spatial::Domain;
using fuse::physics::spatial::Mesh;
using fuse::physics::spatial::Shape;

constexpr double kPi = 3.14159265358979323846;
constexpr double kAlpha = 0.5;
constexpr double kNodalBound = 2.0;  // U of the header comment; checked per level

inline double exact(unsigned dimension, const Point& p) {
  double u = std::sin(kPi * p.x) * std::sin(kPi * p.y);
  if (dimension == 3) u *= std::sin(kPi * p.z);
  return u;
}

struct LevelResult {
  double l2_error = 0.0;
  double algebraic_bound = 0.0;  ///< bound on the L2 norm of the algebraic error
  fuse::num::NewtonResult newton;
};

inline LevelResult solve_level(unsigned dimension, unsigned degree, unsigned n) {
  Mesh mesh(dimension);
  if (dimension == 2)
    mesh.generate_2d_box(n, n, Point(0, 0, 0), Point(1, 1, 0));
  else
    mesh.generate_box(n, n, n, Point(0, 0, 0), Point(1, 1, 1));
  BasisRegistry registry;
  DofMap dof_map(mesh, {Variable{"u", FieldBasisFamily::LagrangeNodal, degree, BasisContinuity::cG, 1}}, registry);
  const Domain domain(dimension, dimension == 2 ? Shape::Square : Shape::Cube, 1.0, 1.0, dimension == 2 ? 0.0 : 1.0,
                      false);
  const auto basis = registry.create({dimension == 2 ? CellFamily::Quadrilateral : CellFamily::Hexahedron,
                                      FieldBasisFamily::LagrangeNodal, degree, BasisContinuity::cG});
  DiffusionReactionSystem system(mesh, domain, dof_map, basis,
                                 dimension == 2 ? GeometryVariant::Quad4 : GeometryVariant::Hex8,
                                 /*quadrature_degree=*/static_cast<int>(2 * degree + 2));
  DiffusionReactionCoefficients coefficients;
  coefficients.reaction = [](const Point&, double u) { return kAlpha * u * u; };
  coefficients.reaction_derivative = [](const Point&, double u) { return 2.0 * kAlpha * u; };
  system.set_coefficients(coefficients);
  system.set_forcing([dimension](const Point& p, double) {
    const double u = exact(dimension, p);
    return dimension * kPi * kPi * u + kAlpha * u * u;
  });
  std::vector<BoundaryId> sides{BoundaryId::XMin, BoundaryId::XMax, BoundaryId::YMin, BoundaryId::YMax};
  if (dimension == 3) sides.insert(sides.end(), {BoundaryId::ZMin, BoundaryId::ZMax});
  for (const auto id : sides)
    system.set_boundary_condition(id, BoundaryCondition::steady_dirichlet([](const Point&, double) { return 0.0; }));
  system.initialize();

  const std::size_t n_dofs = static_cast<std::size_t>(dof_map.n_dofs());
  fuse::linalg::DistVector x(n_dofs, fuse::linalg::make_serial_exchanger(), fuse::linalg::make_serial_reducer());
  x.set_zero();
  fuse::num::NewtonKrylovOptions options;
  options.rtol = 1e-10;
  options.gmres.restart = 200;
  options.gmres.max_iterations = 4000;
  const auto solved = fuse::num::solve(
      [&](fuse::linalg::DistVector& xv, fuse::linalg::DistVector& fv) { return system.assemble_residual(xv, fv); }, x,
      nullptr, nullptr, options);
  if (!solved.ok()) throw std::runtime_error("steady solve failed: " + solved.error().message);

  LevelResult out;
  out.newton = solved.value();
  EXPECT_TRUE(fuse::num::is_converged(out.newton.status)) << fuse::num::to_cstring(out.newton.status);

  // (a) of the header comment.
  for (const auto& d : system.boundary_conditions().dirichlet_dofs())
    EXPECT_EQ(x.data()[static_cast<std::size_t>(d.dof)], 0.0) << "Dirichlet dof " << d.dof;

  const std::vector<double> state(x.data(), x.data() + n_dofs);
  out.l2_error = system.l2_error(state, 0.0, [dimension](const Point& p, double) { return exact(dimension, p); });

  const double h = 1.0 / n;
  const double c = degree == 1 ? 1.0 / 3.0 : 2.0 / 15.0;
  const double mass_condition_1d = degree == 1 ? 3.0 : 5.0;
  const double shift = 2.0 * kAlpha * kNodalBound * std::pow(mass_condition_1d, 0.5 * dimension);
  const double coercivity = dimension * kPi * kPi - shift;
  if (coercivity <= 0.0) throw std::logic_error("the coercivity estimate of the algebraic bound is not positive");
  const double sqrt_lambda_min_mass = std::pow(c * h, 0.5 * dimension);
  out.algebraic_bound = out.newton.final_residual_norm / (coercivity * sqrt_lambda_min_mass);
  double max_nodal = 0.0;
  for (const double v : state) max_nodal = std::max(max_nodal, std::fabs(v));
  EXPECT_LE(max_nodal + out.algebraic_bound / sqrt_lambda_min_mass, kNodalBound)
      << "the nodal bound U the algebraic bound was derived with does not hold";
  return out;
}

inline void expect_order(const char* name, unsigned dimension, unsigned degree, std::vector<unsigned> sizes) {
  std::vector<LevelResult> levels;
  for (const unsigned n : sizes) levels.push_back(solve_level(dimension, degree, n));
  auto rate = [&](std::size_t k) { return std::log2(levels[k].l2_error / levels[k + 1].l2_error); };
  auto algebraic = [&](std::size_t k) {
    const double coarse = levels[k].algebraic_bound / levels[k].l2_error;
    const double fine = levels[k + 1].algebraic_bound / levels[k + 1].l2_error;
    return fine < 1.0 ? std::log2((1.0 + coarse) / (1.0 - fine)) : std::numeric_limits<double>::infinity();
  };
  const double r1 = rate(0), r2 = rate(1);
  const double expected = degree + 1.0;
  const double band = 2.0 * std::fabs(r1 - r2) + algebraic(0) + algebraic(1);
  std::cout << "[ ORDER    ] " << name << ":";
  for (std::size_t k = 0; k < levels.size(); ++k)
    std::cout << " n=" << sizes[k] << " error=" << levels[k].l2_error << " (algebraic <= " << levels[k].algebraic_bound
              << ", newton " << levels[k].newton.iterations << ")";
  std::cout << " | r1=" << r1 << " r2=" << r2 << " band=" << band << "\n";
  EXPECT_LT(std::fabs(r2 - expected), std::fabs(r1 - expected)) << name;
  EXPECT_NEAR(r2, expected, band) << name;
}

}  // namespace steady_order_study
