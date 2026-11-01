#pragma once

/// Geometry-independent, physics-agnostic element operators: the mass
/// and stiffness (diffusion) bilinear forms every physics built on
/// fuse_physics::fe shares. Adapted from KARMA's own
/// include/physics/common/generic_mass_operator.h (docs/dev/physics-kit-
/// disposition.md's own D-A: "port-adapt"), the abstraction level
/// test/mass_assembly_test.C's own KARMA source actually exercises --
/// see that file's own comment history for why this, not
/// karma_algebraic_system.h itself, is where a per-element bilinear form
/// belongs.
///
/// NOT A LITERAL PORT. KARMA's own element_mass_matrix(system, element,
/// degree) reads its FEBase/basis/quadrature objects out of a
/// KARMAAlgebraicSystem's own per-variable bookkeeping; this port takes
/// an already-reinit-able fe::FEBase directly (fe_base.hpp own
/// evaluator), so the SAME two functions below serve a caller that
/// built its FEBase from fe::AlgebraicSystem's own DofMap or from
/// neither (a standalone quadrature check, as fe_basis_test.cpp and this
/// file's own test do).
///
/// KARMA's include/numerics/karma_diffusion.h (a per-element stiffness
/// operator parallel to generic_mass_operator.h) does not exist in the
/// upstream/karma export this port builds from -- element_stiffness_
/// matrix below is the textbook FE bilinear form integral(grad(phi_i) .
/// grad(phi_j)) dV, derived and cited as such (design_notes), the same
/// posture took for KARMA gaps with no upstream file to port
/// from.

#include <vector>

#include "fuse_physics/fe/fe_base.hpp"
#include "fuse_physics/spatial/mesh.hpp"

namespace fuse::physics::fe {

/// `mass(i, j) = integral(phi_i * phi_j) dV` over `element`, row-major
/// `n x n` (`n = fe.get_phi().size()`, the basis size `fe` was built
/// with). `fe` must already be bound to the SAME mesh `element` lives
/// in; this function calls `fe.reinit(element)` itself. Exact for a
/// polynomial `phi_i * phi_j` of degree <= `fe`'s own quadrature degree
/// on an AFFINE element (constant Jacobian) -- the caller's own
/// responsibility to pick a high enough degree (design_notes states the
/// derivation this wave's tests use: 2 * basis order, exact for `phi_i *
/// phi_j` itself, on the affine box elements those tests build).
[[nodiscard]] std::vector<double> element_mass_matrix(FEBase& fe, spatial::EntityHandle element);

/// `stiffness(i, j) = integral(grad(phi_i) . grad(phi_j)) dV` over
/// `element`, row-major `n x n`. Same reinit and quadrature-degree
/// contract as `element_mass_matrix`; for a P1 (linear) basis the
/// gradient is CONSTANT per element, so `grad(phi_i) . grad(phi_j)` has
/// degree 0 and even a single-point rule integrates it exactly.
[[nodiscard]] std::vector<double> element_stiffness_matrix(FEBase& fe, spatial::EntityHandle element);

}  // namespace fuse::physics::fe
