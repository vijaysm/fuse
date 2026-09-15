#ifndef __advection_reaction_assembly_h__
#define __advection_reaction_assembly_h__

#include "karma_geometry.h"

#include <petscmat.h>
#include <petscvec.h>

#include <functional>

namespace KARMA
{
class KARMASystem;

using AdvectionVelocity = std::function< Vector3( const Point&, int material_id ) >;
using AdvectionScalar = std::function< Real( const Point&, Real field, int material_id ) >;
using AdvectionBoundaryValue = std::function< Real( const Point&, int material_id ) >;

/**
 * Physical flux is F(u)=b(x,material)*u. Interior faces use
 * Fhat=0.5*((bL.n)*uL+(bR.n)*uR)-0.5*alpha*(uR-uL), where
 * alpha=max(abs(bL.n),abs(bR.n)). Both material velocities are evaluated at
 * the same physical point, so normal-velocity continuity is not required.
 * For state-independent velocity, dFhat/duL=0.5*(bL.n+alpha) and
 * dFhat/duR=0.5*(bR.n-alpha); opposite test-side rows use their negatives.
 */
struct AdvectionReactionCoefficients
{
    AdvectionVelocity velocity = []( const Point&, int ) { return Vector3::Zero(); };
    AdvectionScalar reaction = []( const Point&, Real, int ) { return 0.0; };
    AdvectionScalar source = []( const Point&, Real, int ) { return 0.0; };
    AdvectionScalar reaction_derivative;
    AdvectionScalar source_derivative;
    AdvectionBoundaryValue inflow = []( const Point&, int ) { return 0.0; };
};

[[nodiscard]] Real upwind_flux( Real interior, Real exterior, const Vector3& velocity, const Vector3& outward_normal );
[[nodiscard]] Real rusanov_flux( Real interior, Real exterior, const Vector3& interior_velocity,
                                 const Vector3& exterior_velocity, const Vector3& outward_normal );

/** Assemble R_i=-int grad(phi_i).(b u)+int phi_i(r(u)-s(u))+int_boundary phi_i Fhat. */
void assemble_advection_reaction( const KARMASystem& system, Vec solution, Vec residual, Mat jacobian,
                                   const AdvectionReactionCoefficients& coefficients, int quadrature_degree = 5 );
}  // namespace KARMA

#endif
