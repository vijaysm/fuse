#ifndef __diffusion_reaction_assembly_h__
#define __diffusion_reaction_assembly_h__

#include "karma_dense.h"
#include "karma_geometry.h"

#include "moab/Types.hpp"
#include <petscmat.h>
#include <petscvec.h>

#include <functional>

class KARMAPhysicsBCManager;

namespace KARMA
{
class KARMASystem;

using ScalarCoefficient = std::function< Real( const Point&, Real field, int material_id ) >;

struct DiffusionReactionCoefficients
{
    ScalarCoefficient diffusion = []( const Point&, Real, int ) { return 1.0; };
    ScalarCoefficient reaction  = []( const Point&, Real, int ) { return 0.0; };
    ScalarCoefficient source    = []( const Point&, Real, int ) { return 0.0; };
    ScalarCoefficient diffusion_derivative;
    ScalarCoefficient reaction_derivative;
    ScalarCoefficient source_derivative;
};

struct DiffusionReactionElementTerms
{
    RealDenseMatrix jacobian;
    RealDenseVector residual;
    RealDenseVector load;
};

[[nodiscard]] DiffusionReactionElementTerms element_diffusion_reaction(
    const KARMASystem& system, moab::EntityHandle element, const DiffusionReactionCoefficients& coefficients,
    const RealDenseVector* element_solution = nullptr, int quadrature_degree = 5, bool include_jacobian = true );

/** Assemble F(u)=A(u)u-f and dF/du. Neumann g is positive outward flux in f.
 * Typed Robin data are flux-form: a*(k grad(u).n)+b*u=c, contributing +(b/a)u
 * to A and +(c/a) to f. Legacy boundary APIs retain their historical semantics. */
void assemble_diffusion_reaction( const KARMASystem& system, Vec solution, Vec residual, Mat jacobian,
                                  const DiffusionReactionCoefficients& coefficients,
                                  const KARMAPhysicsBCManager* boundary = nullptr, int quadrature_degree = 5 );
}  // namespace KARMA

#endif
