#ifndef KARMA_TRANSIENT_REACTION_DIFFUSION_ASSEMBLY_H
#define KARMA_TRANSIENT_REACTION_DIFFUSION_ASSEMBLY_H

#include "karma_dense.h"
#include "karma_geometry.h"

#include <functional>
#include <petscmat.h>
#include <petscvec.h>
#include <moab/Types.hpp>

namespace KARMA
{
class KARMASystem;

using TransientReaction = std::function<Real( const Point&, Real state, Real coupled, Real time )>;

struct TransientReactionDiffusionModel
{
    Real diffusion = 1;
    TransientReaction reaction;
    TransientReaction state_derivative;
};

struct TransientElementContext
{
    Real time_left = 0;
    Real time_right = 0;
    const RealDenseVector& state_left;
    const RealDenseVector& state_right;
    const RealDenseVector& coupled;
};

struct TransientReactionDiffusionElementTerms
{
    RealDenseMatrix jacobian;
    RealDenseVector residual;
};

[[nodiscard]] TransientReactionDiffusionElementTerms element_transient_reaction_diffusion(
    const KARMASystem&, moab::EntityHandle, const TransientReactionDiffusionModel&, const TransientElementContext&,
    int quadrature_degree = 5 );

void assemble_transient_reaction_diffusion( const KARMASystem&, Vec state_left, Vec state_right, Vec coupled,
                                            Real time_left, Real time_right,
                                            const TransientReactionDiffusionModel&, Vec residual, Mat jacobian,
                                            int quadrature_degree = 5 );
}  // namespace KARMA

#endif
