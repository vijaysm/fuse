#ifndef __generic_mass_operator_h__
#define __generic_mass_operator_h__

#include "karma_dense.h"

#include "moab/Types.hpp"
#include <petscmat.h>
#include <petscvec.h>

#ifdef KARMA_HAVE_LIBMESH
#include "karma_operator.h"
PetscErrorCode generic_mass_matrix_multiply( const KARMAVector&, KARMAVector&, const KARMAParameters&,
                                             KARMAParameters* = nullptr );
PetscErrorCode generic_mass_matrix_solve( const KARMAVector&, KARMAVector&, const KARMAParameters&,
                                          KARMAParameters* = nullptr );
PetscErrorCode generic_mass_matrix( const KARMAVector&, KARMAMatrix&, const KARMAParameters&,
                                    KARMAParameters* = nullptr );
#endif

namespace KARMA
{
class KARMASystem;

/** Consistent element mass matrix in variable-major, component-major, basis-minor order. */
[[nodiscard]] RealDenseMatrix element_mass_matrix( const KARMASystem& system, moab::EntityHandle element,
                                                   int quadrature_degree = 5 );
/** Owner-only additive physical mass assembly. It is unconstrained unless explicit Dirichlet DoFs are supplied. */
void assemble_mass_matrix( const KARMASystem& system, Mat mass, const PetscInt* dirichlet_dofs = nullptr,
                           PetscInt n_dirichlet_dofs = 0, int quadrature_degree = 5 );
void mass_multiply( const KARMASystem& system, Mat mass, Vec input, Vec output );
/** Row-sum lumping requires every lump to be finite and positive above a scale-aware tolerance.
 * No positivity-preserving alternative is currently provided for bases such as P2 simplices. */
void mass_solve( const KARMASystem& system, Mat mass, Vec rhs, Vec solution, bool lumped = false );
}  // namespace KARMA

#endif
