#include "generic_mass_operator.h"

#include "karma_exceptions.h"
#include "karma_system.h"
#include "petsc_raii.h"

#include <petscksp.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace KARMA
{
void mass_solve( const KARMASystem& system, Mat mass, Vec rhs, Vec solution, bool lumped )
{
    system.validate_matrix_layout( mass, "mass solve matrix" );
    system.validate_vector_layout( rhs, "mass solve right-hand side" );
    system.validate_vector_layout( solution, "mass solve output" );
    if( lumped )
    {
        PetscOwner< Vec, VecDestroy > diagonal;
        if( VecDuplicate( rhs, diagonal.put() ) || MatGetRowSum( mass, diagonal.get() ) )
            KARMA_THROW( InvalidArgument( "PETSc error in lumped mass solve" ) );
        PetscReal norm    = 0;
        PetscReal minimum = 0;
        if( VecNorm( diagonal.get(), NORM_INFINITY, &norm ) || VecMin( diagonal.get(), nullptr, &minimum ) )
            KARMA_THROW( InvalidArgument( "PETSc error validating lumped mass" ) );
        const PetscReal tolerance = std::numeric_limits< PetscReal >::epsilon() * norm;
        if( !std::isfinite( norm ) || !std::isfinite( minimum ) || minimum <= tolerance )
            KARMA_THROW( InvalidArgument(
                "row-sum lumped mass requires every lump to be finite and strictly positive above tolerance; "
                "no positivity-preserving alternative is available" ) );
        if( VecPointwiseDivide( solution, rhs, diagonal.get() ) )
            KARMA_THROW( InvalidArgument( "PETSc error in lumped mass solve" ) );
        return;
    }

    PetscOwner< KSP, KSPDestroy > solver;
    if( KSPCreate( PetscObjectComm( reinterpret_cast< PetscObject >( mass ) ), solver.put() ) ||
        KSPSetOperators( solver.get(), mass, mass ) || KSPSetType( solver.get(), KSPCG ) ||
        KSPSetTolerances( solver.get(), 1.e-13, PETSC_DEFAULT, PETSC_DEFAULT, 1000 ) ||
        KSPSetFromOptions( solver.get() ) || KSPSolve( solver.get(), rhs, solution ) )
        KARMA_THROW( InvalidArgument( "PETSc error in consistent mass solve" ) );
    KSPConvergedReason reason = KSP_CONVERGED_ITERATING;
    if( KSPGetConvergedReason( solver.get(), &reason ) )
        KARMA_THROW( InvalidArgument( "PETSc error querying consistent mass solve reason" ) );
    if( reason <= 0 ) KARMA_THROW( InvalidArgument( "consistent mass solve did not converge" ) );
}
}  // namespace KARMA
