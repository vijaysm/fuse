#include "diffusion_reaction_preconditioner_operator.h"

#include "karma_exceptions.h"
#include "petsc_raii.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace KARMA
{
void point_jacobi_solve( Mat matrix, Vec rhs, Vec solution )
{
    if( !matrix || !rhs || !solution ) KARMA_THROW( InvalidArgument( "point Jacobi requires non-null PETSc objects" ) );
    PetscOwner< Vec, VecDestroy > diagonal;
    if( VecDuplicate( rhs, diagonal.put() ) || MatGetDiagonal( matrix, diagonal.get() ) )
        KARMA_THROW( InvalidArgument( "PETSc error extracting point-Jacobi diagonal" ) );

    PetscReal norm = 0;
    if( VecNorm( diagonal.get(), NORM_INFINITY, &norm ) || !std::isfinite( norm ) )
        KARMA_THROW( InvalidArgument( "point-Jacobi diagonal is nonfinite" ) );
    const PetscReal tolerance = std::numeric_limits< PetscReal >::epsilon() * norm;
    PetscInt local_size = 0;
    if( VecGetLocalSize( diagonal.get(), &local_size ) )
        KARMA_THROW( InvalidArgument( "PETSc error validating point-Jacobi diagonal" ) );
    PetscVecArrayRead values( diagonal.get() );
    for( PetscInt i = 0; i < local_size; ++i )
        if( !std::isfinite( PetscRealPart( values.get()[i] ) ) || PetscAbsScalar( values.get()[i] ) <= tolerance )
            KARMA_THROW( InvalidArgument( "point-Jacobi diagonal contains a zero or nonfinite entry" ) );

    if( VecPointwiseDivide( solution, rhs, diagonal.get() ) )
        KARMA_THROW( InvalidArgument( "PETSc error applying point-Jacobi inverse" ) );
}
}  // namespace KARMA
