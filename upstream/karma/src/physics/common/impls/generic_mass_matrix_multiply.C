#include "generic_mass_operator.h"

#include "karma_exceptions.h"
#include "karma_system.h"

namespace KARMA
{
void mass_multiply( const KARMASystem& system, Mat mass, Vec input, Vec output )
{
    system.validate_matrix_layout( mass, "mass multiply matrix" );
    system.validate_vector_layout( input, "mass multiply input" );
    system.validate_vector_layout( output, "mass multiply output" );
    if( MatMult( mass, input, output ) )
        KARMA_THROW( InvalidArgument( "PETSc error in MatMult(mass)" ) );
}
}  // namespace KARMA
