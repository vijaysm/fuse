#include "karma_algebraic_system.h"
#include "karma_exceptions.h"

#include <petsc.h>

#include <array>
#include <iostream>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>

namespace
{
int failures = 0;

void check( bool condition, const char* message )
{
    if( !condition )
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template < class Function >
bool throws_invalid_argument( Function&& function )
{
    try
    {
        function();
    }
    catch( const KARMA::InvalidArgument& )
    {
        return true;
    }
    return false;
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    {
        const std::string_view mode = argc > 1 ? argv[1] : "";
        int rank = 0;
        MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
        if( mode == "--asymmetric-empty" || mode == "--asymmetric-name" || mode == "--asymmetric-components" )
        {
            KARMA::KARMAAlgebraicSystem asymmetric( PETSC_COMM_WORLD, "asymmetric" );
            if( !( mode == "--asymmetric-empty" && rank == 1 ) )
            {
                const std::string name = mode == "--asymmetric-name" && rank == 1 ? "other" : "u";
                asymmetric.add_variable( { name, KARMA::FieldBasisFamily::LagrangeNodal, 0,
                                           KARMA::BasisContinuity::dG, 1 } );
                if( mode == "--asymmetric-components" && rank == 1 )
                    asymmetric.add_variable( { "v", KARMA::FieldBasisFamily::LagrangeNodal, 0,
                                               KARMA::BasisContinuity::dG, 1 } );
            }
            check( throws_invalid_argument( [&] { asymmetric.init(); } ) && !asymmetric.is_initialized(),
                   "asymmetric algebraic signature fails collectively before PETSc creation" );
        }
        else
        {
        static_assert( std::is_base_of_v< KARMA::AbstractSystem, KARMA::KARMAAlgebraicSystem > );

        KARMA::KARMAAlgebraicSystem retry( PETSC_COMM_WORLD, "retry" );
        check( throws_invalid_argument( [&] { retry.init(); } ) && !retry.is_initialized(),
               "failed initialization remains retryable" );
        retry.add_variable( { "only", KARMA::FieldBasisFamily::LagrangeNodal, 0,
                              KARMA::BasisContinuity::dG, 1 } );
        retry.init();
        const Vec retry_solution = retry.solution();
        retry.init();
        check( retry.is_initialized() && retry.solution() == retry_solution,
               "algebraic initialization is idempotent" );

        KARMA::KARMAAlgebraicSystem system( PETSC_COMM_WORLD, "algebraic" );
        system.add_variable( { "temperature", KARMA::FieldBasisFamily::LagrangeNodal, 0,
                               KARMA::BasisContinuity::dG, 1 } );
        system.add_variable( { "pressure", KARMA::FieldBasisFamily::LagrangeNodal, 0,
                               KARMA::BasisContinuity::dG, 1 } );
        system.init();
        check( system.n_dofs() == 2 && system.n_variables() == 2, "component count defines algebraic DoFs" );

        PetscInt ownership_sum = 0;
        const PetscInt local_size = system.n_local_dofs();
        MPI_Allreduce( &local_size, &ownership_sum, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD );
        check( ownership_sum == system.n_dofs(), "local ownership sums to the global DoF count" );

        PetscInt first = -1, last = -1;
        VecGetOwnershipRange( system.solution(), &first, &last );
        check( first == system.first_owned_dof() && last - first == system.n_local_dofs(),
               "reported ownership matches PETSc vector range" );
        int ranks = 1;
        MPI_Comm_size( PETSC_COMM_WORLD, &ranks );
        const int zero_owned = local_size == 0 ? 1 : 0;
        int zero_owned_sum = 0;
        MPI_Allreduce( &zero_owned, &zero_owned_sum, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        if( ranks > system.n_dofs() )
            check( zero_owned_sum == ranks - system.n_dofs(), "excess ranks own zero algebraic DoFs" );

        const std::array< Vec, 5 > vectors{ system.solution(), system.current_solution(), system.old_solution(),
                                            system.local_solution(), system.temporary_vector() };
        std::set< Vec > distinct;
        for( Vec vector : vectors )
        {
            PetscReal norm = -1;
            PetscInt size = -1, vector_first = -1, vector_last = -1;
            VecGetSize( vector, &size );
            VecGetOwnershipRange( vector, &vector_first, &vector_last );
            check( vector && VecNorm( vector, NORM_INFINITY, &norm ) == PETSC_SUCCESS && norm == 0,
                   "all five state vectors exist and start zero" );
            check( size == system.n_dofs() && vector_first == first && vector_last == last,
                   "all state vectors have matching global and local dimensions" );
            distinct.insert( vector );
        }
        check( distinct.size() == vectors.size(), "the five state vectors are distinct PETSc objects" );

        Vec vector = system.create_vector();
        Mat matrix = system.create_matrix();
        PetscInt vector_size = 0, rows = 0, columns = 0, local_rows = 0, local_columns = 0;
        VecGetSize( vector, &vector_size );
        MatGetSize( matrix, &rows, &columns );
        MatGetLocalSize( matrix, &local_rows, &local_columns );
        check( vector_size == 2, "created algebraic vector has global component dimension" );
        check( rows == 2 && columns == 2, "algebraic matrix has global component dimensions" );
        check( local_rows == local_size && local_columns == local_size,
               "algebraic matrix local dimensions match ownership" );
        MatDestroy( &matrix );
        VecDestroy( &vector );
        }
    }
    int global_failures = 0;
    MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    PetscFinalize();
    return global_failures == 0 ? 0 : 1;
}
