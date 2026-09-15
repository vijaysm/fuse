#include "karma_dense.h"
#include "petsc_native_scatter.h"

#include <petsc.h>

#include <array>
#include <cmath>
#include <iostream>
#include <span>

namespace
{
int failures = 0;
void check( bool value, const char* message )
{
    if( !value )
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

template < typename Function > bool throws( Function&& function )
{
    try
    {
        function();
    }
    catch( const std::logic_error& )
    {
        return true;
    }
    return false;
}
}  // namespace

int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) ) return 1;
    int rank = 0, ranks = 1;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    MPI_Comm_size( PETSC_COMM_WORLD, &ranks );

    Mat matrix = nullptr;
    Vec vector = nullptr;
    MatCreateAIJ( PETSC_COMM_WORLD, 2, 2, 2 * ranks, 2 * ranks, 4, nullptr, 4, nullptr, &matrix );
    VecCreateMPI( PETSC_COMM_WORLD, 2, 2 * ranks, &vector );

    const PetscInt next = ( rank + 1 ) % ranks;
    const std::array< PetscInt, 2 > rows{ 2 * rank, 2 * next + 1 };
    const std::array< PetscInt, 2 > cols{ 2 * next, 2 * rank + 1 };
    KARMA::RealDenseMatrix block( 2, 2 );
    block( 0, 0 ) = 1;
    block( 0, 1 ) = 2;
    block( 1, 0 ) = 7;
    block( 1, 1 ) = 11;
    KARMA::scatter( matrix, block, std::span< const PetscInt >( rows ), std::span< const PetscInt >( cols ),
                    INSERT_VALUES );
    KARMA::scatter( matrix, block, std::span< const PetscInt >( rows ), std::span< const PetscInt >( cols ),
                    INSERT_VALUES );

    KARMA::RealDenseVector load( 2 );
    load( 0 ) = 3;
    load( 1 ) = 5;
    KARMA::scatter( vector, load, std::span< const PetscInt >( rows ), INSERT_VALUES );
    KARMA::scatter( vector, load, std::span< const PetscInt >( rows ), INSERT_VALUES );
    MatAssemblyBegin( matrix, MAT_FLUSH_ASSEMBLY );
    MatAssemblyEnd( matrix, MAT_FLUSH_ASSEMBLY );
    VecAssemblyBegin( vector );
    VecAssemblyEnd( vector );
    KARMA::scatter( matrix, block, std::span< const PetscInt >( rows ), std::span< const PetscInt >( cols ), ADD_VALUES );
    KARMA::scatter( vector, load, std::span< const PetscInt >( rows ), ADD_VALUES );
    MatAssemblyBegin( matrix, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( matrix, MAT_FINAL_ASSEMBLY );
    VecAssemblyBegin( vector );
    VecAssemblyEnd( vector );

    PetscScalar value = 0;
    const PetscInt previous = ( rank + ranks - 1 ) % ranks;
    const std::array< PetscInt, 2 > owned_rows{ 2 * rank, 2 * rank + 1 };
    const std::array< std::array< PetscInt, 2 >, 2 > owned_columns{
        std::array< PetscInt, 2 >{ 2 * next, 2 * rank + 1 },
        std::array< PetscInt, 2 >{ 2 * rank, 2 * previous + 1 } };
    const std::array< std::array< PetscScalar, 2 >, 2 > expected_matrix{
        std::array< PetscScalar, 2 >{ 2 * block( 0, 0 ), 2 * block( 0, 1 ) },
        std::array< PetscScalar, 2 >{ 2 * block( 1, 0 ), 2 * block( 1, 1 ) } };
    for( std::size_t i = 0; i < owned_rows.size(); ++i )
        for( std::size_t j = 0; j < owned_columns[i].size(); ++j )
        {
            MatGetValues( matrix, 1, &owned_rows[i], 1, &owned_columns[i][j], &value );
            const PetscScalar expected = expected_matrix[i][j];
            check( std::abs( PetscRealPart( value ) - PetscRealPart( expected ) ) < 1.e-12,
                   "owned rows contain local and off-process row-major matrix entries" );
        }
    for( std::size_t i = 0; i < owned_rows.size(); ++i )
    {
        VecGetValues( vector, 1, &owned_rows[i], &value );
        const PetscScalar expected = i == 0 ? 2 * load( 0 ) : 2 * load( 1 );
        check( std::abs( PetscRealPart( value ) - PetscRealPart( expected ) ) < 1.e-12,
               "owned vector entries include off-process insertion" );
    }

    Mat local_matrix = nullptr;
    Vec local_vector = nullptr;
    MatCreateSeqAIJ( PETSC_COMM_SELF, 2, 2, 2, nullptr, &local_matrix );
    VecCreateSeq( PETSC_COMM_SELF, 2, &local_vector );
    const std::array< PetscInt, 2 > mapping_indices{ 1, 0 };
    ISLocalToGlobalMapping mapping = nullptr;
    ISLocalToGlobalMappingCreate( PETSC_COMM_SELF, 1, 2, mapping_indices.data(), PETSC_COPY_VALUES, &mapping );
    MatSetLocalToGlobalMapping( local_matrix, mapping, mapping );
    VecSetLocalToGlobalMapping( local_vector, mapping );
    const std::array< PetscInt, 2 > local_indices{ 0, 1 };
    KARMA::scatter_local( local_matrix, block, local_indices, local_indices, INSERT_VALUES );
    KARMA::scatter_local( local_vector, load, local_indices, INSERT_VALUES );
    MatAssemblyBegin( local_matrix, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( local_matrix, MAT_FINAL_ASSEMBLY );
    VecAssemblyBegin( local_vector );
    VecAssemblyEnd( local_vector );
    const PetscInt global_zero = 0;
    MatGetValues( local_matrix, 1, &global_zero, 1, &global_zero, &value );
    check( std::abs( PetscRealPart( value ) - 11 ) < 1.e-12, "local matrix insertion honors local ordering" );
    VecGetValues( local_vector, 1, &global_zero, &value );
    check( std::abs( PetscRealPart( value ) - 5 ) < 1.e-12, "local vector insertion honors local ordering" );
    const std::array< PetscInt, 1 > negative{ -1 };
    const std::array< PetscInt, 1 > global_out_of_range{ 2 * ranks };
    const std::array< PetscInt, 1 > local_out_of_range{ 2 };
    KARMA::RealDenseVector singleton( 1 );
    singleton( 0 ) = 1;
    check( throws( [&] { KARMA::scatter( vector, singleton, negative, ADD_VALUES ); } ),
           "global vector scatter rejects negative indices" );
    check( throws( [&] { KARMA::scatter( vector, singleton, global_out_of_range, ADD_VALUES ); } ),
           "global vector scatter rejects out-of-range indices" );
    check( throws( [&] { KARMA::scatter_local( local_vector, singleton, local_out_of_range, ADD_VALUES ); } ),
           "local vector scatter rejects indices outside its mapping" );
    ISLocalToGlobalMappingDestroy( &mapping );
    MatDestroy( &local_matrix );
    VecDestroy( &local_vector );

    MatDestroy( &matrix );
    VecDestroy( &vector );
    const int local_failures = failures;
    int global_failures      = 0;
    MPI_Allreduce( &local_failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
    PetscFinalize();
    return global_failures == 0 ? 0 : 1;
}
