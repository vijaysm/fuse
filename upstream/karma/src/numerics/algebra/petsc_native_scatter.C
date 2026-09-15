#include "petsc_native_scatter.h"

#include "karma_exceptions.h"

#include <algorithm>
#include <string>
#include <vector>

namespace
{
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

std::vector< PetscScalar > row_major( const KARMA::RealDenseMatrix& values )
{
    std::vector< PetscScalar > result( values.rows() * values.cols() );
    for( std::size_t row = 0; row < values.rows(); ++row )
        for( std::size_t column = 0; column < values.cols(); ++column )
            result[row * values.cols() + column] = static_cast< PetscScalar >( values( row, column ) );
    return result;
}

PetscInt checked_size( std::size_t size )
{
    if( size > static_cast< std::size_t >( PETSC_MAX_INT ) )
        KARMA_THROW( KARMA::InvalidArgument( "scatter extent exceeds PetscInt" ) );
    return static_cast< PetscInt >( size );
}

void validate_indices( std::span< const PetscInt > indices, PetscInt size, const char* kind )
{
    if( std::any_of( indices.begin(), indices.end(), [size]( PetscInt index ) { return index < 0 || index >= size; } ) )
        KARMA_THROW( KARMA::InvalidArgument( std::string( kind ) + " scatter index is out of range" ) );
}

void validate( const KARMA::RealDenseMatrix& values, std::span< const PetscInt > rows,
               std::span< const PetscInt > columns )
{
    if( values.rows() != rows.size() || values.cols() != columns.size() )
        KARMA_THROW( KARMA::InvalidArgument( "dense matrix and index extents differ" ) );
}

std::vector< PetscScalar > packed( const KARMA::RealDenseVector& values )
{
    std::vector< PetscScalar > result( values.size() );
    for( std::size_t i = 0; i < values.size(); ++i ) result[i] = static_cast< PetscScalar >( values( i ) );
    return result;
}
}  // namespace

namespace KARMA
{
void scatter( Mat matrix, const RealDenseMatrix& values, std::span< const PetscInt > rows,
              std::span< const PetscInt > columns, InsertMode mode )
{
    validate( values, rows, columns );
    PetscInt global_rows = 0, global_columns = 0;
    check( MatGetSize( matrix, &global_rows, &global_columns ), "MatGetSize" );
    validate_indices( rows, global_rows, "global matrix row" );
    validate_indices( columns, global_columns, "global matrix column" );
    const auto values_row_major = row_major( values );
    check( MatSetValues( matrix, checked_size( rows.size() ), rows.data(), checked_size( columns.size() ), columns.data(),
                         values_row_major.data(), mode ),
           "MatSetValues" );
}

void scatter_local( Mat matrix, const RealDenseMatrix& values, std::span< const PetscInt > rows,
                    std::span< const PetscInt > columns, InsertMode mode )
{
    validate( values, rows, columns );
    ISLocalToGlobalMapping row_mapping = nullptr, column_mapping = nullptr;
    check( MatGetLocalToGlobalMapping( matrix, &row_mapping, &column_mapping ), "MatGetLocalToGlobalMapping" );
    PetscInt row_size = 0, column_size = 0;
    check( ISLocalToGlobalMappingGetSize( row_mapping, &row_size ), "ISLocalToGlobalMappingGetSize(row)" );
    check( ISLocalToGlobalMappingGetSize( column_mapping, &column_size ), "ISLocalToGlobalMappingGetSize(column)" );
    validate_indices( rows, row_size, "local matrix row" );
    validate_indices( columns, column_size, "local matrix column" );
    const auto values_row_major = row_major( values );
    check( MatSetValuesLocal( matrix, checked_size( rows.size() ), rows.data(), checked_size( columns.size() ),
                              columns.data(), values_row_major.data(), mode ),
           "MatSetValuesLocal" );
}

void scatter( Vec vector, const RealDenseVector& values, std::span< const PetscInt > indices, InsertMode mode )
{
    if( values.size() != indices.size() ) KARMA_THROW( InvalidArgument( "dense vector and index extents differ" ) );
    PetscInt global_size = 0;
    check( VecGetSize( vector, &global_size ), "VecGetSize" );
    validate_indices( indices, global_size, "global vector" );
    const auto scalar_values = packed( values );
    check( VecSetValues( vector, checked_size( indices.size() ), indices.data(), scalar_values.data(), mode ),
           "VecSetValues" );
}

void scatter_local( Vec vector, const RealDenseVector& values, std::span< const PetscInt > indices, InsertMode mode )
{
    if( values.size() != indices.size() ) KARMA_THROW( InvalidArgument( "dense vector and index extents differ" ) );
    ISLocalToGlobalMapping mapping = nullptr;
    check( VecGetLocalToGlobalMapping( vector, &mapping ), "VecGetLocalToGlobalMapping" );
    PetscInt local_size = 0;
    check( ISLocalToGlobalMappingGetSize( mapping, &local_size ), "ISLocalToGlobalMappingGetSize" );
    validate_indices( indices, local_size, "local vector" );
    const auto scalar_values = packed( values );
    check( VecSetValuesLocal( vector, checked_size( indices.size() ), indices.data(), scalar_values.data(), mode ),
           "VecSetValuesLocal" );
}
}  // namespace KARMA
