#include "karma_algebraic_system.h"

#include "karma_exceptions.h"
#include "../physics/common/impls/collective_signature.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace
{
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

void collective_validate( MPI_Comm communicator, const std::string& local_error, const std::string& signature )
{
    KARMA::detail::require_collective_signature(
        communicator, local_error, [&] { return signature; }, "algebraic signature preparation failed",
        "algebraic variable signature differs from rank 0" );
}
}  // namespace

namespace KARMA
{
KARMAAlgebraicSystem::KARMAAlgebraicSystem( MPI_Comm communicator, std::string_view name )
    : _communicator( communicator ), _name( name )
{
    if( _name.empty() ) KARMA_THROW( InvalidArgument( "system name must not be empty" ) );
}

unsigned KARMAAlgebraicSystem::add_variable( VariableSpec variable )
{
    if( _initialized ) KARMA_THROW( InvalidArgument( "add_variable called after init" ) );
    if( variable.name.empty() ) KARMA_THROW( InvalidArgument( "variable name must not be empty" ) );
    if( variable.components == 0 ) KARMA_THROW( InvalidArgument( "variable component count must be positive" ) );
    if( has_variable( variable.name ) ) KARMA_THROW( InvalidArgument( "duplicate variable name" ) );
    _variables.push_back( std::move( variable ) );
    return static_cast< unsigned >( _variables.size() - 1 );
}

const VariableSpec& KARMAAlgebraicSystem::variable( unsigned index ) const
{
    if( index >= _variables.size() ) KARMA_THROW( InvalidArgument( "variable index is out of range" ) );
    return _variables[index];
}

unsigned KARMAAlgebraicSystem::variable_number( std::string_view name ) const
{
    const auto found = std::find_if( _variables.begin(), _variables.end(),
                                     [name]( const VariableSpec& variable ) { return variable.name == name; } );
    if( found == _variables.end() ) KARMA_THROW( InvalidArgument( "unknown variable name" ) );
    return static_cast< unsigned >( std::distance( _variables.begin(), found ) );
}

bool KARMAAlgebraicSystem::has_variable( std::string_view name ) const
{
    return std::any_of( _variables.begin(), _variables.end(),
                        [name]( const VariableSpec& variable ) { return variable.name == name; } );
}

Vec KARMAAlgebraicSystem::create_vector() const
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "create_vector called before init" ) );
    Vec vector = nullptr;
    check( VecCreateMPI( _communicator, _local_size, _global_size, &vector ), "VecCreateMPI(algebraic)" );
    check( VecSet( vector, 0 ), "VecSet(algebraic)" );
    return vector;
}

Mat KARMAAlgebraicSystem::create_matrix() const
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "create_matrix called before init" ) );
    Mat matrix = nullptr;
    check( MatCreateAIJ( _communicator, _local_size, _local_size, _global_size, _global_size, _local_size, nullptr,
                         _global_size - _local_size, nullptr, &matrix ),
           "MatCreateAIJ(algebraic)" );
    return matrix;
}

void KARMAAlgebraicSystem::init()
{
    if( _initialized ) return;
    std::size_t components = 0;
    std::string local_error;
    std::string signature;
    if( _variables.empty() ) local_error = "algebraic system requires at least one variable";
    for( const auto& variable : _variables )
    {
        signature += std::to_string( variable.name.size() ) + ":" + variable.name + ":" +
                     std::to_string( variable.components ) + ";";
        if( variable.name.empty() && local_error.empty() ) local_error = "variable name must not be empty";
        if( variable.components == 0 && local_error.empty() ) local_error = "variable component count must be positive";
        if( variable.components > static_cast< std::size_t >( std::numeric_limits< PetscInt >::max() ) - components )
            local_error = "algebraic component count exceeds PetscInt";
        else components += variable.components;
    }
    collective_validate( _communicator, local_error, signature );
    const PetscInt global_size = static_cast< PetscInt >( components );
    PetscOwner< Vec, VecDestroy > prototype;
    check( VecCreateMPI( _communicator, PETSC_DECIDE, global_size, prototype.put() ), "VecCreateMPI(algebraic init)" );
    PetscInt first = 0, last = 0;
    check( VecGetOwnershipRange( prototype.get(), &first, &last ), "VecGetOwnershipRange(algebraic)" );
    check( VecSet( prototype.get(), 0 ), "VecSet(algebraic init)" );

    auto duplicate = [&] {
        PetscOwner< Vec, VecDestroy > vector;
        check( VecDuplicate( prototype.get(), vector.put() ), "VecDuplicate(algebraic)" );
        check( VecSet( vector.get(), 0 ), "VecSet(algebraic duplicate)" );
        return vector;
    };
    auto current = duplicate();
    auto old = duplicate();
    auto local = duplicate();
    auto temporary = duplicate();

    _global_size = global_size;
    _local_size = last - first;
    _first_owned = first;
    _solution = std::move( prototype );
    _current_solution = std::move( current );
    _old_solution = std::move( old );
    _local_solution = std::move( local );
    _temporary = std::move( temporary );
    _initialized = true;
}
}  // namespace KARMA
