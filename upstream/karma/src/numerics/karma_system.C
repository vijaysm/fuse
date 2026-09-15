// KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
#include "karma_system.h"

#include "karma_exceptions.h"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

namespace
{
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

void validate_communicator( PetscObject object, MPI_Comm expected, const char* name )
{
    int relation = MPI_UNEQUAL;
    check( MPI_Comm_compare( PetscObjectComm( object ), expected, &relation ), "MPI_Comm_compare(system layout)" );
    if( relation != MPI_IDENT && relation != MPI_CONGRUENT )
        KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " communicator is not congruent with the system" ) );
}

void validate_mapping( ISLocalToGlobalMapping actual, ISLocalToGlobalMapping expected, const char* name )
{
    if( !actual ) KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " mapping must not be null" ) );
    PetscInt actual_size = 0, expected_size = 0;
    check( ISLocalToGlobalMappingGetSize( actual, &actual_size ), "ISLocalToGlobalMappingGetSize(actual)" );
    check( ISLocalToGlobalMappingGetSize( expected, &expected_size ), "ISLocalToGlobalMappingGetSize(expected)" );
    if( actual_size != expected_size )
        KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " mapping has incompatible size" ) );
    const PetscInt* actual_indices = nullptr;
    const PetscInt* expected_indices = nullptr;
    check( ISLocalToGlobalMappingGetIndices( actual, &actual_indices ), "ISLocalToGlobalMappingGetIndices(actual)" );
    const PetscErrorCode expected_get_error = ISLocalToGlobalMappingGetIndices( expected, &expected_indices );
    if( expected_get_error )
    {
        (void)ISLocalToGlobalMappingRestoreIndices( actual, &actual_indices );
        check( expected_get_error, "ISLocalToGlobalMappingGetIndices(expected)" );
    }
    bool equal = true;
    for( PetscInt i = 0; i < actual_size; ++i ) equal = equal && actual_indices[i] == expected_indices[i];
    const PetscErrorCode expected_restore_error = ISLocalToGlobalMappingRestoreIndices( expected, &expected_indices );
    const PetscErrorCode actual_restore_error = ISLocalToGlobalMappingRestoreIndices( actual, &actual_indices );
    check( expected_restore_error, "ISLocalToGlobalMappingRestoreIndices(expected)" );
    check( actual_restore_error, "ISLocalToGlobalMappingRestoreIndices(actual)" );
    if( !equal ) KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " mapping differs from the system mapping" ) );
}

void collective_layout_check( MPI_Comm communicator, const std::string& local_error )
{
    int rank = 0, ranks = 1;
    check( MPI_Comm_rank( communicator, &rank ), "MPI_Comm_rank(system layout)" );
    check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(system layout)" );
    const int candidate = local_error.empty() ? ranks : rank;
    int failing_rank = ranks;
    check( MPI_Allreduce( &candidate, &failing_rank, 1, MPI_INT, MPI_MIN, communicator ),
           "MPI_Allreduce(system layout)" );
    if( failing_rank == ranks ) return;
    int size = rank == failing_rank ? static_cast< int >( local_error.size() ) : 0;
    check( MPI_Bcast( &size, 1, MPI_INT, failing_rank, communicator ), "MPI_Bcast(system layout size)" );
    std::string message( static_cast< std::size_t >( size ), '\0' );
    if( rank == failing_rank ) message = local_error;
    check( MPI_Bcast( message.data(), size, MPI_CHAR, failing_rank, communicator ), "MPI_Bcast(system layout message)" );
    KARMA_THROW( KARMA::InvalidArgument( "rank " + std::to_string( failing_rank ) + ": " + message ) );
}

template < class Handle, PetscErrorCode ( *Destroy )( Handle* ) >
class PetscHandleGuard
{
  public:
    ~PetscHandleGuard()
    {
        if( value ) (void)Destroy( &value );
    }
    PetscHandleGuard( const PetscHandleGuard& )            = delete;
    PetscHandleGuard& operator=( const PetscHandleGuard& ) = delete;
    PetscHandleGuard()                                     = default;
    PetscHandleGuard( PetscHandleGuard&& other ) noexcept : value( other.release() ) {}

    [[nodiscard]] Handle release() noexcept
    { return std::exchange( value, nullptr ); }

    Handle value = nullptr;
};
}  // namespace

namespace KARMA
{
KARMASystem::KARMASystem( KARMAMesh& mesh, std::string_view name ) : _mesh( mesh ), _name( name ) {}

KARMASystem::KARMASystem( KARMAMesh& mesh, std::string_view name, KARMABasisRegistry registry )
    : _mesh( mesh ), _name( name ), _basis_registry( std::move( registry ) )
{
}

KARMASystem::~KARMASystem()
{
    for( Vec* vector : { &_temporary, &_local_solution, &_old_solution, &_current_solution, &_solution } )
        if( *vector ) (void)VecDestroy( vector );
}

unsigned KARMASystem::add_variable( VariableSpec variable )
{
    if( _initialized ) KARMA_THROW( InvalidArgument( "add_variable called after init" ) );
    if( variable.name.empty() ) KARMA_THROW( InvalidArgument( "variable name must not be empty" ) );
    if( variable.components == 0 ) KARMA_THROW( InvalidArgument( "variable component count must be positive" ) );
    if( has_variable( variable.name ) ) KARMA_THROW( InvalidArgument( "duplicate variable name" ) );
    _variables.push_back( std::move( variable ) );
    return static_cast< unsigned >( _variables.size() - 1 );
}

unsigned KARMASystem::add_variable( std::string_view name, Order order, FieldBasisFamily family,
                                    BasisContinuity continuity, unsigned components )
{ return add_variable( { std::string( name ), family, static_cast< unsigned >( order ), continuity, components } ); }

const VariableSpec& KARMASystem::variable( unsigned index ) const
{
    if( index >= _variables.size() ) KARMA_THROW( InvalidArgument( "variable index is out of range" ) );
    return _variables[index];
}

unsigned KARMASystem::variable_number( std::string_view name ) const
{
    const auto found = std::find_if( _variables.begin(), _variables.end(),
                                     [name]( const VariableSpec& variable ) { return variable.name == name; } );
    if( found == _variables.end() ) KARMA_THROW( InvalidArgument( "unknown variable name" ) );
    return static_cast< unsigned >( std::distance( _variables.begin(), found ) );
}

bool KARMASystem::has_variable( std::string_view name ) const
{
    return std::any_of( _variables.begin(), _variables.end(),
                        [name]( const VariableSpec& variable ) { return variable.name == name; } );
}

void KARMASystem::refresh_sparsity()
{
    _sparsity.global_dofs = _dof_map.n_dofs();
    _sparsity.local_dofs  = _dof_map.n_local_dofs();
    _sparsity.n_nz      = _dof_map.diagonal_nnz();
    _sparsity.o_nz      = _dof_map.off_diagonal_nnz();
    _sparsity.send_list = _dof_map.ghost_indices();
    _sparsity.nnz_per_row =
        _sparsity.n_nz.empty() ? 0 : *std::max_element( _sparsity.n_nz.begin(), _sparsity.n_nz.end() );
    _sparsity.onz_per_row =
        _sparsity.o_nz.empty() ? 0 : *std::max_element( _sparsity.o_nz.begin(), _sparsity.o_nz.end() );
}

void KARMASystem::init()
{
    if( _initialized ) return;

    KARMADofMap dof_map( _mesh, _variables, _basis_registry );
    UserSparsity sparsity;
    sparsity.global_dofs = dof_map.n_dofs();
    sparsity.local_dofs  = dof_map.n_local_dofs();
    sparsity.n_nz        = dof_map.diagonal_nnz();
    sparsity.o_nz        = dof_map.off_diagonal_nnz();
    sparsity.send_list   = dof_map.ghost_indices();
    sparsity.nnz_per_row =
        sparsity.n_nz.empty() ? 0 : *std::max_element( sparsity.n_nz.begin(), sparsity.n_nz.end() );
    sparsity.onz_per_row =
        sparsity.o_nz.empty() ? 0 : *std::max_element( sparsity.o_nz.begin(), sparsity.o_nz.end() );

    auto make_vector = [&] {
        PetscHandleGuard< Vec, VecDestroy > vector;
        check( VecCreateGhost( _mesh.parallel_comm()->comm(), dof_map.n_local_dofs(), dof_map.n_dofs(),
                               detail::checked_petsc_int_size( dof_map.ghost_indices().size() ),
                               dof_map.ghost_indices().data(), &vector.value ),
               "VecCreateGhost" );
        check( VecSetLocalToGlobalMapping( vector.value, dof_map.local_to_global_mapping() ),
               "VecSetLocalToGlobalMapping" );
        check( VecSet( vector.value, 0 ), "VecSet" );
        return vector;
    };
    auto solution         = make_vector();
    auto current_solution = make_vector();
    auto old_solution     = make_vector();
    auto local_solution   = make_vector();
    auto temporary        = make_vector();

    _dof_map          = std::move( dof_map );
    _sparsity         = std::move( sparsity );
    _solution         = solution.release();
    _current_solution = current_solution.release();
    _old_solution     = old_solution.release();
    _local_solution   = local_solution.release();
    _temporary        = temporary.release();
    _initialized      = true;
}

void KARMASystem::add_element_adjacency( moab::EntityHandle first, moab::EntityHandle second )
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "add_element_adjacency called before init" ) );
    _dof_map.add_element_adjacency( first, second );
    refresh_sparsity();
}

Vec KARMASystem::create_vector() const
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "create_vector called before init" ) );
    PetscHandleGuard< Vec, VecDestroy > vector;
    check( VecCreateGhost( _mesh.parallel_comm()->comm(), n_local_dofs(), n_dofs(),
                           detail::checked_petsc_int_size( _dof_map.ghost_indices().size() ),
                           _dof_map.ghost_indices().data(), &vector.value ),
           "VecCreateGhost" );
    check( VecSetLocalToGlobalMapping( vector.value, _dof_map.local_to_global_mapping() ),
           "VecSetLocalToGlobalMapping" );
    return vector.release();
}


void KARMASystem::validate_vector_layout( Vec vector, const char* name ) const
{
    std::string local_error;
    try
    {
        if( !vector ) KARMA_THROW( InvalidArgument( std::string( name ) + " must not be null" ) );
        validate_communicator( reinterpret_cast< PetscObject >( vector ), communicator(), name );
        PetscInt global_size = 0, local_size = 0;
        check( VecGetSize( vector, &global_size ), "VecGetSize(system layout)" );
        check( VecGetLocalSize( vector, &local_size ), "VecGetLocalSize(system layout)" );
        if( global_size != n_dofs() || local_size != n_local_dofs() )
            KARMA_THROW( InvalidArgument( std::string( name ) + " has incompatible global or owned local size" ) );
        ISLocalToGlobalMapping mapping = nullptr;
        check( VecGetLocalToGlobalMapping( vector, &mapping ), "VecGetLocalToGlobalMapping(system layout)" );
        validate_mapping( mapping, _dof_map.local_to_global_mapping(), name );
        Vec local_form = nullptr;
        const PetscErrorCode get_error = VecGhostGetLocalForm( vector, &local_form );
        if( get_error || !local_form )
            KARMA_THROW( InvalidArgument( std::string( name ) + " layout must be ghosted" ) );
        check( VecGhostRestoreLocalForm( vector, &local_form ), "VecGhostRestoreLocalForm(system layout)" );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = std::string( "unknown " ) + name + " layout validation failure"; }
    collective_layout_check( communicator(), local_error );
}

void KARMASystem::validate_matrix_layout( Mat matrix, const char* name ) const
{
    std::string local_error;
    try
    {
        if( !matrix ) KARMA_THROW( InvalidArgument( std::string( name ) + " must not be null" ) );
        validate_communicator( reinterpret_cast< PetscObject >( matrix ), communicator(), name );
        PetscInt rows = 0, columns = 0, local_rows = 0, local_columns = 0;
        check( MatGetSize( matrix, &rows, &columns ), "MatGetSize(system layout)" );
        check( MatGetLocalSize( matrix, &local_rows, &local_columns ), "MatGetLocalSize(system layout)" );
        if( rows != n_dofs() || columns != n_dofs() || local_rows != n_local_dofs() ||
            local_columns != n_local_dofs() )
            KARMA_THROW( InvalidArgument( std::string( name ) + " has incompatible global or local dimensions" ) );
        ISLocalToGlobalMapping row_mapping = nullptr, column_mapping = nullptr;
        check( MatGetLocalToGlobalMapping( matrix, &row_mapping, &column_mapping ),
               "MatGetLocalToGlobalMapping(system layout)" );
        validate_mapping( row_mapping, _dof_map.local_to_global_mapping(), name );
        validate_mapping( column_mapping, _dof_map.local_to_global_mapping(), name );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = std::string( "unknown " ) + name + " layout validation failure"; }
    collective_layout_check( communicator(), local_error );
}

Mat KARMASystem::create_matrix() const
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "create_matrix called before init" ) );
    PetscHandleGuard< Mat, MatDestroy > matrix;
    const auto& diagonal     = _dof_map.diagonal_nnz();
    const auto& off_diagonal = _dof_map.off_diagonal_nnz();
    check( MatCreateAIJ( _mesh.parallel_comm()->comm(), n_local_dofs(), n_local_dofs(), n_dofs(), n_dofs(), 0,
                         diagonal.data(), 0, off_diagonal.data(), &matrix.value ),
           "MatCreateAIJ" );
    check( MatSetLocalToGlobalMapping( matrix.value, _dof_map.local_to_global_mapping(),
                                       _dof_map.local_to_global_mapping() ),
           "MatSetLocalToGlobalMapping" );
    check( MatSetOption( matrix.value, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_TRUE ),
           "MatSetOption(MAT_NEW_NONZERO_ALLOCATION_ERR)" );
    return matrix.release();
}
}  // namespace KARMA
