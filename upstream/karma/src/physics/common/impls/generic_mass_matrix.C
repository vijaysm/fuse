#include "generic_mass_operator.h"

#include "collective_signature.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_reference_topology.h"
#include "karma_system.h"
#include "petsc_native_scatter.h"
#include "petsc_raii.h"

#include <algorithm>
#include <exception>
#include <string>
#include <vector>

namespace
{
void petsc_check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

KARMA::GeometryVariant element_variant( const KARMA::KARMAMesh& mesh, moab::EntityHandle element )
{
    const moab::EntityHandle* connectivity = nullptr;
    int count = 0;
    if( mesh.moab()->get_connectivity( element, connectivity, count ) != moab::MB_SUCCESS )
        KARMA_THROW( KARMA::InvalidArgument( "cannot read element connectivity for mass assembly" ) );
    return KARMA::KARMAReferenceTopology::from_moab( mesh.moab()->type_from_handle( element ),
                                                     static_cast< std::size_t >( count ) );
}
}  // namespace

namespace KARMA
{
RealDenseMatrix element_mass_matrix( const KARMASystem& system, moab::EntityHandle element,
                                     int quadrature_degree )
{
    const auto geometry = element_variant( system.mesh(), element );
    const auto cell = reference_cell( geometry );
    std::size_t total_size = 0;
    std::vector< std::size_t > basis_sizes;
    basis_sizes.reserve( system.n_variables() );
    for( unsigned variable = 0; variable < system.n_variables(); ++variable )
    {
        const auto& spec = system.variable( variable );
        auto basis = system.basis_registry().create( { cell, spec.family, spec.order, spec.continuity } );
        basis_sizes.push_back( basis.size() );
        total_size += spec.components * basis.size();
    }

    RealDenseMatrix result( total_size, total_size );
    std::size_t offset = 0;
    for( unsigned variable = 0; variable < system.n_variables(); ++variable )
    {
        const auto& spec = system.variable( variable );
        auto basis = system.basis_registry().create( { cell, spec.family, spec.order, spec.continuity } );
        KARMAFEBase fe( system.mesh(), geometry, std::move( basis ), quadrature_degree );
        fe.reinit( element );
        const auto& phi = fe.get_phi();
        const auto& jxw = fe.get_JxW();
        const std::size_t n_basis = basis_sizes[variable];
        for( unsigned component = 0; component < spec.components; ++component )
        {
            const std::size_t block = offset + component * n_basis;
            for( std::size_t q = 0; q < fe.n_quadrature_points(); ++q )
                for( std::size_t i = 0; i < n_basis; ++i )
                    for( std::size_t j = 0; j < n_basis; ++j )
                        result( block + i, block + j ) += jxw[q] * phi[i][q] * phi[j][q];
        }
        offset += spec.components * n_basis;
    }
    return result;
}

void assemble_mass_matrix( const KARMASystem& system, Mat mass, const PetscInt* dirichlet_dofs,
                            PetscInt n_dirichlet_dofs, int quadrature_degree )
{
    system.validate_matrix_layout( mass, "mass matrix" );
    std::string local_error;
    std::vector< PetscInt > unique_dofs;
    try
    {
        if( quadrature_degree <= 0 ) local_error = "quadrature degree must be positive";
        else if( n_dirichlet_dofs < 0 ) local_error = "Dirichlet DoF count must be nonnegative";
        else if( n_dirichlet_dofs > 0 && !dirichlet_dofs )
            local_error = "Dirichlet DoF pointer must not be null";
        else
        {
            if( n_dirichlet_dofs > 0 ) unique_dofs.assign( dirichlet_dofs, dirichlet_dofs + n_dirichlet_dofs );
            for( const PetscInt dof : unique_dofs )
                if( dof < 0 || dof >= system.n_dofs() )
                {
                    local_error = "Dirichlet DoF is out of range";
                    break;
                }
            std::sort( unique_dofs.begin(), unique_dofs.end() );
            unique_dofs.erase( std::unique( unique_dofs.begin(), unique_dofs.end() ), unique_dofs.end() );
        }
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown mass signature input preparation failure"; }
    int rank = 0, ranks = 1;
    petsc_check( MPI_Comm_rank( system.communicator(), &rank ), "MPI_Comm_rank(mass validation)" );
    petsc_check( MPI_Comm_size( system.communicator(), &ranks ), "MPI_Comm_size(mass validation)" );
    detail::require_collective_signature(
        system.communicator(), local_error,
        [&] {
            std::string signature;
            signature.append( reinterpret_cast< const char* >( &quadrature_degree ), sizeof( quadrature_degree ) );
            for( const PetscInt dof : unique_dofs )
                signature.append( reinterpret_cast< const char* >( &dof ), sizeof( dof ) );
            return signature;
        },
        "unknown mass signature construction failure",
        "mass quadrature or canonical Dirichlet list differs from rank 0" );

    struct Contribution
    {
        std::vector< PetscInt > indices;
        RealDenseMatrix matrix;
    };
    std::vector< Contribution > contributions;
    try
    {
        std::vector< PetscInt > indices;
        for( const auto element : system.dof_map().assembly_elements() )
        {
            system.dof_map().dof_indices( element, indices );
            contributions.push_back( { indices, element_mass_matrix( system, element, quadrature_degree ) } );
        }
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown mass element assembly failure"; }
    const int element_candidate = local_error.empty() ? ranks : rank;
    int failing_rank = ranks;
    petsc_check( MPI_Allreduce( &element_candidate, &failing_rank, 1, MPI_INT, MPI_MIN, system.communicator() ),
                 "MPI_Allreduce(mass element validation)" );
    if( failing_rank != ranks )
    {
        int size = rank == failing_rank ? static_cast< int >( local_error.size() ) : 0;
        petsc_check( MPI_Bcast( &size, 1, MPI_INT, failing_rank, system.communicator() ),
                     "MPI_Bcast(mass element validation size)" );
        std::string message( static_cast< std::size_t >( size ), '\0' );
        if( rank == failing_rank ) message = local_error;
        petsc_check( MPI_Bcast( message.data(), size, MPI_CHAR, failing_rank, system.communicator() ),
                     "MPI_Bcast(mass element validation message)" );
        KARMA_THROW( InvalidArgument( "rank " + std::to_string( failing_rank ) + ": " + message ) );
    }

    PetscOwner< Mat, MatDestroy > staged_mass;
    *staged_mass.put() = system.create_matrix();
    local_error.clear();
    try
    {
        for( const auto& contribution : contributions )
            scatter( staged_mass.get(), contribution.matrix, contribution.indices, contribution.indices, ADD_VALUES );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown mass scatter failure"; }
    const int scatter_candidate = local_error.empty() ? ranks : rank;
    failing_rank = ranks;
    petsc_check( MPI_Allreduce( &scatter_candidate, &failing_rank, 1, MPI_INT, MPI_MIN, system.communicator() ),
                 "MPI_Allreduce(mass scatter validation)" );
    if( failing_rank != ranks )
    {
        int size = rank == failing_rank ? static_cast< int >( local_error.size() ) : 0;
        petsc_check( MPI_Bcast( &size, 1, MPI_INT, failing_rank, system.communicator() ),
                     "MPI_Bcast(mass scatter validation size)" );
        std::string message( static_cast< std::size_t >( size ), '\0' );
        if( rank == failing_rank ) message = local_error;
        petsc_check( MPI_Bcast( message.data(), size, MPI_CHAR, failing_rank, system.communicator() ),
                     "MPI_Bcast(mass scatter validation message)" );
        KARMA_THROW( InvalidArgument( "rank " + std::to_string( failing_rank ) + ": " + message ) );
    }
    petsc_check( MatAssemblyBegin( staged_mass.get(), MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(staged mass)" );
    petsc_check( MatAssemblyEnd( staged_mass.get(), MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(staged mass)" );
    if( !unique_dofs.empty() )
    {
        petsc_check( MatZeroRowsColumns( staged_mass.get(), static_cast< PetscInt >( unique_dofs.size() ),
                                        unique_dofs.data(), 1.0, nullptr, nullptr ),
                     "MatZeroRowsColumns(staged mass)" );
    }
    petsc_check( MatAssemblyBegin( mass, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(empty mass)" );
    petsc_check( MatAssemblyEnd( mass, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(empty mass)" );
    petsc_check( MatCopy( staged_mass.get(), mass, DIFFERENT_NONZERO_PATTERN ), "MatCopy(mass)" );
    petsc_check( MatAssemblyBegin( mass, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(mass)" );
    petsc_check( MatAssemblyEnd( mass, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(mass)" );
}
}  // namespace KARMA
