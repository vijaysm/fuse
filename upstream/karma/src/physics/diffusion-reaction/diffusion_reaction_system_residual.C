#include "diffusion_reaction_assembly.h"

#include "../common/impls/collective_signature.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_physics_bc_manager.h"
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
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

KARMA::GeometryVariant variant( const KARMA::KARMAMesh& mesh, moab::EntityHandle element )
{
    const moab::EntityHandle* connectivity = nullptr;
    int count                              = 0;
    if( mesh.moab()->get_connectivity( element, connectivity, count ) != moab::MB_SUCCESS )
        KARMA_THROW( KARMA::InvalidArgument( "cannot read element connectivity for diffusion assembly" ) );
    return KARMA::KARMAReferenceTopology::from_moab( mesh.moab()->type_from_handle( element ),
                                                     static_cast< std::size_t >( count ) );
}

void collective_check( MPI_Comm communicator, const std::string& local_error )
{
    KARMA::detail::collective_signature_error( communicator, local_error );
}

struct LocalContribution
{
    std::vector< PetscInt > indices;
    KARMA::DiffusionReactionElementTerms terms;
};
}  // namespace

namespace KARMA
{
void assemble_diffusion_reaction( const KARMASystem& system, Vec solution, Vec residual, Mat jacobian,
                                  const DiffusionReactionCoefficients& coefficients,
                                  const KARMAPhysicsBCManager* boundary, int quadrature_degree )
{
    std::string local_error;
    try
    {
        if( system.n_variables() != 1 || system.variable( 0 ).components != 1 )
            KARMA_THROW( InvalidArgument( "diffusion-reaction assembly requires one scalar variable" ) );
        if( system.variable( 0 ).continuity != BasisContinuity::cG )
            KARMA_THROW( InvalidArgument(
                "diffusion-reaction assembly requires cG continuity; interior diffusion fluxes are not implemented" ) );
        if( quadrature_degree <= 0 ) KARMA_THROW( InvalidArgument( "quadrature degree must be positive" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown diffusion assembly validation failure"; }
    collective_check( system.communicator(), local_error );
    system.validate_vector_layout( solution, "diffusion solution" );
    system.validate_vector_layout( residual, "diffusion residual" );
    const int has_jacobian = jacobian ? 1 : 0;
    int minimum_jacobian = 0, maximum_jacobian = 0;
    check( MPI_Allreduce( &has_jacobian, &minimum_jacobian, 1, MPI_INT, MPI_MIN, system.communicator() ),
           "MPI_Allreduce(diffusion Jacobian presence minimum)" );
    check( MPI_Allreduce( &has_jacobian, &maximum_jacobian, 1, MPI_INT, MPI_MAX, system.communicator() ),
           "MPI_Allreduce(diffusion Jacobian presence maximum)" );
    collective_check( system.communicator(), minimum_jacobian == maximum_jacobian
                                                ? std::string{}
                                                : "optional diffusion Jacobian presence differs across ranks" );
    if( jacobian ) system.validate_matrix_layout( jacobian, "diffusion Jacobian" );
    const int has_boundary = boundary ? 1 : 0;
    int minimum_boundary = 0, maximum_boundary = 0;
    check( MPI_Allreduce( &has_boundary, &minimum_boundary, 1, MPI_INT, MPI_MIN, system.communicator() ),
           "MPI_Allreduce(diffusion boundary presence minimum)" );
    check( MPI_Allreduce( &has_boundary, &maximum_boundary, 1, MPI_INT, MPI_MAX, system.communicator() ),
           "MPI_Allreduce(diffusion boundary presence maximum)" );
    collective_check( system.communicator(), minimum_boundary == maximum_boundary
                                                ? std::string{}
                                                : "optional diffusion boundary presence differs across ranks" );
    std::string boundary_signature;
    local_error.clear();
    try
    {
        if( boundary ) boundary_signature = boundary->assembly_signature( system );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown diffusion boundary signature failure"; }
    collective_check( system.communicator(), local_error );
    if( boundary && system.mesh().is_replicated() )
        detail::require_collective_signature(
            system.communicator(), {}, [&] { return boundary_signature; },
            "unknown diffusion boundary signature copy failure",
            "diffusion boundary signature differs from rank 0" );
    PetscOwner< Vec, VecDestroy > staged_residual;
    *staged_residual.put() = system.create_vector();
    PetscOwner< Mat, MatDestroy > staged_jacobian;
    if( jacobian ) *staged_jacobian.put() = system.create_matrix();
    Vec assembled_residual = staged_residual.get();
    Mat assembled_jacobian = staged_jacobian.get();
    PetscOwner< Vec, VecDestroy > owned_effective_solution;
    Vec effective_solution = solution;
    if( boundary && !boundary->dirichlet_dofs().empty() )
    {
        effective_solution              = system.create_vector();
        *owned_effective_solution.put() = effective_solution;
        check( VecCopy( solution, effective_solution ), "VecCopy(effective solution)" );
        for( const PetscInt dof : boundary->dirichlet_dofs() )
        {
            const PetscScalar value = ( *boundary->get_dirichlet_condition( dof ) )[0];
            check( VecSetValue( effective_solution, dof, value, INSERT_VALUES ), "VecSetValue(effective Dirichlet)" );
        }
        check( VecAssemblyBegin( effective_solution ), "VecAssemblyBegin(effective solution)" );
        check( VecAssemblyEnd( effective_solution ), "VecAssemblyEnd(effective solution)" );
    }
    check( VecGhostUpdateBegin( effective_solution, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateBegin(solution)" );
    check( VecGhostUpdateEnd( effective_solution, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateEnd(solution)" );
    {
        PetscVecLocalForm local_solution( effective_solution );
        PetscVecArrayRead local_values( local_solution.get() );

        std::vector< LocalContribution > contributions;
        local_error.clear();
        try
        {
            std::vector< PetscInt > global_indices, local_indices;
            for( const auto element : system.dof_map().assembly_elements() )
            {
                system.dof_map().dof_indices( element, global_indices );
                system.dof_map().local_dof_indices( element, local_indices );
                RealDenseVector values( local_indices.size() );
                for( std::size_t i = 0; i < local_indices.size(); ++i )
                    values( i ) = PetscRealPart( local_values.get()[local_indices[i]] );
                auto terms = element_diffusion_reaction( system, element, coefficients, &values, quadrature_degree,
                                                         assembled_jacobian != nullptr );

                if( boundary )
                {
                    const auto geometry = variant( system.mesh(), element );
                    const auto& spec    = system.variable( 0 );
                    auto basis          = system.basis_registry().create(
                        { reference_cell( geometry ), spec.family, spec.order, spec.continuity } );
                    KARMAFEBase face( system.mesh(), geometry, std::move( basis ), quadrature_degree,
                                      FEIntegrationDomain::Face );
                    for( unsigned side = 0; side < system.mesh().n_sides( element ); ++side )
                    {
                        if( !system.mesh().is_boundary_side( element, side ) ) continue;
                        const auto id = system.mesh().side_id( element, side );
                        face.reinit( element, side );
                        if( const auto* condition = boundary->get_neumann_condition( id.element, id.side ) )
                        {
                            const auto neumann = KARMAPhysicsBCManager::integrate_neumann_load( face, *condition );
                            terms.load += neumann;
                            terms.residual -= neumann;
                        }
                        if( const auto* condition = boundary->get_robin_condition( id.element, id.side ) )
                        {
                            const auto robin = KARMAPhysicsBCManager::integrate_robin( face, *condition );
                            if( assembled_jacobian ) terms.jacobian += robin.matrix;
                            terms.load += robin.load;
                            terms.residual.native().noalias() += robin.matrix.native() * values.native();
                            terms.residual -= robin.load;
                        }
                    }
                }
                contributions.push_back( { global_indices, std::move( terms ) } );
            }
        }
        catch( const std::exception& error )
        {
            local_error = error.what();
        }
        catch( ... )
        {
            local_error = "unknown diffusion element assembly failure";
        }
        const MPI_Comm communicator = system.communicator();
        collective_check( communicator, local_error );
        try
        {
            for( const auto& contribution : contributions )
            {
                if( assembled_jacobian )
                    scatter( assembled_jacobian, contribution.terms.jacobian, contribution.indices,
                             contribution.indices, ADD_VALUES );
                scatter( assembled_residual, contribution.terms.residual, contribution.indices, ADD_VALUES );
            }
        }
        catch( const std::exception& error )
        {
            local_error = error.what();
        }
        catch( ... )
        {
            local_error = "unknown diffusion scatter failure";
        }
        collective_check( communicator, local_error );
    }
    if( assembled_jacobian )
    {
        check( MatAssemblyBegin( assembled_jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(diffusion)" );
        check( MatAssemblyEnd( assembled_jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(diffusion)" );
    }
    check( VecAssemblyBegin( assembled_residual ), "VecAssemblyBegin(residual)" );
    check( VecAssemblyEnd( assembled_residual ), "VecAssemblyEnd(residual)" );

    if( boundary && !boundary->dirichlet_dofs().empty() )
    {
        if( assembled_jacobian )
            check( MatZeroRowsColumns( assembled_jacobian, static_cast< PetscInt >( boundary->dirichlet_dofs().size() ),
                                       boundary->dirichlet_dofs().data(), 1.0, nullptr, nullptr ),
                   "MatZeroRowsColumns(diffusion)" );
        PetscInt first = 0, last = 0;
        check( VecGetOwnershipRange( solution, &first, &last ), "VecGetOwnershipRange(Dirichlet solution)" );
        for( const PetscInt dof : boundary->dirichlet_dofs() )
        {
            if( dof < first || dof >= last ) continue;
            PetscScalar current = 0;
            check( VecGetValues( solution, 1, &dof, &current ), "VecGetValues(Dirichlet residual)" );
            const PetscScalar prescribed_value = ( *boundary->get_dirichlet_condition( dof ) )[0];
            const PetscScalar value            = current - prescribed_value;
            check( VecSetValue( assembled_residual, dof, value, INSERT_VALUES ), "VecSetValue(Dirichlet residual)" );
        }
        check( VecAssemblyBegin( assembled_residual ), "VecAssemblyBegin(Dirichlet residual)" );
        check( VecAssemblyEnd( assembled_residual ), "VecAssemblyEnd(Dirichlet residual)" );
    }

    local_error.clear();
    try
    {
        check( VecCopy( assembled_residual, residual ), "VecCopy(diffusion residual commit)" );
        if( assembled_jacobian )
        {
            check( MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(empty diffusion Jacobian)" );
            check( MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(empty diffusion Jacobian)" );
            check( MatCopy( assembled_jacobian, jacobian, DIFFERENT_NONZERO_PATTERN ),
                   "MatCopy(diffusion Jacobian commit)" );
            check( MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(diffusion Jacobian commit)" );
            check( MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(diffusion Jacobian commit)" );
        }
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown diffusion output commit failure"; }
    collective_check( system.communicator(), local_error );
}
}  // namespace KARMA
