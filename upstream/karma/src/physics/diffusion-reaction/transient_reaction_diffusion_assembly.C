#include "transient_reaction_diffusion_assembly.h"

#include "../common/impls/collective_signature.h"
#include "generic_mass_operator.h"
#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_reference_topology.h"
#include "karma_system.h"
#include "petsc_native_scatter.h"
#include "petsc_raii.h"

#include <cmath>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace
{
using KARMA::GeometryVariant;
using KARMA::InvalidArgument;
using KARMA::KARMAMesh;
using KARMA::KARMAReferenceTopology;
using KARMA::Point;
using KARMA::Real;
using KARMA::TransientReaction;

void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

GeometryVariant variant( const KARMAMesh& mesh, moab::EntityHandle element )
{
    const moab::EntityHandle* connectivity = nullptr;
    int count = 0;
    if( mesh.moab()->get_connectivity( element, connectivity, count ) != moab::MB_SUCCESS )
        KARMA_THROW( InvalidArgument( "cannot read transient element connectivity" ) );
    return KARMAReferenceTopology::from_moab( mesh.moab()->type_from_handle( element ),
                                               static_cast<std::size_t>( count ) );
}

Real callback( const TransientReaction& function, const Point& point, Real state, Real coupled, Real time,
               const char* name )
{
    if( !function ) KARMA_THROW( InvalidArgument( std::string( name ) + " callback is not set" ) );
    const Real value = function( point, state, coupled, time );
    if( !std::isfinite( value ) ) KARMA_THROW( InvalidArgument( std::string( name ) + " callback returned non-finite value" ) );
    return value;
}
}

namespace KARMA
{
TransientReactionDiffusionElementTerms element_transient_reaction_diffusion(
    const KARMASystem& system, moab::EntityHandle element, const TransientReactionDiffusionModel& model,
    const TransientElementContext& context, int quadrature_degree )
{
    if( system.n_variables() != 1 || system.variable( 0 ).components != 1 )
        KARMA_THROW( InvalidArgument( "transient reaction-diffusion requires one scalar variable" ) );
    if( system.variable( 0 ).continuity != BasisContinuity::cG )
        KARMA_THROW( InvalidArgument( "transient reaction-diffusion requires cG continuity" ) );
    if( quadrature_degree <= 0 ) KARMA_THROW( InvalidArgument( "quadrature degree must be positive" ) );
    const Real h = context.time_right - context.time_left;
    if( !std::isfinite( h ) || h <= 0 ) KARMA_THROW( InvalidArgument( "transient interval must be positive and finite" ) );
    if( !std::isfinite( model.diffusion ) || model.diffusion < 0 )
        KARMA_THROW( InvalidArgument( "transient diffusion must be finite and nonnegative" ) );

    const auto geometry = variant( system.mesh(), element );
    const auto& spec = system.variable( 0 );
    auto basis = system.basis_registry().create( { reference_cell( geometry ), spec.family, spec.order, spec.continuity } );
    const std::size_t size = basis.size();
    if( context.state_left.size() != size || context.state_right.size() != size || context.coupled.size() != size )
        KARMA_THROW( InvalidArgument( "transient element vector size does not match basis" ) );
    KARMAFEBase fe( system.mesh(), geometry, std::move( basis ), quadrature_degree );
    fe.reinit( element );
    TransientReactionDiffusionElementTerms result{ RealDenseMatrix( size, size ), RealDenseVector( size ) };
    for( std::size_t q = 0; q < fe.n_quadrature_points(); ++q )
    {
        Real left = 0, right = 0, coupled = 0;
        Gradient left_gradient = Gradient::Zero(), right_gradient = Gradient::Zero();
        for( std::size_t j = 0; j < size; ++j )
        {
            left += fe.get_phi()[j][q] * context.state_left( j );
            right += fe.get_phi()[j][q] * context.state_right( j );
            coupled += fe.get_phi()[j][q] * context.coupled( j );
            left_gradient += fe.get_dphi()[j][q] * context.state_left( j );
            right_gradient += fe.get_dphi()[j][q] * context.state_right( j );
        }
        const auto& point = fe.get_xyz()[q];
        const Real reaction_left = callback( model.reaction, point, left, coupled, context.time_left, "reaction" );
        const Real reaction_right = callback( model.reaction, point, right, coupled, context.time_right, "reaction" );
        const Real derivative_right = callback( model.state_derivative, point, right, coupled, context.time_right,
                                               "state derivative" );
        const Real weight = fe.get_JxW()[q];
        for( std::size_t i = 0; i < size; ++i )
        {
            result.residual( i ) += weight * ( fe.get_phi()[i][q] * ( right - left ) / h +
                model.diffusion * fe.get_dphi()[i][q].dot( left_gradient + right_gradient ) / 2 -
                fe.get_phi()[i][q] * ( reaction_left + reaction_right ) / 2 );
            for( std::size_t j = 0; j < size; ++j )
                result.jacobian( i, j ) += weight * ( fe.get_phi()[i][q] * fe.get_phi()[j][q] / h +
                    model.diffusion * fe.get_dphi()[i][q].dot( fe.get_dphi()[j][q] ) / 2 -
                    derivative_right * fe.get_phi()[i][q] * fe.get_phi()[j][q] / 2 );
        }
    }
    return result;
}

void assemble_transient_reaction_diffusion( const KARMASystem& system, Vec state_left, Vec state_right, Vec coupled,
                                            Real time_left, Real time_right, const TransientReactionDiffusionModel& model,
                                            Vec residual, Mat jacobian, int quadrature_degree )
{
    std::string local_error;
    try
    {
        if( !model.reaction || !model.state_derivative ) KARMA_THROW( InvalidArgument( "transient callbacks must be set" ) );
        if( !std::isfinite( time_left ) || !std::isfinite( time_right ) || time_right <= time_left )
            KARMA_THROW( InvalidArgument( "transient interval must be positive and finite" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown transient assembly validation failure"; }
    detail::collective_signature_error( system.communicator(), local_error );
    system.validate_vector_layout( state_left, "transient left state" );
    system.validate_vector_layout( state_right, "transient right state" );
    system.validate_vector_layout( coupled, "transient coupled input" );
    system.validate_vector_layout( residual, "transient residual" );
    if( jacobian ) system.validate_matrix_layout( jacobian, "transient Jacobian" );

    PetscOwner<Vec, VecDestroy> staged_residual;
    *staged_residual.put() = system.create_vector();
    PetscOwner<Mat, MatDestroy> staged_jacobian;
    if( jacobian ) *staged_jacobian.put() = system.create_matrix();
    check( VecGhostUpdateBegin( state_left, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateBegin(left)" );
    check( VecGhostUpdateEnd( state_left, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateEnd(left)" );
    check( VecGhostUpdateBegin( state_right, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateBegin(right)" );
    check( VecGhostUpdateEnd( state_right, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateEnd(right)" );
    check( VecGhostUpdateBegin( coupled, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateBegin(coupled)" );
    check( VecGhostUpdateEnd( coupled, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateEnd(coupled)" );
    PetscVecLocalForm left_form( state_left ), right_form( state_right ), coupled_form( coupled );
    PetscVecArrayRead left_values( left_form.get() ), right_values( right_form.get() ), coupled_values( coupled_form.get() );
    std::vector<std::pair<std::vector<PetscInt>, TransientReactionDiffusionElementTerms>> contributions;
    local_error.clear();
    try
    {
        for( const auto element : system.dof_map().assembly_elements() )
        {
            std::vector<PetscInt> global_indices, local_indices;
            system.dof_map().dof_indices( element, global_indices );
            system.dof_map().local_dof_indices( element, local_indices );
            RealDenseVector left( local_indices.size() ), right( local_indices.size() ), input( local_indices.size() );
            for( std::size_t i = 0; i < local_indices.size(); ++i )
            {
                left( i ) = PetscRealPart( left_values.get()[local_indices[i]] );
                right( i ) = PetscRealPart( right_values.get()[local_indices[i]] );
                input( i ) = PetscRealPart( coupled_values.get()[local_indices[i]] );
            }
            contributions.emplace_back( global_indices, element_transient_reaction_diffusion(
                system, element, model, { time_left, time_right, left, right, input }, quadrature_degree ) );
        }
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown transient element assembly failure"; }
    detail::collective_signature_error( system.communicator(), local_error );
    for( const auto& contribution : contributions )
    {
        scatter( staged_residual.get(), contribution.second.residual, contribution.first, ADD_VALUES );
        if( jacobian ) scatter( staged_jacobian.get(), contribution.second.jacobian, contribution.first, contribution.first, ADD_VALUES );
    }
    check( VecAssemblyBegin( staged_residual.get() ), "VecAssemblyBegin(transient)" );
    check( VecAssemblyEnd( staged_residual.get() ), "VecAssemblyEnd(transient)" );
    if( jacobian )
    {
        check( MatAssemblyBegin( staged_jacobian.get(), MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(transient)" );
        check( MatAssemblyEnd( staged_jacobian.get(), MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(transient)" );
        check( MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(transient output)" );
        check( MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(transient output)" );
        check( MatCopy( staged_jacobian.get(), jacobian, DIFFERENT_NONZERO_PATTERN ), "MatCopy(transient Jacobian)" );
        check( MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(transient output)" );
        check( MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(transient output)" );
    }
    check( VecCopy( staged_residual.get(), residual ), "VecCopy(transient residual)" );
}
}  // namespace KARMA
