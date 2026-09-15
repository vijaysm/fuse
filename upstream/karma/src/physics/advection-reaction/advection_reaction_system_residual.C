#include "advection_reaction_assembly.h"

#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_reference_topology.h"
#include "karma_system.h"
#include "petsc_native_scatter.h"
#include "petsc_raii.h"
#include "../common/impls/collective_signature.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
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
    int count = 0;
    if( mesh.moab()->get_connectivity( element, connectivity, count ) != moab::MB_SUCCESS )
        KARMA_THROW( KARMA::InvalidArgument( "cannot read element connectivity for advection assembly" ) );
    return KARMA::KARMAReferenceTopology::from_moab( mesh.moab()->type_from_handle( element ),
                                                     static_cast< std::size_t >( count ) );
}

Real derivative( const KARMA::AdvectionScalar& value, const KARMA::AdvectionScalar& exact,
                 const KARMA::Point& point, Real field, int material )
{
    if( exact ) return exact( point, field, material );
    const Real step = std::cbrt( std::numeric_limits< Real >::epsilon() ) * std::max( Real{ 1 }, std::abs( field ) );
    return ( value( point, field + step, material ) - value( point, field - step, material ) ) / ( 2 * step );
}

void require_finite( Real value, const char* name )
{
    if( !std::isfinite( value ) )
        KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " callback returned a non-finite value" ) );
}

KARMA::KARMABasis basis_for( const KARMA::KARMASystem& system, moab::EntityHandle element )
{
    const auto geometry = variant( system.mesh(), element );
    const auto& spec = system.variable( 0 );
    return system.basis_registry().create( { KARMA::reference_cell( geometry ), spec.family, spec.order,
                                              spec.continuity } );
}

KARMA::RealDenseVector values_for( const KARMA::KARMASystem& system, moab::EntityHandle element,
                                   const PetscScalar* local_values )
{
    std::vector< PetscInt > local_indices;
    system.dof_map().local_dof_indices( element, local_indices );
    KARMA::RealDenseVector values( local_indices.size() );
    for( std::size_t i = 0; i < local_indices.size(); ++i )
        values( i ) = PetscRealPart( local_values[local_indices[i]] );
    return values;
}

bool has_duplicate_element_ids( std::vector< KARMA::GlobalId > ids )
{
    std::sort( ids.begin(), ids.end() );
    return std::adjacent_find( ids.begin(), ids.end() ) != ids.end();
}

void collective_check( MPI_Comm communicator, const std::string& local_error )
{
    KARMA::detail::collective_signature_error( communicator, local_error );
}

struct VectorContribution
{
    std::vector< PetscInt > indices;
    KARMA::RealDenseVector values;
};

struct MatrixContribution
{
    std::vector< PetscInt > rows;
    std::vector< PetscInt > columns;
    KARMA::RealDenseMatrix values;
};
}  // namespace

namespace KARMA
{
#ifdef KARMA_TEST_DUPLICATE_INTERIOR_FACE_TRAVERSAL
#define assemble_advection_reaction assemble_advection_reaction_with_duplicate_interior_faces
namespace
{
std::size_t duplicate_interior_face_traversals = 0;
}

std::size_t duplicate_interior_face_traversal_count()
{
    return duplicate_interior_face_traversals;
}
#endif

void assemble_advection_reaction( const KARMASystem& system, Vec solution, Vec residual, Mat jacobian,
                                   const AdvectionReactionCoefficients& coefficients, int quadrature_degree )
{
    const MPI_Comm communicator = system.communicator();
    std::string local_error;
    int ranks = 1;
    check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(advection Jacobian presence)" );
    const int local_presence = jacobian ? 1 : 0;
    std::vector< int > presence;
    KARMA::detail::prepare_collectively( communicator, [&] { presence.resize( static_cast< std::size_t >( ranks ) ); },
                                         "advection Jacobian presence allocation failed" );
    check( MPI_Allgather( &local_presence, 1, MPI_INT, presence.data(), 1, MPI_INT, communicator ),
           "MPI_Allgather(advection Jacobian presence)" );
    for( int rank = 1; rank < ranks; ++rank )
        if( presence[static_cast< std::size_t >( rank )] != presence[0] )
            KARMA_THROW( InvalidArgument( "rank " + std::to_string( rank ) +
                                          ": advection Jacobian presence differs from rank 0" ) );
    try
    {
        system.validate_vector_layout( solution, "advection solution" );
        system.validate_vector_layout( residual, "advection residual" );
        if( jacobian ) system.validate_matrix_layout( jacobian, "advection Jacobian" );
        if( system.n_variables() != 1 || system.variable( 0 ).components != 1 )
            KARMA_THROW( InvalidArgument( "advection-reaction assembly requires one scalar variable" ) );
        if( system.variable( 0 ).continuity != BasisContinuity::dG )
            KARMA_THROW( InvalidArgument( "advection-reaction assembly requires dG continuity" ) );
        if( !coefficients.velocity || !coefficients.reaction || !coefficients.source || !coefficients.inflow )
            KARMA_THROW( InvalidArgument( "advection-reaction callbacks must be set" ) );
        if( quadrature_degree <= 0 ) KARMA_THROW( InvalidArgument( "quadrature degree must be positive" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown advection configuration validation failure"; }
    collective_check( communicator, local_error );

    check( VecGhostUpdateBegin( solution, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateBegin(advection)" );
    check( VecGhostUpdateEnd( solution, INSERT_VALUES, SCATTER_FORWARD ), "VecGhostUpdateEnd(advection)" );

    local_error.clear();
    if( ranks == 1 )
    {
        try
        {
            std::vector< GlobalId > visible_ids;
            system.mesh().global_ids( system.mesh().elements(), visible_ids );
            if( has_duplicate_element_ids( std::move( visible_ids ) ) )
                local_error = "duplicate global element id in advection assembly";
        }
        catch( const std::exception& error )
        {
            local_error = error.what();
        }
        catch( ... )
        {
            local_error = "unknown advection element ID validation failure";
        }
    }
    collective_check( communicator, local_error );
    std::vector< VectorContribution > vector_contributions;
    std::vector< MatrixContribution > matrix_contributions;
#ifdef KARMA_TEST_DUPLICATE_INTERIOR_FACE_TRAVERSAL
    std::size_t local_duplicate_interior_face_traversals = 0;
#endif
    {
        PetscVecLocalForm local_solution( solution );
        PetscVecArrayRead local_values( local_solution.get() );
        try
        {
            std::vector< PetscInt > indices, neighbor_indices;
            for( const auto element : system.dof_map().assembly_elements() )
            {
                system.dof_map().dof_indices( element, indices );
                const auto values   = values_for( system, element, local_values.get() );
                const auto geometry = variant( system.mesh(), element );
                KARMAFEBase volume( system.mesh(), geometry, basis_for( system, element ), quadrature_degree );
                volume.reinit( element );
                RealDenseVector element_residual( indices.size() );
                RealDenseMatrix diagonal( indices.size(), indices.size() );
                const int material = system.mesh().material_id( element );
                for( std::size_t q = 0; q < volume.n_quadrature_points(); ++q )
                {
                    Real field = 0;
                    for( std::size_t j = 0; j < indices.size(); ++j )
                        field += volume.get_phi()[j][q] * values( j );
                    const auto velocity = coefficients.velocity( volume.get_xyz()[q], material );
                    const Real reaction = coefficients.reaction( volume.get_xyz()[q], field, material );
                    const Real source   = coefficients.source( volume.get_xyz()[q], field, material );
                    if( !velocity.allFinite() )
                        KARMA_THROW( InvalidArgument( "velocity callback returned a non-finite value" ) );
                    require_finite( reaction, "reaction" );
                    require_finite( source, "source" );
                    const Real reaction_prime = jacobian ? derivative( coefficients.reaction,
                                                                        coefficients.reaction_derivative,
                                                                        volume.get_xyz()[q], field, material ) : 0;
                    const Real source_prime = jacobian ? derivative( coefficients.source,
                                                                      coefficients.source_derivative,
                                                                      volume.get_xyz()[q], field, material ) : 0;
                    if( jacobian )
                    {
                        require_finite( reaction_prime, "reaction derivative" );
                        require_finite( source_prime, "source derivative" );
                    }
                    for( std::size_t i = 0; i < indices.size(); ++i )
                    {
                        element_residual( i ) += volume.get_JxW()[q] *
                            ( -volume.get_dphi()[i][q].dot( velocity ) * field +
                              volume.get_phi()[i][q] * ( reaction - source ) );
                        if( jacobian )
                            for( std::size_t j = 0; j < indices.size(); ++j )
                                diagonal( i, j ) += volume.get_JxW()[q] *
                                    ( -volume.get_dphi()[i][q].dot( velocity ) * volume.get_phi()[j][q] +
                                      volume.get_phi()[i][q] * ( reaction_prime - source_prime ) *
                                          volume.get_phi()[j][q] );
                    }
                }

                KARMAFEBase face( system.mesh(), geometry, basis_for( system, element ), quadrature_degree,
                                  FEIntegrationDomain::Face );
                for( unsigned side = 0; side < system.mesh().n_sides( element ); ++side )
                {
                    const auto neighbor = system.mesh().neighbor( element, side );
#ifdef KARMA_TEST_DUPLICATE_INTERIOR_FACE_TRAVERSAL
                    if( neighbor ) ++local_duplicate_interior_face_traversals;
#else
                    if( neighbor && system.dof_map().canonical_element_ordinal( element ) >
                                        system.dof_map().canonical_element_ordinal( *neighbor ) )
                        continue;
#endif
                    face.reinit( element, side );
                    if( !neighbor )
                    {
                        for( std::size_t q = 0; q < face.n_quadrature_points(); ++q )
                        {
                            Real interior = 0;
                            for( std::size_t j = 0; j < indices.size(); ++j )
                                interior += face.get_phi()[j][q] * values( j );
                            const auto velocity = coefficients.velocity( face.get_xyz()[q], material );
                            if( !velocity.allFinite() )
                                KARMA_THROW( InvalidArgument( "velocity callback returned a non-finite value" ) );
                            const Real speed = velocity.dot( face.get_normals()[q] );
                            const Real exterior = speed < 0 ? coefficients.inflow( face.get_xyz()[q], material ) : interior;
                            require_finite( exterior, "inflow" );
                            const Real flux = speed * ( speed >= 0 ? interior : exterior );
                            require_finite( flux, "boundary flux" );
                            for( std::size_t i = 0; i < indices.size(); ++i )
                            {
                                element_residual( i ) += face.get_JxW()[q] * face.get_phi()[i][q] * flux;
                                if( jacobian && speed >= 0 )
                                    for( std::size_t j = 0; j < indices.size(); ++j )
                                        diagonal( i, j ) += face.get_JxW()[q] * face.get_phi()[i][q] * speed *
                                                            face.get_phi()[j][q];
                            }
                        }
                        continue;
                    }

                    system.dof_map().dof_indices( *neighbor, neighbor_indices );
                    const auto neighbor_values   = values_for( system, *neighbor, local_values.get() );
                    const auto neighbor_geometry = variant( system.mesh(), *neighbor );
                    KARMAFEBase neighbor_face( system.mesh(), neighbor_geometry, basis_for( system, *neighbor ),
                                               quadrature_degree, FEIntegrationDomain::Face );
                    neighbor_face.reinit_at_physical_points( *neighbor, face.get_xyz() );
                    RealDenseVector neighbor_residual( neighbor_indices.size() );
                    RealDenseMatrix neighbor_diagonal( neighbor_indices.size(), neighbor_indices.size() );
                    RealDenseMatrix element_neighbor( indices.size(), neighbor_indices.size() );
                    RealDenseMatrix neighbor_element( neighbor_indices.size(), indices.size() );
                    const int neighbor_material = system.mesh().material_id( *neighbor );
                    for( std::size_t q = 0; q < face.n_quadrature_points(); ++q )
                    {
                        Real interior = 0, exterior = 0;
                        for( std::size_t j = 0; j < indices.size(); ++j )
                            interior += face.get_phi()[j][q] * values( j );
                        for( std::size_t j = 0; j < neighbor_indices.size(); ++j )
                            exterior += neighbor_face.get_phi()[j][q] * neighbor_values( j );
                        const auto interior_velocity = coefficients.velocity( face.get_xyz()[q], material );
                        const auto exterior_velocity = coefficients.velocity( face.get_xyz()[q], neighbor_material );
                        if( !interior_velocity.allFinite() || !exterior_velocity.allFinite() )
                            KARMA_THROW( InvalidArgument( "velocity callback returned a non-finite value" ) );
                        const Real interior_speed = interior_velocity.dot( face.get_normals()[q] );
                        const Real exterior_speed = exterior_velocity.dot( face.get_normals()[q] );
                        const Real alpha = std::max( std::abs( interior_speed ), std::abs( exterior_speed ) );
                        const Real left_derivative  = 0.5 * ( interior_speed + alpha );
                        const Real right_derivative = 0.5 * ( exterior_speed - alpha );
                        const Real flux = rusanov_flux( interior, exterior, interior_velocity,
                                                        exterior_velocity, face.get_normals()[q] );
                        require_finite( flux, "interior flux" );
                        for( std::size_t i = 0; i < indices.size(); ++i )
                            element_residual( i ) += face.get_JxW()[q] * face.get_phi()[i][q] * flux;
                        for( std::size_t i = 0; i < neighbor_indices.size(); ++i )
                            neighbor_residual( i ) -= face.get_JxW()[q] * neighbor_face.get_phi()[i][q] * flux;
                        if( !jacobian ) continue;
                        for( std::size_t i = 0; i < indices.size(); ++i )
                            for( std::size_t j = 0; j < indices.size(); ++j )
                                diagonal( i, j ) += face.get_JxW()[q] * face.get_phi()[i][q] *
                                                    left_derivative * face.get_phi()[j][q];
                        for( std::size_t i = 0; i < indices.size(); ++i )
                            for( std::size_t j = 0; j < neighbor_indices.size(); ++j )
                                element_neighbor( i, j ) += face.get_JxW()[q] * face.get_phi()[i][q] *
                                                            right_derivative * neighbor_face.get_phi()[j][q];
                        for( std::size_t i = 0; i < neighbor_indices.size(); ++i )
                            for( std::size_t j = 0; j < indices.size(); ++j )
                                neighbor_element( i, j ) -= face.get_JxW()[q] * neighbor_face.get_phi()[i][q] *
                                                            left_derivative * face.get_phi()[j][q];
                        for( std::size_t i = 0; i < neighbor_indices.size(); ++i )
                            for( std::size_t j = 0; j < neighbor_indices.size(); ++j )
                                neighbor_diagonal( i, j ) -= face.get_JxW()[q] * neighbor_face.get_phi()[i][q] *
                                                             right_derivative * neighbor_face.get_phi()[j][q];
                    }
                    vector_contributions.push_back( { neighbor_indices, std::move( neighbor_residual ) } );
                    if( jacobian )
                    {
                        matrix_contributions.push_back(
                            { neighbor_indices, neighbor_indices, std::move( neighbor_diagonal ) } );
                        matrix_contributions.push_back( { indices, neighbor_indices, std::move( element_neighbor ) } );
                        matrix_contributions.push_back( { neighbor_indices, indices, std::move( neighbor_element ) } );
                    }
                }
                vector_contributions.push_back( { indices, std::move( element_residual ) } );
                if( jacobian )
                    matrix_contributions.push_back( { indices, indices, std::move( diagonal ) } );
            }
        }
        catch( const std::exception& error )
        {
            local_error = error.what();
        }
        catch( ... )
        {
            local_error = "unknown advection element or face assembly failure";
        }
        collective_check( communicator, local_error );
    }
#ifdef KARMA_TEST_DUPLICATE_INTERIOR_FACE_TRAVERSAL
    const std::uint64_t local_traversals = local_duplicate_interior_face_traversals;
    std::uint64_t global_traversals = 0;
    check( MPI_Allreduce( &local_traversals, &global_traversals, 1, MPI_UINT64_T, MPI_SUM, communicator ),
           "MPI_Allreduce(duplicate advection interior-face traversals)" );
    duplicate_interior_face_traversals = static_cast< std::size_t >( global_traversals );
#endif

    check( VecSet( residual, 0 ), "VecSet(advection residual)" );
    if( jacobian ) check( MatZeroEntries( jacobian ), "MatZeroEntries(advection)" );
    try
    {
        for( const auto& contribution : vector_contributions )
            scatter( residual, contribution.values, contribution.indices, ADD_VALUES );
        if( jacobian )
            for( const auto& contribution : matrix_contributions )
                scatter( jacobian, contribution.values, contribution.rows, contribution.columns, ADD_VALUES );
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    catch( ... )
    {
        local_error = "unknown advection scatter failure";
    }
    collective_check( communicator, local_error );
    check( VecAssemblyBegin( residual ), "VecAssemblyBegin(advection)" );
    check( VecAssemblyEnd( residual ), "VecAssemblyEnd(advection)" );
    if( jacobian )
    {
        check( MatAssemblyBegin( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(advection)" );
        check( MatAssemblyEnd( jacobian, MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(advection)" );
    }
}
}  // namespace KARMA
