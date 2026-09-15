#include "diffusion_reaction_assembly.h"

#include "karma_exceptions.h"
#include "karma_fe_base.h"
#include "karma_reference_topology.h"
#include "karma_system.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace
{
Real coefficient_value( const KARMA::ScalarCoefficient& coefficient, const KARMA::Point& point, Real field,
                        int material, const char* name )
{
    const Real value = coefficient( point, field, material );
    if( !std::isfinite( value ) )
        KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " callback returned a non-finite value" ) );
    return value;
}

Real coefficient_derivative( const KARMA::ScalarCoefficient& coefficient, const KARMA::ScalarCoefficient& derivative,
                             const KARMA::Point& point, Real field, int material, const char* name )
{
    if( derivative ) return coefficient_value( derivative, point, field, material, name );
    const Real step  = std::cbrt( std::numeric_limits< Real >::epsilon() ) * std::max( Real{ 1 }, std::abs( field ) );
    const Real value = ( coefficient_value( coefficient, point, field + step, material, name ) -
                         coefficient_value( coefficient, point, field - step, material, name ) ) /
                       ( 2 * step );
    if( !std::isfinite( value ) ) KARMA_THROW( KARMA::InvalidArgument( std::string( name ) + " is non-finite" ) );
    return value;
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
}  // namespace

namespace KARMA
{
DiffusionReactionElementTerms element_diffusion_reaction( const KARMASystem& system, moab::EntityHandle element,
                                                          const DiffusionReactionCoefficients& coefficients,
                                                           const RealDenseVector* element_solution,
                                                           int quadrature_degree, bool include_jacobian )
{
    if( system.n_variables() != 1 || system.variable( 0 ).components != 1 )
        KARMA_THROW( InvalidArgument( "diffusion-reaction assembly requires one scalar variable" ) );
    const auto& spec = system.variable( 0 );
    if( spec.continuity != BasisContinuity::cG )
        KARMA_THROW( InvalidArgument(
            "diffusion-reaction assembly requires cG continuity; interior diffusion fluxes are not implemented" ) );
    const auto geometry = variant( system.mesh(), element );
    auto basis =
        system.basis_registry().create( { reference_cell( geometry ), spec.family, spec.order, spec.continuity } );
    const std::size_t size = basis.size();
    if( element_solution && element_solution->size() != size )
        KARMA_THROW( InvalidArgument( "element solution size does not match diffusion basis" ) );
    KARMAFEBase fe( system.mesh(), geometry, std::move( basis ), quadrature_degree );
    fe.reinit( element );
    DiffusionReactionElementTerms result{ RealDenseMatrix( size, size ), RealDenseVector( size ),
                                          RealDenseVector( size ) };
    const int material = system.mesh().material_id( element );
    for( std::size_t q = 0; q < fe.n_quadrature_points(); ++q )
    {
        Real field              = 0;
        Gradient field_gradient = Gradient::Zero();
        if( element_solution )
            for( std::size_t j = 0; j < size; ++j )
            {
                field += fe.get_phi()[j][q] * ( *element_solution )( j );
                field_gradient += fe.get_dphi()[j][q] * ( *element_solution )( j );
            }
        const auto& point    = fe.get_xyz()[q];
        const Real diffusion = coefficient_value( coefficients.diffusion, point, field, material, "diffusion" );
        const Real reaction  = coefficient_value( coefficients.reaction, point, field, material, "reaction" );
        const Real source    = coefficient_value( coefficients.source, point, field, material, "source" );
        Real diffusion_derivative = 0;
        Real reaction_derivative  = 0;
        Real source_derivative    = 0;
        if( include_jacobian )
        {
            diffusion_derivative = coefficient_derivative( coefficients.diffusion, coefficients.diffusion_derivative,
                                                            point, field, material, "diffusion derivative" );
            reaction_derivative = coefficient_derivative( coefficients.reaction, coefficients.reaction_derivative,
                                                           point, field, material, "reaction derivative" );
            source_derivative = coefficient_derivative( coefficients.source, coefficients.source_derivative, point,
                                                         field, material, "source derivative" );
        }
        for( std::size_t i = 0; i < size; ++i )
        {
            result.load( i ) += fe.get_JxW()[q] * source * fe.get_phi()[i][q];
            result.residual( i ) +=
                fe.get_JxW()[q] * ( diffusion * fe.get_dphi()[i][q].dot( field_gradient ) +
                                    reaction * field * fe.get_phi()[i][q] - source * fe.get_phi()[i][q] );
            if( include_jacobian )
                for( std::size_t j = 0; j < size; ++j )
                    result.jacobian( i, j ) +=
                        fe.get_JxW()[q] *
                        ( diffusion * fe.get_dphi()[i][q].dot( fe.get_dphi()[j][q] ) +
                          diffusion_derivative * fe.get_phi()[j][q] * fe.get_dphi()[i][q].dot( field_gradient ) +
                          ( reaction + reaction_derivative * field - source_derivative ) * fe.get_phi()[i][q] *
                              fe.get_phi()[j][q] );
        }
    }
    return result;
}
}  // namespace KARMA
