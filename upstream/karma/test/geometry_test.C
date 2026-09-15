// Tests for reference topology and geometry mappings.

#include "karma_geometry_map.h"
#include "karma_mesh.h"
#include "karma_reference_topology.h"

#include "moab/CN.hpp"
#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
int failures = 0;

void check( bool value, const std::string& message )
{
    if( !value )
    {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void close( KARMA::Real value, KARMA::Real expected, const std::string& message, KARMA::Real tolerance = 2.e-11 )
{
    check( std::isfinite( value ) && std::abs( value - expected ) <= tolerance, message );
}

void close_scaled( KARMA::Real value, KARMA::Real expected, const std::string& message,
                   KARMA::Real relative = 2.e-10, KARMA::Real absolute = 1.e-300 )
{
    check( std::isfinite( value ) && std::isfinite( expected ) &&
               std::abs( value - expected ) <= absolute + relative * std::abs( expected ), message );
}

template < class Function > bool throws( Function&& function )
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

using KARMA::GeometryVariant;
using KARMA::Point;
using KARMA::Real;

constexpr std::array variants{
    GeometryVariant::Edge2, GeometryVariant::Edge3, GeometryVariant::Tri3, GeometryVariant::Tri6,
    GeometryVariant::Quad4, GeometryVariant::Quad8, GeometryVariant::Quad9, GeometryVariant::Tet4, GeometryVariant::Tet10,
    GeometryVariant::Hex8, GeometryVariant::Hex27, GeometryVariant::Prism6, GeometryVariant::Prism15,
    GeometryVariant::Prism18, GeometryVariant::Pyramid5, GeometryVariant::Pyramid13, GeometryVariant::Pyramid14
};

std::vector< Point > expected_nodes( GeometryVariant variant )
{
    switch( variant )
    {
        case GeometryVariant::Edge2: return { { 0, 0, 0 }, { 1, 0, 0 } };
        case GeometryVariant::Edge3: return { { 0, 0, 0 }, { 1, 0, 0 }, { .5, 0, 0 } };
        case GeometryVariant::Tri3: return { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 } };
        case GeometryVariant::Tri6: return { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { .5, 0, 0 }, { .5, .5, 0 }, { 0, .5, 0 } };
        case GeometryVariant::Quad4: return { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 } };
        case GeometryVariant::Quad8: return { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { .5, 0, 0 }, { 1, .5, 0 }, { .5, 1, 0 }, { 0, .5, 0 } };
        case GeometryVariant::Quad9: return { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { .5, 0, 0 }, { 1, .5, 0 }, { .5, 1, 0 }, { 0, .5, 0 }, { .5, .5, 0 } };
        case GeometryVariant::Tet4: return { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 } };
        case GeometryVariant::Tet10: return { { 0, 0, 0 }, { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { .5, 0, 0 }, { .5, .5, 0 }, { 0, .5, 0 }, { 0, 0, .5 }, { .5, 0, .5 }, { 0, .5, .5 } };
        case GeometryVariant::Hex8: return { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 } };
        case GeometryVariant::Hex27: return { { 0, 0, 0 }, { 1, 0, 0 }, { 1, 1, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 0, 1 }, { 1, 1, 1 }, { 0, 1, 1 }, { .5, 0, 0 }, { 1,.5,0 }, { .5,1,0 }, { 0,.5,0 }, { 0,0,.5 }, { 1,0,.5 }, { 1,1,.5 }, { 0,1,.5 }, { .5,0,1 }, { 1,.5,1 }, { .5,1,1 }, { 0,.5,1 }, { .5,0,.5 }, { 1,.5,.5 }, { .5,1,.5 }, { 0,.5,.5 }, { .5,.5,0 }, { .5,.5,1 }, { .5,.5,.5 } };
        case GeometryVariant::Prism6: return { { 0,0,0 }, { 1,0,0 }, { 0,1,0 }, { 0,0,1 }, { 1,0,1 }, { 0,1,1 } };
        case GeometryVariant::Prism15: return { { 0,0,0 }, { 1,0,0 }, { 0,1,0 }, { 0,0,1 }, { 1,0,1 }, { 0,1,1 }, { .5,0,0 }, { .5,.5,0 }, { 0,.5,0 }, { 0,0,.5 }, { 1,0,.5 }, { 0,1,.5 }, { .5,0,1 }, { .5,.5,1 }, { 0,.5,1 } };
        case GeometryVariant::Prism18: return { { 0,0,0 }, { 1,0,0 }, { 0,1,0 }, { 0,0,1 }, { 1,0,1 }, { 0,1,1 }, { .5,0,0 }, { .5,.5,0 }, { 0,.5,0 }, { 0,0,.5 }, { 1,0,.5 }, { 0,1,.5 }, { .5,0,1 }, { .5,.5,1 }, { 0,.5,1 }, { .5,0,.5 }, { .5,.5,.5 }, { 0,.5,.5 } };
        case GeometryVariant::Pyramid5: return { { -1,-1,0 }, { 1,-1,0 }, { 1,1,0 }, { -1,1,0 }, { 0,0,1 } };
        case GeometryVariant::Pyramid13: return { { -1,-1,0 }, { 1,-1,0 }, { 1,1,0 }, { -1,1,0 }, { 0,0,1 }, { 0,-1,0 }, { 1,0,0 }, { 0,1,0 }, { -1,0,0 }, { -.5,-.5,.5 }, { .5,-.5,.5 }, { .5,.5,.5 }, { -.5,.5,.5 } };
        case GeometryVariant::Pyramid14: return { { -1,-1,0 }, { 1,-1,0 }, { 1,1,0 }, { -1,1,0 }, { 0,0,1 }, { 0,-1,0 }, { 1,0,0 }, { 0,1,0 }, { -1,0,0 }, { -.5,-.5,.5 }, { .5,-.5,.5 }, { .5,.5,.5 }, { -.5,.5,.5 }, { 0,0,0 } };
    }
    return {};
}

std::vector< std::vector< std::size_t > > expected_sides( GeometryVariant variant )
{
    switch( variant )
    {
        case GeometryVariant::Edge2: case GeometryVariant::Edge3: return { { 0 }, { 1 } };
        case GeometryVariant::Tri3: return { {0,1}, {1,2}, {2,0} };
        case GeometryVariant::Tri6: return { {0,1,3}, {1,2,4}, {2,0,5} };
        case GeometryVariant::Quad4: return { {0,1}, {1,2}, {2,3}, {3,0} };
        case GeometryVariant::Quad8: case GeometryVariant::Quad9: return { {0,1,4}, {1,2,5}, {2,3,6}, {3,0,7} };
        case GeometryVariant::Tet4: return { {0,1,3}, {1,2,3}, {0,3,2}, {0,2,1} };
        case GeometryVariant::Tet10: return { {0,1,3,4,8,7}, {1,2,3,5,9,8}, {0,3,2,7,9,6}, {0,2,1,6,5,4} };
        case GeometryVariant::Hex8: return { {0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {3,0,4,7}, {0,3,2,1}, {4,5,6,7} };
        case GeometryVariant::Hex27: return { {0,1,5,4,8,13,16,12,20}, {1,2,6,5,9,14,17,13,21}, {2,3,7,6,10,15,18,14,22}, {3,0,4,7,11,12,19,15,23}, {0,3,2,1,11,10,9,8,24}, {4,5,6,7,16,17,18,19,25} };
        case GeometryVariant::Prism6: return { {0,1,4,3}, {1,2,5,4}, {0,3,5,2}, {0,2,1}, {3,4,5} };
        case GeometryVariant::Prism15: return { {0,1,4,3,6,10,12,9}, {1,2,5,4,7,11,13,10}, {0,3,5,2,9,14,11,8}, {0,2,1,8,7,6}, {3,4,5,12,13,14} };
        case GeometryVariant::Prism18: return { {0,1,4,3,6,10,12,9,15}, {1,2,5,4,7,11,13,10,16}, {0,3,5,2,9,14,11,8,17}, {0,2,1,8,7,6}, {3,4,5,12,13,14} };
        case GeometryVariant::Pyramid5: return { {0,1,4}, {1,2,4}, {2,3,4}, {3,0,4}, {0,3,2,1} };
        case GeometryVariant::Pyramid13: return { {0,1,4,5,10,9}, {1,2,4,6,11,10}, {2,3,4,7,12,11}, {3,0,4,8,9,12}, {0,3,2,1,8,7,6,5} };
        case GeometryVariant::Pyramid14: return { {0,1,4,5,10,9}, {1,2,4,6,11,10}, {2,3,4,7,12,11}, {3,0,4,8,9,12}, {0,3,2,1,8,7,6,5,13} };
    }
    return {};
}

Point expected_reference_normal( moab::EntityType topology, unsigned side )
{
    if( topology == moab::MBTRI ) return std::array{ Point(0,-1,0), Point(1,1,0), Point(-1,0,0) }[side];
    if( topology == moab::MBQUAD ) return std::array{ Point(0,-1,0), Point(1,0,0), Point(0,1,0), Point(-1,0,0) }[side];
    if( topology == moab::MBTET ) return std::array{ Point(0,-1,0), Point(1,1,1), Point(-1,0,0), Point(0,0,-1) }[side];
    if( topology == moab::MBHEX ) return std::array{ Point(0,-1,0), Point(1,0,0), Point(0,1,0), Point(-1,0,0), Point(0,0,-1), Point(0,0,1) }[side];
    if( topology == moab::MBPRISM ) return std::array{ Point(0,-1,0), Point(1,1,0), Point(-1,0,0), Point(0,0,-1), Point(0,0,1) }[side];
    return std::array{ Point(0,-1,1), Point(1,0,1), Point(0,1,1), Point(-1,0,1), Point(0,0,-1) }[side];
}

Point affine( const Point& reference )
{
    return Point( 1.0 + 2.0 * reference( 0 ) - 0.5 * reference( 1 ),
                  -2.0 + 0.25 * reference( 0 ) + 1.5 * reference( 1 ) + 0.5 * reference( 2 ),
                  0.75 - reference( 0 ) + 0.4 * reference( 1 ) + 2.0 * reference( 2 ) );
}

moab::EntityHandle add_element( KARMA::KARMAMesh& mesh, moab::EntityType type, std::span< const Point > points )
{
    std::vector< Real > coordinates;
    for( const Point& point : points )
        coordinates.insert( coordinates.end(), point.data(), point.data() + 3 );
    moab::Range vertices;
    check( mesh.moab()->create_vertices( coordinates.data(), static_cast< int >( points.size() ), vertices ) == moab::MB_SUCCESS,
           "create geometry vertices" );
    std::vector< moab::EntityHandle > connectivity( vertices.begin(), vertices.end() );
    moab::EntityHandle element = 0;
    check( mesh.moab()->create_element( type, connectivity.data(), static_cast< int >( connectivity.size() ), element ) ==
               moab::MB_SUCCESS,
           "create geometry element" );
    mesh.update();
    return element;
}

void test_reference_variants()
{
    using namespace KARMA;
    for( const GeometryVariant variant : variants )
    {
        const KARMAReferenceTopology topology( variant );
        const std::string tag = " variant " + std::to_string( static_cast< int >( variant ) );
        check( topology.node_count() == topology.reference_nodes().size(), "variant node count" + tag );
        const auto oracle_nodes = expected_nodes( variant );
        check( topology.reference_nodes().size() == oracle_nodes.size(), "hard-coded reference node count" + tag );
        for( std::size_t i = 0; i < oracle_nodes.size(); ++i )
            close( ( topology.reference_nodes()[i] - oracle_nodes[i] ).norm(), 0.0,
                   "hard-coded reference node ordering" + tag, 0.0 );
        const auto expected_order = []( GeometryVariant value ) {
            switch( value )
            {
                case GeometryVariant::Edge2: case GeometryVariant::Tri3: case GeometryVariant::Quad4:
                case GeometryVariant::Tet4: case GeometryVariant::Hex8: case GeometryVariant::Prism6:
                case GeometryVariant::Pyramid5: return 1u;
                default: return 2u;
            }
        };
        check( topology.order() == expected_order( variant ), "literal variant order" + tag );
        check( KARMAReferenceTopology::from_moab( topology.topology(), topology.node_count() ) == variant,
               "MOAB topology and node count identify variant" + tag );
        for( std::size_t point = 0; point < topology.node_count(); ++point )
        {
            const Point reference_node = topology.reference_nodes()[point];
            const auto nodal_values = topology.evaluate_values( reference_node );
            Real sum = 0.0;
            for( std::size_t node = 0; node < nodal_values.size(); ++node )
            {
                close( nodal_values[node], node == point ? 1.0 : 0.0, "Kronecker property" + tag, 3.e-10 );
                sum += nodal_values[node];
            }
            close( sum, 1.0, "nodal partition unity" + tag, 3.e-10 );
            if( topology.topology() != moab::MBPYRAMID || point != 4 )
            {
                Point derivative_sum = Point::Zero();
                for( const Point& derivative : topology.evaluate( reference_node ).derivatives ) derivative_sum += derivative;
                close( derivative_sum.norm(), 0.0, "nodal derivative sum" + tag, 2.e-9 );
            }
        }
        const std::array< Real, 4 > fractions{ .13, .29, .47, .71 };
        for( std::size_t sample = 0; sample < fractions.size(); ++sample )
        {
            Point interior = topology.center();
            const Point anchor = topology.reference_nodes()[sample % topology.node_count()];
            interior = ( 1.0 - fractions[sample] ) * topology.center() + fractions[sample] * anchor;
            const auto interior_values = topology.evaluate( interior );
            Real sum = 0.0;
            Point derivative_sum = Point::Zero();
            for( std::size_t i = 0; i < topology.node_count(); ++i )
            {
                sum += interior_values.values[i];
                derivative_sum += interior_values.derivatives[i];
            }
            close( sum, 1.0, "deterministic interior partition unity" + tag, 2.e-10 );
            close( derivative_sum.norm(), 0.0, "deterministic interior derivative sum" + tag, 2.e-9 );
            constexpr Real step = 1.e-7;
            for( unsigned component = 0; component < topology.dimension(); ++component )
            {
                Point plus = interior, minus = interior;
                plus( component ) += step;
                minus( component ) -= step;
                const auto plus_values = topology.evaluate_values( plus );
                const auto minus_values = topology.evaluate_values( minus );
                for( std::size_t basis = 0; basis < topology.node_count(); ++basis )
                    close( interior_values.derivatives[basis]( component ),
                           ( plus_values[basis] - minus_values[basis] ) / ( 2 * step ),
                           "per-basis analytic derivative matches independent finite difference" + tag, 3.e-7 );
            }
        }

        const auto side_oracles = expected_sides( variant );
        check( side_oracles.size() == topology.side_count(), "hard-coded side count" + tag );
        for( unsigned side = 0; side < topology.side_count(); ++side )
            check( topology.side_nodes( side ) == side_oracles[side], "hard-coded side ordering" + tag );
        if( topology.dimension() > 1 ) for( unsigned side = 0; side < topology.side_count(); ++side )
        {
            const auto side_nodes = topology.side_nodes( side );
            check( side_nodes.size() == topology.side_topology( side ).node_count(), "side node count matches side variant" + tag );
            const auto side_topology = topology.side_topology( side );
            const std::array< Point, 3 > side_points{ side_topology.center(),
                0.8 * side_topology.center() + 0.2 * side_topology.reference_nodes().front(),
                0.6 * side_topology.center() + 0.4 * side_topology.reference_nodes().back() };
            for( const Point& side_point : side_points )
            {
                const Point parent = topology.side_reference_map( side, side_point );
                const auto parent_values = topology.evaluate( parent );
                const auto trace_values = side_topology.evaluate( side_point );
                for( std::size_t node = 0; node < topology.node_count(); ++node )
                {
                    auto position = std::find( side_nodes.begin(), side_nodes.end(), node );
                    const Real expected = position == side_nodes.end() ? 0.0 : trace_values.values[static_cast< std::size_t >( position - side_nodes.begin() )];
                    close( parent_values.values[node], expected, "parent-to-side basis restriction" + tag, 8.e-10 );
                }
            }
        }
    }
}

void test_affine_and_curved_maps()
{
    using namespace KARMA;
    for( const GeometryVariant variant : variants )
    {
        const KARMAReferenceTopology reference( variant );
        std::vector< Point > coordinates;
        for( const Point& node : reference.reference_nodes() ) coordinates.push_back( affine( node ) );
        KARMAGeometryMap geometry( reference, coordinates );
        check( geometry.node_count() == coordinates.size(), "geometry exposes node count" );
        const Point xi = reference.center();
        close( ( geometry.map( xi ) - affine( xi ) ).norm(), 0.0, "affine reproduction", 3.e-10 );
        const auto jacobian = geometry.jacobian( xi );
        const Real step = 2.e-7;
        for( unsigned d = 0; d < reference.dimension(); ++d )
        {
            Point plus = xi;
            Point minus = xi;
            plus( d ) += step;
            minus( d ) -= step;
            close( ( jacobian.col( d ) - ( geometry.map( plus ) - geometry.map( minus ) ) / ( 2 * step ) ).norm(), 0.0,
                   "analytic Jacobian agrees with finite difference", 2.e-7 );
        }
        const Point physical = geometry.map( xi );
        close( ( geometry.inverse_map( physical ) - xi ).norm(), 0.0, "inverse roundtrip", 2.e-9 );
        check( std::isfinite( geometry.measure( xi ) ) && geometry.measure( xi ) > 0.0, "positive finite metric" );
        for( const Real scaling : { 1.e-8, 1.e8 } )
        {
            std::vector< Point > scaled_nodes;
            for( const Point& coordinate : coordinates ) scaled_nodes.push_back( scaling * coordinate );
            KARMAGeometryMap scaled_map( reference, scaled_nodes );
            const Point expected_map = scaling * geometry.map( xi );
            const Matrix3 expected_jacobian = scaling * geometry.jacobian( xi );
            const Matrix3 expected_metric = scaling * scaling * geometry.metric( xi );
            const Real expected_measure = std::pow( scaling, static_cast< int >( reference.dimension() ) ) * geometry.measure( xi );
            check( expected_map.norm() > 0.0 && expected_jacobian.norm() > 0.0 && expected_metric.norm() > 0.0 && expected_measure > 0.0,
                   "scale covariance expectations are nonzero" );
            close_scaled( ( scaled_map.map( xi ) - expected_map ).norm(), 0.0, "mapping scale invariance",
                          0.0, 2.e-10 * expected_map.norm() );
            close_scaled( ( scaled_map.jacobian( xi ) - expected_jacobian ).norm(), 0.0, "Jacobian scale covariance",
                          0.0, 2.e-10 * expected_jacobian.norm() );
            close_scaled( ( scaled_map.metric( xi ) - expected_metric ).norm(), 0.0, "metric scale covariance",
                          0.0, 3.e-10 * expected_metric.norm() );
            close_scaled( scaled_map.measure( xi ), expected_measure, "measure scale covariance" );
            close_scaled( scaled_map.measure( xi ) / geometry.measure( xi ),
                          std::pow( scaling, static_cast< int >( reference.dimension() ) ), "measure scale ratio" );
            close( ( scaled_map.inverse_map( scaled_map.map( xi ) ) - xi ).norm(), 0.0,
                   "inverse scale covariance", 3.e-9 );
            if( reference.dimension() > 1 )
            {
                const auto side_reference = reference.side_topology( 0 );
                const SideGeometry original_side = geometry.side( 0, side_reference.center() );
                const SideGeometry scaled_side = scaled_map.side( 0, side_reference.center() );
                const Real expected_side_measure =
                    std::pow( scaling, static_cast< int >( reference.dimension() - 1 ) ) * original_side.measure;
                check( expected_side_measure > 0.0, "side scale expectation is nonzero" );
                close_scaled( scaled_side.measure, expected_side_measure, "side measure scale covariance" );
                close_scaled( scaled_side.measure / original_side.measure,
                              std::pow( scaling, static_cast< int >( reference.dimension() - 1 ) ),
                              "side measure scale ratio" );
                close( ( scaled_side.normal - original_side.normal ).norm(), 0.0, "normal scale invariance", 3.e-10 );
            }
        }
        auto invalid_nodes = coordinates;
        invalid_nodes.back()( 0 ) = std::numeric_limits< Real >::quiet_NaN();
        check( throws( [&] { KARMAGeometryMap invalid( reference, invalid_nodes ); } ),
               "every variant rejects nonfinite geometry" );
        std::vector< Point > collapsed_nodes( reference.node_count(), Point::Zero() );
        check( throws( [&] { KARMAGeometryMap invalid( reference, collapsed_nodes ); } ),
               "every variant rejects degenerate geometry" );
        if( reference.dimension() > 1 )
            for( unsigned side = 0; side < reference.side_count(); ++side )
                for( const Real fraction : { .2, .5, .8 } )
                {
                    const auto side_ref = reference.side_topology( side );
                    const Point eta = ( 1 - fraction ) * side_ref.center() + fraction * side_ref.reference_nodes().front();
                    const SideGeometry face = geometry.side( side, eta );
                    const Matrix3 side_jacobian = geometry.jacobian( reference.side_reference_map( side, eta ) );
                    const Eigen::MatrixXd parent_j = side_jacobian.leftCols( reference.dimension() );
                    const Eigen::MatrixXd gram = parent_j.transpose() * parent_j;
                    const Point expected = parent_j * gram.inverse() *
                        expected_reference_normal( reference.topology(), side ).head( reference.dimension() );
                    close( ( face.normal - expected.normalized() ).norm(), 0.0,
                           "canonical affine normal for every side", 3.e-9 );
                    for( unsigned d = 0; d < reference.dimension() - 1; ++d )
                        close( dot( face.normal, face.tangents.col( d ) ), 0.0,
                               "side normal is tangent-orthogonal", 3.e-9 );
                    const Real tangent_measure = reference.dimension() == 2
                        ? face.tangents.col( 0 ).norm()
                        : cross( face.tangents.col( 0 ), face.tangents.col( 1 ) ).norm();
                    close( face.measure, tangent_measure, "side measure matches tangents", 3.e-9 );
                }
        if( reference.order() == 2 )
        {
            auto curved_nodes = coordinates;
            for( std::size_t i = static_cast< std::size_t >( moab::CN::VerticesPerEntity( reference.topology() ) ); i < curved_nodes.size(); ++i )
                curved_nodes[i] += 0.015 * Point( std::sin( 1.7 * i ), std::cos( 2.1 * i ), std::sin( .9 * i ) );
            KARMAGeometryMap curved_map( reference, curved_nodes );
            if( reference.dimension() > 1 ) for( unsigned side = 0; side < reference.side_count(); ++side )
            {
                const auto side_ref = reference.side_topology( side );
                for( const Real fraction : { .1, .45, .8 } )
                {
                    const Point eta = ( 1 - fraction ) * side_ref.center() + fraction * side_ref.reference_nodes().front();
                    const Point nearby_eta = .99 * eta + .01 * side_ref.center();
                    const SideGeometry face = curved_map.side( side, eta );
                    const SideGeometry nearby = curved_map.side( side, nearby_eta );
                    check( dot( face.normal, nearby.normal ) > .99, "curved side normal is locally continuous" );
                    for( unsigned d = 0; d < reference.dimension() - 1; ++d )
                        close( dot( face.normal, face.tangents.col( d ) ), 0.0,
                               "curved side normal is tangent-orthogonal", 2.e-9 );
                    const Real tangent_measure = reference.dimension() == 2
                        ? face.tangents.col( 0 ).norm()
                        : cross( face.tangents.col( 0 ), face.tangents.col( 1 ) ).norm();
                    close( face.measure, tangent_measure, "curved side measure matches tangents", 2.e-9 );
                    const Matrix3 side_jacobian = curved_map.jacobian( reference.side_reference_map( side, eta ) );
                    const Eigen::MatrixXd parent_j = side_jacobian.leftCols( reference.dimension() );
                    const Point canonical = parent_j * ( parent_j.transpose() * parent_j ).inverse() *
                        expected_reference_normal( reference.topology(), side ).head( reference.dimension() );
                    check( dot( face.normal, canonical ) > 0.0, "curved side follows canonical orientation" );
                }
            }
            for( const Real fraction : { .17, .43, .79 } )
            {
                const Point curved_xi = ( 1 - fraction ) * reference.center() + fraction * reference.reference_nodes().front();
                const Matrix3 curved_jacobian = curved_map.jacobian( curved_xi );
                for( unsigned d = 0; d < reference.dimension(); ++d )
                {
                    Point plus = curved_xi, minus = curved_xi;
                    plus( d ) += step; minus( d ) -= step;
                    close( ( curved_jacobian.col( d ) - ( curved_map.map( plus ) - curved_map.map( minus ) ) / ( 2 * step ) ).norm(),
                           0.0, "curved analytic Jacobian", 4.e-7 );
                }
                close( ( curved_map.inverse_map( curved_map.map( curved_xi ) ) - curved_xi ).norm(), 0.0,
                       "curved inverse roundtrip", 2.e-8 );
            }
        }
    }

    const KARMAReferenceTopology unit_edge( GeometryVariant::Edge2 );
    KARMAGeometryMap edge_measure( unit_edge, std::array{ Point( 0, 0, 0 ), Point( 0, 0, 3 ) } );
    close( edge_measure.measure( unit_edge.center() ), 3.0, "known embedded edge metric" );
    const KARMAReferenceTopology unit_tri( GeometryVariant::Tri3 );
    KARMAGeometryMap tri_measure( unit_tri, std::array{ Point( 0, 0, 2 ), Point( 2, 0, 2 ), Point( 0, 3, 2 ) } );
    close( tri_measure.measure( unit_tri.center() ), 6.0, "known surface area density" );
    const Matrix3 tri_metric = tri_measure.metric( unit_tri.center() );
    close( tri_metric( 0, 0 ), 4.0, "surface metric rr" );
    close( tri_metric( 1, 1 ), 9.0, "surface metric ss" );
    close( tri_metric( 0, 1 ), 0.0, "surface metric rs" );
    const KARMAReferenceTopology unit_tet( GeometryVariant::Tet4 );
    KARMAGeometryMap tet_measure( unit_tet, std::array{ Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 0, 3, 0 ), Point( 0, 0, 4 ) } );
    close( tet_measure.measure( unit_tet.center() ) / 6.0, 4.0, "known tetrahedron volume" );

    const KARMAReferenceTopology quad( GeometryVariant::Quad9 );
    const auto identity_bijection = quad.node_bijection( quad.reference_nodes() );
    check( identity_bijection && *identity_bijection == std::vector< std::size_t >{ 0, 1, 2, 3, 4, 5, 6, 7, 8 },
           "complete nodal points map to geometry connectivity" );
    auto reordered_nodes = quad.reference_nodes();
    std::swap( reordered_nodes[0], reordered_nodes[8] );
    const auto reordered_bijection = quad.node_bijection( reordered_nodes );
    check( reordered_bijection && reordered_bijection->front() == 8 && reordered_bijection->back() == 0,
           "nodal mapping records basis-to-connectivity permutation" );
    check( !KARMAReferenceTopology( GeometryVariant::Quad4 ).node_bijection( quad.reference_nodes() ),
           "Q2 points do not map bijectively to QUAD4 geometry" );
    reordered_nodes[1] = reordered_nodes[0];
    check( !quad.node_bijection( reordered_nodes ), "duplicate nodal points reject bijection" );
    reordered_nodes = quad.reference_nodes();
    reordered_nodes[1]( 0 ) += 1.e-12;
    check( !quad.node_bijection( reordered_nodes ), "unmatched nodal point rejects exact bijection" );

    std::vector< Point > curved;
    for( const Point& node : quad.reference_nodes() )
        curved.emplace_back( node( 0 ), node( 1 ), 0.3 * node( 0 ) * ( 1.0 - node( 0 ) ) * node( 1 ) );
    KARMAGeometryMap surface( quad, curved );
    check( surface.map( Point( 0.5, 0.5, 0 ) )( 2 ) > 0.0, "quadratic surface map is curved" );
    const Point near_boundary( 1.e-9, 1.0 - 2.e-9, 0.0 );
    close( ( surface.inverse_map( surface.map( near_boundary ) ) - near_boundary ).norm(), 0.0,
           "embedded curved surface inverse near boundary", 2.e-8 );

    const KARMAReferenceTopology folded_edge( GeometryVariant::Edge3 );
    check( throws( [&] { KARMAGeometryMap invalid( folded_edge,
        std::array{ Point( 0, 0, 0 ), Point( 1, 0, 0 ), Point( .2, 0, 0 ) } ); } ),
        "EDGE3 rejects exact interior zero-speed fold" );
    const Real edge_threshold = 256.0 * std::numeric_limits< Real >::epsilon();
    for( const Real scaling : { 1.e-8, 1.0, 1.e8 } )
    {
        check( throws( [&] { KARMAGeometryMap invalid( folded_edge,
            std::array{ Point( 0,0,0 ), Point( scaling,0,0 ),
                        Point( .25 * scaling, .2 * edge_threshold * scaling, 0 ) } ); } ),
            "EDGE3 rejects minimum speed below scaled threshold" );
        KARMAGeometryMap valid( folded_edge,
            std::array{ Point( 0,0,0 ), Point( scaling,0,0 ),
                        Point( .25 * scaling, 2.0 * edge_threshold * scaling, 0 ) } );
        check( valid.measure( Point( .25,0,0 ) ) > 0.0, "EDGE3 accepts minimum speed above scaled threshold" );
        KARMAGeometryMap endpoint_valid( folded_edge,
            std::array{ Point( 0,0,0 ), Point( scaling,0,0 ), Point( .75 * scaling, .1 * scaling, 0 ) } );
        check( endpoint_valid.measure( Point( 0,0,0 ) ) > 0.0 && endpoint_valid.measure( Point( 1,0,0 ) ) > 0.0,
               "EDGE3 endpoint-clamped stationary case remains valid" );
    }

    const KARMAReferenceTopology edge( GeometryVariant::Edge3 );
    const std::vector< Point > edge_nodes{ Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 1, 0.5, 1 ) };
    KARMAGeometryMap curve( edge, edge_nodes );
    const Point edge_xi( 0.37, 0, 0 );
    close( ( curve.inverse_map( curve.map( edge_xi ) ) - edge_xi ).norm(), 0.0, "embedded edge inverse", 2.e-9 );

    const Point apex( 0, 0, 1 );
    for( const GeometryVariant variant : { GeometryVariant::Pyramid5, GeometryVariant::Pyramid13,
                                           GeometryVariant::Pyramid14 } )
    {
        const KARMAReferenceTopology pyramid_reference( variant );
        const auto apex_values = pyramid_reference.evaluate_values( apex );
        check( apex_values.size() == pyramid_reference.node_count(), "pyramid apex value count" );
        for( std::size_t i = 0; i < apex_values.size(); ++i )
            close( apex_values[i], i == 4 ? 1.0 : 0.0, "pyramid apex value Kronecker property", 2.e-12 );
        check( throws( [&] { (void)pyramid_reference.evaluate( apex ); } ),
               "reference pyramid derivatives reject exact apex" );
    }
    const KARMAReferenceTopology pyramid( GeometryVariant::Pyramid14 );
    std::vector< Point > pyramid_nodes;
    for( const Point& node : pyramid.reference_nodes() ) pyramid_nodes.push_back( affine( node ) );
    pyramid_nodes[9] += Point( .01, -.005, .015 );
    pyramid_nodes[11] += Point( -.008, .006, -.01 );
    KARMAGeometryMap pyramid_map( pyramid, pyramid_nodes );
    close( ( pyramid_map.map( apex ) - pyramid_nodes[4] ).norm(), 0.0, "pyramid apex has exact nodal map" );
    check( throws( [&] { (void)pyramid_map.jacobian( apex ); } ), "pyramid apex Jacobian is path-dependent" );
    check( throws( [&] { (void)pyramid_map.metric( apex ); } ), "pyramid apex metric is undefined" );
    check( throws( [&] { (void)pyramid_map.measure( apex ); } ), "pyramid apex measure is undefined" );
    close( ( pyramid_map.inverse_map( pyramid_nodes[4] ) - apex ).norm(), 0.0, "pyramid apex inverse", 2.e-8 );
    const Real pyramid_scale = [&] { Real value = 0.0; for( const Point& a : pyramid_nodes ) for( const Point& b : pyramid_nodes ) value = std::max( value, ( a - b ).norm() ); return value; }();
    const Vector3 offset_direction = Vector3( 1, -.5, .25 ).normalized();
    close( ( pyramid_map.inverse_map( pyramid_nodes[4] + 1.9e-12 * pyramid_scale * offset_direction ) - apex ).norm(),
           0.0, "pyramid inverse snaps immediately below apex tolerance", 0.0 );
    check( throws( [&] { (void)pyramid_map.inverse_map( pyramid_nodes[4] + 2.1e-12 * pyramid_scale * offset_direction ); } ),
           "pyramid inverse does not snap immediately above apex tolerance" );
    for( const Point& direction : { Point( 0.2, 0.1, 0 ), Point( -0.4, 0.3, 0 ) } )
        for( const Real distance : { 1.e-2, 1.e-5, 1.e-8 } )
        {
            const Point approach( distance * direction( 0 ), distance * direction( 1 ), 1.0 - distance );
            check( pyramid_map.map( approach ).allFinite(), "pyramid approach map remains finite" );
            close( ( pyramid_map.inverse_map( pyramid_map.map( approach ) ) - approach ).norm(), 0.0,
                   "pyramid near-apex inverse roundtrip", 2.e-7 );
        }
}

Real tet10_determinant_oracle( const std::vector< Point >& nodes, const Point& point )
{
    const Real lambda[] = { 1 - point( 0 ) - point( 1 ) - point( 2 ), point( 0 ), point( 1 ), point( 2 ) };
    const Point dl[] = { Point( -1,-1,-1 ), Point( 1,0,0 ), Point( 0,1,0 ), Point( 0,0,1 ) };
    const unsigned edges[][2] = { {0,1}, {1,2}, {2,0}, {0,3}, {1,3}, {2,3} };
    KARMA::Matrix3 jacobian = KARMA::Matrix3::Zero();
    for( unsigned node = 0; node < 4; ++node )
        for( unsigned d = 0; d < 3; ++d ) jacobian.col( d ) += ( 4 * lambda[node] - 1 ) * dl[node]( d ) * nodes[node];
    for( unsigned edge = 0; edge < 6; ++edge )
    {
        const auto i = edges[edge][0], j = edges[edge][1];
        const Point derivative = 4 * ( lambda[j] * dl[i] + lambda[i] * dl[j] );
        for( unsigned d = 0; d < 3; ++d ) jacobian.col( d ) += derivative( d ) * nodes[4 + edge];
    }
    return jacobian.determinant();
}

void test_adversarial_tet10_validity()
{
    using namespace KARMA;
    const KARMAReferenceTopology reference( GeometryVariant::Tet10 );
    const Point center = reference.center();
    // Found by deterministic mt19937 mutation search (seed 3612170) over all six mid-edge nodes.
    const std::vector< Point > fixture{
        {0,0,0}, {1,0,0}, {0,1,0}, {0,0,1},
        {.84102870927517559,.49706385909810002,-.13212003053372251},
        {.88856628519567948,.873707589642851,.23980946502359621},
        {.41038063661420837,.19777878949910277,.12809959171820942},
        {-.099594929581347316,-.48018490664600549,.12375920438690896},
        {.38167953982910352,.51214488380968404,.74438420799753124},
        {-.42873150421419737,.54224340618688172,.91317070439606784}
    };
    bool old_policy_positive = tet10_determinant_oracle( fixture, center ) > 1.e-8;
    for( const Point& reference_node : reference.reference_nodes() )
        old_policy_positive = old_policy_positive &&
            tet10_determinant_oracle( fixture, .99 * reference_node + .01 * center ) > 1.e-8;
    check( old_policy_positive, "Tet10 fixture passes former node-center validity sampling" );
    const Point negative_point( 0, .25, .125 );
    check( tet10_determinant_oracle( fixture, negative_point ) < -.55,
           "closed-form independent Tet10 determinant is negative at literal point" );
    check( throws( [&] { KARMAGeometryMap invalid( reference, fixture ); } ),
           "dense validity policy rejects interior-folded Tet10 missed by node-center sampling" );
}

void test_mesh_loading_sides_and_validity()
{
    using namespace KARMA;
    const KARMAReferenceTopology prism_reference( GeometryVariant::Prism18 );
    std::vector< Point > prism_points;
    for( const Point& node : prism_reference.reference_nodes() ) prism_points.push_back( affine( node ) );
    KARMAMesh mesh( 3, PETSC_COMM_WORLD );
    const auto prism = add_element( mesh, moab::MBPRISM, prism_points );
    KARMAGeometryMap geometry( mesh, prism );
    check( throws( [&] { (void)geometry.side( geometry.reference_topology().side_count(), Point::Zero() ); } ),
           "geometry side rejects out-of-range index" );
    check( throws( [&] { (void)geometry.side_from_parent( geometry.reference_topology().side_count(),
                                                        Point::Zero(), Matrix3::Zero() ); } ),
           "geometry parent-side evaluator rejects out-of-range index before normal lookup" );
    check( geometry.variant() == GeometryVariant::Prism18 && geometry.node_count() == 18, "mesh geometry preserves prism18 variant" );
    check( KARMAReferenceTopology( GeometryVariant::Prism15 ).side_topology( 0 ).variant() == GeometryVariant::Quad8,
           "prism15 quadrilateral face is Quad8" );
    check( KARMAReferenceTopology( GeometryVariant::Prism15 ).side_nodes( 0 ).size() == 8,
           "prism15 quadrilateral face has eight nodes" );
    check( geometry.reference_topology().side_topology( 0 ).variant() == GeometryVariant::Quad9, "prism quadrilateral face is quadratic" );
    check( geometry.reference_topology().side_topology( 3 ).variant() == GeometryVariant::Tri6, "prism triangular face is quadratic" );
    for( unsigned side = 0; side < geometry.reference_topology().side_count(); ++side )
    {
        const auto side_reference = geometry.reference_topology().side_topology( side );
        const Point eta = side_reference.center();
        const auto side_geometry = geometry.side( side, eta );
        check( side_geometry.point.allFinite() && side_geometry.normal.allFinite(), "side map and normal finite" );
        close( side_geometry.normal.norm(), 1.0, "unit outward side normal" );
        const Matrix3 affine_jacobian = geometry.jacobian( geometry.reference_topology().center() );
        const Point expected = affine_jacobian.inverse().transpose() * expected_reference_normal( moab::MBPRISM, side );
        close( ( side_geometry.normal - expected.normalized() ).norm(), 0.0,
               "affine side normal follows canonical reference orientation", 2.e-10 );
    }

    const KARMAReferenceTopology pyramid13_reference( GeometryVariant::Pyramid13 );
    check( pyramid13_reference.side_topology( 4 ).variant() == GeometryVariant::Quad8,
           "pyramid13 base is Quad8" );
    check( pyramid13_reference.side_nodes( 4 ).size() == 8, "pyramid13 base has eight nodes" );

    const KARMAReferenceTopology pyramid_reference( GeometryVariant::Pyramid14 );
    check( pyramid_reference.side_topology( 0 ).variant() == GeometryVariant::Tri6,
           "pyramid side face is triangular" );
    check( pyramid_reference.side_topology( 4 ).variant() == GeometryVariant::Quad9,
           "pyramid base face is quadrilateral" );
    std::vector< Point > pyramid_points;
    for( const Point& node : pyramid_reference.reference_nodes() ) pyramid_points.push_back( affine( node ) );
    KARMAGeometryMap pyramid_geometry( pyramid_reference, pyramid_points );
    for( unsigned side = 0; side < pyramid_reference.side_count(); ++side )
    {
        const Point eta = pyramid_reference.side_topology( side ).center();
        const SideGeometry face = pyramid_geometry.side( side, eta );
        check( face.point.allFinite() && face.tangents.allFinite(), "pyramid side map and tangents finite" );
        close( face.normal.norm(), 1.0, "pyramid side unit normal" );
        check( dot( face.normal, face.point - pyramid_geometry.map( pyramid_reference.center() ) ) > 0.0,
               "pyramid side normal points outward" );
    }

    check( throws( [&] { KARMAGeometryMap bad( KARMAReferenceTopology( GeometryVariant::Tri6 ), std::span< const Point >( prism_points.data(), 5 ) ); } ),
           "invalid node count rejected" );
    auto invalid = prism_points;
    invalid[0]( 1 ) = std::numeric_limits< Real >::quiet_NaN();
    check( throws( [&] { KARMAGeometryMap bad( prism_reference, invalid ); } ), "nonfinite node rejected" );
    std::vector< Point > collapsed( prism_points.size(), Point::Zero() );
    check( throws( [&] { KARMAGeometryMap bad( prism_reference, collapsed ); } ), "degenerate geometry rejected" );

    const KARMAReferenceTopology tet( GeometryVariant::Tet4 );
    const std::vector< Point > inverted{ Point( 0, 0, 0 ), Point( 0, 1, 0 ), Point( 1, 0, 0 ), Point( 0, 0, 1 ) };
    check( throws( [&] { KARMAGeometryMap bad( tet, inverted ); } ), "inverted volume rejected" );

    const KARMAReferenceTopology folded_quad8( GeometryVariant::Quad8 );
    const std::vector< Point > folded_quad8_nodes{
        {0,0,0}, {1,0,0}, {1,1,0}, {0,1,0}, {.5,0,0}, {1,.5,0}, {.5,-.8,0}, {0,.5,0}
    };
    const Point quad_anchor = cross( Point( 1,0,0 ), Point( 0,1,0 ) );
    const auto quad8_values = folded_quad8.evaluate( Point( .5,.875,0 ) );
    Vector3 quad8_dr = Vector3::Zero(), quad8_ds = Vector3::Zero();
    for( std::size_t i = 0; i < folded_quad8_nodes.size(); ++i )
    {
        quad8_dr += quad8_values.derivatives[i]( 0 ) * folded_quad8_nodes[i];
        quad8_ds += quad8_values.derivatives[i]( 1 ) * folded_quad8_nodes[i];
    }
    check( dot( cross( quad8_dr, quad8_ds ), quad_anchor ) < 0.0,
           "independent signed normal detects folded Quad8 region" );
    check( throws( [&] { KARMAGeometryMap invalid( folded_quad8, folded_quad8_nodes ); } ),
           "folded Quad8 sampled reversal rejected" );

    const KARMAReferenceTopology folded_quad9( GeometryVariant::Quad9 );
    auto folded_quad9_nodes = expected_nodes( GeometryVariant::Quad9 );
    folded_quad9_nodes[8] = Point( .5, -.7, 0 );
    const auto quad9_values = folded_quad9.evaluate( Point( .5,.125,0 ) );
    Vector3 quad9_dr = Vector3::Zero(), quad9_ds = Vector3::Zero();
    for( std::size_t i = 0; i < folded_quad9_nodes.size(); ++i )
    {
        quad9_dr += quad9_values.derivatives[i]( 0 ) * folded_quad9_nodes[i];
        quad9_ds += quad9_values.derivatives[i]( 1 ) * folded_quad9_nodes[i];
    }
    check( dot( cross( quad9_dr, quad9_ds ), quad_anchor ) < 0.0,
           "independent signed normal detects folded Quad9 region" );
    check( throws( [&] { KARMAGeometryMap invalid( folded_quad9, folded_quad9_nodes ); } ),
           "folded Quad9 sampled reversal rejected" );

    const KARMAReferenceTopology anisotropic_quad( GeometryVariant::Quad4 );
    const std::array affine_quad_nodes{ Point( 1, 2, 3 ), Point( 3, 2, 4 ), Point( 4, 5, 6 ), Point( 2, 5, 5 ) };
    check( affine_quad_nodes[0] - affine_quad_nodes[1] + affine_quad_nodes[2] - affine_quad_nodes[3] == Point::Zero(),
           "affine Quad4 bilinear cross-term vanishes" );
    KARMAGeometryMap affine_quad( anisotropic_quad, affine_quad_nodes );
    const Point affine_origin = affine_quad_nodes[0];
    const Vector3 affine_r = affine_quad_nodes[1] - affine_origin;
    const Vector3 affine_s = affine_quad_nodes[3] - affine_origin;
    const Matrix3 affine_jacobian = [&] {
        Matrix3 result = Matrix3::Zero();
        result.col( 0 ) = affine_r;
        result.col( 1 ) = affine_s;
        return result;
    }();
    const Matrix3 affine_metric = [&] {
        Matrix3 result = Matrix3::Zero();
        result.topLeftCorner( 2, 2 ) = affine_jacobian.leftCols( 2 ).transpose() * affine_jacobian.leftCols( 2 );
        return result;
    }();
    const Real affine_measure = cross( affine_r, affine_s ).norm();
    for( const Point& reference_point : { Point( .2, .3, 0 ), Point( .8, .7, 0 ) } )
    {
        check( ( affine_quad.map( reference_point ) -
                 ( affine_origin + reference_point( 0 ) * affine_r + reference_point( 1 ) * affine_s ) ).norm() <=
                   16.0 * std::numeric_limits< Real >::epsilon(),
               "affine Quad4 map matches independent coefficient form" );
        check( ( affine_quad.jacobian( reference_point ) - affine_jacobian ).norm() <=
                   16.0 * std::numeric_limits< Real >::epsilon(),
               "affine Quad4 Jacobian matches independent constant form" );
        check( ( affine_quad.metric( reference_point ) - affine_metric ).norm() <=
                   64.0 * std::numeric_limits< Real >::epsilon(),
               "affine Quad4 metric matches independent constant form" );
        close( affine_quad.measure( reference_point ), affine_measure,
               "affine Quad4 measure matches independent constant form" );
    }
    check( ( affine_quad.jacobian( Point( .2, .3, 0 ) ) - affine_quad.jacobian( Point( .8, .7, 0 ) ) ).norm() <=
               16.0 * std::numeric_limits< Real >::epsilon(),
            "affine Quad4 has a constant Jacobian" );

    const Real condition_cutoff = 256.0 * std::numeric_limits< Real >::epsilon();
    KARMAGeometryMap accepted_anisotropy( anisotropic_quad,
        std::array{ Point(0,0,0), Point(1,0,0), Point(1,2 * condition_cutoff,0), Point(0,2 * condition_cutoff,0) } );
    const SideGeometry anisotropic_side = accepted_anisotropy.side( 0, Point(.3,0,0) );
    check( anisotropic_side.normal.allFinite() && anisotropic_side.measure > 0.0,
           "side orientation accepts anisotropy above SVD cutoff" );
    check( throws( [&] { KARMAGeometryMap rejected( anisotropic_quad,
        std::array{ Point(0,0,0), Point(1,0,0), Point(1,.5 * condition_cutoff,0), Point(0,.5 * condition_cutoff,0) } ); } ),
        "validation rejects anisotropy below SVD cutoff" );

    const KARMAReferenceTopology quad( GeometryVariant::Quad4 );
    const std::vector< Point > warped_quad{ Point( 0, 0, 0 ), Point( 2, 0, 0 ), Point( 2.2, 1, 0.3 ), Point( 0, 1, 0 ) };
    check( warped_quad[0] - warped_quad[1] + warped_quad[2] - warped_quad[3] != Point::Zero(),
           "warped Quad4 retains nonzero bilinear cross-term" );
    KARMAGeometryMap quad_map( quad, warped_quad );
    const auto tiny_quad_nodes = [&] { std::vector< Point > result; for( const Point& node : quad.reference_nodes() ) result.push_back( 1.e-20 * node ); return result; }();
    KARMAGeometryMap tiny_quad( quad, tiny_quad_nodes );
    const SideGeometry tiny_side = tiny_quad.side( 0, Point( .37, 0, 0 ) );
    check( tiny_side.measure > 0.0, "side degeneracy threshold is scale-aware" );
    check( ( quad_map.jacobian( Point( 0.2, 0.3, 0 ) ) - quad_map.jacobian( Point( 0.8, 0.7, 0 ) ) ).norm() > 1.e-4,
           "linear quad retains full bilinear map" );

    const KARMAReferenceTopology hex( GeometryVariant::Hex8 );
    std::vector< Point > warped_hex( hex.reference_nodes().begin(), hex.reference_nodes().end() );
    warped_hex[6] += Point( 0.1, 0.05, 0.1 );
    KARMAGeometryMap hex_map( hex, warped_hex );
    check( ( hex_map.jacobian( Point( 0.2, 0.3, 0.4 ) ) - hex_map.jacobian( Point( 0.8, 0.7, 0.6 ) ) ).norm() > 1.e-4,
           "linear hex retains full trilinear map" );
    check( throws( [&] { (void)hex_map.inverse_map( Point( 4, 4, 4 ) ); } ), "inverse map rejects outside volume point" );
    check( throws( [&] { (void)quad_map.inverse_map( Point( 0.5, 0.5, 4 ) ); } ), "inverse map rejects off-surface point" );
}
}  // namespace

/** Sections exist so CTest can schedule this file's independent parts in
 * parallel instead of as one ~19s serial block. They share no state: each
 * builds its own meshes and maps, and the only thing crossing between them
 * is the `failures` counter, which is per-process either way. Running with
 * no argument still runs all of them, so an interactive run is unchanged. */
int main( int argc, char** argv )
{
    if( PetscInitialize( &argc, &argv, nullptr, nullptr ) != PETSC_SUCCESS ) return 2;

    const std::string section = argc > 1 ? argv[1] : "--all";
    bool recognized = false;
    const auto selected = [&]( const char* name ) {
        if( section == "--all" ) return true;
        if( section == name )
        {
            recognized = true;
            return true;
        }
        return false;
    };

    if( selected( "--reference-variants" ) ) test_reference_variants();
    if( selected( "--affine-and-curved" ) ) test_affine_and_curved_maps();
    if( selected( "--adversarial-tet10" ) ) test_adversarial_tet10_validity();
    if( selected( "--mesh-loading" ) ) test_mesh_loading_sides_and_validity();

    // An unrecognized selector must fail loudly rather than silently pass by
    // running nothing at all - a green test that executed no assertions is
    // worse than a red one.
    if( section != "--all" && !recognized )
    {
        std::cerr << "unknown section '" << section
                  << "'; expected one of --reference-variants, --affine-and-curved, "
                     "--adversarial-tet10, --mesh-loading\n";
        PetscFinalize();
        return 2;
    }

    const PetscErrorCode error = PetscFinalize();
    return error ? error : failures == 0 ? 0 : 1;
}
