// Reference cells and analytic geometry interpolation functions.
#include "karma_reference_topology.h"

#include "karma_exceptions.h"
#include "moab/CN.hpp"

#include <Eigen/LU>
#include <Eigen/QR>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

namespace
{
using KARMA::GeometryVariant;
using KARMA::Point;
using KARMA::Real;

Point p( Real x, Real y = 0.0, Real z = 0.0 ) { return Point( x, y, z ); }

std::vector< Point > nodes_for( GeometryVariant variant )
{
    switch( variant )
    {
        case GeometryVariant::Edge2: return { p( 0 ), p( 1 ) };
        case GeometryVariant::Edge3: return { p( 0 ), p( 1 ), p( .5 ) };
        case GeometryVariant::Tri3: return { p( 0, 0 ), p( 1, 0 ), p( 0, 1 ) };
        case GeometryVariant::Tri6:
            return { p( 0, 0 ), p( 1, 0 ), p( 0, 1 ), p( .5, 0 ), p( .5, .5 ), p( 0, .5 ) };
        case GeometryVariant::Quad4: return { p( 0, 0 ), p( 1, 0 ), p( 1, 1 ), p( 0, 1 ) };
        case GeometryVariant::Quad8:
            return { p( 0, 0 ), p( 1, 0 ), p( 1, 1 ), p( 0, 1 ), p( .5, 0 ), p( 1, .5 ), p( .5, 1 ), p( 0, .5 ) };
        case GeometryVariant::Quad9:
            return { p( 0, 0 ), p( 1, 0 ), p( 1, 1 ), p( 0, 1 ), p( .5, 0 ), p( 1, .5 ), p( .5, 1 ),
                     p( 0, .5 ), p( .5, .5 ) };
        case GeometryVariant::Tet4: return { p( 0, 0, 0 ), p( 1, 0, 0 ), p( 0, 1, 0 ), p( 0, 0, 1 ) };
        case GeometryVariant::Tet10:
            return { p( 0, 0, 0 ), p( 1, 0, 0 ), p( 0, 1, 0 ), p( 0, 0, 1 ), p( .5, 0, 0 ), p( .5, .5, 0 ),
                     p( 0, .5, 0 ), p( 0, 0, .5 ), p( .5, 0, .5 ), p( 0, .5, .5 ) };
        case GeometryVariant::Hex8:
        case GeometryVariant::Hex27: {
            std::vector< Point > result{ p( 0, 0, 0 ), p( 1, 0, 0 ), p( 1, 1, 0 ), p( 0, 1, 0 ), p( 0, 0, 1 ),
                                         p( 1, 0, 1 ), p( 1, 1, 1 ), p( 0, 1, 1 ) };
            if( variant == GeometryVariant::Hex8 ) return result;
            const int edges[][2] = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 }, { 0, 4 }, { 1, 5 },
                                     { 2, 6 }, { 3, 7 }, { 4, 5 }, { 5, 6 }, { 6, 7 }, { 7, 4 } };
            for( const auto& edge : edges ) result.push_back( .5 * ( result[edge[0]] + result[edge[1]] ) );
            const int faces[][4] = { { 0, 1, 5, 4 }, { 1, 2, 6, 5 }, { 2, 3, 7, 6 },
                                     { 3, 0, 4, 7 }, { 0, 3, 2, 1 }, { 4, 5, 6, 7 } };
            for( const auto& face : faces )
                result.push_back( .25 * ( result[face[0]] + result[face[1]] + result[face[2]] + result[face[3]] ) );
            result.push_back( p( .5, .5, .5 ) );
            return result;
        }
        case GeometryVariant::Prism6:
        case GeometryVariant::Prism15:
        case GeometryVariant::Prism18: {
            std::vector< Point > result{ p( 0, 0, 0 ), p( 1, 0, 0 ), p( 0, 1, 0 ),
                                         p( 0, 0, 1 ), p( 1, 0, 1 ), p( 0, 1, 1 ) };
            if( variant == GeometryVariant::Prism6 ) return result;
            const int edges[][2] = { { 0, 1 }, { 1, 2 }, { 2, 0 }, { 0, 3 }, { 1, 4 },
                                     { 2, 5 }, { 3, 4 }, { 4, 5 }, { 5, 3 } };
            for( const auto& edge : edges ) result.push_back( .5 * ( result[edge[0]] + result[edge[1]] ) );
            if( variant == GeometryVariant::Prism18 )
            {
                result.push_back( p( .5, 0, .5 ) );
                result.push_back( p( .5, .5, .5 ) );
                result.push_back( p( 0, .5, .5 ) );
            }
            return result;
        }
        case GeometryVariant::Pyramid5:
        case GeometryVariant::Pyramid13:
        case GeometryVariant::Pyramid14: {
            std::vector< Point > result{ p( -1, -1, 0 ), p( 1, -1, 0 ), p( 1, 1, 0 ), p( -1, 1, 0 ), p( 0, 0, 1 ) };
            if( variant == GeometryVariant::Pyramid5 ) return result;
            const int edges[][2] = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 3, 0 }, { 0, 4 }, { 1, 4 }, { 2, 4 }, { 3, 4 } };
            for( const auto& edge : edges ) result.push_back( .5 * ( result[edge[0]] + result[edge[1]] ) );
            if( variant == GeometryVariant::Pyramid14 ) result.push_back( p( 0, 0, 0 ) );
            return result;
        }
    }
    return {};
}

struct Candidate
{
    Real value;
    Point derivative;
};

struct PyramidCandidates
{
    std::array< Candidate, 14 > values{};
    std::size_t size{};
};

PyramidCandidates pyramid_candidates( const Point& point, unsigned order )
{
    const Real x = point( 0 ), y = point( 1 ), z = point( 2 ), w = 1.0 - z;
    PyramidCandidates result;
    for( unsigned i = 0; i <= order; ++i )
        for( unsigned j = 0; j <= order; ++j )
        {
            const unsigned m = std::max( i, j );
            for( unsigned k = 0; k + m <= order; ++k )
            {
                if( w == 0.0 )
                {
                    const Real value = i == 0 && j == 0 ? std::pow( z, static_cast< int >( k ) ) : 0.0;
                    Real dx = 0.0, dy = 0.0, dz = 0.0;
                    if( i == 1 && j == 0 && m == 1 ) dx = std::pow( z, static_cast< int >( k ) );
                    if( j == 1 && i == 0 && m == 1 ) dy = std::pow( z, static_cast< int >( k ) );
                    if( i == 0 && j == 0 && k > 0 ) dz = k * std::pow( z, static_cast< int >( k - 1 ) );
                    result.values[result.size++] = { value, p( dx, dy, dz ) };
                    continue;
                }
                const Real value = std::pow( x, static_cast< int >( i ) ) * std::pow( y, static_cast< int >( j ) ) *
                                   std::pow( w, static_cast< int >( m - i - j ) ) * std::pow( z, static_cast< int >( k ) );
                const int w_power = static_cast< int >( m ) - static_cast< int >( i ) - static_cast< int >( j );
                // Evaluate derivatives directly to preserve removable limits on axes.
                auto direct = [&]( unsigned di, unsigned dj, unsigned dk, int dw ) {
                    if( ( di && x == 0.0 ) || ( dj && y == 0.0 ) || ( dk && z == 0.0 ) ) return 0.0;
                    return std::pow( x, static_cast< int >( di ) ) * std::pow( y, static_cast< int >( dj ) ) *
                           std::pow( w, dw ) * std::pow( z, static_cast< int >( dk ) );
                };
                result.values[result.size++] = { value,
                    p( i ? i * direct( i - 1, j, k, w_power ) : 0.0,
                       j ? j * direct( i, j - 1, k, w_power ) : 0.0,
                       ( k ? k * direct( i, j, k - 1, w_power ) : 0.0 ) -
                           ( w_power ? w_power * direct( i, j, k, w_power - 1 ) : 0.0 ) ) };

            }
        }
    return result;
}

void pyramid_values( GeometryVariant variant, const Point& point, const std::vector< Point >& nodes,
                     std::span< Real > output_values, std::span< Point > output_derivatives )
{
    const unsigned order = variant == GeometryVariant::Pyramid5 ? 1 : 2;
    static const std::vector< Point > pyramid14_nodes = nodes_for( GeometryVariant::Pyramid14 );
    const std::vector< Point >& interpolation_nodes =
        variant == GeometryVariant::Pyramid13 ? pyramid14_nodes : nodes;
    const std::size_t size = interpolation_nodes.size();
    // These two small Vandermonde inverses depend only on reference nodes.
    static const auto linear = [&] { const auto saved = nodes_for( GeometryVariant::Pyramid5 );
        const auto candidate_count = pyramid_candidates( saved.front(), 1 ).size; Eigen::MatrixXd all( saved.size(), candidate_count );
        for( std::size_t r = 0; r < saved.size(); ++r ) { const auto c = pyramid_candidates( saved[r], 1 ); for( std::size_t j = 0; j < candidate_count; ++j ) all( r, j ) = c.values[j].value; }
        Eigen::ColPivHouseholderQR< Eigen::MatrixXd > qr( all ); std::vector< Eigen::Index > selected( saved.size() ); Eigen::MatrixXd square( saved.size(), saved.size() );
        for( std::size_t j = 0; j < saved.size(); ++j ) { selected[j] = qr.colsPermutation().indices()( j ); square.col( j ) = all.col( selected[j] ); } return std::pair{ selected, square.inverse().eval() }; }();
    static const auto quadratic = [&] { const auto saved = nodes_for( GeometryVariant::Pyramid14 );
        const auto candidate_count = pyramid_candidates( saved.front(), 2 ).size; Eigen::MatrixXd all( saved.size(), candidate_count );
        for( std::size_t r = 0; r < saved.size(); ++r ) { const auto c = pyramid_candidates( saved[r], 2 ); for( std::size_t j = 0; j < candidate_count; ++j ) all( r, j ) = c.values[j].value; }
        Eigen::ColPivHouseholderQR< Eigen::MatrixXd > qr( all ); std::vector< Eigen::Index > selected( saved.size() ); Eigen::MatrixXd square( saved.size(), saved.size() );
        for( std::size_t j = 0; j < saved.size(); ++j ) { selected[j] = qr.colsPermutation().indices()( j ); square.col( j ) = all.col( selected[j] ); } return std::pair{ selected, square.inverse().eval() }; }();
    const auto& data = order == 1 ? linear : quadratic;
    const auto candidates = pyramid_candidates( point, order );
    Eigen::Matrix< Real, 1, 14 > values = Eigen::Matrix< Real, 1, 14 >::Zero();
    Eigen::Matrix< Real, 3, 14 > derivatives = Eigen::Matrix< Real, 3, 14 >::Zero();
    for( std::size_t i = 0; i < size; ++i )
    {
        values( i ) = candidates.values[static_cast< std::size_t >( data.first[i] )].value;
        derivatives.col( static_cast< Eigen::Index >( i ) ) = candidates.values[static_cast< std::size_t >( data.first[i] )].derivative;
    }
    std::array< Real, 14 > nodal_values{};
    std::array< Point, 14 > nodal_derivatives{};
    for( std::size_t column = 0; column < size; ++column )
        for( std::size_t row = 0; row < size; ++row )
        {
            const Real coefficient = data.second( static_cast< Eigen::Index >( row ),
                                                  static_cast< Eigen::Index >( column ) );
            nodal_values[column] += values( static_cast< Eigen::Index >( row ) ) * coefficient;
            nodal_derivatives[column] += derivatives.col( static_cast< Eigen::Index >( row ) ) * coefficient;
        }
    for( std::size_t i = 0; i < nodes.size(); ++i )
    {
        output_values[i] = nodal_values[i];
        output_derivatives[i] = nodal_derivatives[i];
    }
    if( variant == GeometryVariant::Pyramid13 )
    {
        constexpr Real weights[] = { -.25, -.25, -.25, -.25, 0, .5, .5, .5, .5, 0, 0, 0, 0 };
        for( std::size_t i = 0; i < nodes.size(); ++i )
        {
            output_values[i] += weights[i] * nodal_values[13];
            output_derivatives[i] += weights[i] * nodal_derivatives[13];
        }
    }
}

std::vector< std::size_t > side_indices( GeometryVariant variant, unsigned side )
{
    using Sides = std::vector< std::vector< std::size_t > >;
    const Sides* sides = nullptr;
    static const Sides edge{ { 0 }, { 1 } };
    static const Sides tri3{ {0,1}, {1,2}, {2,0} };
    static const Sides tri6{ {0,1,3}, {1,2,4}, {2,0,5} };
    static const Sides quad4{ {0,1}, {1,2}, {2,3}, {3,0} };
    static const Sides quad_quadratic{ {0,1,4}, {1,2,5}, {2,3,6}, {3,0,7} };
    static const Sides tet4{ {0,1,3}, {1,2,3}, {0,3,2}, {0,2,1} };
    static const Sides tet10{ {0,1,3,4,8,7}, {1,2,3,5,9,8}, {0,3,2,7,9,6}, {0,2,1,6,5,4} };
    static const Sides hex8{ {0,1,5,4}, {1,2,6,5}, {2,3,7,6}, {3,0,4,7}, {0,3,2,1}, {4,5,6,7} };
    static const Sides hex27{ {0,1,5,4,8,13,16,12,20}, {1,2,6,5,9,14,17,13,21},
        {2,3,7,6,10,15,18,14,22}, {3,0,4,7,11,12,19,15,23},
        {0,3,2,1,11,10,9,8,24}, {4,5,6,7,16,17,18,19,25} };
    static const Sides prism6{ {0,1,4,3}, {1,2,5,4}, {0,3,5,2}, {0,2,1}, {3,4,5} };
    static const Sides prism15{ {0,1,4,3,6,10,12,9}, {1,2,5,4,7,11,13,10},
        {0,3,5,2,9,14,11,8}, {0,2,1,8,7,6}, {3,4,5,12,13,14} };
    static const Sides prism18{ {0,1,4,3,6,10,12,9,15}, {1,2,5,4,7,11,13,10,16},
        {0,3,5,2,9,14,11,8,17}, {0,2,1,8,7,6}, {3,4,5,12,13,14} };
    static const Sides pyramid5{ {0,1,4}, {1,2,4}, {2,3,4}, {3,0,4}, {0,3,2,1} };
    static const Sides pyramid13{ {0,1,4,5,10,9}, {1,2,4,6,11,10}, {2,3,4,7,12,11},
        {3,0,4,8,9,12}, {0,3,2,1,8,7,6,5} };
    static const Sides pyramid14{ {0,1,4,5,10,9}, {1,2,4,6,11,10}, {2,3,4,7,12,11},
        {3,0,4,8,9,12}, {0,3,2,1,8,7,6,5,13} };
    switch( variant )
    {
        case GeometryVariant::Edge2: case GeometryVariant::Edge3: sides = &edge; break;
        case GeometryVariant::Tri3: sides = &tri3; break;
        case GeometryVariant::Tri6: sides = &tri6; break;
        case GeometryVariant::Quad4: sides = &quad4; break;
        case GeometryVariant::Quad8: case GeometryVariant::Quad9: sides = &quad_quadratic; break;
        case GeometryVariant::Tet4: sides = &tet4; break;
        case GeometryVariant::Tet10: sides = &tet10; break;
        case GeometryVariant::Hex8: sides = &hex8; break;
        case GeometryVariant::Hex27: sides = &hex27; break;
        case GeometryVariant::Prism6: sides = &prism6; break;
        case GeometryVariant::Prism15: sides = &prism15; break;
        case GeometryVariant::Prism18: sides = &prism18; break;
        case GeometryVariant::Pyramid5: sides = &pyramid5; break;
        case GeometryVariant::Pyramid13: sides = &pyramid13; break;
        case GeometryVariant::Pyramid14: sides = &pyramid14; break;
    }
    if( side >= sides->size() ) KARMA_THROW( KARMA::InvalidArgument( "side index is out of range" ) );
    return ( *sides )[side];
}
}  // namespace

namespace KARMA
{
KARMAReferenceTopology::KARMAReferenceTopology( GeometryVariant variant ) : _variant( variant ), _nodes( nodes_for( variant ) ) {}

GeometryVariant KARMAReferenceTopology::from_moab( moab::EntityType topology, std::size_t count )
{
    const std::pair< moab::EntityType, std::size_t > key{ topology, count };
    if( key == std::pair{ moab::MBEDGE, std::size_t{ 2 } } ) return GeometryVariant::Edge2;
    if( key == std::pair{ moab::MBEDGE, std::size_t{ 3 } } ) return GeometryVariant::Edge3;
    if( key == std::pair{ moab::MBTRI, std::size_t{ 3 } } ) return GeometryVariant::Tri3;
    if( key == std::pair{ moab::MBTRI, std::size_t{ 6 } } ) return GeometryVariant::Tri6;
    if( key == std::pair{ moab::MBQUAD, std::size_t{ 4 } } ) return GeometryVariant::Quad4;
    if( key == std::pair{ moab::MBQUAD, std::size_t{ 8 } } ) return GeometryVariant::Quad8;
    if( key == std::pair{ moab::MBQUAD, std::size_t{ 9 } } ) return GeometryVariant::Quad9;
    if( key == std::pair{ moab::MBTET, std::size_t{ 4 } } ) return GeometryVariant::Tet4;
    if( key == std::pair{ moab::MBTET, std::size_t{ 10 } } ) return GeometryVariant::Tet10;
    if( key == std::pair{ moab::MBHEX, std::size_t{ 8 } } ) return GeometryVariant::Hex8;
    if( key == std::pair{ moab::MBHEX, std::size_t{ 27 } } ) return GeometryVariant::Hex27;
    if( key == std::pair{ moab::MBPRISM, std::size_t{ 6 } } ) return GeometryVariant::Prism6;
    if( key == std::pair{ moab::MBPRISM, std::size_t{ 15 } } ) return GeometryVariant::Prism15;
    if( key == std::pair{ moab::MBPRISM, std::size_t{ 18 } } ) return GeometryVariant::Prism18;
    if( key == std::pair{ moab::MBPYRAMID, std::size_t{ 5 } } ) return GeometryVariant::Pyramid5;
    if( key == std::pair{ moab::MBPYRAMID, std::size_t{ 13 } } ) return GeometryVariant::Pyramid13;
    if( key == std::pair{ moab::MBPYRAMID, std::size_t{ 14 } } ) return GeometryVariant::Pyramid14;
    KARMA_THROW( InvalidArgument( "unsupported MOAB topology and node-count geometry variant" ) );
}

moab::EntityType KARMAReferenceTopology::topology() const noexcept
{
    if( _variant <= GeometryVariant::Edge3 ) return moab::MBEDGE;
    if( _variant <= GeometryVariant::Tri6 ) return moab::MBTRI;
    if( _variant <= GeometryVariant::Quad9 ) return moab::MBQUAD;
    if( _variant <= GeometryVariant::Tet10 ) return moab::MBTET;
    if( _variant <= GeometryVariant::Hex27 ) return moab::MBHEX;
    if( _variant <= GeometryVariant::Prism18 ) return moab::MBPRISM;
    return moab::MBPYRAMID;
}

unsigned KARMAReferenceTopology::dimension() const noexcept { return static_cast< unsigned >( moab::CN::Dimension( topology() ) ); }
unsigned KARMAReferenceTopology::order() const noexcept { return node_count() == static_cast< std::size_t >( moab::CN::VerticesPerEntity( topology() ) ) ? 1 : 2; }
std::size_t KARMAReferenceTopology::node_count() const noexcept { return _nodes.size(); }
std::optional< std::vector< std::size_t > >
KARMAReferenceTopology::node_bijection( std::span< const Point > points ) const
{
    if( points.size() != _nodes.size() ) return std::nullopt;
    std::vector< std::size_t > mapping( points.size() );
    std::vector< bool > used( _nodes.size() );
    for( std::size_t i = 0; i < points.size(); ++i )
    {
        std::optional< std::size_t > match;
        for( std::size_t j = 0; j < _nodes.size(); ++j )
            if( points[i] == _nodes[j] )
            {
                if( match || used[j] ) return std::nullopt;
                match = j;
            }
        if( !match ) return std::nullopt;
        mapping[i] = *match;
        used[*match] = true;
    }
    return mapping;
}

Point KARMAReferenceTopology::center() const noexcept
{
    switch( topology() )
    {
        case moab::MBEDGE: return p( .5 );
        case moab::MBTRI: return p( 1.0 / 3.0, 1.0 / 3.0 );
        case moab::MBQUAD: return p( .5, .5 );
        case moab::MBTET: return p( .25, .25, .25 );
        case moab::MBHEX: return p( .5, .5, .5 );
        case moab::MBPRISM: return p( 1.0 / 3.0, 1.0 / 3.0, .5 );
        case moab::MBPYRAMID: return p( 0, 0, .25 );
        default: return Point::Zero();
    }
}

std::vector< Real > KARMAReferenceTopology::evaluate_values( const Point& point ) const
{
    std::vector< Real > values( node_count() );
    evaluate_values( point, values );
    return values;
}

void KARMAReferenceTopology::evaluate_values( const Point& point, std::span< Real > values ) const
{
    if( values.size() != node_count() ) KARMA_THROW( InvalidArgument( "reference value storage has wrong size" ) );
    if( topology() == moab::MBPYRAMID && point == Point( 0, 0, 1 ) )
    {
        std::fill( values.begin(), values.end(), 0.0 );
        values[4] = 1.0;
        return;
    }
    std::array< Point, 27 > derivatives{};
    evaluate( point, values, std::span< Point >( derivatives.data(), node_count() ) );
}

ReferenceValues KARMAReferenceTopology::evaluate( const Point& point ) const
{
    ReferenceValues result{ std::vector< Real >( node_count() ), std::vector< Point >( node_count() ) };
    evaluate( point, result.values, result.derivatives );
    return result;
}

void KARMAReferenceTopology::evaluate( const Point& point, std::span< Real > output_values,
                                       std::span< Point > output_derivatives ) const
{
    if( output_values.size() != node_count() || output_derivatives.size() != node_count() )
        KARMA_THROW( InvalidArgument( "reference evaluation storage has wrong size" ) );
    if( !point.allFinite() ) KARMA_THROW( InvalidArgument( "reference point must be finite" ) );
    if( topology() == moab::MBPYRAMID && point == Point( 0, 0, 1 ) )
        KARMA_THROW( InvalidArgument( "pyramid basis derivatives are path-dependent at the exact apex" ) );
    struct Output
    {
        std::span< Real > values;
        std::span< Point > derivatives;
    } result{ output_values, output_derivatives };
    std::fill( result.values.begin(), result.values.end(), 0.0 );
    std::fill( result.derivatives.begin(), result.derivatives.end(), Point::Zero() );
    if( topology() == moab::MBPYRAMID )
    {
        pyramid_values( _variant, point, _nodes, result.values, result.derivatives );
        return;
    }
    const auto line = []( Real x, unsigned order ) {
        if( order == 1 ) return std::array< std::pair< Real, Real >, 3 >{ std::pair{ 1 - x, -1.0 }, { x, 1.0 }, { 0, 0 } };
        return std::array< std::pair< Real, Real >, 3 >{ std::pair{ 2 * ( x - .5 ) * ( x - 1 ), 4 * x - 3 },
                                                        { 2 * x * ( x - .5 ), 4 * x - 1 },
                                                        { 4 * x * ( 1 - x ), 4 - 8 * x } };
    };
    if( topology() == moab::MBEDGE )
    {
        const auto basis = line( point( 0 ), order() );
        for( std::size_t i = 0; i < node_count(); ++i ) { result.values[i] = basis[i].first; result.derivatives[i]( 0 ) = basis[i].second; }
        return;
    }
    if( topology() == moab::MBTRI || topology() == moab::MBTET )
    {
        const std::array< Real, 4 > lambda{ 1 - point( 0 ) - point( 1 ) - ( topology() == moab::MBTET ? point( 2 ) : 0 ),
                                            point( 0 ), point( 1 ), point( 2 ) };
        const std::size_t count = topology() == moab::MBTRI ? 3 : 4;
        const Point dl[] = { p( -1, -1, topology() == moab::MBTET ? -1 : 0 ), p( 1, 0, 0 ),
                             p( 0, 1, 0 ), p( 0, 0, 1 ) };
        for( std::size_t i = 0; i < count; ++i )
        {
            result.values[i] = order() == 1 ? lambda[i] : lambda[i] * ( 2 * lambda[i] - 1 );
            result.derivatives[i] = ( order() == 1 ? 1.0 : 4 * lambda[i] - 1 ) * dl[i];
        }
        constexpr std::array< std::pair< std::size_t, std::size_t >, 6 > edges{
            std::pair{ 0u, 1u }, { 1u, 2u }, { 2u, 0u }, { 0u, 3u }, { 1u, 3u }, { 2u, 3u } };
        const std::size_t edge_count = topology() == moab::MBTRI ? 3 : 6;
        if( order() == 2 ) for( std::size_t e = 0; e < edge_count; ++e )
        {
            const auto [i, j] = edges[e]; const std::size_t n = count + e;
            result.values[n] = 4 * lambda[i] * lambda[j];
            result.derivatives[n] = 4 * ( lambda[j] * dl[i] + lambda[i] * dl[j] );
        }
        return;
    }
    if( _variant == GeometryVariant::Quad8 )
    {
        const Real x = 2 * point( 0 ) - 1, y = 2 * point( 1 ) - 1;
        const Real sx[] = { -1, 1, 1, -1 }, sy[] = { -1, -1, 1, 1 };
        for( std::size_t i = 0; i < 4; ++i )
        {
            result.values[i] = .25 * ( 1 + sx[i] * x ) * ( 1 + sy[i] * y ) * ( sx[i] * x + sy[i] * y - 1 );
            result.derivatives[i]( 0 ) = .5 * sx[i] * ( 1 + sy[i] * y ) * ( 2 * sx[i] * x + sy[i] * y );
            result.derivatives[i]( 1 ) = .5 * sy[i] * ( 1 + sx[i] * x ) * ( sx[i] * x + 2 * sy[i] * y );
        }
        result.values[4] = .5 * ( 1 - x * x ) * ( 1 - y );
        result.values[5] = .5 * ( 1 + x ) * ( 1 - y * y );
        result.values[6] = .5 * ( 1 - x * x ) * ( 1 + y );
        result.values[7] = .5 * ( 1 - x ) * ( 1 - y * y );
        result.derivatives[4] = p( -2 * x * ( 1 - y ), -( 1 - x * x ) );
        result.derivatives[5] = p( 1 - y * y, -2 * y * ( 1 + x ) );
        result.derivatives[6] = p( -2 * x * ( 1 + y ), 1 - x * x );
        result.derivatives[7] = p( -( 1 - y * y ), -2 * y * ( 1 - x ) );
        return;
    }
    if( topology() == moab::MBQUAD || topology() == moab::MBHEX )
    {
        const auto bx = line( point( 0 ), order() ), by = line( point( 1 ), order() ), bz = line( point( 2 ), order() );
        for( std::size_t n = 0; n < node_count(); ++n )
        {
            const Point& q = _nodes[n];
            const auto index = []( Real x ) { return x == 0 ? 0u : x == 1 ? 1u : 2u; };
            const unsigned i = index( q( 0 ) ), j = index( q( 1 ) ), k = index( q( 2 ) );
            result.values[n] = bx[i].first * by[j].first * ( topology() == moab::MBHEX ? bz[k].first : 1.0 );
            result.derivatives[n] = p( bx[i].second * by[j].first * ( topology() == moab::MBHEX ? bz[k].first : 1.0 ),
                                         bx[i].first * by[j].second * ( topology() == moab::MBHEX ? bz[k].first : 1.0 ),
                                         topology() == moab::MBHEX ? bx[i].first * by[j].first * bz[k].second : 0.0 );
        }
        return;
    }
    // Wedges: PRISM18 is Tri6 x Edge3; PRISM15 is the standard edge-serendipity wedge.
    const Real lambda[] = { 1 - point( 0 ) - point( 1 ), point( 0 ), point( 1 ) };
    const Point dl[] = { p( -1, -1, 0 ), p( 1, 0, 0 ), p( 0, 1, 0 ) };
    const Real tau = 2 * point( 2 ) - 1;
    if( _variant == GeometryVariant::Prism6 )
    {
        for( unsigned layer = 0; layer < 2; ++layer )
            for( unsigned i = 0; i < 3; ++i )
            {
                const Real z_value = layer == 0 ? 1 - point( 2 ) : point( 2 );
                const Real z_derivative = layer == 0 ? -1 : 1;
                const std::size_t n = 3 * layer + i;
                result.values[n] = lambda[i] * z_value;
                result.derivatives[n] = z_value * dl[i];
                result.derivatives[n]( 2 ) = lambda[i] * z_derivative;
            }
        return;
    }
    if( _variant == GeometryVariant::Prism15 )
    {
        for( unsigned layer = 0; layer < 2; ++layer ) for( unsigned i = 0; i < 3; ++i )
        {
            const Real sign = layer == 0 ? -1.0 : 1.0; const std::size_t n = 3 * layer + i;
            result.values[n] = .5 * lambda[i] * ( 1 + sign * tau ) * ( 2 * lambda[i] - 2 + sign * tau );
            result.derivatives[n] = .5 * ( 1 + sign * tau ) * ( 4 * lambda[i] - 2 + sign * tau ) * dl[i];
            result.derivatives[n]( 2 ) = sign * lambda[i] * ( 2 * lambda[i] - 1 + 2 * sign * tau );
        }
        const unsigned pairs[][2] = { { 0, 1 }, { 1, 2 }, { 2, 0 } };
        for( unsigned layer = 0; layer < 2; ++layer ) for( unsigned e = 0; e < 3; ++e )
        {
            const Real sign = layer == 0 ? -1.0 : 1.0; const auto i = pairs[e][0], j = pairs[e][1];             const std::size_t n = ( layer == 0 ? 6 : 12 ) + e;

            result.values[n] = 2 * lambda[i] * lambda[j] * ( 1 + sign * tau );
            result.derivatives[n] = 2 * ( 1 + sign * tau ) * ( lambda[j] * dl[i] + lambda[i] * dl[j] );
            result.derivatives[n]( 2 ) = 4 * sign * lambda[i] * lambda[j];
        }
        for( unsigned i = 0; i < 3; ++i )
        {
            const std::size_t n = 9 + i; result.values[n] = lambda[i] * ( 1 - tau * tau );
            result.derivatives[n] = ( 1 - tau * tau ) * dl[i]; result.derivatives[n]( 2 ) = -4 * tau * lambda[i];
        }
        return;
    }
    const auto bz = line( point( 2 ), order() );
    const Real tri_values[] = { lambda[0] * ( 2 * lambda[0] - 1 ), lambda[1] * ( 2 * lambda[1] - 1 ), lambda[2] * ( 2 * lambda[2] - 1 ),
                                4 * lambda[0] * lambda[1], 4 * lambda[1] * lambda[2], 4 * lambda[2] * lambda[0] };
    const Point tri_derivatives[] = { ( 4 * lambda[0] - 1 ) * dl[0], ( 4 * lambda[1] - 1 ) * dl[1], ( 4 * lambda[2] - 1 ) * dl[2],
                                      4 * ( lambda[1] * dl[0] + lambda[0] * dl[1] ), 4 * ( lambda[2] * dl[1] + lambda[1] * dl[2] ),
                                      4 * ( lambda[0] * dl[2] + lambda[2] * dl[0] ) };
    const unsigned tri_index[] = { 0, 1, 2, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 3, 4, 5 };
    const unsigned z_index[] = { 0, 0, 0, 1, 1, 1, 0, 0, 0, 2, 2, 2, 1, 1, 1, 2, 2, 2 };
    for( std::size_t n = 0; n < node_count(); ++n )
    {
        const auto i = tri_index[n], k = z_index[n]; result.values[n] = tri_values[i] * bz[k].first;
        result.derivatives[n] = bz[k].first * tri_derivatives[i]; result.derivatives[n]( 2 ) = tri_values[i] * bz[k].second;
    }
}

unsigned KARMAReferenceTopology::side_count() const noexcept { return static_cast< unsigned >( moab::CN::NumSubEntities( topology(), dimension() - 1 ) ); }
KARMAReferenceTopology KARMAReferenceTopology::side_topology( unsigned side ) const
{
    if( side >= side_count() ) KARMA_THROW( InvalidArgument( "side index is out of range" ) );
    const auto type = moab::CN::SubEntityType( topology(), dimension() - 1, static_cast< int >( side ) );
    const bool quadratic = order() == 2;
    if( type == moab::MBEDGE ) return KARMAReferenceTopology( quadratic ? GeometryVariant::Edge3 : GeometryVariant::Edge2 );
    if( type == moab::MBTRI ) return KARMAReferenceTopology( quadratic ? GeometryVariant::Tri6 : GeometryVariant::Tri3 );
    if( type == moab::MBQUAD ) return KARMAReferenceTopology( !quadratic ? GeometryVariant::Quad4
                                                          : _variant == GeometryVariant::Prism15 || _variant == GeometryVariant::Pyramid13
                                                              ? GeometryVariant::Quad8
                                                              : GeometryVariant::Quad9 );
    KARMA_THROW( InvalidArgument( "unsupported side topology" ) );
}
std::vector< std::size_t > KARMAReferenceTopology::side_nodes( unsigned side ) const
{
    return side_indices( _variant, side );
}
Point KARMAReferenceTopology::side_reference_map( unsigned side, const Point& point ) const
{
    if( side >= side_count() ) KARMA_THROW( InvalidArgument( "side index is out of range" ) );
    const moab::EntityType type = moab::CN::SubEntityType( topology(), dimension() - 1, static_cast< int >( side ) );
    std::array< int, moab::MAX_SUB_ENTITY_VERTICES > corners{};
    moab::CN::SubEntityVertexIndices( topology(), dimension() - 1, static_cast< int >( side ), corners.data() );
    const GeometryVariant linear = type == moab::MBEDGE ? GeometryVariant::Edge2
                                   : type == moab::MBTRI ? GeometryVariant::Tri3
                                                        : GeometryVariant::Quad4;
    const auto values = KARMAReferenceTopology( linear ).evaluate( point );
    Point result = Point::Zero();
    for( std::size_t i = 0; i < values.values.size(); ++i ) result += values.values[i] * _nodes[static_cast< std::size_t >( corners[i] )];
    return result;
}
Matrix3 KARMAReferenceTopology::side_reference_jacobian( unsigned side, const Point& point ) const
{
    if( side >= side_count() ) KARMA_THROW( InvalidArgument( "side index is out of range" ) );
    const moab::EntityType type = moab::CN::SubEntityType( topology(), dimension() - 1, static_cast< int >( side ) );
    std::array< int, moab::MAX_SUB_ENTITY_VERTICES > corners{};
    moab::CN::SubEntityVertexIndices( topology(), dimension() - 1, static_cast< int >( side ), corners.data() );
    const GeometryVariant linear = type == moab::MBEDGE ? GeometryVariant::Edge2
                                   : type == moab::MBTRI ? GeometryVariant::Tri3
                                                        : GeometryVariant::Quad4;
    const auto values = KARMAReferenceTopology( linear ).evaluate( point );
    Matrix3 result = Matrix3::Zero();
    for( std::size_t i = 0; i < values.values.size(); ++i )
        for( unsigned d = 0; d < dimension() - 1; ++d )
            result.col( d ) += values.derivatives[i]( d ) * _nodes[static_cast< std::size_t >( corners[i] )];
    return result;
}
}  // namespace KARMA
