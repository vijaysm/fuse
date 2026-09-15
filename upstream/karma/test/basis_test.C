// Tests for geometry-independent scalar reference bases.
#include "karma_basis.h"
#include "karma_system.h"
#include "karma_reference_topology.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace
{
using KARMA::BasisContinuity;
using KARMA::CoefficientRepresentation;
using KARMA::BasisKey;
using KARMA::FieldBasisFamily;
using KARMA::DofAssociation;
using KARMA::Gradient;
using KARMA::KARMABasis;
using KARMA::KARMABasisRegistry;
using KARMA::OrientationTransform;
using KARMA::Point;
using KARMA::Real;
using KARMA::ReferenceCell;

int failures = 0;
void check( bool value, const std::string& message )
{
    if( !value ) { std::cerr << "FAIL: " << message << '\n'; ++failures; }
}
void close( Real value, Real expected, const std::string& message, Real tolerance = 3.e-10 )
{
    check( std::isfinite( value ) && std::abs( value - expected ) <= tolerance, message );
}
template < class Function > bool throws( Function&& function )
{
    try { function(); }
    catch( const std::logic_error& ) { return true; }
    return false;
}

constexpr std::array cells{ ReferenceCell::Edge, ReferenceCell::Triangle, ReferenceCell::Quadrilateral,
                            ReferenceCell::Tetrahedron, ReferenceCell::Hexahedron, ReferenceCell::Prism,
                            ReferenceCell::Pyramid };
constexpr std::array continuities{ BasisContinuity::cG, BasisContinuity::dG, BasisContinuity::SE };
constexpr std::array< std::array< std::size_t, 3 >, 7 > sizes{{ { 1, 2, 3 }, { 1, 3, 6 }, { 1, 4, 9 },
    { 1, 4, 10 }, { 1, 8, 27 }, { 1, 6, 18 }, { 1, 5, 14 } }};
constexpr std::array dimensions{ 1u, 2u, 2u, 3u, 3u, 3u, 3u };

void test_variable_spec_uses_field_basis_vocabulary()
{
    const BasisKey key{ ReferenceCell::Edge, FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG };
    const KARMA::VariableSpec variable{ "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 3 };
    check( key.family == variable.family && key.order == variable.order &&
               key.continuity == variable.continuity && variable.components == 3,
           "variable specifications use the field-basis key vocabulary" );
}

std::vector< DofAssociation > literal_cg_associations( ReferenceCell cell, unsigned order )
{
    if( order == 0 )
    {
        switch( cell )
        {
            case ReferenceCell::Edge: return { { 1, 0, 0 } };
            case ReferenceCell::Triangle: case ReferenceCell::Quadrilateral: return { { 2, 0, 0 } };
            default: return { { 3, 0, 0 } };
        }
    }
    if( order == 1 )
    {
        switch( cell )
        {
            case ReferenceCell::Edge: return { {0,0,0}, {0,1,0} };
            case ReferenceCell::Triangle: return { {0,0,0}, {0,1,0}, {0,2,0} };
            case ReferenceCell::Quadrilateral: case ReferenceCell::Tetrahedron:
                return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0} };
            case ReferenceCell::Hexahedron:
                return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}, {0,4,0}, {0,5,0}, {0,6,0}, {0,7,0} };
            case ReferenceCell::Prism:
                return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}, {0,4,0}, {0,5,0} };
            case ReferenceCell::Pyramid:
                return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}, {0,4,0} };
        }
    }
    switch( cell )
    {
        case ReferenceCell::Edge: return { {0,0,0}, {0,1,0}, {1,0,0} };
        case ReferenceCell::Triangle:
            return { {0,0,0}, {0,1,0}, {0,2,0}, {1,0,0}, {1,1,0}, {1,2,0} };
        case ReferenceCell::Quadrilateral:
            return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}, {1,0,0}, {1,1,0}, {1,2,0}, {1,3,0}, {2,0,0} };
        case ReferenceCell::Tetrahedron:
            return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}, {1,0,0}, {1,1,0}, {1,2,0}, {1,3,0}, {1,4,0}, {1,5,0} };
        case ReferenceCell::Hexahedron:
            return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}, {0,4,0}, {0,5,0}, {0,6,0}, {0,7,0},
                     {1,0,0}, {1,1,0}, {1,2,0}, {1,3,0}, {1,4,0}, {1,5,0}, {1,6,0}, {1,7,0},
                     {1,8,0}, {1,9,0}, {1,10,0}, {1,11,0}, {2,0,0}, {2,1,0}, {2,2,0}, {2,3,0},
                     {2,4,0}, {2,5,0}, {3,0,0} };
        case ReferenceCell::Prism:
            return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}, {0,4,0}, {0,5,0}, {1,0,0}, {1,1,0},
                     {1,2,0}, {1,3,0}, {1,4,0}, {1,5,0}, {1,6,0}, {1,7,0}, {1,8,0}, {2,0,0},
                     {2,1,0}, {2,2,0} };
        case ReferenceCell::Pyramid:
            return { {0,0,0}, {0,1,0}, {0,2,0}, {0,3,0}, {0,4,0}, {1,0,0}, {1,1,0}, {1,2,0},
                     {1,3,0}, {1,4,0}, {1,5,0}, {1,6,0}, {1,7,0}, {2,4,0} };
    }
    return {};
}

Point interior_point( ReferenceCell cell )
{
    switch( cell )
    {
        case ReferenceCell::Edge: return { .37, 0, 0 };
        case ReferenceCell::Triangle: return { .21, .31, 0 };
        case ReferenceCell::Quadrilateral: return { .21, .63, 0 };
        case ReferenceCell::Tetrahedron: return { .13, .21, .17 };
        case ReferenceCell::Hexahedron: return { .21, .63, .37 };
        case ReferenceCell::Prism: return { .21, .31, .37 };
        case ReferenceCell::Pyramid: return { .11, -.17, .37 };
    }
    return Point::Zero();
}

Real polynomial( ReferenceCell cell, unsigned order, const Point& p )
{
    if( order == 0 ) return 2.5;
    Real value = 1.2 + .7 * p( 0 );
    if( dimensions[static_cast< std::size_t >( cell )] > 1 ) value -= .4 * p( 1 );
    if( dimensions[static_cast< std::size_t >( cell )] > 2 ) value += .3 * p( 2 );
    if( order == 1 ) return value;
    value += .2 * p( 0 ) * p( 0 );
    if( cell == ReferenceCell::Quadrilateral || cell == ReferenceCell::Hexahedron )
        value += .11 * p( 0 ) * p( 0 ) * p( 1 ) * p( 1 );
    else if( dimensions[static_cast< std::size_t >( cell )] > 1 ) value += .11 * p( 0 ) * p( 1 );
    if( dimensions[static_cast< std::size_t >( cell )] > 2 ) value -= .09 * p( 2 ) * p( 2 );
    if( cell == ReferenceCell::Prism || cell == ReferenceCell::Hexahedron )
        value += .07 * p( 0 ) * p( 2 ) * p( 2 );
    return value;
}

void test_all_keys_sizes_and_associations()
{
    KARMABasisRegistry registry;
    std::size_t supported_count = 0;
    for( const auto cell : cells ) for( unsigned order = 0; order <= 2; ++order ) for( const auto continuity : continuities )
    {
        const BasisKey key{ cell, FieldBasisFamily::LagrangeNodal, order, continuity };
        if( order == 0 && continuity == BasisContinuity::cG )
        {
            check( !registry.supports( key ) && throws( [&] { (void)registry.create( key ); } ),
                   "cG P0 is semantically invalid and unregistered" );
            continue;
        }
        ++supported_count;
        const auto basis = registry.create( key );
        const std::string tag = " cell/order/continuity " + std::to_string( static_cast< int >( cell ) ) + "/" +
                                std::to_string( order ) + "/" + std::to_string( static_cast< int >( continuity ) );
        check( registry.supports( key ) && basis.supports( key ) && basis.key() == key, "supported exact key" + tag );
        check( basis.size() == sizes[static_cast< std::size_t >( cell )][order], "literal size" + tag );
        check( basis.coefficient_representation() == CoefficientRepresentation::Nodal,
               "built-in Lagrange basis declares nodal coefficients" + tag );
        check( basis.interpolation_points().size() == basis.size(), "interpolation point count" + tag );
        const auto associations = basis.dof_associations();
        check( associations.size() == basis.size(), "association count" + tag );
        if( continuity == BasisContinuity::cG )
            check( associations == literal_cg_associations( cell, order ), "literal cG association table" + tag );
        else
            for( std::size_t i = 0; i < associations.size(); ++i )
                check( associations[i] == DofAssociation{ dimensions[static_cast< std::size_t >( cell )], 0,
                                                          static_cast< unsigned >( i ) },
                       "dG/SE association is cell-local" + tag );
    }
    check( supported_count == 56, "registry exposes 56 built-in keys, excluding seven cG P0 keys" );
    check( !registry.supports( { ReferenceCell::Quadrilateral, FieldBasisFamily::SpectralElement, 2, BasisContinuity::SE } ),
           "extensible but unimplemented family is not advertised" );
    check( throws( [&] { (void)registry.create( { ReferenceCell::Triangle, FieldBasisFamily::Legendre, 1,
                                                 BasisContinuity::cG } ); } ), "unknown key is rejected" );
    check( KARMA::reference_cell( KARMA::GeometryVariant::Quad8 ) == ReferenceCell::Quadrilateral &&
               KARMA::reference_cell( KARMA::GeometryVariant::Quad9 ) == ReferenceCell::Quadrilateral,
           "geometry variants map to one reference-cell key" );
}

void test_nodal_values_derivatives_and_reproduction()
{
    KARMABasisRegistry registry;
    for( const auto cell : cells ) for( unsigned order = 0; order <= 2; ++order ) for( const auto continuity : continuities )
    {
        if( order == 0 && continuity == BasisContinuity::cG ) continue;
        const auto basis = registry.create( { cell, FieldBasisFamily::LagrangeNodal, order, continuity } );
        const auto points = basis.interpolation_points();
        for( std::size_t point = 0; point < points.size(); ++point )
        {
            std::vector< Real > values( basis.size() );
            basis.evaluate_values( points[point], values );
            for( std::size_t i = 0; i < values.size(); ++i )
                close( values[i], order == 0 || i == point ? 1.0 : 0.0, "Kronecker property", 5.e-10 );
        }
        const Point sample = interior_point( cell );
        std::vector< Real > values( basis.size() );
        std::vector< Gradient > gradients( basis.size() );
        basis.evaluate( sample, values, gradients );
        Real sum = 0.0, interpolated = 0.0;
        Gradient derivative_sum = Gradient::Zero();
        for( std::size_t i = 0; i < basis.size(); ++i )
        {
            sum += values[i]; derivative_sum += gradients[i];
            interpolated += polynomial( cell, order, points[i] ) * values[i];
        }
        close( sum, 1.0, "partition unity" );
        close( derivative_sum.norm(), 0.0, "derivative sum", 2.e-9 );
        close( interpolated, polynomial( cell, order, sample ), "space polynomial reproduction", 2.e-9 );
        std::mt19937 generator( 1207u + 31u * static_cast< unsigned >( cell ) + order );
        std::uniform_real_distribution< Real > fraction( .08, .72 );
        for( unsigned random_sample = 0; random_sample < 4; ++random_sample )
        {
            const Real t = fraction( generator );
            const Point random_point = ( 1 - t ) * sample + t * points[random_sample % points.size()];
            basis.evaluate( random_point, values, gradients );
            Real random_sum = 0.0;
            Gradient random_derivative_sum = Gradient::Zero();
            for( std::size_t i = 0; i < basis.size(); ++i )
            { random_sum += values[i]; random_derivative_sum += gradients[i]; }
            close( random_sum, 1.0, "seeded random partition unity" );
            close( random_derivative_sum.norm(), 0.0, "seeded random derivative sum", 3.e-9 );
        }
        basis.evaluate( sample, values, gradients );
        constexpr Real h = 1.e-7;
        for( unsigned d = 0; d < dimensions[static_cast< std::size_t >( cell )]; ++d )
        {
            Point plus = sample, minus = sample; plus( d ) += h; minus( d ) -= h;
            std::vector< Real > vp( basis.size() ), vm( basis.size() );
            basis.evaluate_values( plus, vp ); basis.evaluate_values( minus, vm );
            for( std::size_t i = 0; i < basis.size(); ++i )
                close( gradients[i]( d ), ( vp[i] - vm[i] ) / ( 2 * h ), "direct derivative finite difference", 2.e-7 );
        }
    }
    const auto pyramid = registry.create( { ReferenceCell::Pyramid, FieldBasisFamily::LagrangeNodal, 2,
                                             BasisContinuity::cG } );
    std::vector< Real > apex_values( pyramid.size() );
    pyramid.evaluate_values( Point( 0, 0, 1 ), apex_values );
    close( apex_values[4], 1.0, "pyramid apex value" );
    check( throws( [&] { std::vector< Gradient > g( pyramid.size() ); pyramid.evaluate( Point( 0, 0, 1 ), apex_values, g ); } ),
           "pyramid derivatives are explicitly undefined at apex" );
}

KARMA::GeometryVariant literal_variant( ReferenceCell cell, unsigned order )
{
    using KARMA::GeometryVariant;
    switch( cell )
    {
        case ReferenceCell::Edge: return order == 1 ? GeometryVariant::Edge2 : GeometryVariant::Edge3;
        case ReferenceCell::Triangle: return order == 1 ? GeometryVariant::Tri3 : GeometryVariant::Tri6;
        case ReferenceCell::Quadrilateral: return order == 1 ? GeometryVariant::Quad4 : GeometryVariant::Quad9;
        case ReferenceCell::Tetrahedron: return order == 1 ? GeometryVariant::Tet4 : GeometryVariant::Tet10;
        case ReferenceCell::Hexahedron: return order == 1 ? GeometryVariant::Hex8 : GeometryVariant::Hex27;
        case ReferenceCell::Prism: return order == 1 ? GeometryVariant::Prism6 : GeometryVariant::Prism18;
        case ReferenceCell::Pyramid: return order == 1 ? GeometryVariant::Pyramid5 : GeometryVariant::Pyramid14;
    }
    return GeometryVariant::Edge2;
}

void test_parent_traces_match_complete_side_spaces()
{
    KARMABasisRegistry registry;
    constexpr std::array parents{ ReferenceCell::Triangle, ReferenceCell::Quadrilateral, ReferenceCell::Tetrahedron,
                                  ReferenceCell::Prism, ReferenceCell::Pyramid, ReferenceCell::Hexahedron };
    for( const auto cell : parents ) for( unsigned order = 1; order <= 2; ++order )
    {
        const KARMA::KARMAReferenceTopology topology( literal_variant( cell, order ) );
        const auto parent = registry.create( { cell, FieldBasisFamily::LagrangeNodal, order, BasisContinuity::cG } );
        for( unsigned side = 0; side < topology.side_count(); ++side )
        {
            const auto side_topology = topology.side_topology( side );
            const auto side_basis = registry.create( { KARMA::reference_cell( side_topology.variant() ),
                                                        FieldBasisFamily::LagrangeNodal, order, BasisContinuity::cG } );
            const Point side_point = side_topology.center() * .73 + side_topology.reference_nodes().front() * .27;
            const Point parent_point = topology.side_reference_map( side, side_point );
            std::vector< Real > parent_values( parent.size() ), side_values( side_basis.size() );
            parent.evaluate_values( parent_point, parent_values );
            side_basis.evaluate_values( side_point, side_values );
            const auto side_nodes = topology.side_nodes( side );
            check( side_nodes.size() == side_values.size(), "trace has complete side-space size" );
            for( std::size_t i = 0; i < side_nodes.size(); ++i )
                close( parent_values[side_nodes[i]], side_values[i], "parent and side trace values agree", 2.e-9 );
            for( std::size_t i = 0; i < parent_values.size(); ++i )
                if( std::find( side_nodes.begin(), side_nodes.end(), i ) == side_nodes.end() )
                    close( parent_values[i], 0.0, "off-side basis has zero trace", 2.e-9 );
        }
    }
}

void test_tetrahedron_parent_side_traces_p1_p2()
{
    KARMABasisRegistry registry;
    for( unsigned order = 1; order <= 2; ++order )
    {
        const KARMA::KARMAReferenceTopology parent_topology( literal_variant( ReferenceCell::Tetrahedron, order ) );
        const auto parent = registry.create( { ReferenceCell::Tetrahedron, FieldBasisFamily::LagrangeNodal, order,
                                                BasisContinuity::cG } );
        const auto side = registry.create( { ReferenceCell::Triangle, FieldBasisFamily::LagrangeNodal, order,
                                              BasisContinuity::cG } );
        for( unsigned face = 0; face < parent_topology.side_count(); ++face )
            for( const Point& side_point : side.interpolation_points() )
            {
                const Point parent_point = parent_topology.side_reference_map( face, side_point );
                std::vector< Real > parent_values( parent.size() ), side_values( side.size() );
                parent.evaluate_values( parent_point, parent_values );
                side.evaluate_values( side_point, side_values );
                const auto side_nodes = parent_topology.side_nodes( face );
                check( side_nodes.size() == side_values.size(), "tetrahedron trace has complete P1/P2 triangle space" );
                for( std::size_t i = 0; i < side_nodes.size(); ++i )
                    close( parent_values[side_nodes[i]], side_values[i], "tetrahedron parent/side nodal trace agrees" );
                for( std::size_t i = 0; i < parent_values.size(); ++i )
                    if( std::find( side_nodes.begin(), side_nodes.end(), i ) == side_nodes.end() )
                        close( parent_values[i], 0.0, "tetrahedron off-face basis has zero trace" );
            }
    }
}

void test_orientation_metadata_is_identity_and_isolated()
{
    KARMABasisRegistry registry;
    check( static_cast< unsigned >( KARMA::EdgeOrientation::Reversed ) == 1 &&
               static_cast< unsigned >( KARMA::TriangleOrientation::ReflectRotate2 ) == 5 &&
               static_cast< unsigned >( KARMA::QuadrilateralOrientation::ReflectRotate3 ) == 7,
           "orientation enums cover 2 edge, 6 triangle, and 8 quadrilateral states" );
    for( const auto cell : cells )
    {
        const auto basis = registry.create( { cell, FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG } );
        auto transforms = basis.orientation_transforms();
        for( const auto& transform : transforms )
        {
            check( transform.orientation_code < ( transform.entity_dimension == 1 ? 2u :
                       cell == ReferenceCell::Tetrahedron ? 6u : 8u ), "orientation code is in range" );
            check( transform.local_to_canonical.size() == transform.signs.size(), "orientation vector sizes agree" );
            for( const int sign : transform.signs ) check( sign == 1, "scalar P2 orientation signs are identity" );
            for( std::size_t i = 0; i < transform.local_to_canonical.size(); ++i )
                check( transform.local_to_canonical[i] == i, "scalar P2 orientation permutation is identity" );
        }
        if( !transforms.empty() )
        {
            transforms.front().orientation_code = 99;
            check( basis.orientation_transforms().front().orientation_code == 0, "orientation metadata return is isolated" );
        }
    }
    const auto tet = registry.create( { ReferenceCell::Tetrahedron, FieldBasisFamily::LagrangeNodal, 2,
                                        BasisContinuity::cG } );
    const auto hex = registry.create( { ReferenceCell::Hexahedron, FieldBasisFamily::LagrangeNodal, 2,
                                        BasisContinuity::cG } );
    const auto tet_transforms = tet.orientation_transforms();
    const auto hex_transforms = hex.orientation_transforms();
    check( std::count_if( tet_transforms.begin(), tet_transforms.end(),
                         []( const auto& t ) { return t.entity_dimension == 2 && t.local_entity_index == 0; } ) == 6,
           "triangular face has six orientation states" );
    check( std::count_if( hex_transforms.begin(), hex_transforms.end(),
                         []( const auto& t ) { return t.entity_dimension == 2 && t.local_entity_index == 0; } ) == 8,
           "quadrilateral face has eight orientation states" );
}

void test_orientation_vertex_permutation_convention()
{
    constexpr std::array< std::array< unsigned, 2 >, 2 > edge{{ {0,1}, {1,0} }};
    constexpr std::array< std::array< unsigned, 3 >, 6 > triangle{{
        {0,1,2}, {1,2,0}, {2,0,1}, {0,2,1}, {1,0,2}, {2,1,0}
    }};
    constexpr std::array< std::array< unsigned, 4 >, 8 > quadrilateral{{
        {0,1,2,3}, {1,2,3,0}, {2,3,0,1}, {3,0,1,2},
        {0,3,2,1}, {1,0,3,2}, {2,1,0,3}, {3,2,1,0}
    }};
    for( unsigned code = 0; code < edge.size(); ++code )
        check( KARMA::local_to_canonical_vertices( static_cast< KARMA::EdgeOrientation >( code ) ) == edge[code],
               "edge orientation literal local-to-canonical table" );
    for( unsigned code = 0; code < triangle.size(); ++code )
    {
        const auto permutation = KARMA::local_to_canonical_vertices(
            static_cast< KARMA::TriangleOrientation >( code ) );
        check( permutation == triangle[code], "triangle orientation literal local-to-canonical table" );
        auto sorted = permutation;
        std::sort( sorted.begin(), sorted.end() );
        check( sorted == triangle[0], "triangle orientation is a vertex bijection" );
        for( unsigned local = 0; local < 3; ++local )
            check( permutation[local] == ( code < 3 ? ( local + code ) % 3 : ( code - 3 + 3 - local ) % 3 ),
                   "triangle rotations and reflected rotations follow the documented action" );
    }
    for( unsigned code = 0; code < quadrilateral.size(); ++code )
    {
        const auto permutation = KARMA::local_to_canonical_vertices(
            static_cast< KARMA::QuadrilateralOrientation >( code ) );
        check( permutation == quadrilateral[code], "quadrilateral orientation literal local-to-canonical table" );
        auto sorted = permutation;
        std::sort( sorted.begin(), sorted.end() );
        check( sorted == quadrilateral[0], "quadrilateral orientation is a vertex bijection" );
        for( unsigned local = 0; local < 4; ++local )
            check( permutation[local] == ( code < 4 ? ( local + code ) % 4 : ( code - 4 + 4 - local ) % 4 ),
                   "quadrilateral rotations and reflected rotations follow the documented action" );
    }
}

struct CustomBasis
{
    BasisKey stored_key;
    std::shared_ptr< int > alive;
    std::string label;
    std::size_t declared_size{ 1 };
    std::vector< Point > points{ Point::Zero() };
    std::vector< DofAssociation > associations{ { 2, 0, 0 } };
    std::vector< OrientationTransform > transforms;
    std::string name() const { return label; }
    std::string compatibility_id() const { return "test.custom-basis.v1:" + label; }
    KARMA::CoefficientRepresentation coefficient_representation() const noexcept
    { return KARMA::CoefficientRepresentation::Modal; }
    BasisKey key() const noexcept { return stored_key; }
    std::size_t size() const noexcept { return declared_size; }
    void evaluate_values( const Point&, std::span< Real > values ) const { std::fill( values.begin(), values.end(), 7 ); }
    void evaluate( const Point&, std::span< Real > values, std::span< Gradient > gradients ) const
    { std::fill( values.begin(), values.end(), 7 ); std::fill( gradients.begin(), gradients.end(), Gradient::Zero() ); }
    std::vector< Point > interpolation_points() const { return points; }
    std::vector< DofAssociation > dof_associations() const { return associations; }
    std::vector< OrientationTransform > orientation_transforms() const { return transforms; }
};

CustomBasis custom_basis( BasisKey key, std::size_t size, std::vector< DofAssociation > associations,
                          std::vector< OrientationTransform > transforms = {} )
{
    CustomBasis basis{ key, {}, "custom", size, {}, {}, {} };
    basis.points.assign( size, Point::Zero() );
    basis.associations = std::move( associations );
    basis.transforms = std::move( transforms );
    return basis;
}

Real quadratic_edge_trace( Real x, const std::array< Real, 3 >& coefficients )
{
    const std::array< Real, 3 > nodes{ .2, .5, .8 };
    Real value = 0.0;
    for( std::size_t i = 0; i < nodes.size(); ++i )
    {
        Real shape = 1.0;
        for( std::size_t j = 0; j < nodes.size(); ++j )
            if( i != j ) shape *= ( x - nodes[j] ) / ( nodes[i] - nodes[j] );
        value += coefficients[i] * shape;
    }
    return value;
}

void test_custom_metadata_validation_and_orientation_application()
{
    const BasisKey key{ ReferenceCell::Triangle, FieldBasisFamily::UserDefined, 5, BasisContinuity::cG };
    const std::vector< DofAssociation > edge_dofs{ {1,0,0}, {1,0,1}, {1,0,2} };
    const std::vector< OrientationTransform > valid{
        { 1, 0, static_cast< unsigned >( KARMA::EdgeOrientation::Direct ), {0,1,2}, {1,1,1} },
        { 1, 0, static_cast< unsigned >( KARMA::EdgeOrientation::Reversed ), {2,1,0}, {1,1,1} }
    };
    KARMABasisRegistry registry;
    registry.register_factory( key, [=] { return KARMABasis( custom_basis( key, 3, edge_dofs, valid ) ); } );
    const auto basis = registry.create( key );
    std::array< Real, 3 > local{ 10, 20, 30 }, canonical{};
    basis.apply_orientation( 1, 0, static_cast< unsigned >( KARMA::EdgeOrientation::Reversed ), local, canonical );
    check( canonical == std::array< Real, 3 >{ 30, 20, 10 }, "reversed edge maps local coefficients to canonical order" );
    constexpr Real trace_point = .37;
    close( quadratic_edge_trace( trace_point, canonical ), quadratic_edge_trace( 1.0 - trace_point, local ),
           "reversed edge transform preserves the physical trace" );
    basis.apply_orientation( 1, 0, static_cast< unsigned >( KARMA::EdgeOrientation::Reversed ), local, local );
    check( local == std::array< Real, 3 >{ 30, 20, 10 }, "orientation application supports in-place reversal" );
    std::array< Real, 5 > overlapping{ 10, 20, 30, -1, -1 };
    basis.apply_orientation( 1, 0, static_cast< unsigned >( KARMA::EdgeOrientation::Reversed ),
                             std::span< const Real >( overlapping.data(), 3 ),
                             std::span< Real >( overlapping.data() + 1, 3 ) );
    check( overlapping == std::array< Real, 5 >{ 10, 30, 20, 10, -1 },
           "orientation application supports partially overlapping spans" );
    auto wrong = valid;
    wrong[1].local_to_canonical = {0,1,2};
    registry.register_factory( key, [=] { return KARMABasis( custom_basis( key, 3, edge_dofs, wrong ) ); },
                               KARMA::DuplicatePolicy::Replace );
    const auto wrongly_oriented = registry.create( key );
    wrongly_oriented.apply_orientation( 1, 0, 1, local, canonical );
    check( std::abs( quadratic_edge_trace( trace_point, canonical ) -
                     quadratic_edge_trace( 1.0 - trace_point, local ) ) > 1.e-8,
           "mutated reversed permutation fails the physical trace oracle" );

    const auto rejects = [&]( BasisKey bad_key, CustomBasis malformed, const std::string& message ) {
        KARMABasisRegistry candidate;
        candidate.register_factory( bad_key, [malformed] { return KARMABasis( malformed ); } );
        check( throws( [&] { (void)candidate.create( bad_key ); } ), message );
    };
    BasisKey bad = key;
    bad.order++;
    rejects( bad, custom_basis( bad, 2, { {1,0,0} } ), "association count must equal size" );
    bad.order++;
    auto point_mismatch = custom_basis( bad, 2, { {1,0,0}, {1,0,1} } ); point_mismatch.points.pop_back();
    rejects( bad, point_mismatch, "interpolation count must equal size" );
    bad.order++;
    rejects( bad, custom_basis( bad, 1, { {3,0,0} } ), "association dimension must fit reference cell" );
    bad.order++;
    rejects( bad, custom_basis( bad, 1, { {1,3,0} } ), "association entity index must fit reference cell" );
    bad.order++;
    rejects( bad, custom_basis( bad, 2, { {1,0,0}, {1,0,0} } ), "DoF associations must be unique" );
    bad.order++;
    rejects( bad, custom_basis( bad, 1, { {1,0,3} }, { {1,0,0,{0},{1}}, {1,0,1,{0},{1}} } ),
             "association slots must be contiguous from zero" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { {1,0,0,{0,1,2},{1,1}} } ),
             "orientation permutation and sign lengths must agree" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { {1,0,0,{0,1},{1,1}} } ),
             "orientation vectors must match the entity-local DoF count" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { {1,0,0,{0,1,3},{1,1,1}} } ), "orientation permutation must be in range" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { {1,0,0,{0,0,2},{1,1,1}} } ), "orientation permutation must be bijective" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { {1,0,0,{0,1,2},{1,0,1}} } ), "orientation signs must be plus or minus one" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { valid[0], valid[0] } ), "orientation keys must be unique" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { {1,0,2,{0,1,2},{1,1,1}} } ), "orientation code must fit subentity" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { {0,0,0,{0,1,2},{1,1,1}} } ),
             "orientation dimension must identify an orientable subentity" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { {1,3,0,{0,1,2},{1,1,1}} } ),
             "orientation entity index must fit reference cell" );
    bad.order++;
    rejects( bad, custom_basis( bad, 3, edge_dofs, { valid[0] } ),
             "orientation states must completely cover an entity with DoFs" );
}

void test_direct_construction_validates_and_snapshots_metadata()
{
    const BasisKey key{ ReferenceCell::Triangle, FieldBasisFamily::UserDefined, 5, BasisContinuity::cG };
    check( throws( [&] { (void)KARMABasis( custom_basis( key, 2, { {1,0,0} } ) ); } ),
           "direct construction rejects inconsistent metadata" );
    check( throws( [&] { (void)KARMABasis( custom_basis( key, 1, { {3,0,0} } ) ); } ),
           "direct construction rejects invalid associations before topology indexing" );

    struct MutableStateBasis
    {
        std::shared_ptr< CustomBasis > source;
        std::string name() const { return source->name(); }
        std::string compatibility_id() const { return source->compatibility_id(); }
        KARMA::CoefficientRepresentation coefficient_representation() const noexcept
        { return source->coefficient_representation(); }
        BasisKey key() const noexcept { return source->key(); }
        std::size_t size() const noexcept { return source->size(); }
        void evaluate_values( const Point& p, std::span< Real > v ) const { source->evaluate_values( p, v ); }
        void evaluate( const Point& p, std::span< Real > v, std::span< Gradient > g ) const { source->evaluate( p, v, g ); }
        std::vector< Point > interpolation_points() const { return source->interpolation_points(); }
        std::vector< DofAssociation > dof_associations() const { return source->dof_associations(); }
        std::vector< OrientationTransform > orientation_transforms() const { return source->orientation_transforms(); }
    };
    const std::vector< DofAssociation > edge_dofs{ {1,0,0}, {1,0,1}, {1,0,2} };
    const std::vector< OrientationTransform > transforms{
        {1,0,0,{0,1,2},{1,1,1}}, {1,0,1,{2,1,0},{1,1,1}}
    };
    auto source = std::make_shared< CustomBasis >( custom_basis( key, 3, edge_dofs, transforms ) );
    const KARMABasis basis( MutableStateBasis{ source } );
    source->stored_key.order = 17;
    source->declared_size = 1;
    source->points.clear();
    source->associations.clear();
    source->transforms[1].local_to_canonical = {0,1,2};
    check( basis.key() == key && basis.size() == 3 && basis.interpolation_points().size() == 3 &&
               basis.dof_associations() == edge_dofs,
           "constructed basis metadata is an immutable snapshot" );
    std::array< Real, 3 > local{ 1, 2, 3 }, canonical{};
    basis.apply_orientation( 1, 0, 1, local, canonical );
    check( canonical == std::array< Real, 3 >{ 3, 2, 1 },
           "orientation application uses the validated transform snapshot" );
}

void test_invalid_basis_keys_are_rejected_before_indexing()
{
    KARMABasisRegistry registry;
    const std::array invalid{
        BasisKey{ static_cast< ReferenceCell >( -1 ), FieldBasisFamily::UserDefined, 1, BasisContinuity::dG },
        BasisKey{ static_cast< ReferenceCell >( 99 ), FieldBasisFamily::UserDefined, 1, BasisContinuity::dG },
        BasisKey{ ReferenceCell::Edge, static_cast< FieldBasisFamily >( -1 ), 1, BasisContinuity::dG },
        BasisKey{ ReferenceCell::Edge, static_cast< FieldBasisFamily >( 99 ), 1, BasisContinuity::dG },
        BasisKey{ ReferenceCell::Edge, FieldBasisFamily::UserDefined, 1, static_cast< BasisContinuity >( -1 ) },
        BasisKey{ ReferenceCell::Edge, FieldBasisFamily::UserDefined, 1, static_cast< BasisContinuity >( 99 ) },
        BasisKey{ ReferenceCell::Edge, FieldBasisFamily::UserDefined, std::numeric_limits< unsigned >::max(),
                  BasisContinuity::dG }
    };
    for( const auto& key : invalid )
    {
        check( !registry.supports( key ), "supports returns false for an invalid basis key" );
        check( throws( [&] { (void)registry.create( key ); } ), "create rejects an invalid basis key" );
        check( throws( [&] { registry.register_factory( key, [=] {
                   return KARMABasis( custom_basis( key, 1, { {1,0,0} } ) );
               } ); } ), "registration rejects an invalid basis key" );
        check( throws( [&] { (void)KARMABasis( custom_basis( key, 1, { {1,0,0} } ) ); } ),
               "direct construction rejects an invalid basis key" );
    }
}

void test_face_orientation_codes_drive_dof_vertex_maps()
{
    const auto exercise = []( BasisKey key, unsigned vertex_count, unsigned state_count ) {
        std::vector< DofAssociation > associations;
        for( unsigned i = 0; i < vertex_count; ++i ) associations.push_back( {2,0,i} );
        std::vector< OrientationTransform > transforms;
        for( unsigned code = 0; code < state_count; ++code )
        {
            std::vector< std::size_t > permutation( vertex_count );
            if( vertex_count == 3 )
            {
                const auto vertices = KARMA::local_to_canonical_vertices(
                    static_cast< KARMA::TriangleOrientation >( code ) );
                std::copy( vertices.begin(), vertices.end(), permutation.begin() );
            }
            else
            {
                const auto vertices = KARMA::local_to_canonical_vertices(
                    static_cast< KARMA::QuadrilateralOrientation >( code ) );
                std::copy( vertices.begin(), vertices.end(), permutation.begin() );
            }
            transforms.push_back( {2,0,code,std::move( permutation ),std::vector< int >( vertex_count, 1 )} );
        }
        const KARMABasis basis( custom_basis( key, vertex_count, associations, transforms ) );
        std::vector< Real > local( vertex_count ), canonical( vertex_count );
        for( unsigned i = 0; i < vertex_count; ++i ) local[i] = 10.0 + i;
        for( unsigned code = 0; code < state_count; ++code )
        {
            std::fill( canonical.begin(), canonical.end(), -1 );
            basis.apply_orientation( 2, 0, code, local, canonical );
            for( unsigned i = 0; i < vertex_count; ++i )
            {
                const unsigned destination = vertex_count == 3
                    ? KARMA::local_to_canonical_vertices( static_cast< KARMA::TriangleOrientation >( code ) )[i]
                    : KARMA::local_to_canonical_vertices( static_cast< KARMA::QuadrilateralOrientation >( code ) )[i];
                check( canonical[destination] == local[i], "face orientation state maps physical vertex DoFs" );
            }
        }
    };
    exercise( { ReferenceCell::Tetrahedron, FieldBasisFamily::UserDefined, 3, BasisContinuity::cG }, 3, 6 );
    exercise( { ReferenceCell::Hexahedron, FieldBasisFamily::UserDefined, 3, BasisContinuity::cG }, 4, 8 );
}

void test_custom_registry_policy_and_lifetime()
{
    const BasisKey key{ ReferenceCell::Triangle, FieldBasisFamily::UserDefined, 7, BasisContinuity::dG };
    auto lifetime = std::make_shared< int >( 1 );
    std::weak_ptr< int > observed = lifetime;
    KARMABasis saved = [&] {
        KARMABasisRegistry registry;
        registry.register_factory( key, [=] { return KARMABasis( CustomBasis{ key, lifetime, "first", 1, { Point::Zero() }, { {2,0,0} }, {} } ); } );
        check( throws( [&] { registry.register_factory( key, [=] { return KARMABasis( CustomBasis{ key, lifetime, "duplicate", 1, { Point::Zero() }, { {2,0,0} }, {} } ); } ); } ),
               "duplicate registration rejects by default" );
        registry.register_factory( key, [=] { return KARMABasis( CustomBasis{ key, lifetime, "replacement", 1, { Point::Zero() }, { {2,0,0} }, {} } ); },
                                   KARMA::DuplicatePolicy::Replace );
        const auto replacement = registry.create( key );
        check( replacement.name() == "replacement", "explicit replacement policy replaces factory" );
        std::array< Real, 2 > oversized{};
        std::array< Gradient, 1 > gradients{};
        check( throws( [&] { replacement.evaluate( Point::Zero(), oversized, gradients ); } ),
               "type erasure enforces caller storage size for custom bases" );
        check( throws( [&] { registry.register_factory( { ReferenceCell::Edge, FieldBasisFamily::UserDefined, 9,
                                                          BasisContinuity::dG }, {} ); } ), "empty factory rejected" );
        const BasisKey mismatched{ ReferenceCell::Edge, FieldBasisFamily::UserDefined, 8, BasisContinuity::dG };
        registry.register_factory( mismatched, [=] { return KARMABasis( CustomBasis{ key, lifetime, "wrong key", 1, { Point::Zero() }, { {2,0,0} }, {} } ); } );
        check( throws( [&] { (void)registry.create( mismatched ); } ), "factory result must match registered key" );
        return registry.create( key );
    }();
    lifetime.reset();
    check( !observed.expired() && saved.name() == "replacement", "basis owns custom implementation beyond registry lifetime" );
}
}  // namespace

int main()
{
    test_variable_spec_uses_field_basis_vocabulary();
    test_all_keys_sizes_and_associations();
    test_nodal_values_derivatives_and_reproduction();
    test_parent_traces_match_complete_side_spaces();
    test_tetrahedron_parent_side_traces_p1_p2();
    test_orientation_metadata_is_identity_and_isolated();
    test_orientation_vertex_permutation_convention();
    test_custom_metadata_validation_and_orientation_application();
    test_direct_construction_validates_and_snapshots_metadata();
    test_invalid_basis_keys_are_rejected_before_indexing();
    test_face_orientation_codes_drive_dof_vertex_maps();
    test_custom_registry_policy_and_lifetime();
    if( failures ) std::cerr << failures << " basis checks failed\n";
    return failures == 0 ? 0 : 1;
}
