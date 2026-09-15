// Geometry-independent scalar finite-element bases.
#include "karma_basis.h"

#include "karma_exceptions.h"
#include "karma_reference_topology.h"
#include "moab/CN.hpp"

#include <algorithm>
#include <array>
#include <set>
#include <tuple>
#include <utility>

namespace
{
using namespace KARMA;

bool valid_key( const BasisKey& key ) noexcept
{
    return static_cast< int >( key.cell ) >= static_cast< int >( ReferenceCell::Edge ) &&
           static_cast< int >( key.cell ) <= static_cast< int >( ReferenceCell::Pyramid ) &&
           static_cast< int >( key.family ) >= static_cast< int >( FieldBasisFamily::LagrangeNodal ) &&
           static_cast< int >( key.family ) <= static_cast< int >( FieldBasisFamily::UserDefined ) &&
           static_cast< int >( key.continuity ) >= static_cast< int >( BasisContinuity::cG ) &&
           static_cast< int >( key.continuity ) <= static_cast< int >( BasisContinuity::SE ) &&
           key.order <= maximum_basis_order;
}

void validate_key( const BasisKey& key )
{
    if( !valid_key( key ) ) KARMA_THROW( InvalidArgument( "invalid basis key" ) );
}

unsigned dimension( ReferenceCell cell )
{
    if( cell == ReferenceCell::Edge ) return 1;
    if( cell == ReferenceCell::Triangle || cell == ReferenceCell::Quadrilateral ) return 2;
    return 3;
}

moab::EntityType topology( ReferenceCell cell )
{
    constexpr std::array types{ moab::MBEDGE, moab::MBTRI, moab::MBQUAD, moab::MBTET,
                                moab::MBHEX, moab::MBPRISM, moab::MBPYRAMID };
    return types[static_cast< std::size_t >( cell )];
}

unsigned orientation_count( moab::EntityType type )
{
    if( type == moab::MBEDGE ) return 2;
    if( type == moab::MBTRI ) return 6;
    if( type == moab::MBQUAD ) return 8;
    return 1;
}

GeometryVariant variant( ReferenceCell cell, unsigned order )
{
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
    KARMA_THROW( InvalidArgument( "unknown reference cell" ) );
}

std::string cell_name( ReferenceCell cell )
{
    switch( cell )
    {
        case ReferenceCell::Edge: return "Edge";
        case ReferenceCell::Triangle: return "Triangle";
        case ReferenceCell::Quadrilateral: return "Quadrilateral";
        case ReferenceCell::Tetrahedron: return "Tetrahedron";
        case ReferenceCell::Hexahedron: return "Hexahedron";
        case ReferenceCell::Prism: return "Prism";
        case ReferenceCell::Pyramid: return "Pyramid";
    }
    return "Unknown";
}

std::string continuity_name( BasisContinuity continuity )
{
    if( continuity == BasisContinuity::cG ) return "cG";
    if( continuity == BasisContinuity::dG ) return "dG";
    return "SE";
}

void validate_metadata( const KARMABasis& basis )
{
    const auto points = basis.interpolation_points();
    const auto associations = basis.dof_associations();
    if( points.size() != basis.size() || associations.size() != basis.size() )
        KARMA_THROW( InvalidArgument( "basis size, interpolation points, and associations must agree" ) );

    const unsigned cell_dimension = dimension( basis.key().cell );
    const auto cell_topology = topology( basis.key().cell );
    std::set< std::tuple< unsigned, unsigned, unsigned > > association_slots;
    for( const auto& association : associations )
    {
        if( association.entity_dimension > cell_dimension ||
            association.local_entity_index >= static_cast< unsigned >( moab::CN::NumSubEntities(
                cell_topology, static_cast< int >( association.entity_dimension ) ) ) ||
            !association_slots.emplace( association.entity_dimension, association.local_entity_index,
                                        association.slot ).second )
            KARMA_THROW( InvalidArgument( "invalid or duplicate basis DoF association" ) );
    }

    for( const auto& association : associations )
    {
        const auto preceding = static_cast< unsigned >( std::count_if(
            associations.begin(), associations.end(), [&]( const DofAssociation& candidate ) {
                return candidate.entity_dimension == association.entity_dimension &&
                       candidate.local_entity_index == association.local_entity_index && candidate.slot < association.slot;
            } ) );
        if( association.slot != preceding )
            KARMA_THROW( InvalidArgument( "basis DoF slots must be contiguous from zero" ) );
    }

    std::set< std::tuple< unsigned, unsigned, unsigned > > transform_keys;
    for( const auto& transform : basis.orientation_transforms() )
    {
        if( transform.entity_dimension == 0 || transform.entity_dimension >= cell_dimension ||
            transform.local_entity_index >= static_cast< unsigned >( moab::CN::NumSubEntities(
                cell_topology, static_cast< int >( transform.entity_dimension ) ) ) )
            KARMA_THROW( InvalidArgument( "orientation transform identifies an invalid subentity" ) );
        const auto entity_type = moab::CN::SubEntityType( cell_topology, static_cast< int >( transform.entity_dimension ),
                                                          static_cast< int >( transform.local_entity_index ) );
        if( transform.orientation_code >= orientation_count( entity_type ) ||
            !transform_keys.emplace( transform.entity_dimension, transform.local_entity_index,
                                     transform.orientation_code ).second )
            KARMA_THROW( InvalidArgument( "invalid or duplicate orientation state" ) );
        const auto dof_count = static_cast< std::size_t >( std::count_if(
            associations.begin(), associations.end(), [&]( const DofAssociation& association ) {
                return association.entity_dimension == transform.entity_dimension &&
                       association.local_entity_index == transform.local_entity_index;
            } ) );
        if( transform.local_to_canonical.size() != dof_count || transform.signs.size() != dof_count )
            KARMA_THROW( InvalidArgument( "orientation vectors must match subentity DoF count" ) );
        std::vector< bool > seen( dof_count );
        for( std::size_t i = 0; i < dof_count; ++i )
        {
            const auto destination = transform.local_to_canonical[i];
            if( destination >= dof_count || seen[destination] || ( transform.signs[i] != -1 && transform.signs[i] != 1 ) )
                KARMA_THROW( InvalidArgument( "orientation permutation or signs are invalid" ) );
            seen[destination] = true;
        }
    }

    for( unsigned entity_dimension = 1; entity_dimension < cell_dimension; ++entity_dimension )
        for( unsigned entity = 0; entity < static_cast< unsigned >( moab::CN::NumSubEntities(
                 cell_topology, static_cast< int >( entity_dimension ) ) ); ++entity )
        {
            const bool has_dofs = std::any_of( associations.begin(), associations.end(), [=]( const DofAssociation& association ) {
                return association.entity_dimension == entity_dimension && association.local_entity_index == entity;
            } );
            if( !has_dofs ) continue;
            const auto entity_type = moab::CN::SubEntityType( cell_topology, static_cast< int >( entity_dimension ),
                                                              static_cast< int >( entity ) );
            for( unsigned code = 0; code < orientation_count( entity_type ); ++code )
                if( !transform_keys.contains( { entity_dimension, entity, code } ) )
                    KARMA_THROW( InvalidArgument( "orientation metadata does not cover every subentity state" ) );
        }
}

class BuiltinBasis
{
  public:
    explicit BuiltinBasis( BasisKey key ) : _key( key ), _reference( variant( key.cell, key.order == 0 ? 1 : key.order ) )
    {
        if( key.order > 2 ) KARMA_THROW( InvalidArgument( "built-in basis order must be P0, P1, or P2" ) );
    }

    [[nodiscard]] std::string name() const
    {
        return "LagrangeNodal " + cell_name( _key.cell ) + " P" + std::to_string( _key.order ) + " " +
               continuity_name( _key.continuity );
    }
    [[nodiscard]] std::string compatibility_id() const
    { return "karma.lagrange-nodal.v1:" + name(); }
    [[nodiscard]] BasisKey key() const noexcept { return _key; }
    [[nodiscard]] CoefficientRepresentation coefficient_representation() const noexcept
    { return CoefficientRepresentation::Nodal; }
    [[nodiscard]] std::size_t size() const noexcept
    {
        return _key.order == 0 ? 1 : _reference.node_count();
    }
    void evaluate_values( const Point& point, std::span< Real > values ) const
    {
        if( values.size() != size() ) KARMA_THROW( InvalidArgument( "basis output storage must equal basis size" ) );
        if( _key.order == 0 ) { values[0] = 1.0; return; }
        _reference.evaluate_values( point, values );
    }
    void evaluate( const Point& point, std::span< Real > values, std::span< Gradient > gradients ) const
    {
        if( values.size() != size() || gradients.size() != size() )
            KARMA_THROW( InvalidArgument( "basis output storage must equal basis size" ) );
        if( _key.order == 0 )
        {
            values[0] = 1.0;
            gradients[0] = Gradient::Zero();
            return;
        }
        _reference.evaluate( point, values, gradients );
    }
    [[nodiscard]] std::vector< Point > interpolation_points() const
    {
        if( _key.order == 0 ) return { KARMAReferenceTopology( variant( _key.cell, 1 ) ).center() };
        return KARMAReferenceTopology( variant( _key.cell, _key.order ) ).reference_nodes();
    }
    [[nodiscard]] std::vector< DofAssociation > dof_associations() const
    {
        if( _key.continuity != BasisContinuity::cG )
        {
            std::vector< DofAssociation > result( size() );
            for( std::size_t i = 0; i < result.size(); ++i )
                result[i] = { dimension( _key.cell ), 0, static_cast< unsigned >( i ) };
            return result;
        }
        if( _key.order == 0 ) return { { dimension( _key.cell ), 0, 0 } };
        const std::size_t vertex_count = KARMAReferenceTopology( variant( _key.cell, 1 ) ).node_count();
        std::vector< DofAssociation > result;
        for( std::size_t i = 0; i < vertex_count; ++i ) result.push_back( { 0, static_cast< unsigned >( i ), 0 } );
        if( _key.order == 1 ) return result;

        const std::array< unsigned, 7 > edge_count{ 1, 3, 4, 6, 12, 9, 8 };
        const unsigned edges = edge_count[static_cast< std::size_t >( _key.cell )];
        for( unsigned i = 0; i < edges; ++i ) result.push_back( { 1, i, 0 } );
        const unsigned face_centers = _key.cell == ReferenceCell::Quadrilateral ? 1
                                      : _key.cell == ReferenceCell::Hexahedron ? 6
                                      : _key.cell == ReferenceCell::Prism ? 3 : 0;
        for( unsigned i = 0; i < face_centers; ++i ) result.push_back( { 2, i, 0 } );
        if( _key.cell == ReferenceCell::Pyramid ) result.push_back( { 2, 4, 0 } );
        if( result.size() < size() ) result.push_back( { dimension( _key.cell ), 0, 0 } );
        return result;
    }
    [[nodiscard]] std::vector< OrientationTransform > orientation_transforms() const
    {
        std::vector< OrientationTransform > result;
        const auto associations = dof_associations();
        for( unsigned entity_dimension = 1; entity_dimension < dimension( _key.cell ); ++entity_dimension )
            for( unsigned entity = 0; entity < static_cast< unsigned >( moab::CN::NumSubEntities(
                     topology( _key.cell ), static_cast< int >( entity_dimension ) ) ); ++entity )
            {
                const auto dof_count = static_cast< std::size_t >( std::count_if(
                    associations.begin(), associations.end(), [=]( const DofAssociation& association ) {
                        return association.entity_dimension == entity_dimension && association.local_entity_index == entity;
                    } ) );
                const auto entity_type = moab::CN::SubEntityType( topology( _key.cell ),
                                                                  static_cast< int >( entity_dimension ),
                                                                  static_cast< int >( entity ) );
                for( unsigned code = 0; code < orientation_count( entity_type ); ++code )
                {
                    std::vector< std::size_t > permutation( dof_count );
                    for( std::size_t i = 0; i < dof_count; ++i ) permutation[i] = i;
                    result.push_back( { entity_dimension, entity, code, std::move( permutation ),
                                        std::vector< int >( dof_count, 1 ) } );
                }
            }
        return result;
    }

  private:
    BasisKey _key;
    KARMAReferenceTopology _reference;
};
}  // namespace

namespace KARMA
{
KARMABasis::KARMABasis( std::shared_ptr< const Concept > self, int ) : _self( std::move( self ) )
{
    _key = _self->key();
    validate_key( _key );
    _size = _self->size();
    _interpolation_points = _self->interpolation_points();
    _dof_associations = _self->dof_associations();
    _orientation_transforms = _self->orientation_transforms();
    validate_metadata( *this );
}

void KARMABasis::apply_orientation_inverse( unsigned entity_dimension, unsigned local_entity_index,
                                             unsigned orientation_code, std::span< const Real > canonical,
                                             std::span< Real > local ) const
{
    const auto found = std::find_if( _orientation_transforms.begin(), _orientation_transforms.end(),
                                     [&]( const OrientationTransform& transform ) {
        return transform.entity_dimension == entity_dimension && transform.local_entity_index == local_entity_index &&
               transform.orientation_code == orientation_code;
    } );
    if( found == _orientation_transforms.end() || canonical.size() != found->local_to_canonical.size() ||
        local.size() != canonical.size() )
        KARMA_THROW( InvalidArgument( "unknown orientation or incorrect coefficient storage size" ) );
    std::fill( local.begin(), local.end(), 0.0 );
    for( std::size_t i = 0; i < canonical.size(); ++i )
        local[i] = static_cast< Real >( found->signs[i] ) * canonical[found->local_to_canonical[i]];
}

void KARMABasis::apply_orientation( unsigned entity_dimension, unsigned local_entity_index, unsigned orientation_code,
                                    std::span< const Real > local, std::span< Real > canonical ) const
{
    const auto found = std::find_if( _orientation_transforms.begin(), _orientation_transforms.end(),
                                     [&]( const OrientationTransform& transform ) {
        return transform.entity_dimension == entity_dimension && transform.local_entity_index == local_entity_index &&
               transform.orientation_code == orientation_code;
    } );
    if( found == _orientation_transforms.end() || local.size() != found->local_to_canonical.size() ||
        canonical.size() != local.size() )
        KARMA_THROW( InvalidArgument( "unknown orientation or incorrect coefficient storage size" ) );
    const std::vector< Real > staged( local.begin(), local.end() );
    for( std::size_t i = 0; i < staged.size(); ++i )
        canonical[found->local_to_canonical[i]] = static_cast< Real >( found->signs[i] ) * staged[i];
}

ReferenceCell reference_cell( GeometryVariant value ) noexcept
{
    if( value <= GeometryVariant::Edge3 ) return ReferenceCell::Edge;
    if( value <= GeometryVariant::Tri6 ) return ReferenceCell::Triangle;
    if( value <= GeometryVariant::Quad9 ) return ReferenceCell::Quadrilateral;
    if( value <= GeometryVariant::Tet10 ) return ReferenceCell::Tetrahedron;
    if( value <= GeometryVariant::Hex27 ) return ReferenceCell::Hexahedron;
    if( value <= GeometryVariant::Prism18 ) return ReferenceCell::Prism;
    return ReferenceCell::Pyramid;
}

KARMABasisRegistry::KARMABasisRegistry()
{
    constexpr std::array cells{ ReferenceCell::Edge, ReferenceCell::Triangle, ReferenceCell::Quadrilateral,
                                ReferenceCell::Tetrahedron, ReferenceCell::Hexahedron, ReferenceCell::Prism,
                                ReferenceCell::Pyramid };
    constexpr std::array continuities{ BasisContinuity::cG, BasisContinuity::dG, BasisContinuity::SE };
    for( const auto cell : cells )
        for( unsigned order = 0; order <= 2; ++order )
            for( const auto continuity : continuities )
            {
                // A globally continuous elementwise constant space is not a cG cell basis.
                if( order == 0 && continuity == BasisContinuity::cG ) continue;
                const BasisKey key{ cell, FieldBasisFamily::LagrangeNodal, order, continuity };
                _factories.emplace( key, [key] { return KARMABasis( BuiltinBasis( key ) ); } );
            }
}

void KARMABasisRegistry::register_factory( const BasisKey& key, Factory factory, DuplicatePolicy policy )
{
    validate_key( key );
    if( !factory ) KARMA_THROW( InvalidArgument( "basis factory must be callable" ) );
    const auto found = _factories.find( key );
    if( found != _factories.end() && policy == DuplicatePolicy::Reject )
        KARMA_THROW( InvalidArgument( "basis key is already registered" ) );
    _factories.insert_or_assign( key, std::move( factory ) );
}

bool KARMABasisRegistry::supports( const BasisKey& key ) const noexcept
{
    return valid_key( key ) && _factories.contains( key );
}

KARMABasis KARMABasisRegistry::create( const BasisKey& key ) const
{
    validate_key( key );
    const auto found = _factories.find( key );
    if( found == _factories.end() ) KARMA_THROW( InvalidArgument( "unknown basis key" ) );
    KARMABasis result = found->second();
    if( result.key() != key ) KARMA_THROW( InvalidArgument( "basis factory returned a mismatched key" ) );
    return result;
}
}  // namespace KARMA
