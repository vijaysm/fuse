#include "karma_physics_bc_manager.h"
#include "impls/collective_signature.h"

#include "karma_dof_map.h"
#include "karma_fe_base.h"
#include "karma_geometry_map.h"
#include "karma_parameters.h"
#include "karma_system.h"

#include "moab/CN.hpp"

#ifdef KARMA_HAVE_LIBMESH
#include "node.h"
#include "point.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>

namespace
{
template < typename Map, typename Key >
KARMABoundaryCondition* find_condition( const Map& conditions, const Key& key )
{
    const auto found = conditions.find( key );
    return found == conditions.end() ? nullptr : found->second.get();
}

template < typename Conditions >
unsigned union_size( const Conditions& modern, const BoundaryNodes& legacy, char prefix )
{
    unsigned count = static_cast< unsigned >( modern.size() );
    for( const auto& [key, condition] : legacy )
    {
        (void)condition;
        unsigned element = 0;
        unsigned side    = 0;
        char parsed_prefix{};
        if( std::sscanf( key.c_str(), "%c;%u;%u", &parsed_prefix, &element, &side ) != 3 || parsed_prefix != prefix ||
            element > static_cast< unsigned >( std::numeric_limits< KARMA::GlobalId >::max() ) ||
            modern.count( typename Conditions::key_type{ static_cast< KARMA::GlobalId >( element ), side } ) == 0 )
            ++count;
    }
    return count;
}

bool association_is_on_side( moab::EntityType topology, unsigned dimension, const KARMA::DofAssociation& association,
                             unsigned side )
{
    if( association.entity_dimension >= dimension ) return false;
    std::array< int, moab::MAX_SUB_ENTITY_VERTICES > side_vertices{};
    std::array< int, moab::MAX_SUB_ENTITY_VERTICES > entity_vertices{};
    const int side_count = moab::CN::VerticesPerEntity(
        moab::CN::SubEntityType( topology, static_cast< int >( dimension - 1 ), static_cast< int >( side ) ) );
    const int entity_count = moab::CN::VerticesPerEntity(
        moab::CN::SubEntityType( topology, static_cast< int >( association.entity_dimension ),
                                 static_cast< int >( association.local_entity_index ) ) );
    moab::CN::SubEntityVertexIndices( topology, static_cast< int >( dimension - 1 ), static_cast< int >( side ),
                                      side_vertices.data() );
    moab::CN::SubEntityVertexIndices( topology, static_cast< int >( association.entity_dimension ),
                                      static_cast< int >( association.local_entity_index ), entity_vertices.data() );
    return std::all_of( entity_vertices.begin(), entity_vertices.begin() + entity_count, [&]( int vertex ) {
        return std::find( side_vertices.begin(), side_vertices.begin() + side_count, vertex ) !=
               side_vertices.begin() + side_count;
    } );
}

bool equal_dirichlet( const KARMABoundaryCondition& first, const KARMABoundaryCondition& second )
{
    constexpr Real tolerance = 1.e-12;
    if( first.values().size() != second.values().size() ) return false;
    for( std::size_t i = 0; i < first.values().size(); ++i )
        if( std::abs( first[i] - second[i] ) >
            tolerance * std::max( { Real{ 1 }, std::abs( first[i] ), std::abs( second[i] ) } ) )
            return false;
    return true;
}

void validate_type( const KARMABoundaryCondition& condition, KARMA::PHYSICS::BCType expected )
{
    if( condition.type() != expected )
        KARMA_THROW( KARMA::InvalidArgument( "boundary callback returned the wrong condition type" ) );
}

void delete_conditions( BoundaryNodes& conditions ) noexcept
{
    for( const auto& [key, condition] : conditions )
    {
        (void)key;
        delete condition;
    }
    conditions.clear();
}

void collective_refresh_error( const KARMA::KARMAMesh& mesh, const std::string& local_error )
{
#ifdef MOAB_HAVE_MPI
    if( mesh.n_ranks() > 1 )
    {
        KARMA::detail::collective_signature_error( mesh.communicator(), local_error );
        return;
    }
#endif
    if( !local_error.empty() ) KARMA_THROW( KARMA::InvalidArgument( local_error ) );
}
}  // namespace

namespace KARMA::detail
{
std::vector< SerializedDirichlet > decode_dirichlet_conditions( std::string_view bytes )
{
    constexpr std::size_t header_size = sizeof( PetscInt ) + sizeof( std::uint8_t ) + sizeof( std::uint64_t );
    std::vector< SerializedDirichlet > result;
    std::size_t offset = 0;
    while( offset < bytes.size() )
    {
        if( bytes.size() - offset < header_size )
            KARMA_THROW( InvalidArgument( "truncated Dirichlet serialization header" ) );
        SerializedDirichlet condition{};
        std::uint8_t time_dependent = 0;
        std::uint64_t count = 0;
        std::memcpy( &condition.dof, bytes.data() + offset, sizeof( condition.dof ) );
        offset += sizeof( condition.dof );
        std::memcpy( &time_dependent, bytes.data() + offset, sizeof( time_dependent ) );
        offset += sizeof( time_dependent );
        if( time_dependent > 1 ) KARMA_THROW( InvalidArgument( "invalid Dirichlet time-dependent flag" ) );
        condition.time_dependent = time_dependent != 0;
        std::memcpy( &count, bytes.data() + offset, sizeof( count ) );
        offset += sizeof( count );
        const std::size_t remaining = bytes.size() - offset;
        if( count > remaining / sizeof( Real ) )
            KARMA_THROW( InvalidArgument( "truncated Dirichlet serialization values" ) );
        const std::size_t value_bytes = static_cast< std::size_t >( count ) * sizeof( Real );
        if( value_bytes != remaining && remaining - value_bytes < header_size )
            KARMA_THROW( InvalidArgument( "trailing Dirichlet serialization bytes" ) );
        condition.values.resize( static_cast< std::size_t >( count ) );
        if( value_bytes != 0 ) std::memcpy( condition.values.data(), bytes.data() + offset, value_bytes );
        offset += value_bytes;
        result.push_back( std::move( condition ) );
    }
    if( offset != bytes.size() ) KARMA_THROW( InvalidArgument( "trailing Dirichlet serialization bytes" ) );
    return result;
}
}  // namespace KARMA::detail

KARMAPhysicsBCManager::KARMAPhysicsBCManager( unsigned variable, bool continuous, std::optional< unsigned > component )
    : _variable( variable ), _continuous( continuous ), _component( component )
{
}

KARMAPhysicsBCManager::KARMAPhysicsBCManager( const KARMAPhysicsBCManager& other )
    : _variable( other._variable ), _continuous( other._continuous ), _component( other._component ),
      _dirichlet_function( other._dirichlet_function ), _neumann_function( other._neumann_function ),
      _robin_function( other._robin_function )
#ifdef KARMA_HAVE_LIBMESH
      ,
      _legacy_dirichlet_function( other._legacy_dirichlet_function ),
      _legacy_neumann_function( other._legacy_neumann_function ), _legacy_robin_function( other._legacy_robin_function )
#endif
{
}

KARMAPhysicsBCManager::~KARMAPhysicsBCManager()
{ clear(); }

KARMAPhysicsBCManager& KARMAPhysicsBCManager::operator=( const KARMAPhysicsBCManager& other )
{
    if( this == &other ) return *this;
    clear();
    _variable           = other._variable;
    _continuous         = other._continuous;
    _component          = other._component;
    _dirichlet_function = other._dirichlet_function;
    _neumann_function   = other._neumann_function;
    _robin_function     = other._robin_function;
#ifdef KARMA_HAVE_LIBMESH
    _legacy_dirichlet_function = other._legacy_dirichlet_function;
    _legacy_neumann_function   = other._legacy_neumann_function;
    _legacy_robin_function     = other._legacy_robin_function;
#endif
    return *this;
}

void KARMAPhysicsBCManager::clear() noexcept
{
    _dirichlet.clear();
    _weak_dirichlet.clear();
    _neumann.clear();
    _robin.clear();
    _dirichlet_dofs.clear();
    delete_conditions( _legacy_dirichlet );
    delete_conditions( _legacy_neumann );
    delete_conditions( _legacy_robin );
}

void KARMAPhysicsBCManager::attach_function( modern_boundary_function function, KARMA::PHYSICS::BCType type )
{
    if( !function ) KARMA_THROW( KARMA::InvalidArgument( "boundary function must not be null" ) );
    switch( type )
    {
        case KARMA::PHYSICS::Dirichlet:
            _dirichlet_function = function;
            break;
        case KARMA::PHYSICS::Neumann:
            _neumann_function = function;
            break;
        case KARMA::PHYSICS::Robin:
            _robin_function = function;
            break;
        case KARMA::PHYSICS::UserBC:
            KARMA_THROW( KARMA::InvalidArgument( "only Dirichlet, Neumann, and Robin functions are supported" ) );
    }
}

#ifdef KARMA_HAVE_LIBMESH
void KARMAPhysicsBCManager::attach_function( boundary_function function, KARMA::PHYSICS::BCType type )
{
    karma_assert( function != nullptr );
    switch( type )
    {
        case KARMA::PHYSICS::Dirichlet:
            _legacy_dirichlet_function = function;
            break;
        case KARMA::PHYSICS::Neumann:
            _legacy_neumann_function = function;
            break;
        case KARMA::PHYSICS::Robin:
            _legacy_robin_function = function;
            break;
        case KARMA::PHYSICS::UserBC:
            karma_error();
    }
}

void KARMAPhysicsBCManager::update_boundary_conditions( const Elem* element, KARMAParameters& parameters,
                                                        bool check_all_sides )
{
    for( unsigned side = 0; side < element->n_sides(); ++side )
    {
        auto element_side = element->build_side( side );
        if( element->neighbor( side ) != nullptr && !check_all_sides ) continue;

        if( element->neighbor( side ) == nullptr )
        {
            const Point centroid          = element_side->centroid();
            parameters.set< Real >( "x" ) = centroid( 0 );
            parameters.set< Real >( "y" ) = centroid( 1 );
            parameters.set< Real >( "z" ) = centroid( 2 );
            if( _legacy_neumann_function )
                if( auto* condition = _legacy_neumann_function( parameters ) )
                    add_neumann_condition( element->id(), side, condition );
            if( _legacy_robin_function )
                if( auto* condition = _legacy_robin_function( parameters ) )
                    add_robin_condition( element->id(), side, condition );
        }

        if( !_legacy_dirichlet_function ) continue;
        for( unsigned node_index = 0; node_index < element_side->n_nodes(); ++node_index )
        {
            const Node* node              = element_side->get_node( node_index );
            parameters.set< Real >( "x" ) = ( *node )( 0 );
            parameters.set< Real >( "y" ) = ( *node )( 1 );
            parameters.set< Real >( "z" ) = ( *node )( 2 );
            unsigned dof                  = 0;
            if( _continuous )
                dof = node->id() * ( _variable + 1 );
            else if( element->n_comp( 0, _variable ) > 0 )
                dof = element->id();

            if( auto* condition = _legacy_dirichlet_function( parameters ) )
            {
                if( _continuous )
                    add_dirichlet_condition( dof, condition );
                else
                {
                    add_dirichlet_condition( dof, side, condition );
                    break;
                }
            }
        }
    }
}
#endif

void KARMAPhysicsBCManager::refresh( const KARMA::KARMAMesh& mesh, const KARMA::KARMASystem& system,
                                     KARMAParameters& parameters )
{
    std::unique_ptr< KARMAPhysicsBCManager > replacement;
    std::string local_error;
#ifdef MOAB_HAVE_MPI
    std::string local;
    std::vector< int > sizes;
    int local_size = 0;
#endif
    try
    {
        replacement = std::make_unique< KARMAPhysicsBCManager >( *this );
        mesh.validate_owned_side_topology();
        for( const auto element : mesh.owned_elements() )
            replacement->update_boundary_conditions( mesh, element, system, parameters );
#ifdef MOAB_HAVE_MPI
        if( mesh.n_ranks() > 1 && !mesh.is_replicated() && replacement->_continuous )
        {
            const auto append = [&]( const auto& value ) {
                const char* bytes = reinterpret_cast< const char* >( &value );
                local.append( bytes, sizeof( value ) );
            };
            for( const auto& [dof, condition] : replacement->_dirichlet )
            {
                append( dof );
                const std::uint8_t time_dependent = condition->is_time_dependent_bcs() ? 1 : 0;
                append( time_dependent );
                const std::uint64_t count = condition->values().size();
                append( count );
                for( const Real value : condition->values() ) append( value );
            }
            local_size = KARMA::detail::checked_mpi_count( local.size() );
            sizes.resize( static_cast< std::size_t >( mesh.n_ranks() ) );
        }
#endif
    }
    catch( const std::exception& error )
    {
        local_error = error.what();
    }
    catch( ... )
    {
        local_error = "unknown boundary refresh discovery failure";
    }
    collective_refresh_error( mesh, local_error );
#ifdef MOAB_HAVE_MPI
    if( mesh.n_ranks() > 1 && !mesh.is_replicated() )
    {
        const int local_continuous = replacement->_continuous ? 1 : 0;
        int minimum_continuous = 0;
        int maximum_continuous = 0;
        if( MPI_Allreduce( &local_continuous, &minimum_continuous, 1, MPI_INT, MPI_MIN, mesh.communicator() ) != MPI_SUCCESS ||
            MPI_Allreduce( &local_continuous, &maximum_continuous, 1, MPI_INT, MPI_MAX, mesh.communicator() ) != MPI_SUCCESS )
            KARMA_THROW( KARMA::InvalidArgument( "MPI_Allreduce failed while validating boundary manager configuration" ) );
        if( minimum_continuous != maximum_continuous )
            KARMA_THROW( KARMA::InvalidArgument( "boundary manager collective configuration mismatch: continuity" ) );
    }
    if( mesh.n_ranks() > 1 && !mesh.is_replicated() && replacement->_continuous )
    {
        if( MPI_Allgather( &local_size, 1, MPI_INT, sizes.data(), 1, MPI_INT, mesh.communicator() ) != MPI_SUCCESS )
            KARMA_THROW( KARMA::InvalidArgument( "MPI_Allgather failed while reconciling Dirichlet values" ) );
        KARMA::detail::MpiLayout layout;
        local_error.clear();
        try { layout = KARMA::detail::checked_mpi_displacements( sizes, 1 ); }
        catch( const std::exception& error ) { local_error = error.what(); }
        catch( ... ) { local_error = "unknown Dirichlet gather layout failure"; }
        collective_refresh_error( mesh, local_error );
        std::string gathered;
        local_error.clear();
        try { gathered.resize( layout.total_count ); }
        catch( const std::exception& error ) { local_error = error.what(); }
        catch( ... ) { local_error = "unknown Dirichlet gather allocation failure"; }
        collective_refresh_error( mesh, local_error );
        if( MPI_Allgatherv( local.data(), local_size, MPI_CHAR, gathered.data(), sizes.data(), layout.displacements.data(),
                            MPI_CHAR, mesh.communicator() ) != MPI_SUCCESS )
            KARMA_THROW( KARMA::InvalidArgument( "MPI_Allgatherv failed while reconciling Dirichlet values" ) );
        std::vector< KARMA::detail::SerializedDirichlet > decoded;
        local_error.clear();
        try
        {
            for( std::size_t rank = 0; rank < sizes.size(); ++rank )
            {
                const auto offset = static_cast< std::size_t >( layout.displacements[rank] );
                const auto size = static_cast< std::size_t >( sizes[rank] );
                auto rank_conditions = KARMA::detail::decode_dirichlet_conditions(
                    std::string_view( gathered ).substr( offset, size ) );
                std::move( rank_conditions.begin(), rank_conditions.end(), std::back_inserter( decoded ) );
            }
        }
        catch( const std::exception& error ) { local_error = error.what(); }
        catch( ... ) { local_error = "unknown Dirichlet serialization failure"; }
        collective_refresh_error( mesh, local_error );
        std::map< PetscInt, std::unique_ptr< KARMABoundaryCondition > > canonical;
        std::vector< PetscInt > canonical_dofs;
        bool conflict = false;
        local_error.clear();
        try
        {
            for( auto& decoded_condition : decoded )
            {
                auto condition = std::make_unique< KARMABoundaryCondition >(
                    KARMA::PHYSICS::Dirichlet, decoded_condition.time_dependent );
                for( std::size_t value = 0; value < decoded_condition.values.size(); ++value )
                    condition->set( static_cast< unsigned >( value ), decoded_condition.values[value] );
                const auto found = canonical.find( decoded_condition.dof );
                if( found == canonical.end() ) canonical.emplace( decoded_condition.dof, std::move( condition ) );
                else if( found->second->is_time_dependent_bcs() != condition->is_time_dependent_bcs() ||
                         !equal_dirichlet( *found->second, *condition ) ) conflict = true;
            }
            canonical_dofs.reserve( canonical.size() );
            for( const auto& [dof, condition] : canonical )
            {
                (void)condition;
                canonical_dofs.push_back( dof );
            }
        }
        catch( const std::exception& error ) { local_error = error.what(); }
        catch( ... ) { local_error = "unknown Dirichlet canonicalization failure"; }
        collective_refresh_error( mesh, local_error );
        int local_conflict = conflict ? 1 : 0;
        int global_conflict = 0;
        if( MPI_Allreduce( &local_conflict, &global_conflict, 1, MPI_INT, MPI_MAX, mesh.communicator() ) != MPI_SUCCESS )
            KARMA_THROW( KARMA::InvalidArgument( "MPI_Allreduce failed while reconciling Dirichlet values" ) );
        if( global_conflict ) KARMA_THROW( KARMA::InvalidArgument( "conflicting Dirichlet values for shared DoF" ) );
        replacement->_dirichlet = std::move( canonical );
        replacement->_dirichlet_dofs = std::move( canonical_dofs );
    }
#endif
    swap_conditions( *replacement );
}

void KARMAPhysicsBCManager::swap_conditions( KARMAPhysicsBCManager& other ) noexcept
{
    _dirichlet.swap( other._dirichlet );
    _weak_dirichlet.swap( other._weak_dirichlet );
    _neumann.swap( other._neumann );
    _robin.swap( other._robin );
    _dirichlet_dofs.swap( other._dirichlet_dofs );
}

void KARMAPhysicsBCManager::update_boundary_conditions( const KARMA::KARMAMesh& mesh, moab::EntityHandle element,
                                                        const KARMA::KARMASystem& system, KARMAParameters& parameters,
                                                        bool check_all_sides )
{
    parameters.set< int >( "material" ) = mesh.material_id( element );
    const auto& variable                = system.variable( _variable );
    if( _component && *_component >= variable.components )
        KARMA_THROW( KARMA::InvalidArgument( "boundary component is out of range" ) );
    std::vector< PetscInt > element_dofs;
    system.dof_map().dof_indices( element, element_dofs, _variable );
    const auto geometry = KARMA::KARMAGeometryMap( mesh, element );
    const auto basis    = system.basis_registry().create(
        { KARMA::reference_cell( geometry.variant() ), variable.family, variable.order, variable.continuity } );
    const auto associations = basis.dof_associations();
    const auto points       = basis.interpolation_points();

    for( unsigned side = 0; side < mesh.n_sides( element ); ++side )
    {
        const bool boundary = mesh.is_boundary_side( element, side );
        if( !boundary && !check_all_sides ) continue;
        const auto id                                     = mesh.side_id( element, side );
        parameters.set< unsigned >( "side_id" )           = side;
        parameters.set< KARMA::GlobalId >( "element_id" ) = id.element;

        const auto side_vertices = mesh.side_vertices( element, side );
        KARMA::Point centroid    = KARMA::Point::Zero();
        std::vector< Real > coordinates( 3 * side_vertices.size() );
        if( mesh.moab()->get_coords( side_vertices.data(), side_vertices.size(), coordinates.data() ) !=
            moab::MB_SUCCESS )
            KARMA_THROW( KARMA::InvalidArgument( "failed to read boundary coordinates" ) );
        for( std::size_t i = 0; i < side_vertices.size(); ++i )
            for( unsigned d = 0; d < 3; ++d )
                centroid( d ) += coordinates[3 * i + d];
        centroid /= static_cast< Real >( side_vertices.size() );
        parameters.set< Real >( "x" ) = centroid( 0 );
        parameters.set< Real >( "y" ) = centroid( 1 );
        parameters.set< Real >( "z" ) = centroid( 2 );

        if( boundary && _neumann_function )
            if( auto condition = _neumann_function( parameters ) )
            {
                validate_type( *condition, KARMA::PHYSICS::Neumann );
                _neumann[{ id.element, id.side }] = std::move( condition );
            }
        if( boundary && _robin_function )
            if( auto condition = _robin_function( parameters ) )
            {
                validate_type( *condition, KARMA::PHYSICS::Robin );
                if( _neumann.count( { id.element, id.side } ) != 0 )
                    KARMA_THROW( KARMA::InvalidArgument( "overlapping natural boundary conditions" ) );
                _robin[{ id.element, id.side }] = std::move( condition );
            }
        if( !_dirichlet_function ) continue;

        const bool has_natural =
            _neumann.count( { id.element, id.side } ) != 0 || _robin.count( { id.element, id.side } ) != 0;
        if( !_continuous )
        {
            if( auto condition = _dirichlet_function( parameters ) )
            {
                validate_type( *condition, KARMA::PHYSICS::Dirichlet );
                if( has_natural )
                    KARMA_THROW( KARMA::InvalidArgument( "overlapping essential and natural boundary conditions" ) );
                _weak_dirichlet[{ id.element, id.side }] = std::move( condition );
            }
            continue;
        }
        for( unsigned component = 0; component < variable.components; ++component )
        {
            if( _component && component != *_component ) continue;
            for( std::size_t basis_index = 0; basis_index < associations.size(); ++basis_index )
            {
                if( !association_is_on_side( geometry.topology(), geometry.reference_dimension(),
                                             associations[basis_index], side ) )
                    continue;
                const auto point              = geometry.map( points[basis_index] );
                parameters.set< Real >( "x" ) = point( 0 );
                parameters.set< Real >( "y" ) = point( 1 );
                parameters.set< Real >( "z" ) = point( 2 );
                auto condition                = _dirichlet_function( parameters );
                if( !condition ) continue;
                validate_type( *condition, KARMA::PHYSICS::Dirichlet );
                if( has_natural )
                    KARMA_THROW( KARMA::InvalidArgument( "overlapping essential and natural boundary conditions" ) );
                const std::size_t offset = component * associations.size() + basis_index;
                const PetscInt dof       = element_dofs.at( offset );
                const auto existing      = _dirichlet.find( dof );
                if( existing != _dirichlet.end() )
                {
                    if( !equal_dirichlet( *existing->second, *condition ) )
                        KARMA_THROW( KARMA::InvalidArgument( "conflicting Dirichlet values for shared DoF" ) );
                    continue;
                }
                _dirichlet.emplace( dof, std::move( condition ) );
            }
        }
    }
    _dirichlet_dofs.clear();
    for( const auto& [dof, condition] : _dirichlet )
    {
        (void)condition;
        _dirichlet_dofs.push_back( dof );
    }
}

std::string KARMAPhysicsBCManager::dirichlet_key( unsigned node )
{ return "D;" + std::to_string( node ); }

std::string KARMAPhysicsBCManager::dirichlet_key( unsigned component, unsigned side )
{ return "D;" + std::to_string( component ) + ";" + std::to_string( side ); }

std::string KARMAPhysicsBCManager::neumann_key( unsigned element, unsigned side )
{ return "N;" + std::to_string( element ) + ";" + std::to_string( side ); }

std::string KARMAPhysicsBCManager::robin_key( unsigned element, unsigned side )
{ return "R;" + std::to_string( element ) + ";" + std::to_string( side ); }

bool KARMAPhysicsBCManager::is_dirichlet_condition( std::string& key ) const
{ return _legacy_dirichlet.count( key ) != 0; }

bool KARMAPhysicsBCManager::is_neumann_condition( std::string& key ) const
{ return _legacy_neumann.count( key ) != 0; }

bool KARMAPhysicsBCManager::is_robin_condition( std::string& key ) const
{ return _legacy_robin.count( key ) != 0; }

namespace
{
bool replace_legacy_condition( BoundaryNodes& conditions, std::string key, KARMABoundaryCondition* condition )
{
    if( !condition ) return false;
    const auto found = conditions.find( key );
    if( found != conditions.end() ) delete found->second;
    conditions[std::move( key )] = condition;
    return true;
}
}  // namespace

bool KARMAPhysicsBCManager::add_dirichlet_condition( unsigned node, KARMABoundaryCondition* condition )
{ return replace_legacy_condition( _legacy_dirichlet, dirichlet_key( node ), condition ); }

bool KARMAPhysicsBCManager::add_dirichlet_condition( unsigned component, unsigned side,
                                                     KARMABoundaryCondition* condition )
{ return replace_legacy_condition( _legacy_dirichlet, dirichlet_key( component, side ), condition ); }

bool KARMAPhysicsBCManager::add_neumann_condition( unsigned element, unsigned side, KARMABoundaryCondition* condition )
{ return replace_legacy_condition( _legacy_neumann, neumann_key( element, side ), condition ); }

bool KARMAPhysicsBCManager::add_robin_condition( unsigned element, unsigned side, KARMABoundaryCondition* condition )
{ return replace_legacy_condition( _legacy_robin, robin_key( element, side ), condition ); }

namespace
{
bool delete_legacy_condition( BoundaryNodes& conditions, const std::string& key )
{
    const auto found = conditions.find( key );
    if( found == conditions.end() ) return false;
    delete found->second;
    conditions.erase( found );
    return true;
}
}  // namespace

bool KARMAPhysicsBCManager::delete_dirichlet_condition( unsigned node )
{ return delete_legacy_condition( _legacy_dirichlet, dirichlet_key( node ) ); }

bool KARMAPhysicsBCManager::delete_dirichlet_condition( unsigned component, unsigned side )
{ return delete_legacy_condition( _legacy_dirichlet, dirichlet_key( component, side ) ); }

bool KARMAPhysicsBCManager::delete_neumann_condition( unsigned element, unsigned side )
{ return delete_legacy_condition( _legacy_neumann, neumann_key( element, side ) ); }

bool KARMAPhysicsBCManager::delete_robin_condition( unsigned element, unsigned side )
{ return delete_legacy_condition( _legacy_robin, robin_key( element, side ) ); }

const BoundaryNodes& KARMAPhysicsBCManager::get_dirichlet_condition() const noexcept
{ return _legacy_dirichlet; }

const BoundaryNodes& KARMAPhysicsBCManager::get_neumann_condition() const noexcept
{ return _legacy_neumann; }

const BoundaryNodes& KARMAPhysicsBCManager::get_robin_condition() const noexcept
{ return _legacy_robin; }

unsigned KARMAPhysicsBCManager::n_dirichlet_nodes() const noexcept
{
    unsigned count = static_cast< unsigned >( _dirichlet.size() + _weak_dirichlet.size() );
    for( const auto& [key, condition] : _legacy_dirichlet )
    {
        (void)condition;
        unsigned first = 0;
        unsigned side  = 0;
        char prefix{};
        const int fields = std::sscanf( key.c_str(), "%c;%u;%u", &prefix, &first, &side );
        const bool overlaps_strong =
            fields == 2 && prefix == 'D' && _dirichlet.count( static_cast< PetscInt >( first ) ) != 0;
        const bool overlaps_weak = fields == 3 && prefix == 'D' &&
                                   first <= static_cast< unsigned >( std::numeric_limits< KARMA::GlobalId >::max() ) &&
                                   _weak_dirichlet.count( { static_cast< KARMA::GlobalId >( first ), side } ) != 0;
        if( !overlaps_strong && !overlaps_weak ) ++count;
    }
    return count;
}

unsigned KARMAPhysicsBCManager::n_neumann_nodes() const noexcept
{ return union_size( _neumann, _legacy_neumann, 'N' ); }

unsigned KARMAPhysicsBCManager::n_robin_nodes() const noexcept
{ return union_size( _robin, _legacy_robin, 'R' ); }

void KARMAPhysicsBCManager::print_conditions() const
{
    for( const auto& [key, condition] : _legacy_dirichlet )
        PetscPrintf( KARMA_COMM_WORLD, "\n%s\t\t%g", key.c_str(), ( *condition )[0] );
    for( const auto& [dof, condition] : _dirichlet )
        if( _legacy_dirichlet.count( dirichlet_key( static_cast< unsigned >( dof ) ) ) == 0 )
            PetscPrintf( KARMA_COMM_WORLD, "\nD;%D\t\t%g", dof, ( *condition )[0] );
    for( const auto& [id, condition] : _weak_dirichlet )
        if( _legacy_dirichlet.count( dirichlet_key( static_cast< unsigned >( id.first ), id.second ) ) == 0 )
            PetscPrintf( KARMA_COMM_WORLD, "\nD;%d;%u\t\t%g", id.first, id.second, ( *condition )[0] );

    for( const auto& [key, condition] : _legacy_neumann )
        PetscPrintf( KARMA_COMM_WORLD, "\n%s\t\t%g", key.c_str(), ( *condition )[0] );
    for( const auto& [id, condition] : _neumann )
        if( _legacy_neumann.count( neumann_key( static_cast< unsigned >( id.first ), id.second ) ) == 0 )
            PetscPrintf( KARMA_COMM_WORLD, "\nN;%d;%u\t\t%g", id.first, id.second, ( *condition )[0] );

    for( const auto& [key, condition] : _legacy_robin )
        PetscPrintf( KARMA_COMM_WORLD, "\n%s\t\t%g\t%g\t%g", key.c_str(), ( *condition )[0], ( *condition )[1],
                     ( *condition )[2] );
    for( const auto& [id, condition] : _robin )
        if( _legacy_robin.count( robin_key( static_cast< unsigned >( id.first ), id.second ) ) == 0 )
            PetscPrintf( KARMA_COMM_WORLD, "\nR;%d;%u\t\t%g\t%g\t%g", id.first, id.second, ( *condition )[0],
                         ( *condition )[1], ( *condition )[2] );
}

std::string KARMAPhysicsBCManager::assembly_signature( const KARMA::KARMASystem& system ) const
{
    std::map< KARMA::GlobalId, moab::EntityHandle > elements_by_id;
    if( !_weak_dirichlet.empty() || !_neumann.empty() || !_robin.empty() )
        for( const auto element : system.mesh().owned_elements() )
        {
            const auto id = system.mesh().global_id( element );
            if( !elements_by_id.emplace( id, element ).second )
                KARMA_THROW( KARMA::InvalidArgument( "duplicate private element ID in boundary signature resolution" ) );
        }

    std::string result;
    const auto append_integer = [&]( auto value ) {
        const char* bytes = reinterpret_cast< const char* >( &value );
        result.append( bytes, sizeof( value ) );
    };
    const auto append_condition = [&]( const KARMABoundaryCondition& condition ) {
        append_integer( static_cast< int >( condition.type() ) );
        append_integer( condition.is_time_dependent_bcs() );
        append_integer( condition.values().size() );
        for( const double value : condition.values() )
        {
            std::uint64_t bits = 0;
            static_assert( sizeof( bits ) == sizeof( value ) );
            std::memcpy( &bits, &value, sizeof( value ) );
            append_integer( bits );
        }
    };
    const auto append_sides = [&]( char kind, const Conditions& conditions ) {
        using CanonicalSide = std::pair< std::uint64_t, unsigned >;
        std::map< CanonicalSide, const KARMABoundaryCondition* > canonical_conditions;
        for( const auto& [key, condition] : conditions )
        {
            const auto found = elements_by_id.find( key.first );
            if( found == elements_by_id.end() )
                KARMA_THROW( KARMA::InvalidArgument( "missing private element ID in boundary signature resolution" ) );
            const CanonicalSide canonical{ system.dof_map().canonical_element_ordinal( found->second ), key.second };
            if( !canonical_conditions.emplace( canonical, condition.get() ).second )
                KARMA_THROW( KARMA::InvalidArgument( "duplicate canonical side in boundary signature resolution" ) );
        }
        for( const auto& [key, condition] : canonical_conditions )
        {
            append_integer( kind );
            append_integer( key.first );
            append_integer( key.second );
            append_condition( *condition );
        }
    };

    for( const auto& [dof, condition] : _dirichlet )
    {
        append_integer( 'D' );
        append_integer( dof );
        append_condition( *condition );
    }
    append_sides( 'W', _weak_dirichlet );
    append_sides( 'N', _neumann );
    append_sides( 'R', _robin );
    const auto append_legacy = [&]( char kind, const BoundaryNodes& conditions ) {
        for( const auto& [key, condition] : conditions )
        {
            append_integer( kind );
            append_integer( key.size() );
            result.append( key );
            append_condition( *condition );
        }
    };
    append_legacy( 'd', _legacy_dirichlet );
    append_legacy( 'n', _legacy_neumann );
    append_legacy( 'r', _legacy_robin );
    return result;
}

KARMABoundaryCondition* KARMAPhysicsBCManager::get_dirichlet_condition( PetscInt dof ) const
{
    if( auto* condition = find_condition( _dirichlet, dof ) ) return condition;
    const auto found = _legacy_dirichlet.find( dirichlet_key( static_cast< unsigned >( dof ) ) );
    return found == _legacy_dirichlet.end() ? nullptr : found->second;
}

KARMABoundaryCondition* KARMAPhysicsBCManager::get_dirichlet_condition( unsigned element, unsigned side ) const
{
    const auto found = _legacy_dirichlet.find( dirichlet_key( element, side ) );
    return found == _legacy_dirichlet.end() ? nullptr : found->second;
}

KARMABoundaryCondition* KARMAPhysicsBCManager::get_neumann_condition( unsigned element, unsigned side ) const
{
    const auto found = _legacy_neumann.find( neumann_key( element, side ) );
    return found == _legacy_neumann.end() ? nullptr : found->second;
}

KARMABoundaryCondition* KARMAPhysicsBCManager::get_robin_condition( unsigned element, unsigned side ) const
{
    const auto found = _legacy_robin.find( robin_key( element, side ) );
    return found == _legacy_robin.end() ? nullptr : found->second;
}

KARMABoundaryCondition* KARMAPhysicsBCManager::get_dirichlet_condition( KARMA::GlobalId element, unsigned side ) const
{
    if( auto* condition = find_condition( _weak_dirichlet, SideKey{ element, side } ) ) return condition;
    if( element < 0 ) return nullptr;
    return get_dirichlet_condition( static_cast< unsigned >( element ), side );
}

KARMABoundaryCondition* KARMAPhysicsBCManager::get_neumann_condition( KARMA::GlobalId element, unsigned side ) const
{
    if( auto* condition = find_condition( _neumann, SideKey{ element, side } ) ) return condition;
    if( element < 0 ) return nullptr;
    return get_neumann_condition( static_cast< unsigned >( element ), side );
}

KARMABoundaryCondition* KARMAPhysicsBCManager::get_robin_condition( KARMA::GlobalId element, unsigned side ) const
{
    if( auto* condition = find_condition( _robin, SideKey{ element, side } ) ) return condition;
    if( element < 0 ) return nullptr;
    return get_robin_condition( static_cast< unsigned >( element ), side );
}

KARMA::RealDenseVector KARMAPhysicsBCManager::integrate_face_load( const KARMA::KARMAFEBase& fe, Real value )
{
    const auto& phi = fe.get_phi();
    const auto& jxw = fe.get_JxW();
    KARMA::RealDenseVector result( phi.size() );
    for( std::size_t i = 0; i < phi.size(); ++i )
        for( std::size_t q = 0; q < jxw.size(); ++q )
            result( i ) += value * phi[i][q] * jxw[q];
    return result;
}

KARMA::RealDenseMatrix KARMAPhysicsBCManager::integrate_face_mass( const KARMA::KARMAFEBase& fe, Real value )
{
    const auto& phi = fe.get_phi();
    const auto& jxw = fe.get_JxW();
    KARMA::RealDenseMatrix result( phi.size(), phi.size() );
    for( std::size_t i = 0; i < phi.size(); ++i )
        for( std::size_t j = 0; j < phi.size(); ++j )
            for( std::size_t q = 0; q < jxw.size(); ++q )
                result( i, j ) += value * phi[i][q] * phi[j][q] * jxw[q];
    return result;
}

KARMA::RealDenseVector KARMAPhysicsBCManager::integrate_neumann_load( const KARMA::KARMAFEBase& fe,
                                                                      const KARMABoundaryCondition& condition )
{
    if( condition.type() != KARMA::PHYSICS::Neumann || condition.values().empty() )
        KARMA_THROW( KARMA::InvalidArgument( "Neumann integration requires a scalar outward flux" ) );
    return integrate_face_load( fe, condition[0] );
}

KARMAPhysicsBCManager::RobinTerms KARMAPhysicsBCManager::integrate_robin( const KARMA::KARMAFEBase& fe,
                                                                          const KARMABoundaryCondition& condition )
{
    if( condition.type() != KARMA::PHYSICS::Robin || condition.values().size() != 3 )
        KARMA_THROW( KARMA::InvalidArgument( "Robin integration requires coefficients a, b, and c" ) );
    const Real a = condition[0];
    if( a == 0 ) KARMA_THROW( KARMA::InvalidArgument( "Robin coefficient a must be nonzero" ) );
    return { integrate_face_mass( fe, condition[1] / a ), integrate_face_load( fe, condition[2] / a ) };
}
