// KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
// Copyright (C) 2007-2011  Vijay S Mahadevan
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//
// Tests for KARMASystem and KARMADofMap -- the seam that nine source files
// referenced through "karma_system.h" before that header existed.

#include "karma_exceptions.h"
#include "karma_moab_io.h"
#include "karma_system.h"
#include "../src/physics/common/impls/collective_signature.h"
#include "karma_system_base.h"
#include "petsc_raii.h"

#include "moab/CN.hpp"

#include <petsc.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

namespace
{
int failures = 0;

void check( bool ok, const std::string& what )
{
    if( !ok )
    {
        std::cerr << "FAIL: " << what << std::endl;
        ++failures;
    }
}

void equal( std::size_t got, std::size_t expected, const std::string& what )
{
    if( got != expected )
    {
        std::cerr << "FAIL: " << what << " (got " << got << ", expected " << expected << ")" << std::endl;
        ++failures;
    }
}

/** @returns true if @p fn threw a KARMA logic error. */
template < typename F >
bool throws( F&& fn )
{
    try
    {
        fn();
    }
    catch( const std::logic_error& )
    {
        return true;
    }
    catch( ... )
    {
        return false;
    }
    return false;
}

template < typename F >
bool throws_containing( F&& fn, std::string_view text )
{
    try
    {
        fn();
    }
    catch( const std::logic_error& error )
    {
        return std::string_view( error.what() ).find( text ) != std::string_view::npos;
    }
    return false;
}

struct ThreeDofsPerEdgeBasis
{
    [[nodiscard]] std::string name() const
    { return "three-dofs-per-edge"; }
    [[nodiscard]] std::string compatibility_id() const { return "test.three-dofs-per-edge.v1"; }
    [[nodiscard]] KARMA::CoefficientRepresentation coefficient_representation() const noexcept
    { return KARMA::CoefficientRepresentation::Modal; }
    [[nodiscard]] KARMA::BasisKey key() const noexcept
    {
        return { KARMA::ReferenceCell::Quadrilateral, KARMA::FieldBasisFamily::UserDefined, 3,
                 KARMA::BasisContinuity::cG };
    }
    [[nodiscard]] std::size_t size() const noexcept
    { return 12; }
    void evaluate_values( const KARMA::Point&, std::span< Real > values ) const
    { std::fill( values.begin(), values.end(), 0 ); }
    void evaluate( const KARMA::Point&, std::span< Real > values, std::span< KARMA::Gradient > gradients ) const
    {
        std::fill( values.begin(), values.end(), 0 );
        std::fill( gradients.begin(), gradients.end(), KARMA::Gradient::Zero() );
    }
    [[nodiscard]] std::vector< KARMA::Point > interpolation_points() const
    { return std::vector< KARMA::Point >( size(), KARMA::Point::Zero() ); }
    [[nodiscard]] std::vector< KARMA::DofAssociation > dof_associations() const
    {
        std::vector< KARMA::DofAssociation > result;
        for( unsigned edge = 0; edge < 4; ++edge )
            for( unsigned slot = 0; slot < 3; ++slot )
                result.push_back( { 1, edge, slot } );
        return result;
    }
    [[nodiscard]] std::vector< KARMA::OrientationTransform > orientation_transforms() const
    {
        std::vector< KARMA::OrientationTransform > result;
        for( unsigned edge = 0; edge < 4; ++edge )
        {
            result.push_back( { 1, edge, 0, { 0, 1, 2 }, { 1, 1, 1 } } );
            result.push_back( { 1, edge, 1, { 2, 1, 0 }, { 1, 1, 1 } } );
        }
        return result;
    }
};

struct MultiSlotFaceBasis
{
    KARMA::ReferenceCell cell;
    unsigned face_count;
    unsigned orientation_count;
    [[nodiscard]] std::string name() const { return "multi-slot-face"; }
    [[nodiscard]] std::string compatibility_id() const { return "test.multi-slot-face.v1"; }
    [[nodiscard]] KARMA::CoefficientRepresentation coefficient_representation() const noexcept
    { return KARMA::CoefficientRepresentation::Modal; }
    [[nodiscard]] KARMA::BasisKey key() const noexcept
    { return { cell, KARMA::FieldBasisFamily::UserDefined, 6, KARMA::BasisContinuity::cG }; }
    [[nodiscard]] std::size_t size() const noexcept { return 2 * face_count; }
    void evaluate_values( const KARMA::Point&, std::span< Real > values ) const
    { std::fill( values.begin(), values.end(), 0 ); }
    void evaluate( const KARMA::Point&, std::span< Real > values, std::span< KARMA::Gradient > gradients ) const
    {
        std::fill( values.begin(), values.end(), 0 );
        std::fill( gradients.begin(), gradients.end(), KARMA::Gradient::Zero() );
    }
    [[nodiscard]] std::vector< KARMA::Point > interpolation_points() const
    { return std::vector< KARMA::Point >( size(), KARMA::Point::Zero() ); }
    [[nodiscard]] std::vector< KARMA::DofAssociation > dof_associations() const
    {
        std::vector< KARMA::DofAssociation > result;
        for( unsigned face = 0; face < face_count; ++face )
            for( unsigned slot = 0; slot < 2; ++slot ) result.push_back( { 2, face, slot } );
        return result;
    }
    [[nodiscard]] std::vector< KARMA::OrientationTransform > orientation_transforms() const
    {
        std::vector< KARMA::OrientationTransform > result;
        for( unsigned face = 0; face < face_count; ++face )
            for( unsigned code = 0; code < orientation_count; ++code )
                result.push_back( { 2, face, code, code % 2 == 0 ? std::vector< std::size_t >{ 0, 1 }
                                                                    : std::vector< std::size_t >{ 1, 0 },
                                    { 1, 1 } } );
        return result;
    }
};

struct LocalAssociatedBasis
{
    KARMA::BasisContinuity continuity;
    [[nodiscard]] std::string name() const { return "local-associated"; }
    [[nodiscard]] std::string compatibility_id() const { return "test.local-associated.v1"; }
    [[nodiscard]] KARMA::CoefficientRepresentation coefficient_representation() const noexcept
    { return KARMA::CoefficientRepresentation::Modal; }
    [[nodiscard]] KARMA::BasisKey key() const noexcept
    {
        return { KARMA::ReferenceCell::Triangle, KARMA::FieldBasisFamily::UserDefined, 4, continuity };
    }
    [[nodiscard]] std::size_t size() const noexcept { return 3; }
    void evaluate_values( const KARMA::Point&, std::span< Real > values ) const
    { std::fill( values.begin(), values.end(), 0 ); }
    void evaluate( const KARMA::Point&, std::span< Real > values, std::span< KARMA::Gradient > gradients ) const
    {
        std::fill( values.begin(), values.end(), 0 );
        std::fill( gradients.begin(), gradients.end(), KARMA::Gradient::Zero() );
    }
    [[nodiscard]] std::vector< KARMA::Point > interpolation_points() const
    { return std::vector< KARMA::Point >( size(), KARMA::Point::Zero() ); }
    [[nodiscard]] std::vector< KARMA::DofAssociation > dof_associations() const
    { return { { 0, 0, 0 }, { 0, 1, 0 }, { 1, 0, 0 } }; }
    [[nodiscard]] std::vector< KARMA::OrientationTransform > orientation_transforms() const
    { return { { 1, 0, 0, { 0 }, { 1 } }, { 1, 0, 1, { 0 }, { 1 } } }; }
};

using ElementDofKeys = std::map< int, std::vector< KARMA::DofKey > >;

int conventional_id( KARMA::KARMAMesh& mesh, moab::EntityHandle entity )
{
    moab::Tag global_id = nullptr;
    check( mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ) == moab::MB_SUCCESS,
           "find oracle GLOBAL_ID" );
    int id = 0;
    check( mesh.moab()->tag_get_data( global_id, &entity, 1, &id ) == moab::MB_SUCCESS && id > 0,
           "read positive oracle GLOBAL_ID" );
    return id;
}

std::pair< std::vector< int >, unsigned > canonical_oracle_cycle( std::span< const int > local_ids )
{
    std::vector< int > best;
    unsigned orientation = 0;
    for( unsigned reflected = 0; reflected < 2; ++reflected )
        for( unsigned rotation = 0; rotation < local_ids.size(); ++rotation )
        {
            std::vector< int > candidate( local_ids.size() );
            for( unsigned local = 0; local < local_ids.size(); ++local )
            {
                const unsigned canonical = reflected == 0
                                               ? ( local + rotation ) % local_ids.size()
                                               : ( rotation + local_ids.size() - local ) % local_ids.size();
                candidate[canonical] = local_ids[local];
            }
            if( best.empty() || candidate < best )
            {
                best = std::move( candidate );
                orientation = reflected == 0 ? rotation : static_cast< unsigned >( local_ids.size() ) + rotation;
            }
        }
    return { std::move( best ), orientation };
}

KARMA::DofKey expected_dof_key( unsigned variable, unsigned component, unsigned dimension,
                                moab::EntityType type, unsigned slot, std::span< const int > ids )
{
    KARMA::DofKey key;
    key.words[0] = variable;
    key.words[1] = component;
    key.words[2] = dimension;
    key.words[3] = static_cast< std::uint64_t >( type );
    key.words[4] = slot;
    key.words[5] = ids.size();
    for( std::size_t i = 0; i < ids.size(); ++i ) key.words[6 + 3 * i] = static_cast< std::uint64_t >( ids[i] );
    return key;
}

ElementDofKeys expected_fixture_dof_keys( KARMA::KARMAMesh& mesh,
                                          const std::vector< KARMA::VariableSpec >& variables,
                                          const KARMA::KARMABasisRegistry& registry )
{
    using namespace KARMA;
    ElementDofKeys result;
    for( const auto element : mesh.elements() )
    {
        const auto type = mesh.moab()->type_from_handle( element );
        const auto cell = reference_cell( KARMAReferenceTopology::from_moab( type, 4 ) );
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        check( mesh.moab()->get_connectivity( element, connectivity, count ) == moab::MB_SUCCESS,
               "read oracle element connectivity" );
        std::vector< DofKey > keys;
        for( unsigned variable = 0; variable < variables.size(); ++variable )
        {
            const auto basis = registry.create(
                { cell, variables[variable].family, variables[variable].order, variables[variable].continuity } );
            const auto associations = basis.dof_associations();
            for( unsigned component = 0; component < variables[variable].components; ++component )
                for( std::size_t ordinal = 0; ordinal < associations.size(); ++ordinal )
                {
                    const auto& association = associations[ordinal];
                    unsigned dimension = mesh.dimension();
                    moab::EntityType entity_type = type;
                    unsigned canonical_slot = static_cast< unsigned >( ordinal );
                    std::vector< int > identities{ conventional_id( mesh, element ) };
                    if( variables[variable].continuity == BasisContinuity::cG )
                    {
                        dimension = association.entity_dimension;
                        canonical_slot = association.slot;
                        if( dimension == 0 )
                        {
                            entity_type = moab::MBVERTEX;
                            identities = { conventional_id( mesh, connectivity[association.local_entity_index] ) };
                        }
                        else if( dimension < mesh.dimension() )
                        {
                            moab::EntityHandle side = 0;
                            check( mesh.moab()->side_element( element, dimension, association.local_entity_index,
                                                              side ) == moab::MB_SUCCESS && side != 0,
                                   "oracle resolves explicit MOAB side" );
                            entity_type = mesh.moab()->type_from_handle( side );
                            const moab::EntityHandle* side_connectivity = nullptr;
                            int side_count = 0;
                            check( mesh.moab()->get_connectivity( side, side_connectivity, side_count ) ==
                                       moab::MB_SUCCESS,
                                   "read oracle side connectivity" );
                            identities.clear();
                            for( int i = 0; i < side_count; ++i )
                                identities.push_back( conventional_id( mesh, side_connectivity[i] ) );
                            auto [canonical_ids, orientation] = canonical_oracle_cycle( identities );
                            identities = std::move( canonical_ids );
                            const auto transforms = basis.orientation_transforms();
                            const auto transform = std::find_if( transforms.begin(), transforms.end(), [&]( const auto& item ) {
                                return item.entity_dimension == dimension &&
                                       item.local_entity_index == association.local_entity_index &&
                                       item.orientation_code == orientation;
                            } );
                            if( transform != transforms.end() )
                                canonical_slot = static_cast< unsigned >(
                                    transform->local_to_canonical.at( association.slot ) );
                        }
                    }
                    keys.push_back( expected_dof_key( variable, component, dimension, entity_type,
                                                      canonical_slot, identities ) );
                }
        }
        result.emplace( conventional_id( mesh, element ), std::move( keys ) );
    }
    return result;
}

ElementDofKeys actual_fixture_dof_keys( KARMA::KARMAMesh& mesh, const KARMA::KARMADofMap& dofs )
{
    moab::Tag global_id = nullptr;
    check( mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ) == moab::MB_SUCCESS,
           "find fixture GLOBAL_ID" );
    ElementDofKeys result;
    for( const auto element : mesh.elements() )
    {
        int id = 0;
        check( mesh.moab()->tag_get_data( global_id, &element, 1, &id ) == moab::MB_SUCCESS,
               "read fixture element GLOBAL_ID" );
        result.emplace( id, dofs.dof_keys( element ) );
    }
    return result;
}

void write_dof_reference( const std::string& mesh_filename, const std::string& reference_filename )
{
    using namespace KARMA;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMAMoabIO( mesh ).read_mesh( mesh_filename );
    const std::vector< VariableSpec > variables{
        { "cg", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 2 },
        { "dg", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::dG, 1 },
        { "se", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::SE, 1 },
    };
    KARMABasisRegistry registry;
    const auto expected = expected_fixture_dof_keys( mesh, variables, registry );
    std::set< DofKey > unique_keys;
    for( const auto& [element, keys] : expected )
    {
        (void)element;
        unique_keys.insert( keys.begin(), keys.end() );
    }
    std::ofstream output( reference_filename );
    check( output.good(), "open serial DoF reference" );
    output << unique_keys.size() << '\n';
    for( const auto& [element, keys] : expected )
        for( std::size_t ordinal = 0; ordinal < keys.size(); ++ordinal )
        {
            output << element << ' ' << ordinal;
            for( const auto word : keys[ordinal].words ) output << ' ' << word;
            output << '\n';
        }
}

void verify_partitioned_dof_ownership( const std::string& filename, const std::string& reference_filename )
{
    using namespace KARMA;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMAMoabIO( mesh ).read_mesh(
        filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    check( mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 1, true ) == moab::MB_SUCCESS,
           "exchange partition fixture ghost elements" );
    mesh.update();

    const std::vector< VariableSpec > variables{
        { "cg", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 2 },
        { "dg", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::dG, 1 },
        { "se", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::SE, 1 },
    };
    KARMABasisRegistry registry;
    KARMADofMap dofs( mesh, variables, registry );

    std::ifstream input( reference_filename );
    check( input.good(), "open serial DoF reference" );
    PetscInt reference_count = 0;
    input >> reference_count;
    check( dofs.n_dofs() == reference_count, "partitioned DoF count matches serial reference" );
    ElementDofKeys reference;
    int element_id = 0;
    std::size_t ordinal = 0;
    DofKey key;
    while( input >> element_id >> ordinal )
    {
        for( auto& word : key.words ) input >> word;
        auto& keys = reference[element_id];
        if( keys.size() <= ordinal ) keys.resize( ordinal + 1 );
        keys[ordinal] = key;
    }
    std::set< DofKey > reference_keys;
    for( const auto& [id, keys] : reference )
    {
        (void)id;
        reference_keys.insert( keys.begin(), keys.end() );
    }
    const auto visible_keys = actual_fixture_dof_keys( mesh, dofs );
    std::map< DofKey, PetscInt > key_indices;
    std::set< DofKey > local_key_set;
    for( const auto& [id, keys] : visible_keys )
    {
        check( reference.contains( id ) && reference.at( id ) == keys,
               "partitioned element ordered DoF keys match serial reference" );
        local_key_set.insert( keys.begin(), keys.end() );
        const auto element = *std::find_if( mesh.elements().begin(), mesh.elements().end(), [&]( auto candidate ) {
            moab::Tag global_id = nullptr;
            int candidate_id = 0;
            return mesh.moab()->tag_get_handle( "GLOBAL_ID", global_id ) == moab::MB_SUCCESS &&
                   mesh.moab()->tag_get_data( global_id, &candidate, 1, &candidate_id ) == moab::MB_SUCCESS &&
                   candidate_id == id;
        } );
        std::vector< PetscInt > indices;
        dofs.dof_indices( element, indices );
        check( indices.size() == keys.size(), "DoF keys align one-to-one with indices" );
        for( std::size_t i = 0; i < keys.size(); ++i )
        {
            const auto [found, inserted] = key_indices.emplace( keys[i], indices[i] );
            check( inserted || found->second == indices[i], "one stable DoF key has one global index" );
        }
    }
    const int local_key_words = detail::checked_mpi_word_count( local_key_set.size(), DofKey::width );
    std::vector< std::uint64_t > local_key_data;
    local_key_data.reserve( static_cast< std::size_t >( local_key_words ) );
    for( const auto& local_key : local_key_set )
        local_key_data.insert( local_key_data.end(), local_key.begin(), local_key.end() );
    std::vector< int > key_counts( static_cast< std::size_t >( mesh.n_ranks() ) );
    check( MPI_Allgather( &local_key_words, 1, MPI_INT, key_counts.data(), 1, MPI_INT,
                          PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather visible stable-key counts" );
    const auto key_layout = detail::checked_mpi_displacements( key_counts, DofKey::width );
    std::vector< std::uint64_t > all_key_data( key_layout.total_count );
    check( MPI_Allgatherv( local_key_data.data(), local_key_words, MPI_UINT64_T, all_key_data.data(),
                           key_counts.data(), key_layout.displacements.data(), MPI_UINT64_T,
                           PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather visible stable keys" );
    std::set< DofKey > global_key_set;
    for( std::size_t offset = 0; offset < all_key_data.size(); offset += DofKey::width )
    {
        DofKey gathered_key;
        std::copy_n( all_key_data.begin() + static_cast< std::ptrdiff_t >( offset ), DofKey::width,
                     gathered_key.begin() );
        global_key_set.insert( gathered_key );
    }
    check( global_key_set == reference_keys, "partitioned global DoF keys equal serial reference" );

    constexpr std::size_t indexed_record_words = DofKey::width + 1;
    const int local_index_words = detail::checked_mpi_word_count( key_indices.size(), indexed_record_words );
    std::vector< std::uint64_t > local_index_data;
    local_index_data.reserve( static_cast< std::size_t >( local_index_words ) );
    for( const auto& [indexed_key, index] : key_indices )
    {
        local_index_data.insert( local_index_data.end(), indexed_key.begin(), indexed_key.end() );
        local_index_data.push_back( static_cast< std::uint64_t >( index ) );
    }
    std::vector< int > index_counts( static_cast< std::size_t >( mesh.n_ranks() ) );
    check( MPI_Allgather( &local_index_words, 1, MPI_INT, index_counts.data(), 1, MPI_INT,
                          PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather stable-key index counts" );
    const auto index_layout = detail::checked_mpi_displacements( index_counts, indexed_record_words );
    std::vector< std::uint64_t > all_index_data( index_layout.total_count );
    check( MPI_Allgatherv( local_index_data.data(), local_index_words, MPI_UINT64_T, all_index_data.data(),
                           index_counts.data(), index_layout.displacements.data(), MPI_UINT64_T,
                           PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather stable-key index records" );
    std::map< DofKey, PetscInt > global_key_indices;
    for( std::size_t offset = 0; offset < all_index_data.size(); offset += indexed_record_words )
    {
        DofKey gathered_key;
        std::copy_n( all_index_data.begin() + static_cast< std::ptrdiff_t >( offset ), DofKey::width,
                     gathered_key.begin() );
        const PetscInt gathered_index = static_cast< PetscInt >( all_index_data[offset + DofKey::width] );
        const auto [found, inserted] = global_key_indices.emplace( gathered_key, gathered_index );
        check( inserted || found->second == gathered_index,
               "stable DoF key has one deterministic partitioned index globally" );
    }

    const std::vector< moab::EntityHandle > expected_assembly( mesh.owned_elements().begin(),
                                                               mesh.owned_elements().end() );
    check( dofs.assembly_elements() == expected_assembly,
           "partitioned assembly elements are exactly MOAB-owned elements" );

    std::set< PetscInt > expected_ghosts;
    for( const auto element : mesh.elements() )
    {
        for( unsigned variable = 0; variable < variables.size(); ++variable )
        {
            const auto basis = registry.create( { ReferenceCell::Quadrilateral, variables[variable].family,
                                                  variables[variable].order, variables[variable].continuity } );
            const auto storage = dofs.storage( element, variable, basis );
            const moab::EntityHandle* connectivity = nullptr;
            int connectivity_size = 0;
            check( mesh.moab()->get_connectivity( element, connectivity, connectivity_size ) == moab::MB_SUCCESS,
                   "read partition element connectivity" );
            for( std::size_t i = 0; i < storage.size(); ++i )
            {
                int expected_owner = -1;
                if( variables[variable].continuity == BasisContinuity::cG &&
                    storage[i].association.entity_dimension == 0 )
                    check( mesh.parallel_comm()->get_owner(
                               connectivity[storage[i].association.local_entity_index], expected_owner ) == moab::MB_SUCCESS,
                           "query associated MOAB vertex owner" );
                else if( variables[variable].continuity == BasisContinuity::cG &&
                         storage[i].association.entity_dimension < mesh.dimension() )
                {
                    moab::EntityHandle associated = 0;
                    check( mesh.moab()->side_element( element,
                                                       static_cast< int >( storage[i].association.entity_dimension ),
                                                       static_cast< int >( storage[i].association.local_entity_index ),
                                                       associated ) == moab::MB_SUCCESS,
                           "query canonical MOAB side" );
                    check( associated != 0, "partition fixture exposes associated MOAB edge" );
                    if( associated != 0 )
                        check( mesh.parallel_comm()->get_owner( associated, expected_owner ) == moab::MB_SUCCESS,
                               "query associated MOAB edge owner" );
                }
                else
                    check( mesh.parallel_comm()->get_owner( element, expected_owner ) == moab::MB_SUCCESS,
                           "query containing MOAB element owner" );
                const bool owned = storage[i].index >= dofs.first_owned_dof() &&
                                   storage[i].index < dofs.end_owned_dof();
                check( owned == ( expected_owner == mesh.rank() ),
                       variables[variable].continuity == BasisContinuity::cG
                           ? "cG DoF follows associated MOAB entity ownership: rank=" + std::to_string( mesh.rank() ) +
                                 " dim=" + std::to_string( storage[i].association.entity_dimension ) +
                                 " owner=" + std::to_string( expected_owner ) + " index=" +
                                 std::to_string( storage[i].index ) + " range=[" +
                                 std::to_string( dofs.first_owned_dof() ) + "," +
                                 std::to_string( dofs.end_owned_dof() ) + ")"
                           : "dG/SE DoF follows containing-element ownership" );
                if( !owned ) expected_ghosts.insert( storage[i].index );
            }
        }
    }
    check( dofs.ghost_indices() == std::vector< PetscInt >( expected_ghosts.begin(), expected_ghosts.end() ),
           "ghost list exactly covers non-owned DoFs on visible elements" );
    for( PetscInt local = 0; local < dofs.n_local_dofs(); ++local )
        check( dofs.local_to_global_indices()[static_cast< std::size_t >( local )] ==
                   dofs.first_owned_dof() + local,
               "partitioned owned range is contiguous" );

    for( const auto element : mesh.elements() )
    {
        const moab::EntityHandle* connectivity = nullptr;
        int count = 0;
        check( mesh.moab()->get_connectivity( element, connectivity, count ) == moab::MB_SUCCESS,
               "read partition connectivity for vertex_dof" );
        std::vector< PetscInt > indices;
        dofs.dof_indices( element, indices, 0 );
        for( int vertex = 0; vertex < count; ++vertex )
            check( dofs.vertex_dof( connectivity[vertex], 0, 0 ) == indices[static_cast< std::size_t >( vertex )],
                   "partitioned vertex_dof agrees with element indices" );
    }
}

using DofKeyPair = std::pair< KARMA::DofKey, KARMA::DofKey >;

void write_sparsity_reference( const std::string& mesh_filename, const std::string& reference_filename )
{
    using namespace KARMA;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMAMoabIO( mesh ).read_mesh( mesh_filename );
    KARMABasisRegistry registry;
    KARMADofMap dofs( mesh, { { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } }, registry );
    std::set< DofKeyPair > reference;
    for( const auto element : mesh.elements() )
    {
        const auto& keys = dofs.dof_keys( element );
        for( const auto& row : keys )
            for( const auto& column : keys ) reference.emplace( row, column );
    }
    std::ofstream output( reference_filename );
    check( output.good(), "open serial sparsity reference" );
    output << reference.size() << '\n';
    for( const auto& [row, column] : reference )
    {
        for( const auto word : row.words ) output << word << ' ';
        for( const auto word : column.words ) output << word << ' ';
        output << '\n';
    }
}

void verify_partitioned_sparsity_without_halo( const std::string& filename, const std::string& reference_filename )
{
    using namespace KARMA;
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMAMoabIO( mesh ).read_mesh(
        filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    mesh.update();

    KARMASystem system( mesh, "partitioned-sparsity-without-halo" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    system.init();

    const auto local_elements = std::vector< moab::EntityHandle >( mesh.elements().begin(), mesh.elements().end() );
    std::map< DofKey, PetscInt > local_indices;
    for( const auto element : local_elements )
    {
        std::vector< PetscInt > indices;
        system.dof_map().dof_indices( element, indices );
        const auto& keys = system.dof_map().dof_keys( element );
        check( keys.size() == indices.size(), "partitioned stable keys align with indices for sparsity reference" );
        for( std::size_t i = 0; i < keys.size(); ++i )
        {
            const auto [found, inserted] = local_indices.emplace( keys[i], indices[i] );
            check( inserted || found->second == indices[i], "one partitioned stable key maps to one index" );
        }
    }
    constexpr std::size_t indexed_key_words = DofKey::width + 1;
    const int local_words = detail::checked_mpi_word_count( local_indices.size(), indexed_key_words );
    std::vector< std::uint64_t > local_data;
    local_data.reserve( static_cast< std::size_t >( local_words ) );
    for( const auto& [key, index] : local_indices )
    {
        local_data.insert( local_data.end(), key.begin(), key.end() );
        local_data.push_back( static_cast< std::uint64_t >( index ) );
    }
    std::vector< int > counts( static_cast< std::size_t >( mesh.n_ranks() ) );
    check( MPI_Allgather( &local_words, 1, MPI_INT, counts.data(), 1, MPI_INT, PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather no-halo stable-key index counts" );
    const auto layout = detail::checked_mpi_displacements( counts, indexed_key_words );
    std::vector< std::uint64_t > gathered( layout.total_count );
    check( MPI_Allgatherv( local_data.data(), local_words, MPI_UINT64_T, gathered.data(), counts.data(),
                           layout.displacements.data(), MPI_UINT64_T, PETSC_COMM_WORLD ) == MPI_SUCCESS,
           "gather no-halo stable-key indices" );
    std::map< DofKey, PetscInt > partitioned_indices;
    for( std::size_t offset = 0; offset < gathered.size(); offset += indexed_key_words )
    {
        DofKey key;
        std::copy_n( gathered.begin() + static_cast< std::ptrdiff_t >( offset ), DofKey::width, key.begin() );
        const PetscInt index = static_cast< PetscInt >( gathered[offset + DofKey::width] );
        const auto [found, inserted] = partitioned_indices.emplace( key, index );
        check( inserted || found->second == index, "one gathered stable key maps to one partitioned index" );
    }

    std::ifstream input( reference_filename );
    check( input.good(), "open serial sparsity reference" );
    std::size_t reference_size = 0;
    input >> reference_size;
    std::set< std::pair< PetscInt, PetscInt > > reference;
    for( std::size_t pair = 0; pair < reference_size; ++pair )
    {
        DofKey row, column;
        for( auto& word : row.words ) input >> word;
        for( auto& word : column.words ) input >> word;
        check( input.good(), "read serial stable-key adjacency pair" );
        const auto row_index = partitioned_indices.find( row );
        const auto column_index = partitioned_indices.find( column );
        check( row_index != partitioned_indices.end() && column_index != partitioned_indices.end(),
               "serial stable-key adjacency maps to partitioned indices" );
        if( row_index != partitioned_indices.end() && column_index != partitioned_indices.end() )
            reference.emplace( row_index->second, column_index->second );
    }
    check( reference.size() == reference_size, "serial stable-key adjacency has no duplicate directed pairs" );

    const auto& dofs = system.dof_map();
    std::vector< PetscInt > expected_diagonal( static_cast< std::size_t >( dofs.n_local_dofs() ), 0 );
    std::vector< PetscInt > expected_off_diagonal( static_cast< std::size_t >( dofs.n_local_dofs() ), 0 );
    for( const auto [row, column] : reference )
        if( row >= dofs.first_owned_dof() && row < dofs.end_owned_dof() )
            ++( column >= dofs.first_owned_dof() && column < dofs.end_owned_dof()
                    ? expected_diagonal[static_cast< std::size_t >( row - dofs.first_owned_dof() )]
                    : expected_off_diagonal[static_cast< std::size_t >( row - dofs.first_owned_dof() )] );
    check( system.sparsity().n_nz == expected_diagonal,
           "no-halo partitioned diagonal sparsity matches the serial stable-key graph" );
    check( system.sparsity().o_nz == expected_off_diagonal,
           "no-halo partitioned off-diagonal sparsity matches the serial stable-key graph" );

    if( !local_elements.empty() )
        system.add_element_adjacency( local_elements.front(), local_elements.back() );
    else
        system.add_element_adjacency( 0, 0 );
    check( !throws( [&] { system.add_element_adjacency( 0, 0 ); } ),
           "rank-local adjacency union keeps repeated no-op calls collective" );

    Mat matrix = system.create_matrix();
    for( const auto [row, column] : reference )
        if( row >= dofs.first_owned_dof() && row < dofs.end_owned_dof() )
            check( MatSetValue( matrix, row, column, 1.0, INSERT_VALUES ) == PETSC_SUCCESS,
                   "every serial stable-key adjacency fits PETSc preallocation" );
    check( MatAssemblyBegin( matrix, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
           "begin no-halo serial stable-key matrix assembly" );
    check( MatAssemblyEnd( matrix, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
           "end no-halo serial stable-key matrix assembly" );
    check( MatDestroy( &matrix ) == PETSC_SUCCESS, "destroy no-halo serial stable-key matrix" );
}

void write_face_fixture( const std::string& filename )
{
    using namespace KARMA;
    KARMAMesh mesh( 3, PETSC_COMM_WORLD );
    mesh.generate_box( 2, 1, 1 );
    moab::Range faces;
    check( mesh.moab()->get_adjacencies( mesh.elements(), 2, true, faces, moab::Interface::UNION ) ==
               moab::MB_SUCCESS,
           "materialize 3D fixture faces" );
    mesh.ensure_global_ids();
    KARMAMoabIO( mesh ).write_mesh( filename );
}

void verify_partitioned_face_ownership( const std::string& filename )
{
    using namespace KARMA;
    KARMAMesh mesh( 3, PETSC_COMM_WORLD );
    KARMAMoabIO( mesh ).read_mesh(
        filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    check( mesh.parallel_comm()->exchange_ghost_cells( 3, 0, 1, 1, true ) == moab::MB_SUCCESS,
           "exchange partition fixture ghost edges" );
    check( mesh.parallel_comm()->exchange_ghost_cells( 3, 0, 1, 2, true ) == moab::MB_SUCCESS,
           "exchange partition fixture ghost faces" );
    mesh.update();
    KARMABasisRegistry registry;
    KARMADofMap dofs( mesh, { { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } }, registry );
    const auto basis = registry.create(
        { ReferenceCell::Hexahedron, FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG } );
    bool saw_face = false;
    for( const auto element : mesh.elements() )
        for( const auto& item : dofs.storage( element, 0, basis ) )
            if( item.association.entity_dimension == 2 )
            {
                saw_face = true;
                moab::EntityHandle face = 0;
                check( mesh.moab()->side_element( element, 2, item.association.local_entity_index, face ) ==
                           moab::MB_SUCCESS && face != 0,
                       "query canonical MOAB face" );
                int owner = -1;
                if( face != 0 )
                    check( mesh.parallel_comm()->get_owner( face, owner ) == moab::MB_SUCCESS,
                           "query associated MOAB face owner" );
                const bool owned = item.index >= dofs.first_owned_dof() && item.index < dofs.end_owned_dof();
                check( owned == ( owner == mesh.rank() ), "cG face DoF follows associated MOAB face ownership" );
            }
    check( saw_face, "partitioned 3D fixture exercises face-associated DoFs" );
}

void verify_asymmetric_missing_subentity( const std::string& filename )
{
    using namespace KARMA;
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    KARMAMoabIO( mesh ).read_mesh(
        filename, "PARALLEL=READ_PART;PARTITION=PARALLEL_PARTITION;PARALLEL_RESOLVE_SHARED_ENTS" );
    check( mesh.parallel_comm()->exchange_ghost_cells( 2, 0, 1, 1, true ) == moab::MB_SUCCESS,
           "exchange subentities for asymmetric test" );
    mesh.update();
    if( rank == 1 )
    {
        const moab::EntityHandle element = *mesh.owned_elements().begin();
        moab::EntityHandle edge = 0;
        check( mesh.moab()->side_element( element, 1, 0, edge ) == moab::MB_SUCCESS && edge != 0,
               "find edge for asymmetric removal" );
        check( mesh.moab()->delete_entities( &edge, 1 ) == moab::MB_SUCCESS,
               "remove one rank's required cG subentity" );
    }
    KARMASystem system( mesh, "asymmetric-subentity" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
    check( throws_containing( [&] { system.init(); }, "rank 1" ),
           "asymmetric missing cG subentity fails collectively" );
}

void verify_heterogeneous_partition_cells()
{
    using namespace KARMA;
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    KARMAMesh mesh( 2, PETSC_COMM_WORLD );
    if( rank == 0 )
    {
        const double coordinates[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coordinates, 3, vertices ) == moab::MB_SUCCESS,
               "create heterogeneous triangle vertices" );
        const std::array< moab::EntityHandle, 3 > connectivity{ vertices[0], vertices[1], vertices[2] };
        moab::EntityHandle element = 0;
        check( mesh.moab()->create_element( moab::MBTRI, connectivity.data(), 3, element ) == moab::MB_SUCCESS,
               "create heterogeneous triangle" );
    }
    else
    {
        const double coordinates[] = { 2, 0, 0, 3, 0, 0, 3, 1, 0, 2, 1, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coordinates, 4, vertices ) == moab::MB_SUCCESS,
               "create heterogeneous quadrilateral vertices" );
        const std::array< moab::EntityHandle, 4 > connectivity{ vertices[0], vertices[1], vertices[2], vertices[3] };
        moab::EntityHandle element = 0;
        check( mesh.moab()->create_element( moab::MBQUAD, connectivity.data(), 4, element ) == moab::MB_SUCCESS,
               "create heterogeneous quadrilateral" );
    }
    mesh.assign_ids();
    mesh.update();
    mesh.resolve_shared_entities( 0 );
    KARMASystem system( mesh, "heterogeneous-partition-cells" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    check( !throws( [&] { system.init(); } ), "heterogeneous partition cell types initialize collectively" );
    if( system.is_initialized() ) check( system.n_dofs() == 7, "heterogeneous partition has all global DoFs" );
}

void verify_empty_partition_rank()
{
    using namespace KARMA;
    int rank = 0;
    MPI_Comm_rank( PETSC_COMM_WORLD, &rank );
    KARMAMesh mesh( 1, PETSC_COMM_WORLD );
    if( rank == 0 )
    {
        const double coordinates[] = { 0, 0, 0, 1, 0, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coordinates, 2, vertices ) == moab::MB_SUCCESS,
               "create nonempty-rank vertices" );
        const std::array< moab::EntityHandle, 2 > connectivity{ vertices[0], vertices[1] };
        moab::EntityHandle element = 0;
        check( mesh.moab()->create_element( moab::MBEDGE, connectivity.data(), 2, element ) == moab::MB_SUCCESS,
               "create nonempty-rank element" );
    }
    mesh.assign_ids();
    mesh.update();
    mesh.resolve_shared_entities( 0 );
    KARMASystem system( mesh, "empty-partition-rank" );
    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 1 } );
    check( !throws( [&] { system.init(); } ), "empty partition rank initializes collectively" );
    if( system.is_initialized() )
        check( system.n_dofs() == 2 && ( rank == 0 ? system.n_local_dofs() == 2 : system.n_local_dofs() == 0 ),
               "empty partition rank receives a valid ownership range" );
}

void add_reference_element( KARMA::KARMAMesh& mesh, KARMA::GeometryVariant variant )
{
    const KARMA::KARMAReferenceTopology topology( variant );
    std::vector< double > coordinates;
    for( const auto& point : topology.reference_nodes() )
        for( unsigned d = 0; d < 3; ++d )
            coordinates.push_back( point( d ) );
    moab::Range vertices;
    check( mesh.moab()->create_vertices( coordinates.data(), static_cast< int >( topology.node_count() ), vertices ) ==
               moab::MB_SUCCESS,
           "create reference vertices" );
    std::vector< moab::EntityHandle > connectivity( vertices.begin(), vertices.end() );
    moab::EntityHandle element = 0;
    check( mesh.moab()->create_element( topology.topology(), connectivity.data(),
                                        static_cast< int >( connectivity.size() ), element ) == moab::MB_SUCCESS,
           "create reference element" );
    mesh.assign_ids();
    mesh.update();
}
}  // namespace

int main( int argc, char** argv )
{
    PetscErrorCode ierr = PetscInitialize( &argc, &argv, NULL, NULL );
    if( ierr ) return ierr;

    using namespace KARMA;
    {
        Vec release_vector = nullptr;
        check( VecCreateSeq( PETSC_COMM_SELF, 2, &release_vector ) == PETSC_SUCCESS,
               "create vector for explicit array release" );
        PetscVecArrayRead values( release_vector );
        check( values.get() != nullptr && values.release() == PETSC_SUCCESS && values.get() == nullptr &&
                   values.release() == PETSC_SUCCESS,
               "PetscVecArrayRead explicitly releases once and becomes empty" );
        check( VecSet( release_vector, 3.0 ) == PETSC_SUCCESS,
               "explicit array release unlocks the vector before guard destruction" );
        VecDestroy( &release_vector );
    }
    {
        Vec release_vector = nullptr;
        check( VecCreateSeq( PETSC_COMM_SELF, 2, &release_vector ) == PETSC_SUCCESS,
               "create vector for explicit writable array release" );
        PetscVecArrayWrite values( release_vector );
        values.get()[0] = 5.0;
        check( values.release() == PETSC_SUCCESS && values.get() == nullptr &&
                   values.release() == PETSC_SUCCESS,
               "PetscVecArrayWrite explicitly releases once and becomes empty" );
        check( VecSet( release_vector, 3.0 ) == PETSC_SUCCESS,
               "explicit writable array release unlocks the vector before guard destruction" );
        VecDestroy( &release_vector );
    }
    {
        Vec empty_vector = nullptr;
        check( VecCreateSeq( PETSC_COMM_SELF, 0, &empty_vector ) == PETSC_SUCCESS,
               "create zero-length vector for array guards" );
        {
            PetscVecArrayRead values( empty_vector );
            check( values.release() == PETSC_SUCCESS && values.release() == PETSC_SUCCESS,
                   "zero-length read array explicitly releases once" );
            PetscVecArrayRead reacquired( empty_vector );
            check( reacquired.release() == PETSC_SUCCESS,
                   "zero-length read array release unlocks the vector" );
        }
        {
            PetscVecArrayWrite values( empty_vector );
            check( values.release() == PETSC_SUCCESS && values.release() == PETSC_SUCCESS,
                   "zero-length writable array explicitly releases once" );
            PetscVecArrayWrite reacquired( empty_vector );
            check( reacquired.release() == PETSC_SUCCESS,
                   "zero-length writable array release unlocks the vector" );
        }
        VecDestroy( &empty_vector );
    }
    static_assert( std::is_base_of_v< AbstractSystem, KARMASystem > );
    static_assert( !std::is_copy_constructible_v< KARMASystem > );
    static_assert( std::is_same_v< KARMADofMap::index_type, PetscInt > );
    static_assert( std::is_same_v< decltype( UserSparsity::global_dofs ), PetscInt > );
    static_assert( std::is_same_v< typename decltype( UserSparsity::n_nz )::value_type, PetscInt > );
    check( detail::checked_mpi_count( static_cast< std::size_t >( std::numeric_limits< int >::max() ) ) ==
               std::numeric_limits< int >::max(),
           "checked MPI count accepts INT_MAX" );
    check( throws( [] {
               (void)detail::checked_mpi_count( static_cast< std::size_t >( std::numeric_limits< int >::max() ) + 1 );
           } ),
           "checked MPI count rejects values above INT_MAX" );
    check( detail::checked_mpi_word_count( 0, 32 ) == 0 && detail::checked_mpi_word_count( 4, 32 ) == 128 &&
               detail::checked_mpi_word_count( std::numeric_limits< int >::max() / 32, 32 ) <=
                   std::numeric_limits< int >::max(),
           "checked MPI word arithmetic accepts representable 32-bit boundary products" );
    check( throws( [] {
               (void)detail::checked_mpi_word_count( std::numeric_limits< int >::max() / 32 + 1ULL, 32 );
           } ),
           "checked MPI word arithmetic rejects overflow" );
    const auto mpi_layout = detail::checked_mpi_displacements( std::array{ 2, 0, 3 }, 1 );
    check( mpi_layout.displacements == std::vector< int >( { 0, 2, 2 } ) && mpi_layout.total_count == 5,
           "checked MPI displacements retain empty contributions" );
    const auto maximum_mpi_layout =
        detail::checked_mpi_displacements( std::array{ std::numeric_limits< int >::max() }, 1 );
    check( maximum_mpi_layout.displacements == std::vector< int >{ 0 } &&
               maximum_mpi_layout.total_count == static_cast< std::size_t >( std::numeric_limits< int >::max() ),
           "checked MPI displacements accept an exact INT_MAX total without allocation" );
    check( throws( [] {
               (void)detail::checked_mpi_displacements( std::array{ std::numeric_limits< int >::max(), 1 }, 1 );
           } ),
           "checked MPI displacements reject cumulative overflow" );
    check( throws( [] { (void)detail::checked_mpi_displacements( std::array{ 3 }, 2 ); } ),
           "checked MPI displacements reject malformed serialized records" );
    bool signature_producer_called = false;
    int signature_test_rank = 0, signature_test_ranks = 1;
    MPI_Comm_rank( PETSC_COMM_WORLD, &signature_test_rank );
    MPI_Comm_size( PETSC_COMM_WORLD, &signature_test_ranks );
    const int injected_rank = signature_test_ranks > 1 ? 1 : 0;
    check( throws_containing( [&] {
               detail::require_collective_signature(
                   PETSC_COMM_WORLD, {},
                   [&]() -> std::string {
                       signature_producer_called = true;
                       if( signature_test_rank == injected_rank )
                           throw std::runtime_error( "injected signature failure" );
                       return "signature";
                   },
                   "unknown test signature failure", "test signature" );
           }, "rank " + std::to_string( injected_rank ) + ": unknown test signature failure" ) &&
               signature_producer_called,
           "collective signature catches an asymmetric producer failure before gathering" );
    bool collective_preparation_called = false;
    check( throws_containing( [&] {
               detail::prepare_collectively(
                   PETSC_COMM_WORLD,
                   [&] {
                       collective_preparation_called = true;
                       if( signature_test_rank == injected_rank )
                           throw std::bad_alloc();
                   },
                   "test collective allocation failed" );
           }, "rank " + std::to_string( injected_rank ) + ": test collective allocation failed" ) &&
               collective_preparation_called,
           "collective preparation catches an asymmetric allocation failure before continuing" );
    check( detail::checked_petsc_int_size( static_cast< std::size_t >( PETSC_MAX_INT ) ) == PETSC_MAX_INT,
           "checked PetscInt size accepts PETSC_MAX_INT" );
    if constexpr( static_cast< std::uintmax_t >( PETSC_MAX_INT ) < std::numeric_limits< std::size_t >::max() )
        check( throws( [] {
                   (void)detail::checked_petsc_int_size( static_cast< std::size_t >( PETSC_MAX_INT ) + 1 );
               } ),
               "checked PetscInt size rejects values above PETSC_MAX_INT" );
    {
        UserSparsity serial;
        serial.global_dofs = 2;
        serial.local_dofs  = 2;
        serial.n_nz        = { 1, 1 };
        serial.o_nz        = { 0, 0 };
        check( serial.is_ready(), "serial exact sparsity is ready without a send list" );
        UserSparsity copied = serial;
        copied.n_nz[0] = 7;
        check( serial.n_nz[0] == 1 && copied.n_nz[0] == 7, "UserSparsity copy owns independent arrays" );
        UserSparsity moved = std::move( copied );
        check( moved.n_nz == std::vector< PetscInt >( { 7, 1 } ), "UserSparsity move preserves partial state" );
        serial.o_nz.pop_back();
        check( !serial.is_ready(), "exact sparsity rejects an off-diagonal array with the wrong size" );
        serial.o_nz.push_back( 0 );
        serial.n_nz.pop_back();
        check( !serial.is_ready(), "exact sparsity rejects a diagonal array with the wrong size" );
        UserSparsity empty_owner;
        empty_owner.global_dofs = 1;
        check( empty_owner.is_ready(), "zero-owned rank accepts empty exact arrays" );
    }
    const std::array< PetscInt, 3 > triangle{ 10, 20, 30 };
    const std::array< PetscInt, 4 > quadrilateral{ 10, 20, 30, 40 };
    std::set< unsigned > triangle_codes, quadrilateral_codes;
    for( unsigned code = 0; code < 6; ++code )
    {
        const auto permutation = local_to_canonical_vertices( static_cast< TriangleOrientation >( code ) );
        std::array< PetscInt, 3 > local{};
        for( unsigned i = 0; i < 3; ++i )
            local[i] = triangle[permutation[i]];
        const auto canonical = canonicalize_cyclic_vertex_ids( local );
        triangle_codes.insert( canonical.orientation_code );
        check( canonical.vertex_ids == std::vector< PetscInt >( { 10, 20, 30 } ),
               "triangle symmetries produce one canonical virtual face key" );
    }
    for( unsigned code = 0; code < 8; ++code )
    {
        const auto permutation = local_to_canonical_vertices( static_cast< QuadrilateralOrientation >( code ) );
        std::array< PetscInt, 4 > local{};
        for( unsigned i = 0; i < 4; ++i )
            local[i] = quadrilateral[permutation[i]];
        const auto canonical = canonicalize_cyclic_vertex_ids( local );
        quadrilateral_codes.insert( canonical.orientation_code );
        check( canonical.vertex_ids == std::vector< PetscInt >( { 10, 20, 30, 40 } ),
               "quadrilateral symmetries produce one canonical virtual face key" );
    }
    equal( triangle_codes.size(), 6, "DoF canonicalization extracts all six triangle orientations" );
    equal( quadrilateral_codes.size(), 8, "DoF canonicalization extracts all eight quad orientations" );

    int n_ranks = 1, rank = 0;
    check( MPI_Comm_size( PETSC_COMM_WORLD, &n_ranks ) == MPI_SUCCESS, "query MPI communicator size" );
    check( MPI_Comm_rank( PETSC_COMM_WORLD, &rank ) == MPI_SUCCESS, "query MPI rank" );
    if( argc == 4 && std::string_view( argv[1] ) == "--partitioned-dof-fixture" )
    {
        verify_partitioned_dof_ownership( argv[2], argv[3] );
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        PetscFinalize();
        return global_failures == 0 ? 0 : 1;
    }
    if( argc == 4 && std::string_view( argv[1] ) == "--write-sparsity-reference" )
    {
        write_sparsity_reference( argv[2], argv[3] );
        PetscFinalize();
        return failures == 0 ? 0 : 1;
    }
    if( argc == 4 && std::string_view( argv[1] ) == "--partitioned-sparsity-no-halo" )
    {
        verify_partitioned_sparsity_without_halo( argv[2], argv[3] );
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        PetscFinalize();
        return global_failures == 0 ? 0 : 1;
    }
    if( argc == 4 && std::string_view( argv[1] ) == "--write-dof-reference" )
    {
        write_dof_reference( argv[2], argv[3] );
        PetscFinalize();
        return failures == 0 ? 0 : 1;
    }
    if( argc == 3 && std::string_view( argv[1] ) == "--write-face-fixture" )
    {
        write_face_fixture( argv[2] );
        PetscFinalize();
        return failures == 0 ? 0 : 1;
    }
    if( argc == 3 && std::string_view( argv[1] ) == "--partitioned-face-fixture" )
    {
        verify_partitioned_face_ownership( argv[2] );
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        PetscFinalize();
        return global_failures == 0 ? 0 : 1;
    }
    if( argc == 3 && std::string_view( argv[1] ) == "--asymmetric-missing-subentity" )
    {
        verify_asymmetric_missing_subentity( argv[2] );
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        PetscFinalize();
        return global_failures == 0 ? 0 : 1;
    }
    if( argc == 2 && std::string_view( argv[1] ) == "--heterogeneous-partition-cells" )
    {
        verify_heterogeneous_partition_cells();
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        PetscFinalize();
        return global_failures == 0 ? 0 : 1;
    }
    if( argc == 2 && std::string_view( argv[1] ) == "--empty-partition-rank" )
    {
        verify_empty_partition_rank();
        int global_failures = 0;
        MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD );
        PetscFinalize();
        return global_failures == 0 ? 0 : 1;
    }

    check( detail::collective_validation_succeeded( PETSC_COMM_WORLD, true ),
           "collective validation succeeds when every rank succeeds" );
    if( n_ranks > 1 )
        check( !detail::collective_validation_succeeded( PETSC_COMM_WORLD, rank != 1 ),
               "rank-specific validation failure is agreed collectively" );

    // ---- variable bookkeeping -------------------------------------------
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 2, 2 );

        KARMASystem sys( mesh, "diffusion" );
        check( sys.name() == "diffusion", "system name" );
        check( !sys.is_initialized(), "system starts uninitialized" );

        const unsigned t = sys.add_variable( "temperature" );
        const unsigned c =
            sys.add_variable( { "concentration", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::dG, 3 } );

        equal( t, 0, "first variable index" );
        equal( c, 1, "second variable index" );
        equal( sys.n_variables(), 2, "variable count" );

        check( sys.has_variable( "temperature" ), "has_variable finds a declared variable" );
        check( !sys.has_variable( "pressure" ), "has_variable rejects an absent one" );
        equal( sys.variable_number( "concentration" ), 1, "variable_number" );
        check( sys.variable( 0 ).name == "temperature", "variable(0) name" );
        check( sys.variable( 1 ).order == 2 && sys.variable( 1 ).components == 3,
               "variable stores basis key and component count" );

        // A duplicate name would silently alias DoFs, so it must be rejected.
        check( throws( [&] { sys.add_variable( "temperature" ); } ), "duplicate variable name is rejected" );

        check( throws( [&] { (void)sys.variable_number( "nonexistent" ); } ), "unknown variable name is rejected" );

        check( throws( [&] { (void)sys.variable( 7 ); } ), "out-of-range variable index is rejected" );
    }

    // ---- abstract system seam and system-owned vector state ---------------
    {
        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 0, 0 );
        auto concrete = std::make_unique< KARMASystem >( mesh, "owned" );
        concrete->add_variable( "u" );
        AbstractSystem& system = *concrete;
        check( system.name() == "owned", "KARMASystem is usable through AbstractSystem" );
        system.init();
        check( system.solution() != nullptr && system.current_solution() != nullptr && system.old_solution() != nullptr &&
                   system.local_solution() != nullptr && system.temporary_vector() != nullptr,
                "initialized system owns all solution state vectors" );
        check( system.solution() != system.current_solution() && system.solution() != system.old_solution() &&
                   system.solution() != system.local_solution() && system.solution() != system.temporary_vector(),
                "solution state vectors have independent PETSc ownership" );
        for( Vec vector : { system.solution(), system.current_solution(), system.old_solution(), system.local_solution(),
                            system.temporary_vector() } )
        {
            PetscReal norm = -1;
            check( VecNorm( vector, NORM_INFINITY, &norm ) == PETSC_SUCCESS && norm == 0,
                   "system-owned vectors start zeroed" );
        }
    }

    // ---- unsupported discretizations are refused, not silently downgraded
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 2, 2 );

        KARMASystem sys( mesh, "continuous-p0" );
        sys.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::cG, 1 } );

        check( throws( [&] { sys.init(); } ), "continuous P0 is rejected by the instance registry" );
        check( !sys.is_initialized() && sys.solution() == nullptr && sys.current_solution() == nullptr &&
                   sys.old_solution() == nullptr && sys.local_solution() == nullptr && sys.temporary_vector() == nullptr,
               "failed init commits no state" );
    }

    {
        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 0, 0 );
        KARMASystem sys( mesh, "retry" );
        sys.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::cG, 1 } );
        check( throws( [&] { sys.init(); } ), "invalid setup fails before retry" );
        sys.basis_registry().register_factory(
            { ReferenceCell::Edge, FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::cG },
            [] {
                return KARMABasis( LocalAssociatedBasis{ BasisContinuity::cG } );
            } );
        check( throws( [&] { sys.init(); } ), "retry validates the replacement basis rather than stale state" );
        check( !sys.is_initialized(), "failed retry remains transactional" );
    }

    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 2, 2 );

        KARMASystem sys( mesh, "empty" );
        check( throws( [&] { sys.init(); } ), "init with no variables is refused" );
        check( !sys.is_initialized() && sys.solution() == nullptr && sys.current_solution() == nullptr &&
                   sys.old_solution() == nullptr && sys.local_solution() == nullptr && sys.temporary_vector() == nullptr,
               "invalid init leaves no committed vectors" );
        sys.add_variable( "u" );
        sys.init();
        check( sys.is_initialized() && sys.solution() != nullptr, "valid retry succeeds after invalid init" );
    }

    // ---- single-variable DoF numbering ----------------------------------
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 2, 2 );  // 8 hexes, 27 vertices

        KARMASystem sys( mesh, "scalar" );
        sys.add_variable( "u" );
        sys.init();

        check( sys.is_initialized(), "system is initialized after init()" );

        // One DoF per vertex per variable.
        if( n_ranks == 1 ) equal( sys.n_dofs(), 27, "single-variable DoF count" );

        check( throws( [&] { sys.add_variable( "late" ); } ), "add_variable after init() is refused" );

        const KARMADofMap& dofs = sys.dof_map();
        equal( dofs.n_variables(), 1, "dof_map variable count" );
        RealDenseMatrix constraint_matrix;
        RealDenseVector constraint_vector;
        dofs.constrain( constraint_matrix, constraint_vector, {} );
        dofs.constrain( constraint_vector, {} );
        check( !dofs.has_constraints(), "DoF map constraints are an explicit no-op" );

        // A hex has 8 vertices, so 8 DoFs for one variable.
        std::vector< PetscInt > indices;
        dofs.dof_indices( *mesh.elements().begin(), indices );
        equal( indices.size(), 8, "hex element DoF count, one variable" );

        // The eight must be distinct: a repeat would mean two vertices share
        // a DoF, which would quietly corrupt every assembled matrix.
        const std::set< PetscInt > unique( indices.begin(), indices.end() );
        equal( unique.size(), 8, "element DoFs are distinct" );
    }

    // ---- serial identity uses entity IDs, never coordinates ---------------
    if( n_ranks == 1 )
    {
        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        const double coordinates[] = { 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coordinates, 4, vertices ) == moab::MB_SUCCESS,
               "create coincident disconnected vertices" );
        const std::vector< moab::EntityHandle > handles( vertices.begin(), vertices.end() );
        moab::EntityHandle elements[2]{};
        const moab::EntityHandle first[]  = { handles[0], handles[1] };
        const moab::EntityHandle second[] = { handles[2], handles[3] };
        check( mesh.moab()->create_element( moab::MBEDGE, first, 2, elements[0] ) == moab::MB_SUCCESS &&
                   mesh.moab()->create_element( moab::MBEDGE, second, 2, elements[1] ) == moab::MB_SUCCESS,
               "create coincident disconnected elements" );
        mesh.assign_ids();
        mesh.update();

        KARMASystem continuous( mesh, "coincident-cg" );
        continuous.add_variable( "u" );
        continuous.init();
        equal( continuous.n_dofs(), 4, "coincident serial vertices retain distinct cG DoFs" );

        KARMASystem discontinuous( mesh, "coincident-dg" );
        discontinuous.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
        discontinuous.init();
        equal( discontinuous.n_dofs(), 2, "coincident serial elements retain distinct local DoFs" );

    }

    if( n_ranks == 1 )
    {
        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        const double coordinates[] = { 0, 0, 0, 1, 0, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coordinates, 2, vertices ) == moab::MB_SUCCESS,
               "create overlapping-cell vertices" );
        const std::vector< moab::EntityHandle > handles( vertices.begin(), vertices.end() );
        moab::EntityHandle elements[2]{};
        check( mesh.moab()->create_element( moab::MBEDGE, handles.data(), 2, elements[0] ) == moab::MB_SUCCESS &&
                   mesh.moab()->create_element( moab::MBEDGE, handles.data(), 2, elements[1] ) == moab::MB_SUCCESS,
               "create duplicate overlapping serial cells" );
        mesh.assign_ids();
        mesh.update();
        KARMASystem system( mesh, "overlapping-dg" );
        system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
        system.init();
        equal( system.n_dofs(), 2, "duplicate overlapping serial cells retain distinct local DoFs" );
    }

    if( n_ranks == 1 )
    {
        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 0, 0 );
        const moab::EntityHandle vertex = *mesh.vertices().begin();
        const GlobalId invalid          = -1;
        check( mesh.moab()->tag_set_data( mesh.global_id_tag(), &vertex, 1, &invalid ) == moab::MB_SUCCESS,
               "invalidate private vertex global ID" );
        KARMASystem system( mesh, "invalid-private-id" );
        system.add_variable( "u" );
        check( throws_containing( [&] { system.init(); }, "valid KARMA global IDs" ),
               "serial DoF identity rejects an invalid private global ID" );
    }

    // ---- nonfinite coordinates are rejected ------------------------------
    for( const double nonfinite :
         { std::numeric_limits< double >::infinity(), std::numeric_limits< double >::quiet_NaN() } )
    {
        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        const double coordinates[] = { 0, 0, 0, nonfinite, 0, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coordinates, 2, vertices ) == moab::MB_SUCCESS,
               "create nonfinite test vertices" );
        const std::vector< moab::EntityHandle > handles( vertices.begin(), vertices.end() );
        moab::EntityHandle element = 0;
        check( mesh.moab()->create_element( moab::MBEDGE, handles.data(), 2, element ) == moab::MB_SUCCESS,
               "create nonfinite test element" );
        mesh.assign_ids();
        mesh.update();
        KARMASystem system( mesh, "nonfinite" );
        system.add_variable( "u" );
        check( throws( [&] { system.init(); } ), "NaN and infinity coordinates are rejected collectively" );
    }

    // ---- mixed variable-major, component-major, basis-minor layout -------
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 2, 2 );

        KARMASystem sys( mesh, "coupled" );
        sys.add_variable( { "cell", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
        sys.add_variable( { "linear", FieldBasisFamily::LagrangeNodal, 1, BasisContinuity::cG, 2 } );
        sys.add_variable( { "quadratic", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
        sys.init();

        if( n_ranks == 1 ) equal( sys.n_dofs(), 8 + 2 * 27 + 125, "mixed P0/P1/P2 component DoF count" );

        const KARMADofMap& dofs       = sys.dof_map();
        const moab::EntityHandle elem = *mesh.elements().begin();

        std::vector< PetscInt > all;
        dofs.dof_indices( elem, all );
        equal( all.size(), 1 + 2 * 8 + 27, "mixed hex element DoF count" );

        const std::set< PetscInt > unique( all.begin(), all.end() );
        equal( unique.size(), all.size(), "mixed element DoFs are distinct" );

        // Variables are concatenated first; within a variable, complete basis
        // lists are emitted component by component.
        for( std::size_t i = 0; i < 8; ++i )
            check( all[1 + 8 + i] != all[1 + i], "components have distinct basis DoFs" );

        for( unsigned f = 0; f < 3; ++f )
        {
            std::vector< PetscInt > per_var;
            dofs.dof_indices( elem, per_var, f );
            const std::array< std::size_t, 3 > sizes{ 1, 16, 27 };
            const std::array< std::size_t, 3 > offsets{ 0, 1, 17 };
            equal( per_var.size(), sizes[f], "per-variable DoF count" );
            check( std::equal( per_var.begin(), per_var.end(), all.begin() + offsets[f] ),
                   "per-variable DoFs agree with variable-major combined list" );
        }

        check( throws( [&] {
                   std::vector< PetscInt > v;
                   dofs.dof_indices( elem, v, 9 );
               } ),
               "out-of-range variable index is rejected" );
    }

    // ---- all topology counts; field topology is independent of geometry order
    {
        struct Case
        {
            GeometryVariant linear;
            GeometryVariant quadratic;
            unsigned dimension;
            PetscInt p1;
            PetscInt p2;
        };
        const std::array cases{ Case{ GeometryVariant::Edge2, GeometryVariant::Edge3, 1, 2, 3 },
                                Case{ GeometryVariant::Tri3, GeometryVariant::Tri6, 2, 3, 6 },
                                Case{ GeometryVariant::Quad4, GeometryVariant::Quad9, 2, 4, 9 },
                                Case{ GeometryVariant::Tet4, GeometryVariant::Tet10, 3, 4, 10 },
                                Case{ GeometryVariant::Hex8, GeometryVariant::Hex27, 3, 8, 27 },
                                Case{ GeometryVariant::Prism6, GeometryVariant::Prism18, 3, 6, 18 },
                                Case{ GeometryVariant::Pyramid5, GeometryVariant::Pyramid14, 3, 5, 14 } };
        for( const auto& item : cases )
            for( const auto geometry : { item.linear, item.quadratic } )
            {
                KARMAMesh mesh( item.dimension, PETSC_COMM_WORLD );
                add_reference_element( mesh, geometry );
                for( const auto [order, expected] : { std::pair{ 1u, item.p1 }, std::pair{ 2u, item.p2 } } )
                {
                    KARMASystem system( mesh, "topology" );
                    system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, order, BasisContinuity::cG, 1 } );
                    system.init();
                    equal( system.n_dofs(), expected, "single-element cG P1/P2 topology count" );
                    std::vector< PetscInt > indices;
                    system.dof_map().dof_indices( *mesh.elements().begin(), indices );
                    equal( indices.size(), expected, "single-element basis ordering size" );
                }
                for( const auto continuity : { BasisContinuity::dG, BasisContinuity::SE } )
                    for( unsigned order = 0; order <= 2; ++order )
                    {
                        KARMASystem system( mesh, "local-topology" );
                        system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, order, continuity, 1 } );
                        system.init();
                        std::vector< PetscInt > indices;
                        system.dof_map().dof_indices( *mesh.elements().begin(), indices );
                        const PetscInt expected = order == 0 ? 1 : ( order == 1 ? item.p1 : item.p2 );
                        equal( indices.size(), expected, "single-element dG/SE P0/P1/P2 topology count" );
                    }
            }
    }

    // ---- shared and element-local topology -------------------------------
    {
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 1, 0 );
        const auto elements = std::vector< moab::EntityHandle >( mesh.elements().begin(), mesh.elements().end() );

        KARMASystem cg( mesh, "p2" );
        cg.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
        cg.init();
        if( n_ranks == 1 ) equal( cg.n_dofs(), 15, "two-quad cG P2 count reuses shared edge" );
        std::vector< PetscInt > left, right;
        cg.dof_map().dof_indices( elements[0], left );
        cg.dof_map().dof_indices( elements[1], right );
        std::set< PetscInt > left_set( left.begin(), left.end() ), right_set( right.begin(), right.end() );
        std::vector< PetscInt > intersection;
        std::set_intersection( left_set.begin(), left_set.end(), right_set.begin(), right_set.end(),
                               std::back_inserter( intersection ) );
        equal( intersection.size(), 3, "P2 neighbors reuse two vertices and one virtual edge" );
        const auto left_orientation  = cg.dof_map().orientations( elements[0], 0 );
        const auto right_orientation = cg.dof_map().orientations( elements[1], 0 );
        const auto left_edge = std::find_if( left_orientation.begin(), left_orientation.end(), []( const auto& item ) {
            return item.association == DofAssociation{ 1, 1, 0 };
        } );
        const auto right_edge =
            std::find_if( right_orientation.begin(), right_orientation.end(),
                          []( const auto& item ) { return item.association == DofAssociation{ 1, 3, 0 }; } );
        check( left_edge != left_orientation.end() && right_edge != right_orientation.end() &&
                   left_edge->orientation_code != right_edge->orientation_code &&
                   left_edge->local_to_canonical_vertices == std::vector< unsigned >( { 0, 1 } ) &&
                   right_edge->local_to_canonical_vertices == std::vector< unsigned >( { 1, 0 } ),
               "oppositely traversed shared edge records local-to-canonical orientation" );
        KARMABasisRegistry custom_registry;
        const BasisKey custom_key{ ReferenceCell::Quadrilateral, FieldBasisFamily::UserDefined, 3,
                                   BasisContinuity::cG };
        custom_registry.register_factory( custom_key, [] { return KARMABasis( ThreeDofsPerEdgeBasis{} ); } );
        KARMASystem oriented( mesh, "oriented-slots", std::move( custom_registry ) );
        oriented.add_variable( { "u", FieldBasisFamily::UserDefined, 3, BasisContinuity::cG, 1 } );
        oriented.init();
        std::vector< PetscInt > oriented_left, oriented_right;
        oriented.dof_map().dof_indices( elements[0], oriented_left );
        oriented.dof_map().dof_indices( elements[1], oriented_right );
        const auto custom_left_orientation  = oriented.dof_map().orientations( elements[0], 0 );
        const auto custom_right_orientation = oriented.dof_map().orientations( elements[1], 0 );
        const auto custom_left_edge =
            std::find_if( custom_left_orientation.begin(), custom_left_orientation.end(),
                          []( const auto& item ) { return item.association.local_entity_index == 1; } );
        const auto custom_right_edge =
            std::find_if( custom_right_orientation.begin(), custom_right_orientation.end(),
                          []( const auto& item ) { return item.association.local_entity_index == 3; } );
        check( custom_left_edge != custom_left_orientation.end() && custom_right_edge != custom_right_orientation.end(),
               "custom shared edge orientation metadata exists" );
        const auto left_transform =
            ThreeDofsPerEdgeBasis{}.orientation_transforms()[2 + custom_left_edge->orientation_code];
        const auto right_transform =
            ThreeDofsPerEdgeBasis{}.orientation_transforms()[6 + custom_right_edge->orientation_code];
        for( unsigned slot = 0; slot < 3; ++slot )
        {
            const auto right_slot = static_cast< unsigned >( std::distance(
                right_transform.local_to_canonical.begin(),
                std::find( right_transform.local_to_canonical.begin(), right_transform.local_to_canonical.end(),
                           left_transform.local_to_canonical[slot] ) ) );
            check( oriented_left[3 + slot] == oriented_right[9 + right_slot],
                   "physical edge slots share after orientation" );
        }
        check( oriented_left[3] != oriented_right[9], "wrong local edge slots do not alias" );

        int edge_count = -1;
        check( mesh.moab()->get_number_entities_by_dimension( mesh.root_set(), 1, edge_count ) == moab::MB_SUCCESS &&
                   edge_count == 0,
               "DoF construction creates no MOAB edges" );

        for( const auto continuity : { BasisContinuity::dG, BasisContinuity::SE } )
            for( unsigned order = 0; order <= 2; ++order )
            {
                KARMASystem local( mesh, "element-local" );
                local.add_variable( { "u", FieldBasisFamily::LagrangeNodal, order, continuity, 1 } );
                local.init();
                std::vector< PetscInt > local_left, local_right;
                local.dof_map().dof_indices( elements[0], local_left );
                local.dof_map().dof_indices( elements[1], local_right );
                std::set< PetscInt > local_left_set( local_left.begin(), local_left.end() );
                std::set< PetscInt > local_right_set( local_right.begin(), local_right.end() );
                intersection.clear();
                std::set_intersection( local_left_set.begin(), local_left_set.end(), local_right_set.begin(),
                                       local_right_set.end(), std::back_inserter( intersection ) );
                check( intersection.empty(), "dG/SE P0/P1/P2 DoFs remain element-local" );
            }
    }

    // Non-cG locality is absolute even when a custom basis advertises shared subentities.
    if( n_ranks == 1 )
    {
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        const double coordinates[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0 };
        moab::Range vertices;
        check( mesh.moab()->create_vertices( coordinates, 4, vertices ) == moab::MB_SUCCESS,
               "create custom-locality vertices" );
        const std::vector< moab::EntityHandle > v( vertices.begin(), vertices.end() );
        const moab::EntityHandle first_nodes[]  = { v[0], v[1], v[2] };
        const moab::EntityHandle second_nodes[] = { v[1], v[0], v[3] };
        moab::EntityHandle elements[2]{};
        check( mesh.moab()->create_element( moab::MBTRI, first_nodes, 3, elements[0] ) == moab::MB_SUCCESS &&
                   mesh.moab()->create_element( moab::MBTRI, second_nodes, 3, elements[1] ) == moab::MB_SUCCESS,
               "create adjacent custom-locality triangles" );
        mesh.assign_ids();
        mesh.update();
        for( const auto continuity : { BasisContinuity::dG, BasisContinuity::SE } )
        {
            KARMABasisRegistry registry;
            const BasisKey key{ ReferenceCell::Triangle, FieldBasisFamily::UserDefined, 4, continuity };
            registry.register_factory( key, [continuity] { return KARMABasis( LocalAssociatedBasis{ continuity } ); } );
            KARMASystem system( mesh, "custom-locality", std::move( registry ) );
            system.add_variable( { "u", FieldBasisFamily::UserDefined, 4, continuity, 1 } );
            system.init();
            std::vector< PetscInt > first, second, shared;
            system.dof_map().dof_indices( elements[0], first );
            system.dof_map().dof_indices( elements[1], second );
            std::sort( first.begin(), first.end() );
            std::sort( second.begin(), second.end() );
            std::set_intersection( first.begin(), first.end(), second.begin(), second.end(),
                                   std::back_inserter( shared ) );
            check( shared.empty(), "custom dG/SE vertex and edge associations never share across elements" );
        }
    }

    // ---- P2 quad-face reuse in 3-D ---------------------------------------
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 1, 1 );
        const auto elements = std::vector< moab::EntityHandle >( mesh.elements().begin(), mesh.elements().end() );
        KARMASystem system( mesh, "face-sharing" );
        system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
        system.init();
        if( n_ranks == 1 ) equal( system.n_dofs(), 45, "two-hex cG P2 count reuses shared quad face" );
        std::vector< PetscInt > first, second;
        system.dof_map().dof_indices( elements[0], first );
        system.dof_map().dof_indices( elements[1], second );
        std::set< PetscInt > first_set( first.begin(), first.end() ), second_set( second.begin(), second.end() );
        std::vector< PetscInt > shared;
        std::set_intersection( first_set.begin(), first_set.end(), second_set.begin(), second_set.end(),
                               std::back_inserter( shared ) );
        equal( shared.size(), 9, "P2 hex neighbors reuse four vertices, four edges, and one quad face" );
        const auto first_orientations  = system.dof_map().orientations( elements[0], 0 );
        const auto second_orientations = system.dof_map().orientations( elements[1], 0 );
        const auto first_face =
            std::find_if( first_orientations.begin(), first_orientations.end(),
                          []( const auto& item ) { return item.association == DofAssociation{ 2, 3, 0 }; } );
        const auto second_face =
            std::find_if( second_orientations.begin(), second_orientations.end(),
                          []( const auto& item ) { return item.association == DofAssociation{ 2, 5, 0 }; } );
        check( first_face != first_orientations.end() && second_face != second_orientations.end() &&
                   first_face->local_to_canonical_vertices.size() == 4 &&
                   second_face->local_to_canonical_vertices.size() == 4,
               "shared quad face carries canonical orientation metadata" );
    }

    // Custom multi-slot face bases share both canonical slots at map level.
    if( n_ranks == 1 )
    {
        for( const auto [cell, variant, face_count, orientation_count] : {
                 std::tuple{ ReferenceCell::Tetrahedron, GeometryVariant::Tet4, 4u, 6u },
                 std::tuple{ ReferenceCell::Hexahedron, GeometryVariant::Hex8, 6u, 8u } } )
        {
            KARMAMesh mesh( 3, PETSC_COMM_WORLD );
            add_reference_element( mesh, variant );
            const auto original = *mesh.elements().begin();
            const moab::EntityHandle* connectivity = nullptr;
            int count = 0;
            check( mesh.moab()->get_connectivity( original, connectivity, count ) == moab::MB_SUCCESS,
                   "read face-orientation connectivity" );
            std::vector< moab::EntityHandle > reversed( connectivity, connectivity + count );
            std::reverse( reversed.begin(), reversed.end() );
            moab::EntityHandle duplicate = 0;
            check( mesh.moab()->create_element( mesh.moab()->type_from_handle( original ), reversed.data(), count,
                                                duplicate ) == moab::MB_SUCCESS,
                   "create oppositely oriented face-sharing element" );
            mesh.assign_ids();
            mesh.update();
            KARMABasisRegistry registry;
            const BasisKey key{ cell, FieldBasisFamily::UserDefined, 6, BasisContinuity::cG };
            registry.register_factory( key, [=] {
                return KARMABasis( MultiSlotFaceBasis{ cell, face_count, orientation_count } );
            } );
            KARMASystem system( mesh, "multi-slot-face", std::move( registry ) );
            system.add_variable( { "u", FieldBasisFamily::UserDefined, 6, BasisContinuity::cG, 1 } );
            system.init();
            std::vector< PetscInt > first, second, shared;
            system.dof_map().dof_indices( original, first );
            system.dof_map().dof_indices( duplicate, second );
            std::sort( first.begin(), first.end() );
            std::sort( second.begin(), second.end() );
            std::set_intersection( first.begin(), first.end(), second.begin(), second.end(),
                                   std::back_inserter( shared ) );
            check( shared.size() >= 2, "custom triangle/quad face slots share canonically at map level" );
        }
    }

    // ---- PETSc objects carry the system's layout ------------------------
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 2, 2 );

        KARMASystem sys( mesh, "vectors" );
        sys.add_variable( "u" );
        sys.add_variable( "v" );
        sys.init();

        Vec x      = sys.create_vector();
        PetscInt n = 0;
        check( VecGetSize( x, &n ) == PETSC_SUCCESS, "query vector length" );
        equal( static_cast< std::size_t >( n ), sys.n_dofs(), "vector length matches the DoF count" );

        // Usable, not merely allocated.
        check( VecSet( x, 2.0 ) == PETSC_SUCCESS, "set vector values" );
        PetscReal norm = 0.0;
        check( VecNorm( x, NORM_1, &norm ) == PETSC_SUCCESS, "compute vector norm" );
        check( std::abs( norm - 2.0 * static_cast< Real >( n ) ) < 1.e-10, "vector from create_vector() is usable" );
        check( VecDestroy( &x ) == PETSC_SUCCESS, "destroy system vector" );
        check( sys.dof_map().local_to_global_mapping() != nullptr, "DoF map exposes a PETSc local-to-global mapping" );

        Mat A         = sys.create_matrix();
        PetscInt rows = 0, cols = 0;
        check( MatGetSize( A, &rows, &cols ) == PETSC_SUCCESS, "query matrix dimensions" );
        equal( static_cast< std::size_t >( rows ), sys.n_dofs(), "matrix row count" );
        equal( static_cast< std::size_t >( cols ), sys.n_dofs(), "matrix column count" );
        std::vector< PetscInt > element_indices;
        sys.dof_map().dof_indices( *mesh.elements().begin(), element_indices );
        std::vector< PetscScalar > values( element_indices.size() * element_indices.size(), 1.0 );
        check( MatSetValues( A, detail::checked_petsc_int_size( element_indices.size() ), element_indices.data(),
                             detail::checked_petsc_int_size( element_indices.size() ), element_indices.data(),
                             values.data(), ADD_VALUES ) == PETSC_SUCCESS,
               "insert element matrix values" );
        check( MatAssemblyBegin( A, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS, "begin element matrix assembly" );
        check( MatAssemblyEnd( A, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS, "end element matrix assembly" );
        check( MatDestroy( &A ) == PETSC_SUCCESS, "destroy system matrix" );

        // Requesting either before init() must fail rather than hand back a
        // misconfigured object.
        KARMAMesh m2( 3, PETSC_COMM_WORLD );
        m2.generate_box( 2, 2, 2 );
        KARMASystem uninit( m2, "uninitialized" );
        uninit.add_variable( "u" );
        check( throws( [&] { (void)uninit.create_vector(); } ), "create_vector before init() is refused" );
    }

    // ---- sparsity descriptor --------------------------------------------
    {
        KARMAMesh mesh( 3, PETSC_COMM_WORLD );
        mesh.generate_box( 3, 3, 3 );

        KARMASystem sys( mesh, "sparsity" );
        sys.add_variable( "u" );
        sys.init();

        const UserSparsity& sp = sys.sparsity();
        equal( sp.global_dofs, sys.n_dofs(), "sparsity global dof count" );
        equal( sp.local_dofs, sys.n_local_dofs(), "sparsity local dof count" );

        check( sp.n_nz.size() == static_cast< std::size_t >( sys.n_local_dofs() ),
               "exact diagonal sparsity is stored per owned row" );
        check( sp.is_ready(), "exact serial/MPI sparsity is ready" );
    }

    // ---- explicit adjacency refreshes exact matrix preallocation ----------
    {
        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 0, 0 );
        const auto elements = std::vector< moab::EntityHandle >( mesh.elements().begin(), mesh.elements().end() );
        KARMASystem system( mesh, "extra-adjacency" );
        system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
        system.init();
        const PetscInt local_before = std::accumulate( system.sparsity().n_nz.begin(), system.sparsity().n_nz.end(),
                                                       PetscInt{} ) +
                                      std::accumulate( system.sparsity().o_nz.begin(), system.sparsity().o_nz.end(),
                                                       PetscInt{} );
        PetscInt global_before = 0;
        check( MPI_Allreduce( &local_before, &global_before, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "sum initial dG sparsity" );
        system.add_element_adjacency( elements[0], elements[1] );
        const auto diagonal_once = system.sparsity().n_nz;
        const auto off_diagonal_once = system.sparsity().o_nz;
        const PetscInt local_after = std::accumulate( diagonal_once.begin(), diagonal_once.end(), PetscInt{} ) +
                                     std::accumulate( off_diagonal_once.begin(), off_diagonal_once.end(), PetscInt{} );
        PetscInt global_after = 0;
        check( MPI_Allreduce( &local_after, &global_after, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "sum refreshed dG sparsity" );
        system.add_element_adjacency( elements[0], elements[1] );
        check( diagonal_once == system.sparsity().n_nz && off_diagonal_once == system.sparsity().o_nz,
               "extra adjacency refresh is idempotent" );
        check( global_after == global_before + 2, "cross-element dG adjacency adds exactly two global nonzeros" );
        const PetscInt expected_diagonal_max =
            diagonal_once.empty() ? 0 : *std::max_element( diagonal_once.begin(), diagonal_once.end() );
        const PetscInt expected_off_diagonal_max =
            off_diagonal_once.empty() ? 0 : *std::max_element( off_diagonal_once.begin(), off_diagonal_once.end() );
        check( system.sparsity().nnz_per_row == expected_diagonal_max &&
                   system.sparsity().onz_per_row == expected_off_diagonal_max,
               "adjacency refresh updates exact sparsity maxima" );
        std::vector< PetscInt > first, second;
        system.dof_map().dof_indices( elements[0], first );
        system.dof_map().dof_indices( elements[1], second );
        Mat matrix              = system.create_matrix();
        const PetscScalar value = 2.0;
        if( system.dof_map().owns_element_for_assembly( elements[0] ) )
            check( MatSetValue( matrix, first[0], second[0], value, INSERT_VALUES ) == PETSC_SUCCESS,
                   "matrix accepts declared forward off-element coupling" );
        if( system.dof_map().owns_element_for_assembly( elements[1] ) )
            check( MatSetValue( matrix, second[0], first[0], value, INSERT_VALUES ) == PETSC_SUCCESS,
                   "matrix accepts declared reverse off-element coupling" );
        check( MatAssemblyBegin( matrix, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS, "begin adjacency matrix assembly" );
        check( MatAssemblyEnd( matrix, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS, "end adjacency matrix assembly" );
        check( MatDestroy( &matrix ) == PETSC_SUCCESS, "destroy adjacency matrix" );
    }

    // ---- dimensional stencils --------------------------------------------
    {
        KARMAMesh mesh( 2, PETSC_COMM_WORLD );
        mesh.generate_box( 3, 3, 0 );

        KARMASystem sys( mesh, "two-d" );
        sys.add_variable( "u" );
        sys.init();

        // A quad has 4 vertices.
        std::vector< PetscInt > indices;
        sys.dof_map().dof_indices( *mesh.elements().begin(), indices );
        equal( indices.size(), 4, "quad element DoF count" );
    }

    // Replicated ranks have private handles and private KARMA ids. Geometry-
    // derived virtual keys nevertheless describe one global algebraic mesh.
    {
        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        mesh.generate_box( 2, 0, 0 );
        KARMASystem sys( mesh, "replicated" );
        sys.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 2, BasisContinuity::cG, 1 } );
        sys.init();
        equal( sys.n_dofs(), 5, "replicated MPI mesh has one global P2 edge layout" );
        const auto assembly           = sys.dof_map().assembly_elements();
        PetscInt local_element_count  = static_cast< PetscInt >( assembly.size() );
        PetscInt global_element_count = 0;
        check( MPI_Allreduce( &local_element_count, &global_element_count, 1, MPIU_INT, MPI_SUM,
                              PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "sum assembly element ownership" );
        equal( global_element_count, 2, "replicated elements have one assembly owner" );
        for( const auto element : mesh.elements() )
            check( sys.dof_map().owns_element_for_assembly( element ) ==
                       ( std::find( assembly.begin(), assembly.end(), element ) != assembly.end() ),
                   "assembly element list agrees with ownership query" );
        PetscInt minimum = sys.n_dofs(), maximum = sys.n_dofs();
        check( MPI_Allreduce( MPI_IN_PLACE, &minimum, 1, MPIU_INT, MPI_MIN, PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "reduce minimum global DoF count" );
        check( MPI_Allreduce( MPI_IN_PLACE, &maximum, 1, MPIU_INT, MPI_MAX, PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "reduce maximum global DoF count" );
        check( minimum == maximum, "replicated ranks agree on global DoF count" );
        PetscInt owned_sum           = 0;
        const PetscInt locally_owned = sys.n_local_dofs();
        check( MPI_Allreduce( &locally_owned, &owned_sum, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "sum owned DoF counts" );
        equal( owned_sum, sys.n_dofs(), "owned DoF ranges partition the global range" );
        const auto& local_to_global = sys.dof_map().local_to_global_indices();
        for( PetscInt i = 0; i < sys.n_local_dofs(); ++i )
            check( local_to_global[static_cast< std::size_t >( i )] == sys.dof_map().first_owned_dof() + i,
                   "owned local mapping is contiguous" );
        for( const PetscInt ghost : sys.dof_map().ghost_indices() )
            check( ghost < sys.dof_map().first_owned_dof() || ghost >= sys.dof_map().end_owned_dof(),
                   "ghost index lies outside owned range" );
        for( const auto element : mesh.elements() )
        {
            std::vector< PetscInt > global_indices, local_indices;
            sys.dof_map().dof_indices( element, global_indices );
            sys.dof_map().local_dof_indices( element, local_indices );
            check( global_indices.size() == local_indices.size(), "local and global element index counts agree" );
            for( std::size_t i = 0; i < local_indices.size(); ++i )
                check( local_indices[i] >= 0 &&
                           static_cast< std::size_t >( local_indices[i] ) < local_to_global.size() &&
                           local_to_global[static_cast< std::size_t >( local_indices[i] )] == global_indices[i],
                       "element local index maps back to its global index" );
        }
        std::vector< PetscInt > range_starts( static_cast< std::size_t >( n_ranks ) );
        std::vector< PetscInt > range_ends( static_cast< std::size_t >( n_ranks ) );
        const PetscInt local_start = sys.dof_map().first_owned_dof();
        const PetscInt local_end   = sys.dof_map().end_owned_dof();
        check( MPI_Allgather( &local_start, 1, MPIU_INT, range_starts.data(), 1, MPIU_INT,
                              PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "gather owned range starts" );
        check( MPI_Allgather( &local_end, 1, MPIU_INT, range_ends.data(), 1, MPIU_INT,
                              PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "gather owned range ends" );
        for( int owner = 1; owner < n_ranks; ++owner )
            check( range_ends[static_cast< std::size_t >( owner - 1 )] ==
                       range_starts[static_cast< std::size_t >( owner )],
                   "owned ranges are contiguous and disjoint" );

        std::vector< PetscScalar > expected_vector( static_cast< std::size_t >( sys.n_dofs() ), 0.0 );
        std::vector< PetscScalar > expected_matrix(
            static_cast< std::size_t >( sys.n_dofs() * sys.n_dofs() ), 0.0 );
        for( const auto element : mesh.elements() )
        {
            std::vector< PetscInt > indices;
            sys.dof_map().dof_indices( element, indices );
            for( std::size_t i = 0; i < indices.size(); ++i )
            {
                expected_vector[static_cast< std::size_t >( indices[i] )] += static_cast< PetscScalar >( i + 1 );
                for( std::size_t j = 0; j < indices.size(); ++j )
                    expected_matrix[static_cast< std::size_t >( indices[i] * sys.n_dofs() + indices[j] )] +=
                        static_cast< PetscScalar >( 10 * ( i + 1 ) + j + 1 );
            }
        }

        Vec vector = sys.create_vector();
        check( VecSet( vector, 0.0 ) == PETSC_SUCCESS, "zero replicated assembly vector" );
        for( const auto element : assembly )
        {
            std::vector< PetscInt > indices;
            sys.dof_map().dof_indices( element, indices );
            std::vector< PetscScalar > values( indices.size() );
            for( std::size_t i = 0; i < values.size(); ++i ) values[i] = static_cast< PetscScalar >( i + 1 );
            check( VecSetValues( vector, static_cast< PetscInt >( indices.size() ), indices.data(), values.data(),
                                 ADD_VALUES ) == PETSC_SUCCESS,
                   "insert owner-only replicated vector contribution" );
        }
        check( VecAssemblyBegin( vector ) == PETSC_SUCCESS, "begin replicated vector assembly" );
        check( VecAssemblyEnd( vector ) == PETSC_SUCCESS, "end replicated vector assembly" );
        for( PetscInt global = sys.dof_map().first_owned_dof(); global < sys.dof_map().end_owned_dof(); ++global )
        {
            PetscScalar actual = 0.0;
            check( VecGetValues( vector, 1, &global, &actual ) == PETSC_SUCCESS,
                   "read owned replicated vector entry" );
            check( PetscAbsScalar( actual - expected_vector[static_cast< std::size_t >( global )] ) < 1.e-12,
                   "owner-only replicated vector assembly has exact serial entry" );
        }
        PetscReal vector_norm = 0.0;
        check( VecNorm( vector, NORM_1, &vector_norm ) == PETSC_SUCCESS, "compute replicated vector norm" );
        const PetscReal expected_vector_norm = std::accumulate(
            expected_vector.begin(), expected_vector.end(), PetscReal{},
            []( PetscReal sum, PetscScalar value ) { return sum + PetscAbsScalar( value ); } );
        check( std::abs( vector_norm - expected_vector_norm ) < 1.e-12,
               "owner-only replicated vector norm matches serial" );

        for( PetscInt global = sys.dof_map().first_owned_dof(); global < sys.dof_map().end_owned_dof(); ++global )
        {
            const PetscScalar owner_value = static_cast< PetscScalar >( global + 100 );
            check( VecSetValue( vector, global, owner_value, INSERT_VALUES ) == PETSC_SUCCESS,
                   "set owned value for ghost update" );
        }
        check( VecAssemblyBegin( vector ) == PETSC_SUCCESS, "begin owner-value assembly" );
        check( VecAssemblyEnd( vector ) == PETSC_SUCCESS, "end owner-value assembly" );
        check( VecGhostUpdateBegin( vector, INSERT_VALUES, SCATTER_FORWARD ) == PETSC_SUCCESS,
               "begin forward ghost update" );
        check( VecGhostUpdateEnd( vector, INSERT_VALUES, SCATTER_FORWARD ) == PETSC_SUCCESS,
               "end forward ghost update" );
        Vec local_form = nullptr;
        check( VecGhostGetLocalForm( vector, &local_form ) == PETSC_SUCCESS, "get ghosted local vector form" );
        const PetscScalar* local_values = nullptr;
        check( VecGetArrayRead( local_form, &local_values ) == PETSC_SUCCESS, "read ghosted local vector form" );
        for( std::size_t local = 0; local < local_to_global.size(); ++local )
            check( PetscRealPart( local_values[local] ) ==
                       static_cast< PetscReal >( local_to_global[local] + 100 ),
                   "VecGhost update transfers exact owner value" );
        check( VecRestoreArrayRead( local_form, &local_values ) == PETSC_SUCCESS,
               "restore ghosted local vector array" );
        check( VecGhostRestoreLocalForm( vector, &local_form ) == PETSC_SUCCESS,
               "restore ghosted local vector form" );
        check( VecDestroy( &vector ) == PETSC_SUCCESS, "destroy replicated vector" );

        Mat matrix = sys.create_matrix();
        for( const auto element : assembly )
        {
            std::vector< PetscInt > indices;
            sys.dof_map().dof_indices( element, indices );
            std::vector< PetscScalar > values( indices.size() * indices.size() );
            for( std::size_t i = 0; i < indices.size(); ++i )
                for( std::size_t j = 0; j < indices.size(); ++j )
                    values[i * indices.size() + j] = static_cast< PetscScalar >( 10 * ( i + 1 ) + j + 1 );
            check( MatSetValues( matrix, static_cast< PetscInt >( indices.size() ), indices.data(),
                                 static_cast< PetscInt >( indices.size() ), indices.data(), values.data(),
                                 ADD_VALUES ) == PETSC_SUCCESS,
                   "insert owner-only replicated matrix contribution" );
        }
        check( MatAssemblyBegin( matrix, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
               "begin replicated matrix assembly" );
        check( MatAssemblyEnd( matrix, MAT_FINAL_ASSEMBLY ) == PETSC_SUCCESS,
               "end replicated matrix assembly" );
        std::vector< PetscInt > columns( static_cast< std::size_t >( sys.n_dofs() ) );
        std::iota( columns.begin(), columns.end(), PetscInt{} );
        std::vector< PetscScalar > actual_row( columns.size() );
        for( PetscInt row = sys.dof_map().first_owned_dof(); row < sys.dof_map().end_owned_dof(); ++row )
        {
            check( MatGetValues( matrix, 1, &row, sys.n_dofs(), columns.data(), actual_row.data() ) == PETSC_SUCCESS,
                   "read owned replicated matrix row" );
            for( PetscInt column = 0; column < sys.n_dofs(); ++column )
                check( PetscAbsScalar( actual_row[static_cast< std::size_t >( column )] -
                                 expected_matrix[static_cast< std::size_t >(
                                     row * sys.n_dofs() + column )] ) < 1.e-12,
                       "owner-only replicated matrix assembly has exact serial entry" );
        }
        PetscReal matrix_norm = 0.0;
        check( MatNorm( matrix, NORM_FROBENIUS, &matrix_norm ) == PETSC_SUCCESS,
               "compute replicated matrix norm" );
        const PetscReal expected_matrix_norm = std::sqrt( std::accumulate(
            expected_matrix.begin(), expected_matrix.end(), PetscReal{}, []( PetscReal sum, PetscScalar value ) {
                return sum + PetscRealPart( value * PetscConj( value ) );
            } ) );
        check( std::abs( matrix_norm - expected_matrix_norm ) < 1.e-12,
               "owner-only replicated matrix norm matches serial" );
        check( MatDestroy( &matrix ) == PETSC_SUCCESS, "destroy replicated matrix" );
    }

    if( n_ranks > 1 )
    {
        KARMAMesh signed_zero_mesh( 1, PETSC_COMM_WORLD );
        signed_zero_mesh.generate_box( 1, 0, 0 );
        if( rank == 1 )
        {
            const moab::EntityHandle vertex = *signed_zero_mesh.vertices().begin();
            const double negative_zero[]    = { -0.0, 0.0, 0.0 };
            check( signed_zero_mesh.moab()->set_coords( &vertex, 1, negative_zero ) == moab::MB_SUCCESS,
                   "create signed-zero replica" );
        }
        KARMASystem signed_zero_system( signed_zero_mesh, "signed-zero" );
        signed_zero_system.add_variable( "u" );
        check( !throws( [&] { signed_zero_system.init(); } ), "signed zero has one replicated canonical identity" );

        KARMAMesh mesh( 1, PETSC_COMM_WORLD );
        mesh.generate_box( 1, 0, 0 );
        if( rank == 1 )
        {
            const moab::EntityHandle vertex = *mesh.vertices().begin();
            const double shifted[]          = { 0.25, 0.0, 0.0 };
            check( mesh.moab()->set_coords( &vertex, 1, shifted ) == moab::MB_SUCCESS, "create rank-different mesh" );
        }
        KARMASystem system( mesh, "partitioned-rejected" );
        system.add_variable( "u" );
        check( throws_containing( [&] { system.init(); },
                                  "partitioned DoF numbering requires a stable partition identity" ),
               "rank-different mesh is rejected collectively with a stable partition identity diagnostic" );

        KARMAMesh duplicate_mesh( 1, PETSC_COMM_WORLD );
        const double duplicate_coordinates[] = { 0, 0, 0, 1, 0, 0, 0, 0, 0 };
        moab::Range duplicate_vertices;
        check( duplicate_mesh.moab()->create_vertices( duplicate_coordinates, 3, duplicate_vertices ) ==
                   moab::MB_SUCCESS,
               "create duplicate-coordinate replica" );
        const std::vector< moab::EntityHandle > duplicate_handles( duplicate_vertices.begin(),
                                                                   duplicate_vertices.end() );
        const moab::EntityHandle duplicate_connectivity[] = { duplicate_handles[0], duplicate_handles[1] };
        moab::EntityHandle duplicate_element              = 0;
        check( duplicate_mesh.moab()->create_element( moab::MBEDGE, duplicate_connectivity, 2, duplicate_element ) ==
                   moab::MB_SUCCESS,
               "create duplicate-coordinate element" );
        duplicate_mesh.assign_ids();
        duplicate_mesh.update();
        KARMASystem duplicate_system( duplicate_mesh, "duplicate-rejected" );
        duplicate_system.add_variable( "u" );
        check( throws_containing( [&] { duplicate_system.init(); }, "unique vertex coordinates" ),
               "replicated duplicate coordinates are rejected collectively" );

        KARMAMesh duplicate_cell_mesh( 1, PETSC_COMM_WORLD );
        const double cell_coordinates[] = { 0, 0, 0, 1, 0, 0 };
        moab::Range cell_vertices;
        check( duplicate_cell_mesh.moab()->create_vertices( cell_coordinates, 2, cell_vertices ) == moab::MB_SUCCESS,
               "create duplicate-cell vertices" );
        const std::vector< moab::EntityHandle > cell_handles( cell_vertices.begin(), cell_vertices.end() );
        moab::EntityHandle duplicate_cells[2]{};
        check( duplicate_cell_mesh.moab()->create_element( moab::MBEDGE, cell_handles.data(), 2, duplicate_cells[0] ) ==
                       moab::MB_SUCCESS &&
                   duplicate_cell_mesh.moab()->create_element( moab::MBEDGE, cell_handles.data(), 2,
                                                               duplicate_cells[1] ) == moab::MB_SUCCESS,
               "create duplicate coincident cells" );
        duplicate_cell_mesh.assign_ids();
        duplicate_cell_mesh.update();
        KARMASystem duplicate_cell_system( duplicate_cell_mesh, "duplicate-cell-rejected" );
        duplicate_cell_system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
        check( throws_containing( [&] { duplicate_cell_system.init(); }, "stable partition identity" ),
               "replicated duplicate cells are rejected with the stable partition identity diagnostic" );

        KARMAMesh signature_mesh( 1, PETSC_COMM_WORLD );
        signature_mesh.generate_box( 1, 0, 0 );
        KARMASystem signature_system( signature_mesh, "signature" );
        signature_system.add_variable( { rank == 0 ? "u" : "v", FieldBasisFamily::LagrangeNodal, 1,
                                         BasisContinuity::cG, 1 } );
        check( throws_containing( [&] { signature_system.init(); }, "discretization signature" ),
               "rank-different valid variable registries are rejected collectively" );

        KARMAMesh unsupported_mesh( 1, PETSC_COMM_WORLD );
        unsupported_mesh.generate_box( 1, 0, 0 );
        KARMASystem unsupported_system( unsupported_mesh, "rank-only-unsupported" );
        unsupported_system.add_variable( { "u", FieldBasisFamily::LagrangeNodal,
                                           rank == 1 ? 99u : 1u, BasisContinuity::cG, 1 } );
        check( throws_containing( [&] { unsupported_system.init(); }, "discretization signature" ),
               "rank-only unsupported basis declaration is rejected collectively" );

        KARMAMesh unsupported_cell_mesh( 2, PETSC_COMM_WORLD );
        const double unsupported_coordinates[] = { 0, 0, 0, 1, 0, 0, 0, 1, 0 };
        moab::Range unsupported_vertices;
        check( unsupported_cell_mesh.moab()->create_vertices( unsupported_coordinates, 3, unsupported_vertices ) ==
                   moab::MB_SUCCESS,
               "create rank-only unsupported-cell vertices" );
        const std::vector< moab::EntityHandle > unsupported_handles( unsupported_vertices.begin(),
                                                                     unsupported_vertices.end() );
        moab::EntityHandle unsupported_element = 0;
        const moab::EntityType rank_type = rank == 1 ? moab::MBPOLYGON : moab::MBTRI;
        check( unsupported_cell_mesh.moab()->create_element( rank_type, unsupported_handles.data(), 3,
                                                             unsupported_element ) == moab::MB_SUCCESS,
               "create rank-only unsupported cell" );
        unsupported_cell_mesh.assign_ids();
        unsupported_cell_mesh.update();
        KARMASystem unsupported_cell_system( unsupported_cell_mesh, "rank-only-unsupported-cell" );
        unsupported_cell_system.add_variable( "u" );
        check( throws_containing( [&] { unsupported_cell_system.init(); }, "rank 1" ),
               "rank-only unsupported topology fails collectively without deadlock" );

        KARMAMesh permuted_mesh( 1, PETSC_COMM_WORLD );
        permuted_mesh.generate_box( 1, 0, 0 );
        if( rank == 1 )
        {
            const moab::EntityHandle element = *permuted_mesh.elements().begin();
            const moab::EntityHandle* connectivity = nullptr;
            int count = 0;
            check( permuted_mesh.moab()->get_connectivity( element, connectivity, count ) == moab::MB_SUCCESS,
                   "read connectivity for role permutation" );
            std::vector< moab::EntityHandle > reversed( connectivity, connectivity + count );
            std::reverse( reversed.begin(), reversed.end() );
            check( permuted_mesh.moab()->set_connectivity( element, reversed.data(), count ) == moab::MB_SUCCESS,
                   "permute replicated connectivity roles" );
        }
        KARMASystem permuted_system( permuted_mesh, "permuted-rejected" );
        permuted_system.add_variable( "u" );
        check( throws_containing( [&] { permuted_system.init(); },
                                  "partitioned DoF numbering requires a stable partition identity" ),
               "rank-local connectivity role permutation is rejected" );

        KARMAMesh adjacency_mesh( 1, PETSC_COMM_WORLD );
        adjacency_mesh.generate_box( 2, 0, 0 );
        const std::vector< moab::EntityHandle > adjacency_elements( adjacency_mesh.elements().begin(),
                                                                    adjacency_mesh.elements().end() );
        KARMASystem adjacency_system( adjacency_mesh, "collective-adjacency" );
        adjacency_system.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
        adjacency_system.init();
        const PetscInt local_adjacency_before =
            std::accumulate( adjacency_system.sparsity().n_nz.begin(), adjacency_system.sparsity().n_nz.end(), PetscInt{} ) +
            std::accumulate( adjacency_system.sparsity().o_nz.begin(), adjacency_system.sparsity().o_nz.end(), PetscInt{} );
        PetscInt global_adjacency_before = 0;
        check( MPI_Allreduce( &local_adjacency_before, &global_adjacency_before, 1, MPIU_INT, MPI_SUM,
                              PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "sum adjacency before rank-local submission" );
        check( !throws( [&] {
                    if( rank == 0 )
                        adjacency_system.add_element_adjacency( adjacency_elements[0], adjacency_elements[1] );
                    else
                        adjacency_system.add_element_adjacency( 0, 0 );
                } ), "subset-rank adjacency request contributes a collective union" );
        const PetscInt local_adjacency_after =
            std::accumulate( adjacency_system.sparsity().n_nz.begin(), adjacency_system.sparsity().n_nz.end(), PetscInt{} ) +
            std::accumulate( adjacency_system.sparsity().o_nz.begin(), adjacency_system.sparsity().o_nz.end(), PetscInt{} );
        PetscInt global_adjacency_after = 0;
        check( MPI_Allreduce( &local_adjacency_after, &global_adjacency_after, 1, MPIU_INT, MPI_SUM,
                              PETSC_COMM_WORLD ) == MPI_SUCCESS,
               "sum adjacency after rank-local submission" );
        check( global_adjacency_after == global_adjacency_before + 2,
               "rank-local adjacency union adds both directed dG couplings" );
        check( !throws( [&] {
                    adjacency_system.add_element_adjacency( adjacency_elements[0], adjacency_elements[1] );
                } ), "identical collective adjacency request succeeds" );

        KARMAMesh one_cell( 1, PETSC_COMM_WORLD );
        one_cell.generate_box( 1, 0, 0 );
        KARMASystem one_dof( one_cell, "one-dof" );
        one_dof.add_variable( { "u", FieldBasisFamily::LagrangeNodal, 0, BasisContinuity::dG, 1 } );
        one_dof.init();
        check( one_dof.n_dofs() == 1 && one_dof.sparsity().is_ready(),
               "more ranks than DoFs keeps zero-owned sparsity ready" );
        PetscInt owned_sum = 0;
        const PetscInt locally_owned = one_dof.n_local_dofs();
        check( MPI_Allreduce( &locally_owned, &owned_sum, 1, MPIU_INT, MPI_SUM, PETSC_COMM_WORLD ) == MPI_SUCCESS &&
                   owned_sum == 1,
               "one P0 DoF has exactly one owner across all ranks" );
    }

    int global_failures = 0;
    if( MPI_Allreduce( &failures, &global_failures, 1, MPI_INT, MPI_SUM, PETSC_COMM_WORLD ) != MPI_SUCCESS )
        ++failures;
    if( rank == 0 )
    {
        if( global_failures == 0 )
            std::cout << "system tests passed on " << n_ranks << " rank(s)." << std::endl;
        else
            std::cout << "system tests: " << global_failures << " failure(s)." << std::endl;
    }

    ierr = PetscFinalize();
    if( ierr ) return ierr;

    return failures == 0 && global_failures == 0 ? 0 : 1;
}
