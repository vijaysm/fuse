#include "advection_reaction_system.h"

#include "generic_mass_operator.h"
#include "../common/impls/collective_signature.h"
#include "karma_exceptions.h"

#include <algorithm>
#include <cstdint>
#include <exception>
#include <map>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace
{
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

void collective_error( MPI_Comm communicator, const std::string& local_error )
{
    KARMA::detail::collective_signature_error( communicator, local_error );
}

using SideKey = std::pair< std::uint64_t, std::uint64_t >;
using FaceKey = std::vector< KARMA::GlobalId >;

struct FaceRecord
{
    FaceKey face;
    SideKey source;
};

std::vector< std::pair< std::uint64_t, std::uint64_t > > global_interior_faces(
    const KARMA::KARMAMesh& mesh, const KARMA::KARMASystem& system, MPI_Comm communicator )
{
    mesh.validate_owned_side_topology();
    std::vector< FaceRecord > local_records;
    std::string local_error;
    try
    {
        for( const auto element : mesh.owned_elements() )
            for( unsigned side = 0; side < mesh.n_sides( element ); ++side )
                if( const auto neighbor = mesh.neighbor( element, side ); neighbor )
                {
                    const auto source_identity = mesh.canonical_side_vertex_ids( element, side );
                    unsigned reciprocal_side = mesh.n_sides( *neighbor );
                    for( unsigned candidate = 0; candidate < mesh.n_sides( *neighbor ); ++candidate )
                    {
                        if( mesh.canonical_side_vertex_ids( *neighbor, candidate ) != source_identity ) continue;
                        if( reciprocal_side != mesh.n_sides( *neighbor ) )
                            KARMA_THROW( KARMA::InvalidArgument( "nonmanifold reciprocal advection face" ) );
                        reciprocal_side = candidate;
                    }
                    if( reciprocal_side == mesh.n_sides( *neighbor ) )
                        KARMA_THROW( KARMA::InvalidArgument( "conflicting reciprocal advection face" ) );
                    local_records.push_back( { source_identity,
                                               { system.dof_map().canonical_element_ordinal( element ), side } } );
                }
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown advection topology validation failure"; }
    collective_error( communicator, local_error );

    constexpr std::size_t max_face_vertices = 4;
    constexpr std::size_t record_words = 1 + max_face_vertices + 2;
    int ranks = 1;
    check( MPI_Comm_size( communicator, &ranks ), "MPI_Comm_size(advection face records)" );
    int local_words = 0;
    try
    {
        if( std::any_of( local_records.begin(), local_records.end(), []( const FaceRecord& record ) {
                return record.face.empty() || record.face.size() > max_face_vertices;
            } ) )
            KARMA_THROW( KARMA::InvalidArgument( "unsupported advection face vertex count" ) );
        local_words = KARMA::detail::checked_mpi_word_count( local_records.size(), record_words );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, local_error );
    std::vector< std::uint64_t > local;
    std::vector< int > counts;
    KARMA::detail::prepare_collectively( communicator, [&] {
        local.reserve( static_cast< std::size_t >( local_words ) );
        for( const auto& record : local_records )
        {
            local.push_back( record.face.size() );
            for( std::size_t vertex = 0; vertex < max_face_vertices; ++vertex )
                local.push_back( vertex < record.face.size() ? static_cast< std::uint64_t >( record.face[vertex] ) : 0 );
            local.push_back( record.source.first );
            local.push_back( record.source.second );
        }
        counts.resize( static_cast< std::size_t >( ranks ) );
    }, "advection face record preparation failed" );
    check( MPI_Allgather( &local_words, 1, MPI_INT, counts.data(), 1, MPI_INT, communicator ),
           "MPI_Allgather(advection face record counts)" );
    KARMA::detail::MpiLayout layout;
    local_error.clear();
    try { layout = KARMA::detail::checked_mpi_displacements( counts, record_words ); }
    catch( const std::exception& error ) { local_error = error.what(); }
    collective_error( communicator, local_error );
    std::vector< std::uint64_t > gathered;
    KARMA::detail::prepare_collectively( communicator, [&] { gathered.resize( layout.total_count ); },
                                         "advection face gather allocation failed" );
    check( MPI_Allgatherv( local.data(), local_words, MPI_UINT64_T, gathered.data(), counts.data(),
                           layout.displacements.data(), MPI_UINT64_T, communicator ),
           "MPI_Allgatherv(advection face records)" );

    std::map< FaceKey, std::set< SideKey > > faces;
    std::set< std::pair< std::uint64_t, std::uint64_t > > pairs;
    local_error.clear();
    try
    {
        for( std::size_t offset = 0; offset < gathered.size(); offset += record_words )
        {
            const std::size_t vertex_count = static_cast< std::size_t >( gathered[offset] );
            if( vertex_count == 0 || vertex_count > max_face_vertices )
                KARMA_THROW( KARMA::InvalidArgument( "invalid gathered advection face identity" ) );
            const FaceKey face( gathered.begin() + static_cast< std::ptrdiff_t >( offset + 1 ),
                                gathered.begin() + static_cast< std::ptrdiff_t >( offset + 1 + vertex_count ) );
            const SideKey source{ gathered[offset + 1 + max_face_vertices],
                                  gathered[offset + 2 + max_face_vertices] };
            faces[face].insert( source );
        }
        for( const auto& [face, records] : faces )
        {
            (void)face;
            if( records.size() != 2 )
            {
                local_error = records.size() < 2 ? "missing reciprocal advection face record"
                                                 : "nonmanifold advection face record";
                local_error += " for GLOBAL_ID vertices";
                for( const auto id : face ) local_error += " " + std::to_string( id );
                local_error += " (incident sides=" + std::to_string( records.size() );
                for( const auto& [element, side] : records )
                    local_error += " " + std::to_string( element ) + ":" + std::to_string( side );
                local_error += ")";
                break;
            }
            const auto first = records.begin();
            const auto second = std::next( first );
            if( first->first == second->first )
            {
                local_error = "advection interior face connects one global element";
                break;
            }
            if( !pairs.emplace( std::min( first->first, second->first ),
                                std::max( first->first, second->first ) ).second )
            {
                local_error = "nonmanifold duplicate advection element pair";
                break;
            }
        }
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown gathered advection face validation failure"; }
    collective_error( communicator, local_error );
    return { pairs.begin(), pairs.end() };
}
}  // namespace

namespace KARMA
{
namespace detail
{
namespace
{
bool advection_local_element_map_fault = false;
}

void set_advection_local_element_map_fault( bool enabled )
{
    advection_local_element_map_fault = enabled;
}
}

AdvectionReactionSystem::AdvectionReactionSystem( KARMAMesh& mesh, AdvectionReactionSystemPropertyManager properties )
    : AdvectionReactionSystem( mesh, std::move( properties ), Configuration{} )
{}

AdvectionReactionSystem::AdvectionReactionSystem( KARMAMesh& mesh, AdvectionReactionSystemPropertyManager properties,
                                                   Configuration configuration )
    : _mesh( mesh ), _properties( std::move( properties ) ), _configuration( std::move( configuration ) )
{}

AdvectionReactionSystem::~AdvectionReactionSystem() = default;

void AdvectionReactionSystem::init()
{
    if( _initialized ) return;
    const MPI_Comm communicator = _mesh.parallel_comm()->comm();
    std::string local_error;
    try
    {
        if( _configuration.variable_name.empty() ) local_error = "variable name must not be empty";
        else if( _configuration.continuity != BasisContinuity::dG )
            local_error = "advection-reaction requires discontinuous Galerkin continuity";
        else if( _configuration.quadrature_degree <= 0 ) local_error = "quadrature degree must be positive";
        else if( _properties.empty() ) local_error = "at least one material model must be registered";
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown advection initialization validation failure"; }
    detail::require_collective_signature(
        communicator, local_error,
        [&] {
            std::string signature = std::to_string( _configuration.variable_name.size() ) + ":" +
                                    _configuration.variable_name + ":" +
                                    std::to_string( static_cast< unsigned >( _configuration.order ) ) + ":" +
                                    std::to_string( static_cast< unsigned >( _configuration.family ) ) + ":" +
                                    std::to_string( static_cast< unsigned >( _configuration.continuity ) ) + ":" +
                                    std::to_string( _configuration.quadrature_degree ) + ";";
            for( const int material : _properties.material_ids() )
                signature += "m" + std::to_string( material ) + ";";
            return signature;
        },
        "unknown advection signature construction failure",
        "advection configuration or material signature differs from rank 0" );

    auto system = std::make_unique< KARMASystem >( _mesh, "AdvectionReactionSystem" );
    system->add_variable( _configuration.variable_name, _configuration.order, _configuration.family,
                           _configuration.continuity );
    system->init();
    std::map< std::uint64_t, moab::EntityHandle > local_elements;
    local_error.clear();
    try
    {
        if( detail::advection_local_element_map_fault )
            KARMA_THROW( InvalidArgument( "injected advection local element map construction failure" ) );
        for( const auto element : _mesh.elements() )
            local_elements.emplace( system->dof_map().canonical_element_ordinal( element ), element );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown advection local element map construction failure"; }
    collective_error( communicator, local_error );
    const auto interior_faces = global_interior_faces( _mesh, *system, communicator );
    for( const auto& [first, second] : interior_faces )
    {
        const auto first_element = local_elements.find( first );
        const auto second_element = local_elements.find( second );
        if( first_element != local_elements.end() && second_element != local_elements.end() )
            system->add_element_adjacency( first_element->second, second_element->second );
        else
            system->add_element_adjacency( 0, 0 );
    }

    PetscOwner< Vec, VecDestroy > residual;
    PetscOwner< Mat, MatDestroy > jacobian;
    PetscOwner< Mat, MatDestroy > mass;
    PetscOwner< SNES, SNESDestroy > solver;
    *residual.put() = system->create_vector();
    *jacobian.put() = system->create_matrix();
    *mass.put() = system->create_matrix();
    assemble_mass_matrix( *system, mass.get(), nullptr, 0, _configuration.quadrature_degree );
    check( SNESCreate( PetscObjectComm( reinterpret_cast< PetscObject >( system->solution() ) ), solver.put() ),
           "SNESCreate(advection)" );
    check( SNESSetFunction( solver.get(), residual.get(), snes_residual, this ), "SNESSetFunction(advection)" );
    check( SNESSetJacobian( solver.get(), jacobian.get(), jacobian.get(), snes_jacobian, this ),
           "SNESSetJacobian(advection)" );
    check( SNESSetFromOptions( solver.get() ), "SNESSetFromOptions(advection)" );
    _system = std::move( system );
    _residual = std::move( residual );
    _jacobian = std::move( jacobian );
    _mass = std::move( mass );
    _solver = std::move( solver );
    _initialized = true;
}

void AdvectionReactionSystem::clear() noexcept
{
    _solver.reset(); _mass.reset(); _jacobian.reset(); _residual.reset(); _system.reset(); _initialized = false;
}
void AdvectionReactionSystem::require_initialized() const
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "advection-reaction system is not initialized" ) );
}
KARMASystem& AdvectionReactionSystem::system() { require_initialized(); return *_system; }
const KARMASystem& AdvectionReactionSystem::system() const { require_initialized(); return *_system; }
Vec AdvectionReactionSystem::solution() const { require_initialized(); return _system->solution(); }
Mat AdvectionReactionSystem::jacobian() const { require_initialized(); return _jacobian.get(); }
Mat AdvectionReactionSystem::mass() const { require_initialized(); return _mass.get(); }
void AdvectionReactionSystem::assemble_residual( Vec solution, Vec residual ) const
{
    require_initialized();
    _system->validate_vector_layout( solution, "advection solution" );
    _system->validate_vector_layout( residual, "advection residual" );
    assemble_advection_reaction( *_system, solution, residual, nullptr, _properties.coefficients(),
                                 _configuration.quadrature_degree );
}
void AdvectionReactionSystem::assemble_jacobian( Vec solution, Mat jacobian ) const
{
    require_initialized();
    _system->validate_vector_layout( solution, "advection solution" );
    _system->validate_matrix_layout( jacobian, "advection Jacobian" );
    PetscOwner< Vec, VecDestroy > residual;
    *residual.put() = _system->create_vector();
    assemble_advection_reaction( *_system, solution, residual.get(), jacobian, _properties.coefficients(),
                                 _configuration.quadrature_degree );
}
void AdvectionReactionSystem::assemble_mass( Mat mass ) const
{
    require_initialized();
    _system->validate_matrix_layout( mass, "advection mass matrix" );
    assemble_mass_matrix( *_system, mass, nullptr, 0, _configuration.quadrature_degree );
}
void AdvectionReactionSystem::solve()
{
    require_initialized();
    check( SNESSolve( _solver.get(), nullptr, solution() ), "SNESSolve(advection)" );
    SNESConvergedReason reason = SNES_CONVERGED_ITERATING;
    check( SNESGetConvergedReason( _solver.get(), &reason ), "SNESGetConvergedReason(advection)" );
    if( reason <= 0 ) KARMA_THROW( InvalidArgument( "advection-reaction nonlinear solve did not converge" ) );
}
PetscErrorCode AdvectionReactionSystem::snes_residual( SNES, Vec solution, Vec residual, void* context ) noexcept
{
    try { static_cast< AdvectionReactionSystem* >( context )->assemble_residual( solution, residual ); return PETSC_SUCCESS; }
    catch( const std::exception& error )
    {
        return PetscError( PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_LIB,
                           PETSC_ERROR_INITIAL, "%s", error.what() );
    }
    catch( ... )
    {
        return PetscError( PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_LIB,
                           PETSC_ERROR_INITIAL, "%s", "unknown advection residual failure" );
    }
}
PetscErrorCode AdvectionReactionSystem::snes_jacobian( SNES, Vec solution, Mat matrix, Mat preconditioner,
                                                       void* context ) noexcept
{
    try
    {
        auto* system = static_cast< AdvectionReactionSystem* >( context );
        system->assemble_jacobian( solution, matrix );
        if( matrix != preconditioner ) system->assemble_jacobian( solution, preconditioner );
        return PETSC_SUCCESS;
    }
    catch( const std::exception& error )
    {
        return PetscError( PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_LIB,
                           PETSC_ERROR_INITIAL, "%s", error.what() );
    }
    catch( ... )
    {
        return PetscError( PETSC_COMM_SELF, __LINE__, PETSC_FUNCTION_NAME, __FILE__, PETSC_ERR_LIB,
                           PETSC_ERROR_INITIAL, "%s", "unknown advection Jacobian failure" );
    }
}
}  // namespace KARMA
