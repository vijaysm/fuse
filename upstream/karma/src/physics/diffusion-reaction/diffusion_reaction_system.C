#include "diffusion_reaction_system.h"

#include "diffusion_reaction_assembly.h"
#include "../common/impls/collective_signature.h"
#include "generic_mass_operator.h"
#include "karma_exceptions.h"
#include "karma_physics_bc_manager.h"
#include "karma_system.h"

#include <petscksp.h>

#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace
{
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}

}  // namespace

namespace KARMA
{
DiffusionReactionSystem::DiffusionReactionSystem( KARMAMesh& mesh,
                                                  DiffusionReactionSystemPropertyManager properties )
    : DiffusionReactionSystem( mesh, std::move( properties ), Configuration{} )
{}

DiffusionReactionSystem::DiffusionReactionSystem( KARMAMesh& mesh,
                                                  DiffusionReactionSystemPropertyManager properties,
                                                  Configuration configuration )
    : _mesh( mesh ), _properties( std::move( properties ) ), _configuration( std::move( configuration ) ),
      _boundary( std::make_unique< KARMAPhysicsBCManager >( 0, _configuration.continuity == BasisContinuity::cG ) )
{}

DiffusionReactionSystem::~DiffusionReactionSystem() = default;

void DiffusionReactionSystem::init()
{
    if( _initialized ) return;
    std::string local_error;
    try
    {
        if( _configuration.variable_name.empty() ) KARMA_THROW( InvalidArgument( "variable name must not be empty" ) );
        if( _configuration.quadrature_degree <= 0 ) KARMA_THROW( InvalidArgument( "quadrature degree must be positive" ) );
        if( _configuration.continuity != BasisContinuity::cG )
            KARMA_THROW( InvalidArgument(
                "diffusion-reaction requires continuous Galerkin continuity until interior diffusion flux is implemented" ) );
        if( _properties.empty() ) KARMA_THROW( InvalidArgument( "at least one material model must be registered" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown diffusion initialization validation failure"; }
    detail::require_collective_signature(
        _mesh.communicator(), local_error,
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
        "unknown diffusion signature construction failure",
        "diffusion configuration or material signature differs from rank 0" );

    auto system = std::make_unique< KARMASystem >( _mesh, "DiffusionReactionSystem" );
    system->add_variable( _configuration.variable_name, _configuration.order, _configuration.family,
                          _configuration.continuity );
    system->init();

    PetscOwner< Vec, VecDestroy > residual;
    PetscOwner< Mat, MatDestroy > jacobian;
    PetscOwner< Mat, MatDestroy > mass;
    PetscOwner< SNES, SNESDestroy > solver;
    *residual.put() = system->create_vector();
    *jacobian.put() = system->create_matrix();
    *mass.put() = system->create_matrix();
    assemble_mass_matrix( *system, mass.get() );
    check( SNESCreate( PetscObjectComm( reinterpret_cast< PetscObject >( system->solution() ) ), solver.put() ),
           "SNESCreate" );
    check( SNESSetFunction( solver.get(), residual.get(), snes_residual, this ), "SNESSetFunction" );
    check( SNESSetJacobian( solver.get(), jacobian.get(), jacobian.get(), snes_jacobian, this ), "SNESSetJacobian" );
    check( SNESSetFromOptions( solver.get() ), "SNESSetFromOptions" );

    _system  = std::move( system );
    _residual = std::move( residual );
    _jacobian = std::move( jacobian );
    _mass     = std::move( mass );
    _solver   = std::move( solver );
    _initialized = true;
}

void DiffusionReactionSystem::clear() noexcept
{
    _solver.reset();
    _mass.reset();
    _jacobian.reset();
    _residual.reset();
    _system.reset();
    _boundary->clear();
    _initialized = false;
}

void DiffusionReactionSystem::require_initialized() const
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "diffusion-reaction system is not initialized" ) );
}

KARMASystem& DiffusionReactionSystem::system()
{
    require_initialized();
    return *_system;
}

const KARMASystem& DiffusionReactionSystem::system() const
{
    require_initialized();
    return *_system;
}

Vec DiffusionReactionSystem::solution() const
{
    require_initialized();
    return _system->solution();
}

Mat DiffusionReactionSystem::jacobian() const
{
    require_initialized();
    return _jacobian.get();
}

Mat DiffusionReactionSystem::mass() const
{
    require_initialized();
    return _mass.get();
}

KARMAPhysicsBCManager& DiffusionReactionSystem::boundary_conditions() noexcept
{
    return *_boundary;
}

const KARMAPhysicsBCManager& DiffusionReactionSystem::boundary_conditions() const noexcept
{
    return *_boundary;
}

void DiffusionReactionSystem::assemble_residual( Vec solution_vector, Vec residual ) const
{
    require_initialized();
    _system->validate_vector_layout( solution_vector, "diffusion solution" );
    _system->validate_vector_layout( residual, "diffusion residual" );
    assemble_diffusion_reaction( *_system, solution_vector, residual, nullptr, _properties.coefficients(),
                                 _boundary.get(), _configuration.quadrature_degree );
}

void DiffusionReactionSystem::assemble_jacobian( Vec solution_vector, Mat jacobian_matrix ) const
{
    require_initialized();
    _system->validate_vector_layout( solution_vector, "diffusion solution" );
    _system->validate_matrix_layout( jacobian_matrix, "diffusion Jacobian" );
    PetscOwner< Vec, VecDestroy > residual;
    *residual.put() = _system->create_vector();
    assemble_diffusion_reaction( *_system, solution_vector, residual.get(), jacobian_matrix,
                                 _properties.coefficients(), _boundary.get(), _configuration.quadrature_degree );
}

void DiffusionReactionSystem::assemble_mass( Mat mass_matrix ) const
{
    require_initialized();
    _system->validate_matrix_layout( mass_matrix, "diffusion mass matrix" );
    assemble_mass_matrix( *_system, mass_matrix, nullptr, 0, _configuration.quadrature_degree );
}

void DiffusionReactionSystem::solve()
{
    require_initialized();
    check( SNESSolve( _solver.get(), nullptr, _system->solution() ), "SNESSolve" );
    SNESConvergedReason reason = SNES_CONVERGED_ITERATING;
    check( SNESGetConvergedReason( _solver.get(), &reason ), "SNESGetConvergedReason" );
    if( reason <= 0 ) KARMA_THROW( InvalidArgument( "diffusion-reaction nonlinear solve did not converge" ) );
}

PetscErrorCode DiffusionReactionSystem::snes_residual( SNES, Vec solution_vector, Vec residual, void* context ) noexcept
{
    try
    {
        static_cast< DiffusionReactionSystem* >( context )->assemble_residual( solution_vector, residual );
        return PETSC_SUCCESS;
    }
    catch( ... )
    {
        return PETSC_ERR_LIB;
    }
}

PetscErrorCode DiffusionReactionSystem::snes_jacobian( SNES, Vec solution_vector, Mat jacobian_matrix,
                                                       Mat preconditioner, void* context ) noexcept
{
    try
    {
        auto& physics = *static_cast< DiffusionReactionSystem* >( context );
        physics.assemble_jacobian( solution_vector, jacobian_matrix );
        if( preconditioner != jacobian_matrix )
            physics.assemble_jacobian( solution_vector, preconditioner );
        return PETSC_SUCCESS;
    }
    catch( ... )
    {
        return PETSC_ERR_LIB;
    }
}
}  // namespace KARMA
