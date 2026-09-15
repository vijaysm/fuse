#include "transient_reaction_diffusion_component.h"

#include "../common/impls/collective_signature.h"
#include "generic_mass_operator.h"
#include "karma_exceptions.h"
#include "karma_mesh.h"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <exception>
#include <string>

namespace
{
void check( PetscErrorCode error, const char* operation )
{
    if( error ) KARMA_THROW( KARMA::InvalidArgument( std::string( "PETSc error in " ) + operation ) );
}
}

namespace KARMA
{
TransientReactionDiffusionComponent::TransientReactionDiffusionComponent(
    KARMAMesh& mesh, TransientReactionDiffusionModel model )
    : TransientReactionDiffusionComponent( mesh, std::move( model ), Configuration{} )
{}

TransientReactionDiffusionComponent::TransientReactionDiffusionComponent(
    KARMAMesh& mesh, TransientReactionDiffusionModel model, Configuration configuration )
    : _mesh( mesh ), _model( std::move( model ) ), _configuration( std::move( configuration ) )
{}

TransientReactionDiffusionComponent::~TransientReactionDiffusionComponent() = default;

void TransientReactionDiffusionComponent::init()
{
    if( _initialized ) return;
    std::string local_error;
    try
    {
        if( _configuration.variable_name.empty() ) KARMA_THROW( InvalidArgument( "transient variable name must not be empty" ) );
        if( _configuration.quadrature_degree <= 0 ) KARMA_THROW( InvalidArgument( "transient quadrature degree must be positive" ) );
        if( !_model.reaction || !_model.state_derivative ) KARMA_THROW( InvalidArgument( "transient callbacks must be set" ) );
        if( !std::isfinite( _model.diffusion ) || _model.diffusion < 0 )
            KARMA_THROW( InvalidArgument( "transient diffusion must be finite and nonnegative" ) );
    }
    catch( const std::exception& error ) { local_error = error.what(); }
    catch( ... ) { local_error = "unknown transient initialization failure"; }
    detail::collective_signature_error( _mesh.communicator(), local_error );

    auto system = std::make_unique<KARMASystem>( _mesh, "TransientReactionDiffusion" );
    system->add_variable( _configuration.variable_name, Order::First, FieldBasisFamily::LagrangeNodal, BasisContinuity::cG );
    system->init();
    PetscOwner<Vec, VecDestroy> committed, trial, old, input, checkpoint_state, checkpoint_input, residual;
    PetscOwner<Mat, MatDestroy> jacobian, preconditioner, mass;
    *committed.put() = system->create_vector();
    *trial.put() = system->create_vector();
    *old.put() = system->create_vector();
    *input.put() = system->create_vector();
    *checkpoint_state.put() = system->create_vector();
    *checkpoint_input.put() = system->create_vector();
    *residual.put() = system->create_vector();
    *jacobian.put() = system->create_matrix();
    *preconditioner.put() = system->create_matrix();
    *mass.put() = system->create_matrix();
    check( MatAssemblyBegin( preconditioner.get(), MAT_FINAL_ASSEMBLY ), "MatAssemblyBegin(transient preconditioner)" );
    check( MatAssemblyEnd( preconditioner.get(), MAT_FINAL_ASSEMBLY ), "MatAssemblyEnd(transient preconditioner)" );
    check( VecSet( committed.get(), 0 ), "VecSet(committed)" );
    check( VecCopy( committed.get(), trial.get() ), "VecCopy(trial)" );
    assemble_mass_matrix( *system, mass.get(), nullptr, 0, _configuration.quadrature_degree );
    PetscOwner<SNES, SNESDestroy> snes;
    check( SNESCreate( _mesh.communicator(), snes.put() ), "SNESCreate(transient)" );
    check( SNESSetFunction( snes.get(), residual.get(), snes_residual, this ), "SNESSetFunction(transient)" );
    check( SNESSetJacobian( snes.get(), jacobian.get(), preconditioner.get(), snes_jacobian, this ), "SNESSetJacobian(transient)" );
    check( SNESSetOptionsPrefix( snes.get(), "karma_transient_reaction_diffusion_" ),
           "SNESSetOptionsPrefix(transient)" );
    check( SNESSetFromOptions( snes.get() ), "SNESSetFromOptions(transient)" );
    _system = std::move( system ); _committed = std::move( committed ); _trial = std::move( trial ); _old = std::move( old );
    _input = std::move( input ); _checkpoint_state = std::move( checkpoint_state ); _checkpoint_input = std::move( checkpoint_input );
    _residual = std::move( residual ); _jacobian = std::move( jacobian ); _preconditioner = std::move( preconditioner );
    _mass = std::move( mass ); _snes = std::move( snes );
    _initialized = true;
}

void TransientReactionDiffusionComponent::require_ready() const
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "transient reaction-diffusion component is not initialized" ) );
    if( !_input_bound ) KARMA_THROW( InvalidArgument( "transient coupled input is not bound" ) );
}

void TransientReactionDiffusionComponent::bind_input( Vec input )
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "transient component is not initialized" ) );
    _system->validate_vector_layout( input, "transient coupled input" );
    check( VecCopy( input, _input.get() ), "VecCopy(transient input)" );
    _input_bound = true;
}

void TransientReactionDiffusionComponent::checkpoint()
{
    require_ready();
    check( VecCopy( _committed.get(), _checkpoint_state.get() ), "VecCopy(transient checkpoint state)" );
    check( VecCopy( _input.get(), _checkpoint_input.get() ), "VecCopy(transient checkpoint input)" );
    _checkpoint_time = _committed_time;
    _checkpointed = true;
    _trial_active = false;
}

void TransientReactionDiffusionComponent::advance( Real time_left, Real time_right )
{
    const auto started = std::chrono::steady_clock::now();
    require_ready();
    if( !_checkpointed ) KARMA_THROW( InvalidArgument( "transient advance requires a checkpoint" ) );
    if( !std::isfinite( time_left ) || !std::isfinite( time_right ) || time_right <= time_left )
        KARMA_THROW( InvalidArgument( "transient interval must be positive and finite" ) );
    if( !_trial_active )
    {
        check( VecCopy( _committed.get(), _old.get() ), "VecCopy(transient old state)" );
        check( VecCopy( _committed.get(), _trial.get() ), "VecCopy(transient trial state)" );
    }
    else
        check( VecCopy( _trial.get(), _old.get() ), "VecCopy(transient old state)" );
    _advance_left = time_left; _advance_right = time_right;
    check( SNESSolve( _snes.get(), nullptr, _trial.get() ), "SNESSolve(transient)" );
    SNESConvergedReason reason = SNES_CONVERGED_ITERATING;
    check( SNESGetConvergedReason( _snes.get(), &reason ), "SNESGetConvergedReason(transient)" );
    if( reason <= 0 ) KARMA_THROW( InvalidArgument( "transient nonlinear solve did not converge" ) );
    PetscInt iterations = 0;
    check( SNESGetIterationNumber( _snes.get(), &iterations ), "SNESGetIterationNumber(transient)" );
    assemble_transient_reaction_diffusion( *_system, _old.get(), _trial.get(), _input.get(), time_left, time_right,
                                           _model, _residual.get(), nullptr, _configuration.quadrature_degree );
    PetscReal norm = 0, old_norm = 0, trial_norm = 0, input_norm = 0;
    check( VecNorm( _residual.get(), NORM_INFINITY, &norm ), "VecNorm(transient residual)" );
    check( VecNorm( _old.get(), NORM_INFINITY, &old_norm ), "VecNorm(transient old state)" );
    check( VecNorm( _trial.get(), NORM_INFINITY, &trial_norm ), "VecNorm(transient trial state)" );
    check( VecNorm( _input.get(), NORM_INFINITY, &input_norm ), "VecNorm(transient input)" );
    const PetscReal scale = std::max<PetscReal>( 1, std::max( { old_norm, trial_norm, input_norm } ) );
    if( !std::isfinite( norm ) || !std::isfinite( scale ) || norm > 1.e-8 * scale )
        KARMA_THROW( InvalidArgument( "transient accepted residual is too large" ) );
    _trial_time = time_right;
    _trial_active = true;
    _diagnostics = { reason, iterations, norm, scale,
                     std::chrono::duration<double>( std::chrono::steady_clock::now() - started ).count() };
}

void TransientReactionDiffusionComponent::commit()
{
    require_ready();
    if( !_checkpointed ) KARMA_THROW( InvalidArgument( "transient commit requires a checkpoint" ) );
    check( VecCopy( _trial.get(), _committed.get() ), "VecCopy(transient commit)" );
    _committed_time = _trial_time;
}

void TransientReactionDiffusionComponent::release_checkpoint() noexcept
{
    _checkpointed = false;
    _trial_active = false;
}

bool TransientReactionDiffusionComponent::restore() noexcept
{
    if( !_initialized || !_checkpointed ) return true;
    if( VecCopy( _checkpoint_state.get(), _committed.get() ) ) return false;
    if( VecCopy( _checkpoint_state.get(), _trial.get() ) ) return false;
    if( VecCopy( _checkpoint_input.get(), _input.get() ) ) return false;
    _committed_time = _checkpoint_time;
    _trial_time = _checkpoint_time;
    _trial_active = false;
    return true;
}

KARMASystem& TransientReactionDiffusionComponent::system()
{
    if( !_initialized ) KARMA_THROW( InvalidArgument( "transient reaction-diffusion component is not initialized" ) );
    return *_system;
}
Vec TransientReactionDiffusionComponent::solution() const { require_ready(); return _committed.get(); }
Vec TransientReactionDiffusionComponent::trial_solution() const { require_ready(); return _trial.get(); }
Vec TransientReactionDiffusionComponent::input() const { require_ready(); return _input.get(); }
Mat TransientReactionDiffusionComponent::mass() const { require_ready(); return _mass.get(); }

void TransientReactionDiffusionComponent::restore_committed_solution( Vec staged, Real time )
{
    require_ready();
    if( !std::isfinite( time ) || time < 0 )
        KARMA_THROW( InvalidArgument( "transient restart time must be finite and nonnegative" ) );
    _system->validate_vector_layout( staged, "transient restart state" );
    check( VecCopy( staged, _committed.get() ), "VecCopy(transient restart committed state)" );
    check( VecCopy( staged, _trial.get() ), "VecCopy(transient restart trial state)" );
    _committed_time = time;
    _trial_time = time;
    _checkpointed = false;
    _trial_active = false;
}

PetscErrorCode TransientReactionDiffusionComponent::snes_residual( SNES, Vec solution, Vec residual, void* context ) noexcept
{
    try
    {
        auto& component = *static_cast<TransientReactionDiffusionComponent*>( context );
        assemble_transient_reaction_diffusion( *component._system, component._old.get(), solution, component._input.get(),
                                               component._advance_left, component._advance_right, component._model,
                                               residual, nullptr, component._configuration.quadrature_degree );
        return PETSC_SUCCESS;
    }
    catch( ... ) { return PETSC_ERR_LIB; }
}

PetscErrorCode TransientReactionDiffusionComponent::snes_jacobian( SNES, Vec solution, Mat jacobian, Mat preconditioner,
                                                                    void* context ) noexcept
{
    try
    {
        auto& component = *static_cast<TransientReactionDiffusionComponent*>( context );
        assemble_transient_reaction_diffusion( *component._system, component._old.get(), solution, component._input.get(),
                                               component._advance_left, component._advance_right, component._model,
                                               component._residual.get(), jacobian, component._configuration.quadrature_degree );
        if( preconditioner != jacobian ) MatCopy( jacobian, preconditioner, DIFFERENT_NONZERO_PATTERN );
        return PETSC_SUCCESS;
    }
    catch( ... ) { return PETSC_ERR_LIB; }
}
}  // namespace KARMA
