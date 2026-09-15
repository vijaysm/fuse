#ifndef KARMA_TRANSIENT_REACTION_DIFFUSION_COMPONENT_H
#define KARMA_TRANSIENT_REACTION_DIFFUSION_COMPONENT_H

#include "transient_reaction_diffusion_assembly.h"
#include "petsc_raii.h"
#include "karma_system.h"

#include <petscsnes.h>
#include <string>

namespace KARMA
{
class KARMAMesh;

struct TransientSolveDiagnostics
{
    SNESConvergedReason reason = SNES_CONVERGED_ITERATING;
    PetscInt nonlinear_iterations = 0;
    Real residual_norm = 0;
    Real residual_scale = 0;
    double solve_seconds = 0;
};

class TransientReactionDiffusionComponent
{
  public:
    struct Configuration
    {
        std::string variable_name{ "state" };
        int quadrature_degree = 5;
    };

    TransientReactionDiffusionComponent( KARMAMesh&, TransientReactionDiffusionModel );
    TransientReactionDiffusionComponent( KARMAMesh&, TransientReactionDiffusionModel, Configuration );
    ~TransientReactionDiffusionComponent();
    TransientReactionDiffusionComponent( const TransientReactionDiffusionComponent& ) = delete;
    TransientReactionDiffusionComponent& operator=( const TransientReactionDiffusionComponent& ) = delete;

    void init();
    void bind_input( Vec );
    void checkpoint();
    void advance( Real time_left, Real time_right );
    void commit();
    void release_checkpoint() noexcept;
    [[nodiscard]] bool restore() noexcept;

    [[nodiscard]] bool is_initialized() const noexcept { return _initialized; }
    [[nodiscard]] bool has_input() const noexcept { return _input_bound; }
    [[nodiscard]] Real committed_time() const noexcept { return _committed_time; }
    [[nodiscard]] Real trial_time() const noexcept { return _trial_time; }
    [[nodiscard]] KARMASystem& system();
    [[nodiscard]] Vec solution() const;
    [[nodiscard]] Vec trial_solution() const;
    [[nodiscard]] Vec input() const;
    [[nodiscard]] Mat mass() const;
    [[nodiscard]] const TransientSolveDiagnostics& diagnostics() const noexcept { return _diagnostics; }
    void restore_committed_solution( Vec, Real time );

  private:
    static PetscErrorCode snes_residual( SNES, Vec, Vec, void* ) noexcept;
    static PetscErrorCode snes_jacobian( SNES, Vec, Mat, Mat, void* ) noexcept;
    void require_ready() const;

    KARMAMesh& _mesh;
    TransientReactionDiffusionModel _model;
    Configuration _configuration;
    std::unique_ptr<KARMASystem> _system;
    PetscOwner<Vec, VecDestroy> _committed;
    PetscOwner<Vec, VecDestroy> _trial;
    PetscOwner<Vec, VecDestroy> _old;
    PetscOwner<Vec, VecDestroy> _input;
    PetscOwner<Vec, VecDestroy> _checkpoint_state;
    PetscOwner<Vec, VecDestroy> _checkpoint_input;
    PetscOwner<Vec, VecDestroy> _residual;
    PetscOwner<Mat, MatDestroy> _jacobian;
    PetscOwner<Mat, MatDestroy> _preconditioner;
    PetscOwner<Mat, MatDestroy> _mass;
    PetscOwner<SNES, SNESDestroy> _snes;
    Real _committed_time = 0;
    Real _trial_time = 0;
    Real _checkpoint_time = 0;
    Real _advance_left = 0;
    Real _advance_right = 0;
    bool _initialized = false;
    bool _input_bound = false;
    bool _checkpointed = false;
    bool _trial_active = false;
    TransientSolveDiagnostics _diagnostics;
};
}  // namespace KARMA

#endif
