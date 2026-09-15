#ifndef __diffusion_reaction_system_h__
#define __diffusion_reaction_system_h__

#include "diffusion_reaction_system_property_manager.h"
#include "karma_system.h"
#include "petsc_raii.h"

#include <petscmat.h>
#include <petscsnes.h>
#include <petscvec.h>

#include <memory>
#include <string>

class KARMAPhysicsBCManager;

namespace KARMA
{
class KARMAMesh;
class KARMASystem;

class DiffusionReactionSystem
{
  public:
    struct Configuration
    {
        std::string variable_name{ "u" };
        Order order{ Order::First };
        FieldBasisFamily family{ FieldBasisFamily::LagrangeNodal };
        BasisContinuity continuity{ BasisContinuity::cG };
        int quadrature_degree{ 5 };
    };

    DiffusionReactionSystem( KARMAMesh& mesh, DiffusionReactionSystemPropertyManager properties );
    DiffusionReactionSystem( KARMAMesh& mesh, DiffusionReactionSystemPropertyManager properties,
                             Configuration configuration );
    ~DiffusionReactionSystem();
    DiffusionReactionSystem( const DiffusionReactionSystem& ) = delete;
    DiffusionReactionSystem& operator=( const DiffusionReactionSystem& ) = delete;
    DiffusionReactionSystem( DiffusionReactionSystem&& ) = delete;
    DiffusionReactionSystem& operator=( DiffusionReactionSystem&& ) = delete;

    void init();
    void clear() noexcept;
    void assemble_residual( Vec solution, Vec residual ) const;
    void assemble_jacobian( Vec solution, Mat jacobian ) const;
    /** Assemble the unconstrained physical mass matrix; boundary constraints belong to solve/operator stages. */
    void assemble_mass( Mat mass ) const;
    void solve();

    [[nodiscard]] bool is_initialized() const noexcept { return _initialized; }
    [[nodiscard]] KARMASystem& system();
    [[nodiscard]] const KARMASystem& system() const;
    [[nodiscard]] Vec solution() const;
    [[nodiscard]] Mat jacobian() const;
    [[nodiscard]] Mat mass() const;
    [[nodiscard]] DiffusionReactionSystemPropertyManager& properties() noexcept { return _properties; }
    [[nodiscard]] const DiffusionReactionSystemPropertyManager& properties() const noexcept { return _properties; }
    [[nodiscard]] KARMAPhysicsBCManager& boundary_conditions() noexcept;
    [[nodiscard]] const KARMAPhysicsBCManager& boundary_conditions() const noexcept;
    [[nodiscard]] int quadrature_degree() const noexcept { return _configuration.quadrature_degree; }

  private:
    static PetscErrorCode snes_residual( SNES, Vec solution, Vec residual, void* context ) noexcept;
    static PetscErrorCode snes_jacobian( SNES, Vec solution, Mat jacobian, Mat preconditioner,
                                         void* context ) noexcept;
    void require_initialized() const;

    KARMAMesh& _mesh;
    DiffusionReactionSystemPropertyManager _properties;
    Configuration _configuration;
    std::unique_ptr< KARMASystem > _system;
    std::unique_ptr< KARMAPhysicsBCManager > _boundary;
    PetscOwner< Vec, VecDestroy > _residual;
    PetscOwner< Mat, MatDestroy > _jacobian;
    PetscOwner< Mat, MatDestroy > _mass;
    PetscOwner< SNES, SNESDestroy > _solver;
    bool _initialized = false;
};
}  // namespace KARMA

#endif
