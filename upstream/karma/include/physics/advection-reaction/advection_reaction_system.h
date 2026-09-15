#ifndef __advection_reaction_system_h__
#define __advection_reaction_system_h__

#include "advection_reaction_system_property_manager.h"
#include "karma_system.h"
#include "petsc_raii.h"

#include <petscsnes.h>

#include <memory>
#include <string>

namespace KARMA
{
class AdvectionReactionSystem
{
  public:
    struct Configuration
    {
        std::string variable_name{ "u" };
        Order order{ Order::Constant };
        FieldBasisFamily family{ FieldBasisFamily::LagrangeNodal };
        BasisContinuity continuity{ BasisContinuity::dG };
        int quadrature_degree{ 5 };
    };

    AdvectionReactionSystem( KARMAMesh& mesh, AdvectionReactionSystemPropertyManager properties );
    AdvectionReactionSystem( KARMAMesh& mesh, AdvectionReactionSystemPropertyManager properties,
                             Configuration configuration );
    ~AdvectionReactionSystem();
    AdvectionReactionSystem( const AdvectionReactionSystem& ) = delete;
    AdvectionReactionSystem& operator=( const AdvectionReactionSystem& ) = delete;

    void init();
    void clear() noexcept;
    void assemble_residual( Vec solution, Vec residual ) const;
    void assemble_jacobian( Vec solution, Mat jacobian ) const;
    void assemble_mass( Mat mass ) const;
    void solve();

    [[nodiscard]] bool is_initialized() const noexcept { return _initialized; }
    [[nodiscard]] KARMASystem& system();
    [[nodiscard]] const KARMASystem& system() const;
    [[nodiscard]] Vec solution() const;
    [[nodiscard]] Mat jacobian() const;
    [[nodiscard]] Mat mass() const;
    [[nodiscard]] AdvectionReactionSystemPropertyManager& properties() noexcept { return _properties; }
    [[nodiscard]] const AdvectionReactionSystemPropertyManager& properties() const noexcept { return _properties; }

  private:
    static PetscErrorCode snes_residual( SNES, Vec, Vec, void* ) noexcept;
    static PetscErrorCode snes_jacobian( SNES, Vec, Mat, Mat, void* ) noexcept;
    void require_initialized() const;

    KARMAMesh& _mesh;
    AdvectionReactionSystemPropertyManager _properties;
    Configuration _configuration;
    std::unique_ptr< KARMASystem > _system;
    PetscOwner< Vec, VecDestroy > _residual;
    PetscOwner< Mat, MatDestroy > _jacobian;
    PetscOwner< Mat, MatDestroy > _mass;
    PetscOwner< SNES, SNESDestroy > _solver;
    bool _initialized = false;
};
}  // namespace KARMA

#endif
