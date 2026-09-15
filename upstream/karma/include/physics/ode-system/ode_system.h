#ifndef __ode_system_h__
#define __ode_system_h__

#include "karma_algebraic_system.h"
#include "ode_system_property_manager.h"
#include "petsc_raii.h"

#include <petscts.h>

#include <string>
#include <vector>

namespace KARMA
{
class ODESystem
{
  public:
    ODESystem( MPI_Comm communicator, std::vector< std::string > unknowns, ODEModel model );
    ~ODESystem() = default;
    ODESystem( const ODESystem& ) = delete;
    ODESystem& operator=( const ODESystem& ) = delete;

    void init();
    void clear() noexcept;
    void solve( Real initial_time, Real final_time, Real initial_step );
    void residual( Real time, Vec state, Vec time_derivative, Vec value ) const;
    void jacobian( Real time, Vec state, Real shift, Mat value ) const;
    void assemble_mass( Real time, Vec state, Mat value ) const;

    [[nodiscard]] bool is_initialized() const noexcept { return _initialized; }
    [[nodiscard]] KARMAAlgebraicSystem& system();
    [[nodiscard]] const KARMAAlgebraicSystem& system() const;
    [[nodiscard]] Vec solution() const;
    [[nodiscard]] Mat mass() const;
    [[nodiscard]] Mat operator_matrix() const;
    void validate_petsc_vector( Vec value, const char* name, bool finite = false ) const;
    void validate_petsc_matrix( Mat value, const char* name ) const;
    [[nodiscard]] ODESystemPropertyManager& properties() noexcept { return _properties; }
    [[nodiscard]] const ODESystemPropertyManager& properties() const noexcept { return _properties; }

  private:
    static PetscErrorCode ts_residual( TS, PetscReal, Vec, Vec, Vec, void* ) noexcept;
    static PetscErrorCode ts_jacobian( TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void* ) noexcept;
    void require_initialized() const;

    MPI_Comm _communicator;
    std::vector< std::string > _unknowns;
    ODESystemPropertyManager _properties;
    std::unique_ptr< KARMAAlgebraicSystem > _system;
    PetscOwner< Vec, VecDestroy > _residual;
    PetscOwner< Mat, MatDestroy > _mass;
    PetscOwner< Mat, MatDestroy > _operator;
    PetscOwner< TS, TSDestroy > _solver;
    bool _initialized = false;
};
}  // namespace KARMA

#endif
