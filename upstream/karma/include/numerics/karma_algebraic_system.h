#ifndef __karma_algebraic_system_h__
#define __karma_algebraic_system_h__

#include "karma_dof_map.h"
#include "karma_system_base.h"
#include "petsc_raii.h"

#include <mpi.h>

#include <string>
#include <string_view>
#include <vector>

namespace KARMA
{
/** Mesh-free algebraic unknown layout with contiguous PETSc ownership. */
class KARMAAlgebraicSystem final : public AbstractSystem
{
  public:
    explicit KARMAAlgebraicSystem( MPI_Comm communicator, std::string_view name );
    ~KARMAAlgebraicSystem() override = default;

    [[nodiscard]] const std::string& name() const noexcept override { return _name; }
    unsigned add_variable( VariableSpec variable ) override;
    [[nodiscard]] unsigned n_variables() const noexcept override
    { return static_cast< unsigned >( _variables.size() ); }
    [[nodiscard]] const VariableSpec& variable( unsigned index ) const override;
    [[nodiscard]] unsigned variable_number( std::string_view name ) const override;
    [[nodiscard]] bool has_variable( std::string_view name ) const override;
    void init() override;
    [[nodiscard]] bool is_initialized() const noexcept override { return _initialized; }
    [[nodiscard]] PetscInt n_dofs() const noexcept override { return _global_size; }
    [[nodiscard]] PetscInt n_local_dofs() const noexcept override { return _local_size; }
    [[nodiscard]] PetscInt first_owned_dof() const noexcept { return _first_owned; }
    [[nodiscard]] Vec create_vector() const override;
    [[nodiscard]] Mat create_matrix() const override;

    [[nodiscard]] Vec solution() const noexcept override { return _solution.get(); }
    [[nodiscard]] Vec current_solution() const noexcept override { return _current_solution.get(); }
    [[nodiscard]] Vec old_solution() const noexcept override { return _old_solution.get(); }
    [[nodiscard]] Vec local_solution() const noexcept override { return _local_solution.get(); }
    [[nodiscard]] Vec temporary_vector() const noexcept override { return _temporary.get(); }
    [[nodiscard]] MPI_Comm communicator() const noexcept { return _communicator; }

  private:
    MPI_Comm _communicator;
    std::string _name;
    std::vector< VariableSpec > _variables;
    PetscInt _global_size = 0;
    PetscInt _local_size = 0;
    PetscInt _first_owned = 0;
    PetscOwner< Vec, VecDestroy > _solution;
    PetscOwner< Vec, VecDestroy > _current_solution;
    PetscOwner< Vec, VecDestroy > _old_solution;
    PetscOwner< Vec, VecDestroy > _local_solution;
    PetscOwner< Vec, VecDestroy > _temporary;
    bool _initialized = false;
};
}  // namespace KARMA

#endif
