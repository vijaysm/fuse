// KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
#ifndef __karma_system_h__
#define __karma_system_h__

#include "karma_dof_map.h"
#include "karma_mesh.h"
#include "karma_system_base.h"
#include "user_sparsity.h"

#include <petscdm.h>
#include <petscmat.h>
#include <petscvec.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace KARMA
{
enum class Order : unsigned
{
    Constant = 0,
    First    = 1,
    Second   = 2
};

class KARMASystem final : public AbstractSystem
{
  public:
    KARMASystem( KARMAMesh& mesh, std::string_view name );
    KARMASystem( KARMAMesh& mesh, std::string_view name, KARMABasisRegistry registry );
    ~KARMASystem() override;
    KARMASystem( const KARMASystem& )            = delete;
    KARMASystem& operator=( const KARMASystem& ) = delete;
    KARMASystem( KARMASystem&& )                 = delete;
    KARMASystem& operator=( KARMASystem&& )      = delete;

    [[nodiscard]] const std::string& name() const noexcept override
    { return _name; }
    unsigned add_variable( VariableSpec variable ) override;
    unsigned add_variable( std::string_view name, Order order = Order::First,
                           FieldBasisFamily family    = FieldBasisFamily::LagrangeNodal,
                           BasisContinuity continuity = BasisContinuity::cG, unsigned components = 1 );
    [[nodiscard]] unsigned n_variables() const noexcept override
    { return static_cast< unsigned >( _variables.size() ); }
    [[nodiscard]] const VariableSpec& variable( unsigned i ) const override;
    [[nodiscard]] unsigned variable_number( std::string_view name ) const override;
    [[nodiscard]] bool has_variable( std::string_view name ) const override;

    void init() override;
    [[nodiscard]] bool is_initialized() const noexcept override
    { return _initialized; }
    [[nodiscard]] KARMAMesh& mesh() noexcept
    { return _mesh; }
    [[nodiscard]] const KARMAMesh& mesh() const noexcept
    { return _mesh; }
    [[nodiscard]] const KARMADofMap& dof_map() const noexcept
    { return _dof_map; }
    /** Collective: every rank must submit the same global element pair, or (0,0). */
    void add_element_adjacency( moab::EntityHandle first, moab::EntityHandle second );
    [[nodiscard]] const KARMABasisRegistry& basis_registry() const noexcept
    { return _basis_registry; }
    [[nodiscard]] KARMABasisRegistry& basis_registry() noexcept
    { return _basis_registry; }
    [[nodiscard]] const UserSparsity& sparsity() const noexcept
    { return _sparsity; }
    /** Transitional mesh-only DM accessor. Algebraic layout does not use it. */
    [[nodiscard]] DM dm() const noexcept
    { return nullptr; }
    [[nodiscard]] PetscInt n_dofs() const noexcept override
    { return _dof_map.n_dofs(); }
    [[nodiscard]] PetscInt n_local_dofs() const noexcept override
    { return _dof_map.n_local_dofs(); }
    [[nodiscard]] MPI_Comm communicator() const noexcept { return _mesh.communicator(); }

    [[nodiscard]] Vec create_vector() const override;
    [[nodiscard]] Mat create_matrix() const override;
    /** Collective: validates an object against this system before any output mutation. */
    void validate_vector_layout( Vec vector, const char* name ) const;
    /** Collective: validates dimensions and both local-to-global mappings. */
    void validate_matrix_layout( Mat matrix, const char* name ) const;
    [[nodiscard]] Vec solution() const noexcept override { return _solution; }
    [[nodiscard]] Vec current_solution() const noexcept override { return _current_solution; }
    [[nodiscard]] Vec old_solution() const noexcept override { return _old_solution; }
    [[nodiscard]] Vec local_solution() const noexcept override { return _local_solution; }
    [[nodiscard]] Vec temporary_vector() const noexcept override { return _temporary; }

  private:
    void refresh_sparsity();

    KARMAMesh& _mesh;
    std::string _name;
    KARMABasisRegistry _basis_registry;
    std::vector< VariableSpec > _variables;
    KARMADofMap _dof_map;
    UserSparsity _sparsity;
    Vec _solution         = nullptr;
    Vec _current_solution = nullptr;
    Vec _old_solution     = nullptr;
    Vec _local_solution   = nullptr;
    Vec _temporary        = nullptr;
    bool _initialized = false;
};
}  // namespace KARMA
#endif
