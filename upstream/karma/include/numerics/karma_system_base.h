#ifndef __karma_system_base_h__
#define __karma_system_base_h__

#include "karma_basis.h"

#include <petscmat.h>
#include <petscvec.h>

#include <string>
#include <string_view>

namespace KARMA
{
struct VariableSpec;

class AbstractSystem
{
  public:
    virtual ~AbstractSystem()                          = default;
    AbstractSystem( const AbstractSystem& )            = delete;
    AbstractSystem& operator=( const AbstractSystem& ) = delete;

    [[nodiscard]] virtual const std::string& name() const noexcept                = 0;
    virtual unsigned add_variable( VariableSpec variable )                        = 0;
    [[nodiscard]] virtual unsigned n_variables() const noexcept                   = 0;
    [[nodiscard]] virtual const VariableSpec& variable( unsigned index ) const    = 0;
    [[nodiscard]] virtual unsigned variable_number( std::string_view name ) const = 0;
    [[nodiscard]] virtual bool has_variable( std::string_view name ) const        = 0;
    virtual void init()                                                           = 0;
    [[nodiscard]] virtual bool is_initialized() const noexcept                    = 0;
    [[nodiscard]] virtual PetscInt n_dofs() const noexcept                        = 0;
    [[nodiscard]] virtual PetscInt n_local_dofs() const noexcept                  = 0;
    [[nodiscard]] virtual Vec create_vector() const                               = 0;
    [[nodiscard]] virtual Mat create_matrix() const                               = 0;

    [[nodiscard]] virtual Vec solution() const noexcept         = 0;
    [[nodiscard]] virtual Vec current_solution() const noexcept = 0;
    [[nodiscard]] virtual Vec old_solution() const noexcept     = 0;
    [[nodiscard]] virtual Vec local_solution() const noexcept   = 0;
    [[nodiscard]] virtual Vec temporary_vector() const noexcept = 0;

  protected:
    AbstractSystem() = default;
};
}  // namespace KARMA

#endif
