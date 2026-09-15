#include "karma_physics_bc_manager.h"

#include <type_traits>

static_assert( std::is_same_v< decltype( static_cast< KARMABoundaryCondition* ( KARMAPhysicsBCManager::* )(
                                           unsigned, unsigned ) const >(
                                  &KARMAPhysicsBCManager::get_neumann_condition ) ),
                               KARMABoundaryCondition* ( KARMAPhysicsBCManager::* )( unsigned, unsigned ) const > );

int main()
{
    KARMAPhysicsBCManager manager( 0 );
    auto* dirichlet = new KARMABoundaryCondition( KARMA::PHYSICS::Dirichlet );
    auto* neumann   = new KARMABoundaryCondition( KARMA::PHYSICS::Neumann );
    auto* robin     = new KARMABoundaryCondition( KARMA::PHYSICS::Robin );

    manager.add_dirichlet_condition( 1u, dirichlet );
    manager.add_neumann_condition( 2u, 0u, neumann );
    manager.add_robin_condition( 2u, 1u, robin );
    KARMABoundaryCondition* first = manager.get_dirichlet_condition( 1u );
    KARMABoundaryCondition* second = manager.get_neumann_condition( 2u, 0u );
    KARMABoundaryCondition* third = manager.get_robin_condition( 2u, 1u );
    manager.print_conditions();
    return first && second && third && manager.n_dirichlet_nodes() == 1 &&
                   manager.n_neumann_nodes() == 1 && manager.n_robin_nodes() == 1
               ? 0
               : 1;
}
