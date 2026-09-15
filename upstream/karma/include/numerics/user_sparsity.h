// KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
// Copyright (C) 2007-2011  Vijay S Mahadevan
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published
// by the Free Software Foundation; either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//

#ifndef __user_sparsity_h__
#define __user_sparsity_h__

// C++ includes
#include <petscsys.h>

#include <cstddef>
#include <vector>

/**
 * This class provides a user sparsity information container.
 * i.e., when the user wants a matrix to be created using a
 * specific sparsity, this can serve as a generic initializer
 * for creating a matrix or a vector in \p ISystem.
 */

// ------------------------------------------------------------
// UserSparsity class definition
struct UserSparsity
{
  public:

    /**
     * Helper method to check if all the necessary information is available to infer sparsity.
     */
    bool is_ready() const;

    /**
     * The vital information necessary to describe a discrete system, namely
     * the global and the local degrees-of-freedom.
     */
    PetscInt global_dofs = 0;
    PetscInt local_dofs  = 0;

    /**
     * The crude estimate of the number of non-zeros in the current processor and
     * belonging to other processors.
     */
    PetscInt nnz_per_row = 0;
    PetscInt onz_per_row = 0;

    /**
     * A reference to the sparsity distribution that is needed to
     * initialize matrices in the current system. Usually, this is set by
     * the user based on a connectivity map depending on the spatial discretization
     * and mesh distribution.
     */
    std::vector< PetscInt > n_nz;
    std::vector< PetscInt > o_nz;

    /**
     * Keep a reference to the send_list that can map the dependency of the local DOFs
     * on non-local DOFs residing on other processors.
     */
    std::vector< PetscInt > send_list;
};

// Inline methods

inline bool UserSparsity::is_ready() const
{
    if( global_dofs <= 0 || local_dofs < 0 || local_dofs > global_dofs ) return false;
    return n_nz.size() == static_cast< std::size_t >( local_dofs ) &&
           o_nz.size() == static_cast< std::size_t >( local_dofs );
}

#endif  // #define __user_sparsity_h__
