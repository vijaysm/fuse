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


#ifndef __petsc_snes_utils_h__
#define __petsc_snes_utils_h__

#include "petscsnes.h"

/*
 *  print_snes_converged_reason - Print the convergence reason based on the value passed in
 *
 *  Input Parameters:
 *  reason - SNESConvergedReason enum containing the convergence reason of Newton method
 */
void print_snes_converged_reason(SNESConvergedReason reason) ;
 
/*
     snes_jacobian_compare - Tests whether a hand computed Jacobian 
     matches one compute via finite differences.
*/
PetscErrorCode snes_jacobian_compare(SNES snes) ;

#endif // #define __petsc_snes_utils_h__
