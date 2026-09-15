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


#ifndef __karma_wrappers_h__
#define __karma_wrappers_h__

#include "petsc.h"

#define MASS_STAGE_SOLVE

namespace KARMA
{ 
	// Compute the residual for all physics through the interface ; Only S.S
	PetscErrorCode karma_wrapper_petsc_snes_residual (SNES , Vec , Vec , void *) ;

	// Compute the global jacobian approximation for all physics through the interface ; Only S.S
	PetscErrorCode karma_wrapper_petsc_snes_jacobian (SNES , Vec , Mat *, Mat *, MatStructure *, void *) ;

	// Compute the residual for all physics through the interface ; Temporal case
	PetscErrorCode karma_wrapper_temporal_residual (SNES , Vec , Vec , void *) ;

	// Compute the global jacobian approximation for all physics through the interface ; Temporal case
	PetscErrorCode karma_wrapper_temporal_jacobian (SNES , Vec , Mat *, Mat *, MatStructure *, void *) ;
	
	// A dummy placeholder to pass to SNES when both Jac and Prec are handled through a shell matrix !
	PetscErrorCode karma_wrapper_temporal_jacobian_dummy (SNES , Vec , Mat *, Mat *, MatStructure *, void *) ;
	
	// Solve the preconditioner and multiply a vector to it; y = inv(P)*x
	PetscErrorCode karma_wrapper_petsc_shell_preconditioner_solve (PC , Vec x, Vec y) ;

	// Multiply the jacobian and a given vector ; y = J*x
	PetscErrorCode karma_wrapper_petsc_shell_jacobian_matvec (Mat , Vec x, Vec y) ;
	
	// Multiply the preconditioner and a given vector ; y = P*x
	PetscErrorCode karma_wrapper_petsc_shell_preconditioner_matvec (Mat , Vec x, Vec y) ;

	// Solve the preconditioner and multiply a vector to it; y = inv(J)*x
	PetscErrorCode karma_wrapper_petsc_shell_jacobian_solve (PC , Vec x, Vec y) ;
	
	PetscErrorCode karma_wrapper_petsc_shell_linear_operations (Mat A, Vec x, Vec y) ;

}

#endif // #define __karma_wrappers_h__
