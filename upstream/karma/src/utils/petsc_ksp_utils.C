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


#include "karma.h"
#include "petsc_ksp_utils.h"

/*
   print_ksp_converged_reason - Print the convergence reason based on the value passed in

   Input Parameters:
   reason - KSPConvergedReason enum containing the convergence reason of Linear solver
 */
#undef __FUNCT__  
#define __FUNCT__ "print_ksp_converged_reason"
void print_ksp_converged_reason(KSPConvergedReason reason)
{  
  /*
	typedef enum {// converged 
              KSP_CONVERGED_RTOL               =  2,
              KSP_CONVERGED_ATOL               =  3,
              KSP_CONVERGED_ITS                =  4,
              KSP_CONVERGED_CG_NEG_CURVE       =  5,
              KSP_CONVERGED_CG_CONSTRAINED     =  6,
              KSP_CONVERGED_STEP_LENGTH        =  7,
              KSP_CONVERGED_HAPPY_BREAKDOWN    =  8,
              // diverged 
              KSP_DIVERGED_NULL                = -2,
              KSP_DIVERGED_ITS                 = -3,
              KSP_DIVERGED_DTOL                = -4,
              KSP_DIVERGED_BREAKDOWN           = -5,
              KSP_DIVERGED_BREAKDOWN_BICG      = -6,
              KSP_DIVERGED_NONSYMMETRIC        = -7,
              KSP_DIVERGED_INDEFINITE_PC       = -8,
              KSP_DIVERGED_NAN                 = -9,
              KSP_DIVERGED_INDEFINITE_MAT      = -10,
 
              KSP_CONVERGED_ITERATING          =  0} KSPConvergedReason;
    */

	switch (reason)
	{
		case KSP_CONVERGED_ITERATING:
		{
			// This flag is returned if you call KSPGetConvergedReason() while the KSPSolve() is still running.
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver might be un-converged, still iterating.\n");
			break;
		}
		case KSP_CONVERGED_ATOL:
		{
			// 2-norm(F) <= abstol
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver converged, absolute tolerance satisfied.\n" ) ;
			break;
		}
		case KSP_CONVERGED_RTOL:
		{
			// 2-norm(F) <= rtol*2-norm(F(x_0)) where x_0 is the initial guess
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver converged, relative tolerance satisfied.\n");
			break;
		}
		case KSP_CONVERGED_ITS:
		{
			// used by the preonly preconditioner that always uses ONE iteration, or when the KSPSkipConverged() convergence
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver converged, when using PREONLY solver.\n");
			break;
		}
		case KSP_CONVERGED_HAPPY_BREAKDOWN:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver converged, happy breakdown with exact convergence.\n");
			break;
		}
		   
		  // Divergence
		case KSP_DIVERGED_ITS:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, max no. of iterations reached.\n") ;
			break;
		}
		case KSP_DIVERGED_DTOL:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, residual norm increase by dtol (default 1.e5).\n") ;
			break;
		}
		case KSP_DIVERGED_NAN:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, residual norm is NaN.\n") ;
			break;
		}
		case KSP_DIVERGED_BREAKDOWN:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, generic breakdown in the method.\n") ;
			break;
		}
		case KSP_DIVERGED_INDEFINITE_PC:
		{
			// It appears the preconditioner is indefinite (has both positive and negative eigenvalues) 
			// and this Krylov method (KSPCG) requires it to be positive definite
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, indefinite preconditioner in use.\n") ;
			break;
		}
		case KSP_DIVERGED_INDEFINITE_MAT:
		{
			// It appears the linear system being solved is indefinite (has both positive and negative eigenvalues) 
			// and this Krylov method (KSPCG) requires it to be positive definite
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, indefinite linear system.\n") ;
			break;
		}
		case KSP_DIVERGED_NONSYMMETRIC:
		{
			// It appears the operator or preconditioner is not symmetric and this 
			// Krylov method (KSPCG, KSPMINRES, KSPCR) requires symmetry
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, non-symmetric operator used.\n") ;
			break;
		}
		
		default:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Unknown/unsupported con(di)vergence reason for linear solver: %D\n", reason);
		}
    }
}

