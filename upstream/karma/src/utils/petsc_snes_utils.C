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
#include "petsc_snes_utils.h"

// #include "private/snesimpl.h"

/*
   print_snes_converged_reason - Print the convergence reason based on the value passed in

   Input Parameters:
   reason - SNESConvergedReason enum containing the convergence reason of Newton method
 */
#undef __FUNCT__  
#define __FUNCT__ "print_snes_converged_reason"
void print_snes_converged_reason(SNESConvergedReason reason)
{
	PetscPrintf(KARMA_COMM_WORLD, "Nonlinear solver convergence/divergence reason: %s\n", 
				SNESConvergedReasons[reason]) ;

}


/*
     snes_jacobian_compare - Tests whether a hand computed Jacobian 
     matches one compute via finite differences.
*/
#undef __FUNCT__  
#define __FUNCT__ "snes_jacobian_compare"
PetscErrorCode snes_jacobian_compare(SNES snes)
{
	Mat            A,B;
	Vec            x, f, pert;
	PetscErrorCode ierr;
	PetscInt       i;
	MatStructure   flg;
	PetscReal      nrm,gnorm;
	PetscBool complete_print ;
	PetscScalar eps = 1e-7 ;
	PetscErrorCode (*funP)(void)  ;

	PetscFunctionBegin;

	ierr = SNESGetSolution(snes, &x) ;CHKERRQ(ierr);
	ierr = SNESGetFunction(snes, &f, PETSC_NULL, PETSC_NULL) ;CHKERRQ(ierr);
	ierr = SNESGetJacobian(snes, &A, PETSC_NULL, PETSC_NULL, PETSC_NULL) ;CHKERRQ(ierr);

	ierr = PetscOptionsHasName(0, "-main_snes_test_display", &complete_print) ;CHKERRQ(ierr);

	ierr = VecView(x, 0) ;CHKERRQ(ierr);
	const VecType vect ;
	// ierr = VecGetType(x, &vect) ;CHKERRQ(ierr);
	ierr = VecDuplicate(x, &pert) ;CHKERRQ(ierr);
	ierr = VecSet(pert,eps);CHKERRQ(ierr);

	ierr = PetscPrintf(((PetscObject)snes)->comm,"Testing hand-coded Jacobian, if the ratio is\n");CHKERRQ(ierr);
	ierr = PetscPrintf(((PetscObject)snes)->comm,"O(1.e-8), the hand-coded Jacobian is probably correct.\n");CHKERRQ(ierr);
	if (!complete_print) {
	ierr = PetscPrintf(((PetscObject)snes)->comm,"Run with -snes_test_display to show difference\n");CHKERRQ(ierr);
	ierr = PetscPrintf(((PetscObject)snes)->comm,"of hand-coded and finite difference Jacobian.\n");CHKERRQ(ierr);
	}

	// x0  = vec_sol provided by user.
	for (i=0; i<3; i++) {
	if (i == 1){	// x = x0 - eps
		ierr = VecAXPY(x, -1.0, pert);CHKERRQ(ierr);
	}
	else if (i == 2) {	// x = x0 + eps
		ierr = VecAXPY(x, 2.0, pert);CHKERRQ(ierr);
	}

	/* evaluate the function at this point because SNESDefaultComputeJacobianColor() assumes that the function has been evaluated and put into snes->vec_func */
	ierr = SNESComputeFunction(snes,x,f);CHKERRQ(ierr);

	/* compute both versions of Jacobian */
	ierr = SNESComputeJacobian(snes,x,&A,&A,&flg);CHKERRQ(ierr);
	if (!i) {ierr = MatConvert(A,MATSAME,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);}
	ierr = SNESDefaultComputeJacobian(snes,x,&B,&B,&flg,(void *)(funP));CHKERRQ(ierr);
	if (complete_print) {
	  MPI_Comm    comm;
	  PetscViewer viewer;
	  ierr = PetscPrintf(PETSC_COMM_WORLD,"Finite difference Jacobian\n");CHKERRQ(ierr);
	  ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
	  ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
	  ierr = MatView(B,viewer);CHKERRQ(ierr);
	}
	/* compare */
	ierr = MatAXPY(B,-1.0,A,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
	ierr = MatNorm(B,NORM_FROBENIUS,&nrm);CHKERRQ(ierr);
	ierr = MatNorm(A,NORM_FROBENIUS,&gnorm);CHKERRQ(ierr);
	if (complete_print) {
	  MPI_Comm    comm;
	  PetscViewer viewer;
	  ierr = PetscPrintf(((PetscObject)snes)->comm,"Hand-coded Jacobian\n");CHKERRQ(ierr);
	  ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
	  ierr = PetscViewerASCIIGetStdout(comm,&viewer);CHKERRQ(ierr);
	  ierr = MatView(A,viewer);CHKERRQ(ierr);
	}
	if (!gnorm) gnorm = 1; /* just in case */
	ierr = PetscPrintf(((PetscObject)snes)->comm,"Norm of matrix ratio %G difference %G\n",nrm/gnorm,nrm);CHKERRQ(ierr);
	}
	ierr = MatDestroy(&B);CHKERRQ(ierr);
	/*
		 Return error code cause Jacobian not good
	*/
	PetscFunctionReturn(PETSC_ERR_ARG_WRONGSTATE);
}
