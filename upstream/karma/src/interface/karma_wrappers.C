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


#include <typeinfo>

#include "karma.h"
#include "karma_context.h"
#include "karma_interface.h"
#include "karma_wrappers.h"
#include "karma_linear_solver.h"
#include "karma_preconditioner.h"
#include "temporal_integrator.h"

#include "karma_vector.h"
#include "karma_matrix.h"

// Petsc nonlinear classes
#include "petscsnes.h"
#include "petsc_vec_utils.h"

#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_petsc_snes_residual"
PetscErrorCode KARMA::karma_wrapper_petsc_snes_residual (SNES , Vec x, Vec r, void *)
{
	karma_assert (x   != NULL);
	karma_assert (r   != NULL);
	
	PetscErrorCode ierr ;
	
#ifdef USE_PETSC_BLOCK_EXTENSIONS
	KARMABlockVector X_global(x), R(r);
#else
	KARMAVector X_global(x), R(r);
#endif
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	
	ierr = interface->system_residual (X_global, R);	CHKERRQ(ierr);
	
	PetscFunctionReturn(ierr);
}


#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_petsc_snes_jacobian"
PetscErrorCode KARMA::karma_wrapper_petsc_snes_jacobian(SNES , Vec x, Mat *jac, Mat *prec, MatStructure *msflag, void *)
{
	karma_assert (x   != NULL);
	karma_assert (prec   != NULL);
	
	PetscErrorCode ierr ;
	
#ifdef USE_PETSC_BLOCK_EXTENSIONS
	KARMABlockVector X_global(x);
#else
	KARMAVector X_global(x);
#endif
	KARMAInterface* interface = KARMAInterface::get_interface  () ;
	KARMAContext* _CTX = interface->get_context() ;
	
	bool mat_available = true ;
	if (_CTX->is_jacobian_matrix_free ())
	{
		if (_CTX->is_preconditioner_matrix_free ())
		{
			// If this is a compeltely matrix-free scheme, let us just close the 
			// matrices and return
			mat_available = false ;
			
			ierr = MatAssemblyBegin(*prec, MAT_FINAL_ASSEMBLY) ; 	CHKERRQ(ierr);
			ierr = MatAssemblyEnd(*prec, MAT_FINAL_ASSEMBLY) ;	CHKERRQ(ierr);
		}
	}
	
	if  (mat_available)
	{
#ifdef USE_PETSC_BLOCK_EXTENSIONS
		// by default - always consider only the preconditioner matrix.
		KARMABlockMatrix Jac_global(*prec) ;
#else
		// by default - always consider only the preconditioner matrix.
		KARMAMatrix Jac_global(*prec) ;
#endif
		ierr = interface->jacobian_matrix (X_global, Jac_global);	CHKERRQ(ierr);
	}
	
	// If performing Picard iterations, then Jacobian=Preconditioner. Use
	// algebraic/numerical preconditioners such as ILU/ICC on the matrix.
	// If matrices are different, then Jacobian is complicated and so use 
	// MF-FD for it and assemble only the preconditioner.
	if (jac == prec)
		*msflag = SAME_NONZERO_PATTERN;
	else
		*msflag = DIFFERENT_NONZERO_PATTERN;

	PetscFunctionReturn(ierr);
}


#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_temporal_residual"
PetscErrorCode KARMA::karma_wrapper_temporal_residual (SNES snes, Vec x, Vec r, void *)
{
	PetscErrorCode 	ierr = 0;
	TemporalIntegrator* integrator = NULL ;
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	KARMAContext* _CTX = interface->get_context() ;
	
	const bool is_transient = interface->is_transient() ;
	
#ifdef MASS_STAGE_SOLVE
	// first, get the SS residual for all physics and store it in "r"
	ierr = karma_wrapper_petsc_snes_residual (snes, x, r, NULL) ;	CHKERRQ(ierr);
#else
	if (is_transient)
	{
		// get the temporal integrator reference from the context
		ierr = SNESGetApplicationContext(snes, (void **)&integrator) ;CHKERRQ(ierr);
		
		// TODO: This is completely unoptimized. Since it is at the core  of the solver,
		// any alternative implementation should provide good speedup !
		
		ierr = VecCopy(x, _CTX->Work_1->vec());CHKERRQ(ierr);
		_CTX->Work_1->close() ;
		
		Vec& current_solution = integrator->get_current_solution();
		ierr = VecAXPY(_CTX->Work_1->vec(), 1.0, current_solution) ;CHKERRQ(ierr);
		_CTX->Work_1->close() ;
		
		// first, get the SS residual for all physics and store it in "_CTX->Work_1"
		ierr = karma_wrapper_petsc_snes_residual (snes, _CTX->Work_1->vec(), r, NULL) ;	CHKERRQ(ierr);
	}
	else
	{
		// first, get the SS residual for all physics and store it in "r"
		ierr = karma_wrapper_petsc_snes_residual (snes, x, r, NULL) ;	CHKERRQ(ierr);
	}
#endif

	// now if we are doing time-dependent problem, need to find the temporal residual 
	// for simplified Newton iteration.
	if (is_transient)
	{
	
#ifdef MASS_STAGE_SOLVE
		ierr = SNESGetApplicationContext(snes, (void **)&integrator) ;CHKERRQ(ierr);
#endif

		double **A = integrator->Acoeff () ;
	
		int currstage = integrator->current_stage() ;
		
#ifdef MASS_STAGE_SOLVE
		{
			// Zi = un + delt*sum(aij*M^{-1}*F(Zj), j=1..i) 
			// Use the definition F(trans) = M*Z - delt*aii*F(Z)
#ifdef USE_PETSC_BLOCK_EXTENSIONS
			KARMABlockVector vec3(x) ;
#else
			KARMAVector vec3(x) ;
#endif
			ierr = interface->mass_multiply (vec3, *_CTX->Work_1) ;CHKERRQ(ierr);
			
			ierr = VecAYPX(r, -(_CTX->delt * A[currstage][currstage]), _CTX->Work_1->vec()) ;	CHKERRQ(ierr);
		}
#else
		{
			// M*Zi = delt*sum(aij*F(Zj), j=1..i) 
			// Use the definition F(trans) = M*Z - delt*aii*F(Z)
#ifdef USE_PETSC_BLOCK_EXTENSIONS
			KARMABlockVector vec3(x) ;
#else
			KARMAVector vec3(x) ;
#endif
			ierr = interface->mass_multiply (vec3, *_CTX->Work_1) ;CHKERRQ(ierr);
			
			ierr = VecAYPX(r, -(_CTX->delt * A[currstage][currstage]), _CTX->Work_1->vec()) ;	CHKERRQ(ierr);
		}
#endif
		
		ierr = VecAssemblyBegin(r) ;	CHKERRQ(ierr);
		ierr = VecAssemblyEnd(r) ;	CHKERRQ(ierr);
		
		integrator = NULL ;
	}
	
	interface = NULL ;
	_CTX = NULL ;
	
	return ierr ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_temporal_jacobian"
PetscErrorCode KARMA::karma_wrapper_temporal_jacobian_dummy (SNES , Vec , Mat *jac, Mat *prec, MatStructure *, void *)
{
	karma_assert(jac != NULL) ;
	karma_assert(prec != NULL) ;
	
	PetscErrorCode ierr = 0;
	
	ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyBegin(*prec,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	ierr = MatAssemblyEnd(*prec,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	
	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_temporal_jacobian"
PetscErrorCode KARMA::karma_wrapper_temporal_jacobian (SNES snes, Vec x, Mat *jac, Mat *prec, MatStructure *msflag, void *)
{
	karma_assert(jac != NULL) ;
	karma_assert(prec != NULL) ;
	
	PetscErrorCode ierr = 0;
		
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	KARMAContext* _CTX = interface->get_context() ;
		
	if (_CTX->is_jacobian_matrix_free ())
	{
		if (_CTX->is_preconditioner_matrix_free ())
		{
			// This is a dummy jacobian wrapper. User is responsible for solving
			// the nonlienar system using custom "SHELL" preconditioning.
			// Note: This is for advanced users only and if you do not understand
			// the implementation, avoid the usage.
			ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
			ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
			
			ierr = MatAssemblyBegin(*prec,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
			ierr = MatAssemblyEnd(*prec,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
			
			return ierr ;
		}
	}
	
	// create an identity matrix for the jacobian ; use it to test snes_mf_operator with snes_mf ! They should behave the same
#ifdef USE_PETSC_BLOCK_EXTENSIONS
	KARMABlockMatrix P(*prec) ;
#else
	KARMAMatrix P(*prec) ;
#endif

	// Jacobian = Preconditioner = SS Jacobian approximation
	ierr = karma_wrapper_petsc_snes_jacobian (snes, x, jac, prec, msflag, NULL) ;	CHKERRQ(ierr);

	if (interface->is_transient())
	{
		TemporalIntegrator* integrator = NULL ;
		
		// by default - always consider only the preconditioner ; 
		// since I will never build the matrix for the jacobian, there is no point ;
		
		ierr = SNESGetApplicationContext(snes, (void **)&integrator) ;CHKERRQ(ierr);

		double **A = integrator->Acoeff () ;
		
		int currstage = integrator->current_stage() ;
		
		{
			// J(transient) = -delt * aii * J(SS)
			//ierr = MatScale(*prec, -(_CTX->delt * A[currstage][currstage])) ;	CHKERRQ(ierr);
			
			// J(transient) = M - delt * aii * J(SS)
			// TODO: Currently, this the preconditioner is Block-Jacobi, the sparsity pattern
			// are the same for both Mass and Preconditioner. But if the preconditioner used
			// is Block-Gauss-Seidel or something else, this could be a problem.
			// Need to chekc this further.
			ierr = interface->mass_matrix_subtract (P, -(_CTX->delt * A[currstage][currstage])) ;CHKERRQ(ierr);
			
		}
		
		A = NULL ;
		integrator = NULL ;
	}
	
	P.close() ;
	if (jac != prec)
	{
		ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
		ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
	}
	
	interface = NULL ;	
	_CTX = NULL ;
	
	PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_petsc_shell_preconditioner_matvec"
PetscErrorCode KARMA::karma_wrapper_petsc_shell_preconditioner_matvec (Mat P, Vec x, Vec y)
{
	karma_assert (x != NULL);
	karma_assert (y != NULL);
	karma_assert (P != NULL);
	
#ifdef USE_PETSC_BLOCK_EXTENSIONS
	KARMABlockVector X_global(x);
	KARMABlockVector Y_global(y);
#else
	KARMAVector X_global(x);
	KARMAVector Y_global(y);
#endif
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	
	// compute the operation of the preconditioner on a vector X
	PetscErrorCode ierr = interface->preconditioner_matvec (X_global, Y_global);	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_petsc_shell_preconditioner_solve"
PetscErrorCode KARMA::karma_wrapper_petsc_shell_preconditioner_solve (PC pc, Vec x, Vec y)
{
	karma_assert (x != NULL);
	karma_assert (y != NULL);
	
#ifdef USE_PETSC_BLOCK_EXTENSIONS
	KARMABlockVector X_global(x);
	KARMABlockVector Y_global(y);
#else
	KARMAVector X_global(x);
	KARMAVector Y_global(y);
#endif
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	
	// compute the operation of the preconditioner solve on a vector X
	PetscErrorCode ierr = interface->preconditioner_solve (X_global, Y_global);	CHKERRQ(ierr);

	PetscFunctionReturn(0);
}



#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_petsc_shell_jacobian_matvec"
PetscErrorCode KARMA::karma_wrapper_petsc_shell_jacobian_matvec (Mat J, Vec x, Vec y)
{
	// Use the idea that you can mimic the product of the jacobian on a vector using MatShell matrix representation
	karma_assert (x != NULL);
	karma_assert (y != NULL);
	karma_assert (J != NULL);
	
#ifdef USE_PETSC_BLOCK_EXTENSIONS
	KARMABlockVector X_global(x);
	KARMABlockVector Y_global(y);
#else
	KARMAVector X_global(x);
	KARMAVector Y_global(y);
#endif
	
	PetscErrorCode ierr = 0 ;
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	
	// compute product of SS Jacobian matrix with the given matrix
	interface->jacobian_matvec (X_global, Y_global);
	
	KARMAContext* _CTX = interface->get_context() ;
		
	if (interface->is_transient())
	{
		TemporalIntegrator* integrator = NULL ;
		ierr = MatShellGetContext(J, (void**)&integrator);	CHKERRQ(ierr);
		
		karma_assert (integrator != NULL);
		
		double **A = integrator->Acoeff () ;
	
		int currstage = integrator->current_stage() ;
		
		// Use the definition J(trans)*v = M*v - delt*aii*J*v
#ifdef USE_PETSC_BLOCK_EXTENSIONS
		KARMABlockVector vec3(x);
#else
		KARMAVector vec3(x) ;
#endif
	
		ierr = interface->mass_multiply (vec3, *_CTX->Work_1) ;CHKERRQ(ierr);
		
		ierr = VecAYPX(y, -(_CTX->delt * A[currstage][currstage]), _CTX->Work_1->vec()) ;	CHKERRQ(ierr);
			
	}
	
	Y_global.close() ;

	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_petsc_shell_jacobian_solve"
PetscErrorCode KARMA::karma_wrapper_petsc_shell_jacobian_solve (PC pc, Vec x, Vec y)
{
	karma_error() ;
	
	karma_assert (x   != NULL);
	karma_assert (y   != NULL);
	
#ifdef USE_PETSC_BLOCK_EXTENSIONS
	KARMABlockVector X_global(x);
	KARMABlockVector Y_global(y);
#else
	KARMAVector X_global(x);
	KARMAVector Y_global(y);
#endif
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	
	// compute the operation of the jacobian solve on a vector X
	PetscErrorCode ierr = interface->jacobian_solve (X_global, Y_global);	CHKERRQ(ierr);
	
	Y_global.close() ;

	PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KARMA::karma_wrapper_petsc_shell_linear_operations "
PetscErrorCode KARMA::karma_wrapper_petsc_shell_linear_operations (Mat A, Vec x, Vec y)
{
	karma_assert (x   != NULL);
	karma_assert (y   != NULL);
	
	KARMAVector X_global(x);
	KARMAVector Y_global(y);
	
	KARMAParameters *ctx ;
	PetscErrorCode ierr = MatShellGetContext(A, (void**)&ctx) ;

	KARMALinearOperator* optr = ctx->get<KARMALinearOperator*>("operator") ;
	const KARMA::KARMALinearOperations optr_function = ctx->get<KARMA::KARMALinearOperations>("operation") ;

	if (optr_function == KARMA::Matrix)
	{
		const KARMA::KARMALinearOperations sub_optr_function = ctx->get<KARMA::KARMALinearOperations>("sub_operation") ;
		bool recompute_matrix = ctx->get<bool>("recompute_matrix") ;

		if (!optr->is_matrix())
		{
			KARMA_LOG(ERROR) << "No matvec operation found for supplied LinearOperator." ;
			karma_error() ;
		}

		KARMAPhysicsBase* physics = ctx->get<KARMAPhysicsBase*>("physics") ;
		
		KARMAMatrix& linsystem = physics->get_jacobian() ;

		// now recompute the matrix as requested
		if (recompute_matrix)
		{
			KARMAVector& solution = physics->get_solution() ;

			ierr = optr->matrix(solution, linsystem, *ctx, NULL) ;	CHKERRQ(ierr);
		}
		
		if (optr_function == KARMA::MatVec)
		{
			// since we have the matrix, just perform a vector multiply
			ierr = MatMult(linsystem.mat(), x, y); CHKERRQ(ierr);
		}
		else
		{
			KARMALinearSolver linsolver ;
			linsolver.name() = "wrapper_" ;
			linsolver.same_preconditioner = true ;
			KSPSetInitialGuessNonzero(linsolver.ksp(), PETSC_TRUE) ;

			// should make the tolerance and iteration specifiable by the user
			linsolver.solve(linsystem, linsystem, X_global, Y_global, 1e-8, 20) ;
			
			// we will rely on the inbuilt matrix solver for now.
			// if this gets expensive, maybe an alternative is to store a ksp context
			// and then appropriately solve that systme.
			// ierr = MatSolve(linsystem.mat(), x, y); CHKERRQ(ierr);
		}
	}
	else
	{
		if (optr_function == KARMA::MatVec)
		{
			if (!optr->is_matvec())
			{
				KARMA_LOG(ERROR) << "No matvec operation found for supplied LinearOperator." ;
				karma_error() ;
			}
			ierr = optr->matvec(X_global, Y_global, *ctx, NULL) ;	CHKERRQ(ierr);
		}
		else if (optr_function == KARMA::Solve)
		{
			if (!optr->is_solve())
			{
				KARMA_LOG(ERROR) << "No solve operation found for supplied LinearOperator." ;
				karma_error() ;
			}
			ierr = optr->solve(X_global, Y_global, *ctx, NULL) ;	CHKERRQ(ierr);
		}
	}
	Y_global.close() ;

	PetscFunctionReturn(0);
}


