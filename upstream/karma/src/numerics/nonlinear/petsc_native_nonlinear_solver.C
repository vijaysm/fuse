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


// Local Includes
#include "karma.h"
#include "karma_nonlinear_operator.h"
#include "petsc_native_linear_solver.h"
#include "petsc_native_nonlinear_solver.h"

template <typename T>
void PetscNativeNonlinearSolver<T>::clear ()
{
  if (this->initialized())
    {
      this->_is_initialized = false;

      int ierr=0;

      ierr = SNESDestroy(&_snes);     CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
}

template <typename T>
void PetscNativeNonlinearSolver<T>::init ()
{  
	// Initialize the data structures if not done so already.
	if (!this->initialized())
	{
		this->_is_initialized = true;

		int ierr=0;

		ierr = SNESCreate(KARMA_COMM_WORLD,&_snes);       CHKERRABORT(KARMA_COMM_WORLD,ierr);

		// set a monitor to output the progress for the nonlinear solve
		ierr = SNESMonitorSet (_snes, __karma_simple_petsc_snes_monitor, this, PETSC_NULL);
			 CHKERRABORT(KARMA_COMM_WORLD,ierr);

		ierr = SNESAppendOptionsPrefix(_snes, Parent::_name.c_str()) ;  CHKERRABORT(KARMA_COMM_WORLD,ierr);

// 		if(this->_preconditioner)
// 		{
// 			KSP ksp;	 
// 			ierr = SNESGetKSP (_snes, &ksp);
// 				   CHKERRABORT(KARMA_COMM_WORLD,ierr);
// 			PC pc;
// 			ierr = KSPGetPC(ksp,&pc);
// 				   CHKERRABORT(KARMA_COMM_WORLD,ierr);
// 				   
// 			PCSetType(pc, PCSHELL);
// 			PCShellSetContext(pc,(void*)this->_preconditioner);
// 
// 			//Re-Use the shell functions from petsc_linear_solver
// 			KARMAPreconditioner::set_shell_routine_references(pc) ;
// 		}
	}
}


// void PetscNativeNonlinearSolver::attach_preconditioner(KARMAPreconditioner * preconditioner)
// {
// 	if(this->_is_initialized)
// 	{
// 		PetscPrintf(KARMA_COMM_WORLD,"\nPreconditioner must be attached before the solver is initialized!");
// 		karma_error();
// 	}
// 
// 	_preconditioner = preconditioner;
// }

template <typename T>
void PetscNativeNonlinearSolver<T>::set_discrete_references(VectorType& res, 	// Residual vector
				MatrixType& jac	  // System Jacobian Matrix
		)
{
	this->set_discrete_references(res, jac, jac) ;
}

template <typename T>
void PetscNativeNonlinearSolver<T>::set_discrete_references(VectorType& res, 	// Residual vector
				MatrixType& jac,	  // System Jacobian Matrix,
				MatrixType& prec	  // System Preconditioner Matrix
		)
{
	PetscErrorCode ierr = 0 ;
	
	ierr = SNESSetFunction (_snes, res.vec(), residual, this);
		 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = SNESSetJacobian (_snes, jac.mat(), prec.mat(), jacobian, this);
		 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// @todo
	// NOTE:
	// should probably consider 3 different cases for preconditioner
	// 1) prec = jac and matrix is allocated and is in memory; not matrix free
	//		performance should be good if ICC is used for preconditioning linear solves
	// 2) jac = mfree ; prec = mfree ; completely matrix free; 
	//		performance will suck for this but is useful for comparison and testing
	// 3) jac = mfree ; prec = approximate jacobian; 
	//		a shell PC is provided; user provides a preconditioner that represents jacobian
// 	if(this->_preconditioner)
// 		this->_preconditioner->set_matrix(prec);

}

template <typename T>
void PetscNativeNonlinearSolver<T>::setup()
{
	KARMA_LOG_PUSH("setup()", "PetscNativeNonlinearSolver");
	
	int ierr=0;
	
	// let us use Inexact Newton method by default; 
	if (_is_inexact_newton)
	{
		// this is the Eisenstat-Walker scheme where the linear tolerance changes based on nonlinear residual norm
		ierr = SNESKSPSetUseEW(_snes, PETSC_TRUE) ;	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}

	// Have the Krylov subspace method use our good initial guess rather than 0
	KSP ksp;	 
	ierr = SNESGetKSP (_snes, &ksp);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set the tolerances for the iterative solver.  Use the user-supplied
	// tolerance for the relative residual & leave the others at default values
	ierr = KSPSetTolerances (ksp, this->initial_linear_tolerance, PETSC_DEFAULT,
						   PETSC_DEFAULT, this->max_linear_iterations);
		 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set the tolerances for the non-linear solver.
	ierr = SNESSetTolerances(_snes, this->_absolute_tolerance, this->_relative_tolerance,
						   this->absolute_step_tolerance, this->_max_iterations, this->max_function_evaluations);
		 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	//Pull in command-line options to allow user to set options to override defaults
	ierr = SNESSetFromOptions(_snes);    CHKERRABORT(KARMA_COMM_WORLD,ierr);

	KARMA_LOG_POP("setup()", "PetscNativeNonlinearSolver");
}

template <typename T>
std::pair<unsigned int, Real> 
PetscNativeNonlinearSolver<T>::solve (VectorType& x,    // Solution vector
				VectorType& res, 	// Residual vector
				MatrixType& jac,	  // System Jacobian Matrix
				VectorType* rhs
				)
{
	this->set_discrete_references(res, jac) ;
	
	return this->solve(x, rhs) ;
}

template <typename T>
std::pair<unsigned int, Real> 
PetscNativeNonlinearSolver<T>::solve (VectorType& x,    // Solution vector
				VectorType& res, 	// Residual vector
				MatrixType& jac,	  // System Jacobian Matrix
				MatrixType& prec,	  // System Preconditioner Matrix
				VectorType* rhs
				)
{
	this->set_discrete_references(res, jac, prec) ;
	
	return this->solve(x, rhs) ;
}

template <typename T>
std::pair<unsigned int, Real> 
PetscNativeNonlinearSolver<T>::solve (VectorType& x, VectorType* rhs)
{
	KARMA_LOG_PUSH("solve()", "PetscNativeNonlinearSolver");
	
	PetscErrorCode ierr = 0 ;
	
	// @todo check if user has called setup for the res, jac pair
	setup() ;
	
	KSP _ksp ;
	ierr = SNESGetKSP (_snes, &_ksp);  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = KSPSetInitialGuessNonzero (_ksp, PETSC_TRUE);  CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Hopefully, x has been preset by user and is closer to the true nonlinear solution
	if (rhs == NULL)
	{
		ierr = SNESSolve (_snes, PETSC_NULL, x.vec());	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	else
	{
		karma_assert(rhs->size() == x.size()) ;
		
		ierr = SNESSolve (_snes, rhs->vec(), x.vec());	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}

	SNESConvergedReason reason;
	SNESGetConvergedReason(_snes,&reason);

	// Based on Petsc 2.3.3 documentation all diverged reasons are negative
	this->_is_converged = reason >= 0;

	int n_iterations =0;
	ierr = SNESGetIterationNumber(_snes, &n_iterations);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	double fnorm =0.0;
	ierr = SNESGetFunctionNorm(_snes, &fnorm);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	KARMA_LOG_POP("solve()", "PetscNativeNonlinearSolver");
	
	// return the # of its. and the final residual norm.
	return std::make_pair(n_iterations, fnorm);
}



// static methods : different wrapper functions for SNES monitor, operator action mimicking

//-------------------------------------------------------------------
// this function is called by PETSc at the end of each nonlinear step
template <typename T>
PetscErrorCode  PetscNativeNonlinearSolver<T>::__karma_simple_petsc_snes_monitor (SNES, PetscInt its, PetscReal fnorm, void *)
{
	KARMA_LOG(NONE) << "  NL step " << its << std::scientific << ", |residual|_2 = " << fnorm ;

	return 0;
}



//---------------------------------------------------------------
// this function is called by PETSc to evaluate the residual at X
template <typename T>
PetscErrorCode PetscNativeNonlinearSolver<T>::__karma_petsc_snes_residual (SNES, Vec x, Vec r, void *ctx)
{
	int ierr=0;

	karma_assert (x   != NULL);
	karma_assert (r   != NULL);
	karma_assert (ctx != NULL);

	PetscNativeNonlinearSolver* solver = static_cast<PetscNativeNonlinearSolver*> (ctx);
	const KARMANonlinearOperator& nl_op = solver->get_operator() ;

	VectorType X_global(x), R(r);

	nl_op.residual (X_global, R, nl_op.get_params());

	R.close();

	return ierr;
}



//---------------------------------------------------------------
// this function is called by PETSc to evaluate the Jacobian at X
template <typename T>
PetscErrorCode  PetscNativeNonlinearSolver<T>::__karma_petsc_snes_jacobian (SNES, Vec x, Mat *jac, Mat *pc, MatStructure *msflag, void *ctx)
{
	int ierr=0;

	karma_assert (x   != NULL);
	karma_assert (jac   != NULL);
	karma_assert (ctx != NULL);

	PetscNativeNonlinearSolver* solver = static_cast<PetscNativeNonlinearSolver*> (ctx);
	const KARMANonlinearOperator& nl_op = solver->get_operator() ;

	VectorType X_global(x) ;
	MatrixType Jac(*jac), PC(*pc) ;

	// if a routine to compute matrix is available, call it and update jac
	if (nl_op.is_matrix())
	{
		nl_op.matrix (X_global, PC, nl_op.get_params());

		*msflag = SAME_NONZERO_PATTERN;
	}
	else
	{
		// matrix creation routine is unavailable

		// check if jac is a shell matrix and if the operator has matvec operation defined
		bool is_valid = nl_op.is_matvec() ;

		// const MatType type ;
#if PETSC_VERSION_LESS_THAN(3,0,0)
		MatType type ;
		ierr = MatGetType(*jac, &type) ;
#else
		const MatType type ;
		ierr = MatGetType(*jac, &type) ;
#endif
		if (!strcmp(type, "shell"))
{
	is_valid = is_valid && true ;
}
		else
			is_valid = false ;

		if (!is_valid)
			karma_error() ;

		*msflag = DIFFERENT_NONZERO_PATTERN;
	}

	PC.close();
	Jac.close();

	return ierr;

}




//------------------------------------------------------------------
// Explicit instantiations
template class PetscNativeNonlinearSolver<PetscScalar>;
