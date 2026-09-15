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


// KARMA Includes
#include "karma.h"
#include "petsc_native_linear_solver.h"
#include "petsc_native_preconditioner.h"


template<typename T>
PetscErrorCode PetscNativeLinearSolver<T>::__petsc_shell_matrix_mult(Mat mat, Vec arg, Vec dest)
{
	/* Get the matrix context.  */
	PetscErrorCode ierr=0;
	void* ctx;
	
	ierr = MatShellGetContext(mat,&ctx); 	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	/* Get user linear matrix operator.  */
	KARMALinearOperator& shell_matrix = *static_cast<KARMALinearOperator *>(ctx);

	/* Make \p NumericVector instances around the vectors.  */
	VectorType arg_global(arg);
	VectorType dest_global(dest);

	/* Call the user function.  */
	shell_matrix.matvec(dest_global, arg_global, shell_matrix.get_params());

	return ierr;
}

template<typename T>
PetscErrorCode PetscNativeLinearSolver<T>::__petsc_shell_matrix_solve(PC pc, Vec arg, Vec dest)
{
	PetscErrorCode ierr=0;
	
	/* Get user linear matrix operator from the PC context.  */
	KARMALinearOperator* shell_op ;
	
	ierr = PCShellGetContext(pc, (void**)&shell_op); 	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	/* Make \p VectorType wrapper instances around the given vectors.  */
	VectorType arg_global(arg);
	VectorType dest_global(dest);

	/* Call the user function.  */
	shell_op->solve(dest_global, arg_global, shell_op->get_params());

	return ierr;
}


template<typename T>
void PetscNativeLinearSolver<T>::clear ()
{
	if (this->initialized())
	{
		this->_is_initialized = false;

		PetscErrorCode ierr=0;

		ierr = KSPDestroy(&_ksp);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

		// Mimic PETSc default solver and preconditioner
		this->_solver_type           = GMRES;

		if(!this->_preconditioner)
		{
			if (libMesh::n_processors() == 1)
				this->_preconditioner_type = ILU_PRECOND;
			else
				this->_preconditioner_type = BLOCK_JACOBI_PRECOND;
		}
	}
}

template<typename T>
void PetscNativeLinearSolver<T>::init ()
{
  // Initialize the data structures if not done so already.
  if (!this->initialized())
    {
      this->_is_initialized = true;
	  
      int ierr=0;
  
      // Create the linear solver context
      ierr = KSPCreate (KARMA_COMM_WORLD, &_ksp);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
      
      // Create the preconditioner context
      ierr = KSPGetPC (_ksp, &_pc);             CHKERRABORT(KARMA_COMM_WORLD,ierr);

      // Have the Krylov subspace method use our good initial guess rather than 0
      ierr = KSPSetInitialGuessNonzero (_ksp, PETSC_TRUE);     CHKERRABORT(KARMA_COMM_WORLD,ierr);
      
      // Set user-specified  solver and preconditioner types
      this->set_petsc_solver_type();

	  // set a prefix for options
		KSPAppendOptionsPrefix(_ksp, Parent::_name.c_str()) ;

      // Notify PETSc of location to store residual history.
      // This needs to be called before any solves, since
      // it sets the residual history length to zero.  The default
      // behavior is for PETSc to allocate (internally) an array
      // of size 1000 to hold the residual norm history.
      ierr = KSPSetResidualHistory(_ksp,
				   PETSC_NULL,   // pointer to the array which holds the history
				   PETSC_DECIDE, // size of the array holding the history
				   PETSC_TRUE);  // Whether or not to reset the history for each solve. 
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
		
		PetscNativePreconditioner<T>::set_type(this->_preconditioner_type,_pc);
		
		//If there is a preconditioner object we need to do the internal setup and apply routines
		if(this->_preconditioner)
		{
			// this->_preconditioner->set_matrix(*matrix);
			PCShellSetContext(_pc,(void*)this->_preconditioner);
			
			PetscNativePreconditioner<T>::set_shell_routines(_pc) ;
		}
		
    }
}

template<typename T>
void PetscNativeLinearSolver<T>::init ( MatrixType* matrix_in )
{
	// Initialize the data structures if not done so already.
	if (!this->initialized())
    {
		this->_is_initialized = true;

		PetscNativeMatrix<T>* matrix = dynamic_cast<PetscNativeMatrix<T>*>(matrix_in) ;
		
		matrix->close() ;

		int ierr=0;

		// Create the linear solver context
		ierr = KSPCreate (KARMA_COMM_WORLD, &_ksp);
			 CHKERRABORT(KARMA_COMM_WORLD,ierr);

		//ierr = PCCreate (KARMA_COMM_WORLD, &_pc);
		//     CHKERRABORT(KARMA_COMM_WORLD,ierr);

		// Create the preconditioner context
		ierr = KSPGetPC (_ksp, &_pc);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

		// Set operators. The input matrix works as the preconditioning matrix
		ierr = KSPSetOperators(_ksp, matrix->mat(), matrix->mat(), (same_preconditioner ? SAME_PRECONDITIONER : SAME_NONZERO_PATTERN)); CHKERRABORT(KARMA_COMM_WORLD,ierr);

		// Have the Krylov subspace method use our good initial guess rather than 0
		// ierr = KSPSetInitialGuessNonzero (_ksp, PETSC_TRUE);	 CHKERRABORT(KARMA_COMM_WORLD,ierr);

		// Set user-specified  solver and preconditioner types
		this->set_petsc_solver_type();
		
		// set a prefix for options
		KSPAppendOptionsPrefix(_ksp, Parent::_name.c_str()) ;

		// Notify PETSc of location to store residual history.
		// This needs to be called before any solves, since
		// it sets the residual history length to zero.  The default
		// behavior is for PETSc to allocate (internally) an array
		// of size 1000 to hold the residual norm history.
		ierr = KSPSetResidualHistory(_ksp,
				   PETSC_NULL,   // pointer to the array which holds the history
				   PETSC_DECIDE, // size of the array holding the history
				   PETSC_TRUE);  // Whether or not to reset the history for each solve. 
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
		
		PetscNativePreconditioner<T>::set_type(this->_preconditioner_type,_pc);
		
		//If there is a preconditioner object we need to do the internal setup and apply routines
		if(this->_preconditioner)
		{
			this->_preconditioner->set_matrix(*matrix);
			
			PCShellSetContext(_pc,(void*)this->_preconditioner);
			
			PetscNativePreconditioner<T>::set_shell_routines(_pc) ;
		}
		
    }
	
}

template<typename T>
void PetscNativeLinearSolver<T>::attach_preconditioner(PetscNativePreconditioner<T>* preconditioner)
{
	if(this->_is_initialized)
	{
		std::cerr<<"Preconditioner must be attached before the solver is initialized!"<<std::endl;
		karma_error();
	}

	_preconditioner_type = SHELL_PRECOND;
	_preconditioner = preconditioner;
}


template<typename T>
std::pair<unsigned int, double> 
PetscNativeLinearSolver<T>::solve (MatrixType&  matrix,
			     MatrixType&  precond,
			     VectorType& solution,
			     const VectorType& rhs,
			     const double tol,
			     const unsigned int m_its)
{
  KARMA_LOG_PUSH("solve()", "PetscNativeLinearSolver");
  
  // Make sure the data passed in are really of Petsc types

  int ierr=0;
  int its=0, max_its = static_cast<int>(m_its);
  double final_resid=0.;

  // Close the matrices and vectors in case this wasn't already done.
  matrix.close ();
  precond.close ();
  solution.close ();
  // rhs.close ();

  this->init (&matrix);

//   // If matrix != precond, then this means we have specified a
//   // special preconditioner, so reset preconditioner type to PCMAT.
//   if (matrix != precond)
//     {
//       this->_preconditioner_type = USER_PRECOND;
//       this->set_petsc_preconditioner_type ();
//     }

  if(this->_preconditioner)
	this->_preconditioner->set_matrix(precond);

  
  // Set operators. The input matrix works as the preconditioning matrix
  ierr = KSPSetOperators(_ksp, matrix.mat(), precond.mat(), (same_preconditioner ? SAME_PRECONDITIONER : DIFFERENT_NONZERO_PATTERN));
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Set the tolerances for the iterative solver.  Use the user-supplied
  // tolerance for the relative residual & leave the others at default values.
  ierr = KSPSetTolerances (_ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_its);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set the options from user-input
	// Set runtime options, e.g.,
	//      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
	//  These options will override those specified above as long as
	//  KSPSetFromOptions() is called _after_ any other customization
	//  routines.
  ierr = KSPSetFromOptions (_ksp);      CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Solve the linear system
  ierr = KSPSolve (_ksp, rhs.vec(), solution.vec()); CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Get the number of iterations required for convergence
  ierr = KSPGetIterationNumber (_ksp, &its);     CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
  // Get the norm of the final residual to return to the user.
  ierr = KSPGetResidualNorm (_ksp, &final_resid);    CHKERRABORT(KARMA_COMM_WORLD,ierr);

  KARMA_LOG_POP("solve()", "PetscNativeLinearSolver");
  
  // return the # of its. and the final residual norm.
  return std::make_pair(its, final_resid);
}


template<typename T>
std::pair<unsigned int, double> 
PetscNativeLinearSolver<T>::solve (const KARMAOperator& shell_matrix,
			     VectorType& solution,
			     const VectorType& rhs,
			     const double tol,
			     const unsigned int m_its)
{
	KARMA_LOG_PUSH("solve()", "PetscNativeLinearSolver");

	this->init ();
	
	int ierr=0;
	int its=0, max_its = static_cast<int>(m_its);
	double final_resid=0.;

	// Close the matrices and vectors in case this wasn't already done.
	solution.close ();
	// rhs.close ();

	// Prepare the matrix.
	Mat mat;
	ierr = MatCreateShell(KARMA_COMM_WORLD,
			rhs.local_size(),
			solution.local_size(),
			rhs.size(),
			solution.size(),
			const_cast<void*>(static_cast<const void*>(&shell_matrix)),
			&mat);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	/* Note that the const_cast above is only necessary because PETSc
	 does not accept a const void*.  Inside the member function
	 _petsc_shell_matrix() below, the pointer is casted back to a
	 const ShellMatrix<T>*.  */

	ierr = MatShellSetOperation(mat,MATOP_MULT,reinterpret_cast<void(*)(void)>(__petsc_shell_matrix_mult));
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set operators. The input matrix works as the preconditioning matrix
	ierr = KSPSetOperators(_ksp, mat, mat, SAME_NONZERO_PATTERN);  CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set the tolerances for the iterative solver.  Use the user-supplied
	// tolerance for the relative residual & leave the others at default values.
	ierr = KSPSetTolerances (_ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_its);  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	// Set the options from user-input
	// Set runtime options, e.g.,
	//      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
	//  These options will override those specified above as long as
	//  KSPSetFromOptions() is called _after_ any other customization
	//  routines.
	ierr = KSPSetFromOptions (_ksp);      CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Solve the linear system
	ierr = KSPSolve (_ksp, rhs.vec(), solution.vec());  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	// Get the number of iterations required for convergence
	ierr = KSPGetIterationNumber (_ksp, &its);  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	// Get the norm of the final residual to return to the user.
	ierr = KSPGetResidualNorm (_ksp, &final_resid);  CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Destroy the matrix.
	ierr = MatDestroy(&mat);  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	KARMA_LOG_POP("solve()", "PetscNativeLinearSolver");

	// return the # of its. and the final residual norm.
	return std::make_pair(its, final_resid);

}


template<typename T>
std::pair<unsigned int, double> 
PetscNativeLinearSolver<T>::solve (const KARMAOperator& shell_matrix,
			     MatrixType& precond_matrix,
			     VectorType &solution,
			     const VectorType &rhs,
			     const double tol,
			     const unsigned int m_its)
{

	KARMA_LOG_PUSH("solve()", "PetscNativeLinearSolver");
	
	this->init ();

	int ierr=0;
	int its=0, max_its = static_cast<int>(m_its);
	double final_resid=0.;

	// Close the matrices and vectors in case this wasn't already done.
	solution.close ();
	// rhs.close ();

	// Prepare the matrix.
	Mat mat;
	ierr = MatCreateShell(KARMA_COMM_WORLD,
			rhs.local_size(),
			solution.local_size(),
			rhs.size(),
			solution.size(),
			const_cast<void*>(static_cast<const void*>(&shell_matrix)),
			&mat);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	/* Note that the const_cast above is only necessary because PETSc
	 does not accept a const void*.  Inside the member function
	 _petsc_shell_matrix() below, the pointer is casted back to a
	 const ShellMatrix<T>*.  */

	ierr = MatShellSetOperation(mat,MATOP_MULT,reinterpret_cast<void(*)(void)>(__petsc_shell_matrix_mult));
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set operators. The input matrix works as the preconditioning matrix
	ierr = KSPSetOperators(_ksp, mat, precond_matrix.mat(), DIFFERENT_NONZERO_PATTERN);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	if(this->_preconditioner)
		this->_preconditioner->set_matrix(precond_matrix);

	// Set the tolerances for the iterative solver.  Use the user-supplied
	// tolerance for the relative residual & leave the others at default values.
	ierr = KSPSetTolerances (_ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_its);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set the options from user-input
	// Set runtime options, e.g.,
	//      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
	//  These options will override those specified above as long as
	//  KSPSetFromOptions() is called _after_ any other customization
	//  routines.
	ierr = KSPSetFromOptions (_ksp);      CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Solve the linear system
	ierr = KSPSolve (_ksp, rhs.vec(), solution.vec());	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	// Get the number of iterations required for convergence
	ierr = KSPGetIterationNumber (_ksp, &its);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	// Get the norm of the final residual to return to the user.
	ierr = KSPGetResidualNorm (_ksp, &final_resid);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Destroy the matrix.
	ierr = MatDestroy(&mat);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	KARMA_LOG_POP("solve()", "PetscNativeLinearSolver");

	// return the # of its. and the final residual norm.
	return std::make_pair(its, final_resid);

}


template<typename T>
std::pair<unsigned int, Real>
PetscNativeLinearSolver<T>::solve ( const KARMAOperator& shell_matrix,
					const KARMAOperator& shell_preconditioner_in,
					VectorType& solution_in,
					const VectorType& rhs_in,
					const double tol,
					const unsigned int m_its)
{

	KARMA_LOG_PUSH("solve()", "PetscNativeLinearSolver");
	
	this->init ();

	int ierr=0;
	int its=0, max_its = static_cast<int>(m_its);
	double final_resid=0.;
	
	PetscNativeVector<T>& solution = dynamic_cast<PetscNativeVector<T>&>(solution_in) ;
	const PetscNativeVector<T>& rhs = dynamic_cast<const PetscNativeVector<T>&>(rhs_in) ;
	const KARMALinearOperator& shell_preconditioner = dynamic_cast<const KARMALinearOperator&>(shell_preconditioner_in) ;

	// Close the matrices and vectors in case this wasn't already done.
	solution.close ();
	// rhs.close ();

	// Prepare the matrix.
	Mat mat ;
	ierr = MatCreateShell(KARMA_COMM_WORLD,
						  rhs.local_size(),
						  solution.local_size(),
						  rhs.size(),
						  solution.size(),
						  const_cast<void*>(static_cast<const void*>(&shell_matrix)),
						  &mat);
	
	/* Note that the const_cast above is only necessary because PETSc
	does not accept a const void*.  Inside the member function
	_petsc_shell_matrix() below, the pointer is casted back to a
	const ShellMatrix<T>*.  */

	ierr = MatShellSetOperation(mat,MATOP_MULT,reinterpret_cast<void(*)(void)>(__petsc_shell_matrix_mult));
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set operators. The input matrix works as the preconditioning matrix
	ierr = KSPSetOperators(_ksp, mat, mat, DIFFERENT_NONZERO_PATTERN);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// set the PC type to be user-specified
	ierr = PCSetType (_pc, PCSHELL);      CHKERRABORT(KARMA_COMM_WORLD,ierr);
	// let us set the operator to the context in order to retreive it later during
	// preconditioner solve phase.
	ierr = PCShellSetContext(_pc, (void*)(&shell_preconditioner)) ;
	// set the PC solver routine 
	ierr = PCShellSetApply(_pc, __petsc_shell_matrix_solve) ;
	
	// Set the tolerances for the iterative solver.  Use the user-supplied
	// tolerance for the relative residual & leave the others at default values.
	ierr = KSPSetTolerances (_ksp, tol, PETSC_DEFAULT, PETSC_DEFAULT, max_its);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Set the options from user-input
	// Set runtime options, e.g.,
	//      -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
	//  These options will override those specified above as long as
	//  KSPSetFromOptions() is called _after_ any other customization
	//  routines.
	ierr = KSPSetFromOptions (_ksp);      CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Solve the linear system
	ierr = KSPSolve (_ksp, rhs.vec(), solution.vec());	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	// Get the number of iterations required for convergence
	ierr = KSPGetIterationNumber (_ksp, &its);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	// Get the norm of the final residual to return to the user.
	ierr = KSPGetResidualNorm (_ksp, &final_resid);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Destroy the matrix.
	ierr = MatDestroy(&mat);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	 
	KARMA_LOG_POP("solve()", "PetscNativeLinearSolver");

	// return the # of its. and the final residual norm.
	return std::make_pair(its, final_resid);

}


template<typename T>
void PetscNativeLinearSolver<T>::get_history(std::vector<double>& hist)
{
	int ierr = 0;
	int its  = 0;

	// Fill the residual history vector with the residual norms
	// Note that GetResidualHistory() does not copy any values, it
	// simply sets the pointer p.  Note that for some Krylov subspace
	// methods, the number of residuals returned in the history
	// vector may be different from what you are expecting.  For
	// example, TFQMR returns two residual values per iteration step.
	double* p;
	ierr = KSPGetResidualHistory(_ksp, &p, &its);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Check for early return
	if (its == 0) return;

	// Create space to store the result
	hist.resize(its);

	// Copy history into the vector provided by the user.
	for (int i=0; i<its; ++i)
    {
		hist[i] = *p;
		p++;
    }
}

template<typename T>
double PetscNativeLinearSolver<T>::get_initial_residual()
{
	int ierr = 0;
	int its  = 0;

	// Fill the residual history vector with the residual norms
	// Note that GetResidualHistory() does not copy any values, it
	// simply sets the pointer p.  Note that for some Krylov subspace
	// methods, the number of residuals returned in the history
	// vector may be different from what you are expecting.  For
	// example, TFQMR returns two residual values per iteration step.
	double* p;
	ierr = KSPGetResidualHistory(_ksp, &p, &its);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Check no residual history
	if (its == 0)
	{
	  std::cerr << "No iterations have been performed, returning 0." << std::endl;
	  return 0.;
	}

	// Otherwise, return the value pointed to by p.
	return *p;
}

template<typename T>
void PetscNativeLinearSolver<T>::set_petsc_solver_type()
{
	int ierr = 0;

	switch (this->_solver_type)
	{
		case CG:
		  ierr = KSPSetType (_ksp, (char*) KSPCG);         CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case CR:
		  ierr = KSPSetType (_ksp, (char*) KSPCR);         CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case CGS:
		  ierr = KSPSetType (_ksp, (char*) KSPCGS);        CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case BICG:
		  ierr = KSPSetType (_ksp, (char*) KSPBICG);       CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case TCQMR:
		  ierr = KSPSetType (_ksp, (char*) KSPTCQMR);      CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case TFQMR:
		  ierr = KSPSetType (_ksp, (char*) KSPTFQMR);      CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case LSQR:
		  ierr = KSPSetType (_ksp, (char*) KSPLSQR);       CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case BICGSTAB:
		  ierr = KSPSetType (_ksp, (char*) KSPBCGS);       CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case MINRES:
		  ierr = KSPSetType (_ksp, (char*) KSPMINRES);     CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case GMRES:
		  ierr = KSPSetType (_ksp, (char*) KSPGMRES);      CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case RICHARDSON:
		  ierr = KSPSetType (_ksp, (char*) KSPRICHARDSON); CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case CHEBYSHEV: 
		  ierr = KSPSetType (_ksp, (char*) KSPCHEBYCHEV);  CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		default:
		  std::cerr << "ERROR:  Unsupported PETSC Solver: "
			<< this->_solver_type               << std::endl
			<< "Continuing with PETSC defaults" << std::endl;
	}
}

template<typename T>
void PetscNativeLinearSolver<T>::print_converged_reason()
{
	KSPConvergedReason reason;
	KSPGetConvergedReason(_ksp, &reason);

	//  KSP_CONVERGED_RTOL (residual 2-norm decreased by a factor of rtol, from 2-norm of right hand side)
	//  KSP_CONVERGED_ATOL (residual 2-norm less than abstol)
	//  KSP_CONVERGED_ITS (used by the preonly preconditioner that always uses ONE iteration) 
	//  KSP_CONVERGED_STEP_LENGTH
	//  KSP_DIVERGED_ITS  (required more than its to reach convergence)
	//  KSP_DIVERGED_DTOL (residual norm increased by a factor of divtol)
	//  KSP_DIVERGED_NAN (residual norm became Not-a-number likely do to 0/0)
	//  KSP_DIVERGED_BREAKDOWN (generic breakdown in method)

	switch (reason)
	{
		// Convergence
		case KSP_CONVERGED_RTOL:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver converged, relative tolerance reached." ) ;
			break;
		}
		case KSP_CONVERGED_ATOL:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver converged, absolute tolerance reached." ) ;
			break;
		}

		// Divergence
		case KSP_DIVERGED_ITS:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, max no. of iterations reached." ) ;
			break;
		}
		case KSP_DIVERGED_DTOL:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, residual norm increase by dtol (default 1.e5)." ) ;
			break;
		}
		case KSP_DIVERGED_NAN:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, residual norm is NaN." ) ;
			break;
		}
		case KSP_DIVERGED_BREAKDOWN:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Linear solver diverged, generic breakdown in the method." ) ;
			break;
		}
		default:
		{
			PetscPrintf(KARMA_COMM_WORLD, "Unknown/unsupported con(di)vergence reason: %d", reason ) ;
		}
	}

}





//------------------------------------------------------------------
// Explicit instantiations
template class PetscNativeLinearSolver<PetscScalar>;
