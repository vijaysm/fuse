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



#ifndef __petsc_native_nonlinear_solver_h__
#define __petsc_native_nonlinear_solver_h__

// Local includes
#include "isolver.h"
#include "petsc_macro.h"

// Petsc include files.
# include "petscsnes.h"

// Forward declarations
class KARMANonlinearOperator ;
template <typename T>
class PetscNativeLinearSolver ;
template <typename T>
class PetscNativeNonlinearSolver ;

// create a Traits class for Solvers
template<typename T>
struct SolverTraits<PetscNativeNonlinearSolver<T> >
{
	/**
	 *  Create some traits to define Scalar and Composite types.
	 */
	typedef T ScalarType ;
	typedef typename PetscNativeLinearSolver<T>::VectorType VectorType ;
	typedef typename PetscNativeLinearSolver<T>::MatrixType MatrixType ;
	typedef PetscNativeNonlinearSolver<T> SolverType ;
};


/**
 * This class provides an interface to PETSc
 * iterative solvers that is compatible with the \p KARMA.
 * This code is general and is based on LibMesh's implementation
 * of PetscNativeNonlinearSolver with quite a few modifications specific
 * to the applications on hand.
 * As long as you can provide a NonlinearOperator, we can solve the 
 * nonlinear problem; And if you provide a KARMAPreconditioner, that
 * will be applied to accelerate the linear iterations.
 */
template <typename T>
class PetscNativeNonlinearSolver : public ISolver<PetscNativeNonlinearSolver<T> >
{
public:
	
	typedef ISolver<PetscNativeNonlinearSolver<T> > Parent ;
	typedef typename SolverTraits<PetscNativeNonlinearSolver<T> >::ScalarType ScalarType ;
	typedef typename SolverTraits<PetscNativeNonlinearSolver<T> >::VectorType VectorType ;
	typedef typename SolverTraits<PetscNativeNonlinearSolver<T> >::MatrixType MatrixType ;
	typedef typename SolverTraits<PetscNativeNonlinearSolver<T> >::SolverType SolverType ;
	
	/**
	* @brief
	* 
	*  Constructor. Initializes Petsc data structures
	*/
	PetscNativeNonlinearSolver ();

	/**
	* @brief
	* 
	* Destructor.
	*/
	~PetscNativeNonlinearSolver ();

	/**
	* @brief
	* 
	* Release all memory and clear data structures.
	*/
	virtual void clear ();

	/**
	* @brief
	* 
	* Initialize data structures if not done so already.
	*/
	virtual void init ();

	/**
	* @brief
	* 
	* Returns the raw PETSc snes context pointer.
	*/
	SNES snes() { this->init(); return _snes; }

	/**
	* @brief
	* 
	* Set the operator to solve the current nonline system
	*/
	virtual void set_operator(KARMANonlinearOperator& nlinop) ;

	/**
	* @brief
	* 
	* Return the operator that is being used to solve the current nonlinear system
	*/
	virtual const KARMANonlinearOperator& get_operator() const ;
	
	/**
	* @brief
	* 
	* Set the residual vector, jacobian matrix and a separate preconditioner matrix used to 
	* solve the current nonlinear system.
	* 
	* It is necessary to call this routine to set residual vector and jacobian matrix reference.
	* If either of these are changed, this routine needs to be called again 
	* because SNES stores this reference and hence could lead to writing solution to wrong 
	* vector/matrix.
	*/
	virtual void set_discrete_references(VectorType& res, MatrixType& jac, MatrixType& prec) ;

	/**
	 * @brief
	 *
	 * Same as above but uses the Jacobian matrix as the preconditioner also.
	 */
	virtual void set_discrete_references(VectorType& res, MatrixType& jac) ;
	
	/**
	* @brief
	* 
	* Sets up the Petsc SNES object we created so that all user specified options are set.  
	* This is essential before solve is called and should result in an error if user bypasses 
	* this step.
	*/
	void setup () ;

	/**
	* @brief
	* 
	* Raw public function pointer that can be provided to use custom residual evaluation
	* By default it points to the private static function implemented in PetscNativeNonlinearSolver.
	*/
	PetscErrorCode (*residual) (SNES, Vec x, Vec r, void *ctx) ;

	/**
	* @brief
	* 
	* Raw public function pointer that can be provided to use custom jacobian evaluation
	* By default it points to the private static function implemented in PetscNativeNonlinearSolver.
	*/
	PetscErrorCode  (*jacobian) (SNES, Vec x, Mat *jac, Mat *pc, MatStructure *msflag, void *ctx) ;

	/**
	* @brief
	* 
	* Call the Petsc solver.  It calls the method below, using the
	* same matrix for the system and preconditioner matrices.
	*/
	virtual std::pair<unsigned int, double> solve (VectorType& , VectorType* rhs=NULL);

	/**
	* @brief
	* 
	* Call the Petsc solver.  It calls the method below, using the
	* same matrix for the system and preconditioner matrices.
	*/
	virtual std::pair<unsigned int, double> solve (VectorType& , VectorType& ,
						MatrixType& jac, VectorType* rhs=NULL);

	/**
	* @brief
	* 
	* Call the Petsc solver.  It calls the method below, using the
	* same matrix for the system and preconditioner matrices.
	*/
	virtual std::pair<unsigned int, double> solve (VectorType& , VectorType& ,
						MatrixType& jac, MatrixType& prec, VectorType* rhs=NULL);


	/**
	* @brief
	*
	* Attaches a Preconditioner object to be used during the linear solves.
	*/
	// PetscLinearSolver<T>& get_linear_solver();

	/**
	* @brief
	* 
	* Attaches a Preconditioner object to be used during the linear solves.
	*/
	// void attach_preconditioner(KARMAPreconditioner* preconditioner);

	/**
	* @brief
	* 
	* Maximum number of non-linear iterations.
	*/
	// unsigned int max_nonlinear_iterations; replaced with _max_iterations

	/**
	* @brief
	* 
	* Maximum number of function evaluations.
	*/
	unsigned int max_function_evaluations;

	/**
	* @brief
	* 
	* The NonlinearSolver should exit after the full nonlinear step norm is
	* reduced to either less than absolute_step_tolerance
	* or less than relative_step_tolerance times the largest
	* nonlinear solution which has been seen so far.
	*
	* Users should increase any of these tolerances that they want to use for a
	* stopping condition.
	*
	* Note that not all NonlinearSolvers support relative_step_tolerance!
	*/
	double absolute_step_tolerance;
	double relative_step_tolerance;

	/**
	* @brief
	* 
	* Each linear solver step should exit after \p max_linear_iterations
	* is exceeded.
	*/
	unsigned int max_linear_iterations;

	/**
	* @brief
	* 
	* Any required linear solves will at first be done with this tolerance;
	* the NonlinearSolver may tighten the tolerance for later solves.
	*/
	double initial_linear_tolerance;

	/**
	* @brief
	* 
	* The tolerance for linear solves is kept above this minimum
	*/
	double minimum_linear_tolerance;

	/**
	* @brief
	* 
	* After a call to solve this will reflect whether or not the nonlinear
	* solve was successful.
	*/
	// bool converged;	// replaced with _is_converged
  
	/**
	* @brief
	* 
	* Sets the flag to perform exact Newton method : i.e., solve the linear problem
	* exactly, no matter how far the current solution is from the true nonlinear solution.
	*/
	void use_exact_newton_method() ;
  
private:

	/**
	* @brief
	* 
	* KARMA Nonlinear operator context
	*/
	KARMANonlinearOperator* _nlin_operator ;
	
	/**
	* @brief
	* 
	* Nonlinear solver context
	*/
	SNES _snes;

	/**
	* @brief
	* 
	* Flag indicating whether we are performing inexact Newton iteration.
	*/
	bool _is_inexact_newton ;

	/**
	* @brief
	* 
	* Holds the Preconditioner object to be used for the linear solves.
	*/
	// KARMAPreconditioner* _preconditioner;
	
	/**
	* @brief
	* 
	* Internal function if shell matrix mode is used.
	*/
	static PetscErrorCode  __karma_simple_petsc_snes_monitor (SNES, PetscInt its, PetscReal fnorm, void *) ;

	/**
	* @brief
	* 
	* Internal function if shell matrix mode is used.
	*/
	static PetscErrorCode __karma_petsc_snes_residual (SNES, Vec x, Vec r, void *ctx) ;

	/**
	* @brief
	* 
	* Internal function if shell matrix mode is used.
	*/
	static PetscErrorCode  __karma_petsc_snes_jacobian (SNES, Vec x, Mat *jac, Mat *pc, MatStructure *msflag, void *ctx) ;

};


// inline functions for PetscNativeNonlinearSolver
template <typename T>
inline
PetscNativeNonlinearSolver<T>::PetscNativeNonlinearSolver () :
		ISolver<PetscNativeNonlinearSolver<T> >(),
		// max_nonlinear_iterations(30),
		max_function_evaluations(500),
		// absolute_residual_tolerance(1e-7),
		// relative_residual_tolerance(1e-6),
		absolute_step_tolerance(1e-10),
		relative_step_tolerance(1e-8),
		max_linear_iterations(60),
		initial_linear_tolerance(1e-3),
		minimum_linear_tolerance(1e-14),
		_nlin_operator(NULL),
		_is_inexact_newton(true)
		// _preconditioner (NULL)
{
	residual = __karma_petsc_snes_residual ;
	jacobian = __karma_petsc_snes_jacobian ;
}

template <typename T>
inline
void PetscNativeNonlinearSolver<T>::set_operator(KARMANonlinearOperator& nlinop)
{
	_nlin_operator = &nlinop ;
}


template <typename T>
inline
const KARMANonlinearOperator& PetscNativeNonlinearSolver<T>::get_operator() const
{
	karma_assert (_nlin_operator != NULL) ;
	
	return *_nlin_operator ;
}


template <typename T>
inline
PetscNativeNonlinearSolver<T>::~PetscNativeNonlinearSolver ()
{
	this->clear ();
}

template <typename T>
inline
void PetscNativeNonlinearSolver<T>::use_exact_newton_method()
{
	_is_inexact_newton = false ; 
}


#endif // #ifdef __petsc_native_nonlinear_solver_h__
