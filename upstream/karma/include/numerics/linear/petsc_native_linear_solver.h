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



#ifndef __petsc_native_linear_solver_h__
#define __petsc_native_linear_solver_h__

// Local includes
#include "isolver.h"
#include "petsc_macro.h"
#include "karma_linear_operator.h"

/**
 * Petsc include files.
 */
#include <petscksp.h>


// Forward declarations
// class KARMALinearOperator;
template <typename VectorType>
class PetscNativeVector ;
template <typename MatrixType>
class PetscNativeMatrix ;
template<typename T>
class PetscNativePreconditioner ;
template<typename T>
class PetscNativeLinearSolver ;


// create a Traits class for Solvers
template<typename T>
struct SolverTraits<PetscNativeLinearSolver<T> >
{
	/**
	 *  Create some traits to define Scalar and Composite types.
	 */
	typedef T ScalarType ;
	typedef PetscNativeVector<T> VectorType ;
	typedef PetscNativeMatrix<T> MatrixType ;
	// typedef PetscNativeLinearSolver<T> SolverType ;
};


/**
 * This class provides an interface to PETSc
 * iterative solvers that is compatible with the \p KARMA
 * \p Originally written by Benjamin Kirk for libMesh but
 * this is an adapted and modified version for KARMA.
 *
 * @author Vijay Mahadevan, 2008-2009
 */
template<typename T>
class PetscNativeLinearSolver : public ISolver<PetscNativeLinearSolver<T> >
{
public:
	
	typedef ISolver<PetscNativeLinearSolver<T> > Parent ;
	typedef typename SolverTraits<PetscNativeLinearSolver<T> >::ScalarType ScalarType ;
	typedef typename SolverTraits<PetscNativeLinearSolver<T> >::VectorType VectorType ;
	typedef typename SolverTraits<PetscNativeLinearSolver<T> >::MatrixType MatrixType ;
	//typedef typename SolverTraits<PetscNativeLinearSolver<T> >::SolverType SolverType ;
	
	/**
	*  Constructor. Initializes Petsc data structures
	*/
	PetscNativeLinearSolver ();
	
	/**
	*  Copy Constructor. Duplicates a linear solver with all
	*  the options from the \p other solver.
	*/
	PetscNativeLinearSolver (const PetscNativeLinearSolver& other);

	/**
	* Destructor.
	*/
	~PetscNativeLinearSolver ();
	
	/**
	* Release all memory and clear data structures.
	*/
	virtual void clear ();
	
	/**
	* Initialize data structures if not done so already.
	* This expects a copy constructor to have been passed
	* before or else create a default KSP object ?
	*/
	virtual void init ();
	
	/**
	* Initialize data structures if not done so already plus much more
	*/
	virtual void init (MatrixType* matrix);
	
	/**
	* Returns the type of preconditioner to use.
	*/
	virtual PreconditionerType preconditioner_type () const;
	
	/**
	* Sets the type of preconditioner to use.
	*/
	virtual PreconditionerType& preconditioner_type ();
	
	/**
	* Attaches a Preconditioner object to be used
	*/
	void attach_preconditioner(PetscNativePreconditioner<T>* preconditioner);
	
	/**
	* Returns the type of solver to use.
	*/
	virtual SolverType solver_type () const { return _solver_type; }
	
	/**
	* Sets the type of solver to use.
	*/
	virtual SolverType& solver_type () { return _solver_type; }
	
	/**
	* This method allows you to call a linear solver while specifying
	* the matrix to use as the (left) preconditioning matrix.  Note
	* that the linear solver will not compute a preconditioner in this
	* case, and will instead premultiply by the matrix you provide.
	*
	* In PETSc, this is accomplished by calling
	*
	* PCSetType(_pc, PCMAT);
	*
	* before invoking KSPSolve().
	*/
	virtual std::pair<unsigned int, Real>
	solve (MatrixType  &matrix,
		MatrixType  &preconditioner,
		VectorType &solution,
		const VectorType &rhs,
		const double tol,
		const unsigned int m_its);
	
	/**
	* This function solves a system whose matrix is a shell matrix.
	* Valid only for single system of equations.
	*/
	virtual std::pair<unsigned int, Real>
	solve ( const KARMAOperator& shell_matrix,
			VectorType& solution_in,
			const VectorType& rhs_in,
			const double tol,
			const unsigned int m_its);
	
	/**
	* This function solves a system whose matrix is a shell matrix, but
	* a sparse matrix is used as preconditioning matrix, this allowing
	* other preconditioners than JACOBI.
	*/
	virtual std::pair<unsigned int, Real>
	solve ( const KARMAOperator& shell_matrix,
			MatrixType& precond_matrix,
			VectorType& solution_in,
			const VectorType& rhs_in,
			const double tol,
			const unsigned int m_its);
	
	/**
	 * This function solves a system whose matrix and the preconditioner
	 * are both shell matrices.
	 */
	virtual std::pair<unsigned int, Real>
	solve ( const KARMAOperator& shell_matrix,
			const KARMAOperator& shell_preconditioner,
			VectorType& solution_in,
			const VectorType& rhs_in,
			const double tol,
			const unsigned int m_its) ;

	/**
	* Set the operator to solve the current system
	*/
	virtual void set_operator (const KARMAOperator& linop) ;
	
	/**
	* Return the operator that is being used to solve the current system
	*/
	virtual const KARMALinearOperator& get_operator() const ;
	
	/**
	* Returns the raw PETSc preconditioner context pointer.  This allows
	* you to specify the PCShellSetApply() and PCShellSetSetUp() functions
	* if you desire.  Just don't do anything crazy like calling PCDestroy()!
	*/
	PC pc() { this->init(); return _pc; }
	
	/**
	* Returns the raw PETSc ksp context pointer.  This is useful if
	* you are for example setting a custom convergence test with
	* KSPSetConvergenceTest().
	*/
	KSP ksp() { this->init(); return _ksp; }
	
	/**
	* Fills the input vector with the sequence of residual norms
	* from the latest iterative solve.
	*/
	virtual void get_history(std::vector<Real>& hist);
	
	/**
	* Returns just the initial residual for the solve just
	* completed with this interface.  Use this method instead
	* of the one above if you just want the starting residual
	* and not the entire history.
	*/
	virtual Real get_initial_residual();
	
	/**
	* Prints a useful message about why the latest linear solve
	* con(di)verged.
	*/
	virtual void print_converged_reason();
	
	bool same_preconditioner;

  
private:

	/**
	* KARMA Linear operator context
	*/
	KARMALinearOperator* _lin_operator ;

	/**
	* Tells PETSC to use the user-specified solver stored in
	* \p _solver_type
	*/
	void set_petsc_solver_type ();

	/**
	* Internal function if shell matrix mode is used for linear operator.
	*/
	static PetscErrorCode __petsc_shell_matrix_mult(Mat mat, Vec arg, Vec dest);

	/**
	 * Internal function if shell matrix mode is used for preconditioner.
	 */
	static PetscErrorCode __petsc_shell_matrix_solve(PC pc, Vec arg, Vec dest) ;

	/**
	* Preconditioner context
	*/
	PC _pc; 

	/**
	* Krylov subspace context
	*/
	KSP _ksp;
	
	/**
	* Enum stating which type of iterative solver to use.
	*/
	SolverType _solver_type;

	/**
	* Enum statitng with type of preconditioner to use.
	*/
	PreconditionerType _preconditioner_type;

	/**
	* Flag indicating if the data structures have been initialized.
	*/
	bool _is_initialized;

	/**
	* Holds the Preconditioner object to be used for the linear solves.
	*/
	PetscNativePreconditioner<T>* _preconditioner;
};


// inline function for PetscNativeLinearSolver

template<typename T>
inline
PetscNativeLinearSolver<T>::PetscNativeLinearSolver () :
		ISolver<PetscNativeLinearSolver<T> >(KARMA::PETSC, KARMA::LIBMESH),
		_lin_operator(NULL),
		_solver_type         (GMRES),
		_preconditioner_type (ILU_PRECOND),
		_is_initialized      (false),
		_preconditioner      (NULL)
{
	if (KARMA::num_processors > 1)
		this->_preconditioner_type = BLOCK_JACOBI_PRECOND;
}


template<typename T>
inline
void PetscNativeLinearSolver<T>::set_operator(const KARMAOperator& linop)
{
	// _lin_operator = dynamic_cast<const KARMALinearOperator>(&linop) ;
	_lin_operator = dynamic_cast<KARMALinearOperator*>(&const_cast<KARMAOperator&>(linop)) ;

	// check if our Operator was really a Linear operator
	// karma_assert(_lin_operator != NULL) ;
}


template<typename T>
inline
const KARMALinearOperator& PetscNativeLinearSolver<T>::get_operator() const
{
	// check if our Operator was really a Linear operator
	karma_assert(_lin_operator != NULL) ;
	
	return *_lin_operator ;

}


template<typename T>
inline
PetscNativeLinearSolver<T>::~PetscNativeLinearSolver ()
{
	this->clear ();
}


template<typename T>
inline
PreconditionerType PetscNativeLinearSolver<T>::preconditioner_type () const
{
	if(_preconditioner)
		return _preconditioner->type();
    
	return _preconditioner_type;
}

template<typename T>
inline
PreconditionerType& PetscNativeLinearSolver<T>::preconditioner_type ()
{
	if(_preconditioner)
		_preconditioner_type = _preconditioner->type();
    
	return _preconditioner_type;
}

#endif // #ifdef __petsc_native_linear_solver_h__
