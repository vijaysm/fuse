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



#ifndef __petsc_block_linear_solver_h__
#define __petsc_block_linear_solver_h__

#ifdef USE_PETSC_BLOCK_EXTENSIONS

// KARMA enum types
#include "enum_solver_package.h"
#include "enum_solver_type.h"
#include "enum_preconditioner_type.h"

// Local includes
#include "isolver.h"
#include "petsc_macro.h"

/**
 * Petsc include files.
 */
#include <petscksp.h>

// Forward declarations
class KARMALinearOperator;
template<typename T, typename AGGREGATE>
class PetscPreconditioner;


/**
 * This class provides an interface to PETSc
 * iterative solvers that is compatible with the \p KARMA
 * \p Originally written by Benjamin Kirk for libMesh but
 * this is an adapted and modified version for KARMA.
 *
 * @author Vijay Mahadevan, 2008-2009
 */
template<typename T>
class PetscBlockLinearSolver : public ISolver<T, PetscLinearSolver<T,T>* >
{
public:
	/**
	*  Constructor. Initializes Petsc data structures
	*/
	PetscBlockLinearSolver ();
	
	/**
	*  Copy Constructor. Duplicates a linear solver with all
	*  the options from the \p other solver.
	*/
	PetscBlockLinearSolver (const PetscBlockLinearSolver& other);

	/**
	* Destructor.
	*/
	~PetscBlockLinearSolver ();
	
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
	virtual void init (IMatrix* matrix);
	
	
	/**
	* Returns the type of preconditioner to use.
	*/
	PreconditionerType preconditioner_type () const;
	
	/**
	* Sets the type of preconditioner to use.
	*/
	void set_preconditioner_type (const PreconditionerType pct);
	
	/**
	* Attaches a Preconditioner object to be used
	*/
	void attach_preconditioner(PetscPreconditioner* preconditioner);
	
	/**
	* Call the Petsc solver.  It calls the method below, using the
	* same matrix for the system and preconditioner matrices. Note
	* that the linear solver will compute the preconditioner in this
	* case, and will depend on the type specified through
	* set_preconditioner_type(const PreconditionerType pct) call.
	*/
	std::pair<unsigned int, Real>
	solve (KARMAMatrix  &matrix_in,
		KARMAVector &solution_in,
		const KARMAVector &rhs_in,
		const double tol,
		const unsigned int m_its)
	{
		return this->solve(matrix_in, matrix_in, solution_in, rhs_in, tol, m_its);
	}
	
	/**
	* Returns the type of solver to use.
	*/
	SolverType solver_type () const { return _solver_type; }
	
	/**
	* Sets the type of solver to use.
	*/
	void set_solver_type (const SolverType st)
	{ _solver_type = st; }
	
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
	solve (KARMAMatrix  &matrix,
		KARMAMatrix  &preconditioner,
		KARMAVector &solution,
		const KARMAVector &rhs,
		const double tol,
		const unsigned int m_its);
	
	/**
	* This function solves a system whose matrix is a shell matrix.
	* Valid only for single system of equations.
	*/
	virtual std::pair<unsigned int, Real>
	solve (const KARMALinearOperator& shell_matrix,
		KARMAVector& solution_in,
		const KARMAVector& rhs_in,
		const double tol,
		const unsigned int m_its);
	
	/**
	* This function solves a system whose matrix is a shell matrix, but
	* a sparse matrix is used as preconditioning matrix, this allowing
	* other preconditioners than JACOBI.
	*/
	virtual std::pair<unsigned int, Real>
	solve (const KARMALinearOperator& shell_matrix,
		KARMAMatrix& precond_matrix,
		KARMAVector& solution_in,
		const KARMAVector& rhs_in,
		const double tol,
		const unsigned int m_its);
	
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
	* Internal function if shell matrix mode is used.
	*/
	static PetscErrorCode __petsc_block_shell_matrix_mult(Mat mat, Vec arg, Vec dest);

	/**
	* Preconditioner context
	*/
	PC _pc; 

	/**
	* Krylov subspace context
	*/
	KSP _ksp;

	/**
	* Krylov solver options prefix
	*/
	std::string _options_prefix ;

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
	PetscBlockPreconditioner<T>* _preconditioner;
};


// inline function for PetscBlockLinearSolver


inline
PetscBlockLinearSolver::PetscBlockLinearSolver () :
		ISolver<T,PetscLinearSolver<T>*>(KARMA::PETSC, KARMA::LIBMESH),
	_lin_operator(NULL),
	_solver_type         (GMRES),
	_preconditioner_type (ILU_PRECOND),
	_is_initialized      (false),
	_preconditioner      (NULL)
{
	if (KARMA::num_procs() > 1)
		this->_preconditioner_type = BLOCK_JACOBI_PRECOND;
}


template<typename T>
inline
void PetscLinearSolver<T>::set_operator(KARMAOperator& linop)
{
	_lin_operator = dynamic_cast<KARMALinearOperator>(&linop) ;

	// check if our Operator was really a Linear operator
	karma_assert(_lin_operator != NULL) ;
}


inline
PetscBlockLinearSolver::~PetscBlockLinearSolver ()
{
	this->clear ();
}

#endif // #ifdef USE_PETSC_BLOCK_EXTENSIONS

#endif // #ifdef __petsc_block_linear_solver_h__
