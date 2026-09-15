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



#ifndef __isolver_h__
#define __isolver_h__

// C++ includes
#include "supporting_packages.h"
#include "karma_error.h"

// KARMA enum types <- From libMesh
#include "enum_solver_type.h"
#include "enum_preconditioner_type.h"

// Forward declarations
template <typename VectorType>
class IVector ;
template <typename MatrixType>
class IMatrix ;

class KARMAOperator ;

// create a Traits class for Solvers
template<typename derived_solver_type>
struct SolverTraits
{};

/**
 * This class provides an abstract solver class.  It is generic
 * enough to yield for both linear/nonlinear/eigen problems.
 * All concrete instances will inherit from this abstract base
 * and implement the pure virtual methods necessary to create
 * a flexible framework that does not depend on the linear solver
 * package.
 */

// ------------------------------------------------------------
// ISolver class definition
template <typename derived_solver_type>
class ISolver : public IKARMA<ISolver<derived_solver_type> >
{
public:
	
	typedef typename SolverTraits<derived_solver_type>::ScalarType ScalarType ;
	typedef typename SolverTraits<derived_solver_type>::VectorType VectorType ;
	typedef typename SolverTraits<derived_solver_type>::MatrixType MatrixType ;
	// typedef typename SolverTraits<derived_solver_type>::SolverType SolverType ;
	
	/**
	* Constructor.  Initializes required
	* data structures.
	*/
	ISolver (KARMA::LAPackage solver=KARMA::PETSC, KARMA::SpatialDiscretizationPackage discretization=KARMA::LIBMESH);

	/**
	* Copy Constructor.  Explicitly disallow copy
	* by value.
	*/
	ISolver(const ISolver<derived_solver_type>& other) ;

	/**
	* Destructor.
	*/
	virtual ~ISolver ();

	/**
	* @returns true if the data structures are
	* initialized, false otherwise.
	*/
	bool initialized () const { return _is_initialized; }

	/**
	 * @returns true if the data structures are
	 * initialized, false otherwise.
	 */
	KARMA::LAPackage solver_package () const { return _solver; }

	/**
	 * @returns true if the data structures are
	 * initialized, false otherwise.
	 */
	KARMA::SpatialDiscretizationPackage discretization_package () const { return _discretization; }

	/**
	 * Initialize data structures if not done so already.
	 * This expects a copy constructor to have been passed
	 * before or else create a default KSP object ?
	 */
	virtual void init () = 0;
	
	/**
	* Initialize data structures if not done so already plus much more
	*/
	virtual void init (MatrixType* matrix) = 0;

	/**
	 * Release all memory and clear data structures.
	 */
	virtual void clear () = 0;
	
	/**
	 * Returns the type of preconditioner to use.
	 */
	virtual PreconditionerType preconditioner_type () const = 0;
	
	/**
	 * Sets the type of preconditioner to use.
	 */
	virtual PreconditionerType& preconditioner_type () = 0;
	
	/**
	 * Returns the type of solver to use.
	 */
	virtual SolverType solver_type () const = 0;
	
	/**
	 * Sets the type of solver to use.
	 */
	virtual SolverType& solver_type () = 0;
	
	/**
	* Call the solver.  It calls the method below, using the
	* same matrix for the system and preconditioner matrices. Note
	* that the linear solver will compute the preconditioner in this
	* case, and will depend on the type specified through
	* set_preconditioner_type(const PreconditionerType pct) call.
	*/    
	virtual std::pair<unsigned int, Real> 
	solve ( VectorType& solution_in,
			const VectorType& rhs_in,
			const double tol,
			const unsigned int m_its)
	{
		return this->solve(*J, *P, solution_in, rhs_in, tol, m_its);
	}

	/**
	* Call the corresponding solver.  It calls the method below, using the
	* same matrix for the system and preconditioner matrices. Note
	* that the linear solver will compute the preconditioner in this
	* case, and will depend on the type specified through
	* set_preconditioner_type(const PreconditionerType pct) call.
	*/    
	virtual std::pair<unsigned int, Real> 
	solve (	MatrixType& matrix_in,
			VectorType& solution_in,
			const VectorType& rhs_in,
			const double tol,
			const unsigned int m_its)
	{
		return this->solve(matrix_in, matrix_in, solution_in, rhs_in, tol, m_its);
	}

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
	solve ( MatrixType& matrix,
			MatrixType& preconditioner,
			VectorType& solution,
			const VectorType& rhs,
			const double tol,
			const unsigned int m_its) = 0;

	/**
	* This function solves a system whose matrix is a shell matrix.
	* Valid only for single system of equations.
	*/
	virtual std::pair<unsigned int, Real>
	solve ( const KARMAOperator& shell_matrix,
			VectorType& solution_in,
			const VectorType& rhs_in,
			const double tol,
			const unsigned int m_its) = 0;

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
			const unsigned int m_its) = 0;
	
	/**
	* This function solves a system whose matrix is a shell matrix.
	* Valid only for single system of equations.
	*/
	virtual std::pair<unsigned int, Real>
	solve ( const KARMAOperator& shell_matrix,
			const KARMAOperator& shell_preconditioner,
			VectorType& solution_in,
			const VectorType& rhs_in,
			const double tol,
			const unsigned int m_its) = 0;

	/**
	* Fills the input vector with the sequence of residual norms
	* from the latest iterative solve.
	*/
	virtual void get_history(std::vector<Real>& hist) = 0;

	/**
	* Returns the initial residual for the solve just
	* completed with this interface.  Use this method instead
	* of the one above if you just want the starting residual
	* and not the entire history.
	*/
	virtual Real get_initial_residual() = 0;

	/**
	 * Set the name for the current solver.
	 */
	virtual std::string& name() ;
	
	/**
	 * Get the name for the current solver.
	 */
	virtual const std::string& name() const ;

	/**
	* Prints a useful message about why the latest solve
	* con(di)verged.
	*/
	virtual void print_converged_reason() = 0;

	/**
	* Set the operator to solve the current nonline system
	*/
	virtual void set_operator(const KARMAOperator& op) = 0 ;

	/**
	* Return the operator that is being used to solve the current nonlinear system
	*/
	virtual const KARMAOperator& get_operator() const = 0 ;
	
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
	// virtual void set_discrete_references(VectorType& B, MatrixType& A, MatrixType& P) = 0 ;

protected:

	KARMA::LAPackage _solver ;
	KARMA::SpatialDiscretizationPackage _discretization ;
	
	/**
	 * Store the reference to the linear system matrix.
	 */
	MatrixType *J ;
	
	/**
	 * Store the reference to the preconditioner of the linear
	 * system matrix (if different).
	 */
	MatrixType *P ;
	
	/**
	 * Store the name or the prefix for the current solver.
	 * Useful when setting options from command-line or file
	 */
	std::string _name;

	/**
	* Flag indicating if the data structures have been initialized.
	*/
	bool _is_initialized;

	/**
	* Maximum number of linear/nonlinear iterations.
	*/
	unsigned int _max_iterations;

	/**
	* The solver should exit after the residual is
	* reduced to either less than absolute tolerance
	* or less than relative tolerance times the
	* initial residual L_2 norm.
	*
	* Users should increase any of these tolerances that they want to use for a
	* stopping condition.
	*
	*/
	double _absolute_tolerance;
	double _relative_tolerance;

	/**
	* After a call to solve this will reflect whether or not the 
	* solver was successful.
	*/
	bool _is_converged;
  
};


// ------------------------------------------------------------
// ISolver inline methods

template <typename derived_solver_type>
inline
ISolver<derived_solver_type>::ISolver (KARMA::LAPackage solver, KARMA::SpatialDiscretizationPackage discretization)
	:	_solver(solver),
		_discretization(discretization),
		_max_iterations(50),
		_absolute_tolerance(1e-6),
		_relative_tolerance(1e-8)
{
}

template <typename derived_solver_type>
inline
ISolver<derived_solver_type>::~ISolver ()
{
}

template <typename derived_solver_type>
inline
const std::string& ISolver<derived_solver_type>::name() const
{
	return _name ;
}


template <typename derived_solver_type>
inline
std::string& ISolver<derived_solver_type>::name() 
{
	return _name ;
}


#endif	// __isolver_h__
