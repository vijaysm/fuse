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


#ifndef __petsc_native_preconditioner_h__
#define __petsc_native_preconditioner_h__

// C++ includes

// Local includes
#include "enum_solver_package.h"
#include "enum_preconditioner_type.h"
#include "isolver.h"

// Petsc include files.
#include "petscpc.h"

// Forward declarations
class KARMALinearOperator;
template <typename VectorType>
class PetscNativeVector ;
template <typename MatrixType>
class PetscNativeMatrix ;
template<typename T>
class PetscNativePreconditioner ;


// create a Traits class for Solvers
template<typename T>
struct SolverTraits<PetscNativePreconditioner<T> >
{
	/**
	 *  Create some traits to define Scalar and Composite types.
	 */
	typedef T ScalarType ;
	typedef PetscNativeVector<T> VectorType ;
	typedef PetscNativeMatrix<T> MatrixType ;
	// typedef PetscNativePreconditioner<T> SolverType ;
};



/**
 * This class provides an interface to the preconditioners available in Petsc.
 * 
 * In the below comments P is the matrix to be preconditioned with Apply()
 * performing the equivalent of the matrix vector product P^-1 x.  This
 * can also be thought of as (usually approximately) solving for Py=x.
 *
 * Modifications by @author Vijay Mahadevan, 2009 for KARMA
 */
template<typename T>
class PetscNativePreconditioner : public ISolver<PetscNativePreconditioner<T> >
{
public:
	
	typedef ISolver<PetscNativePreconditioner<T> > Parent ;
	typedef typename SolverTraits<PetscNativePreconditioner<T> >::ScalarType ScalarType ;
	typedef typename SolverTraits<PetscNativePreconditioner<T> >::VectorType VectorType ;
	typedef typename SolverTraits<PetscNativePreconditioner<T> >::MatrixType MatrixType ;
	// typedef typename SolverTraits<PetscNativePreconditioner<T> >::SolverType SolverType ;
	
	/**
	*  Constructor. Initializes PetscNativePreconditioner data structures
	*/
	PetscNativePreconditioner ();

	/**
	* Destructor.
	*/
	virtual ~PetscNativePreconditioner ();

	/**
	* Computes the preconditioned vector "y" based on input "x".
	* Usually by solving Py=x to get the action of P^-1 x.
	*/
	void apply(const VectorType& x, VectorType& y);

	/**
	* Release all memory and clear data structures.
	*/
	virtual void clear () {}

	/**
	* Initialize data structures if not done so already.
	*/
	virtual void init ();

	/**
	* Returns the actual Petsc PC struct.  Useful for more advanced
	* purposes
	*/
	PC pc() { return _pc; }

	/**
	* Sets the matrix P to be preconditioned.
	*/
	void set_matrix(MatrixType& mat);

	/**
	* Returns the type of preconditioner to use.
	*/
	virtual PreconditionerType type () const
	{
		return _preconditioner_type;
	}
	
	/**
	 * Returns the type of preconditioner to use.
	 */
	virtual PreconditionerType& type () 
	{
		// if the preconditioner changes, re-initialize the structures
		_is_initialized = false;
		return _preconditioner_type;
	}
	
	/**
	 * Set the name for the current solver.
	 */
	virtual std::string& name() ;
	
	virtual const std::string& name() const ;
	
	/**
	* Set the operator to solve the current system
	*/
	void set_operator(KARMALinearOperator& linop) ;

	/**
	* Return the operator that is being used to solve the current system
	*/
	KARMALinearOperator& get_operator() const ;

	/**
	* Tells PETSC to use the user-specified preconditioner
	*/
	static void set_type (const PreconditionerType preconditioner_type, PC pc);
	
	/**
	* Sets the references for a shell preconditioner
	*/
	static void set_shell_routines(PC pc) ;
	
private:
  
	/**
	* KARMA Linear operator context
	*/
	KARMALinearOperator* _lin_operator ;

	/**
	* The matrix P... ie the matrix to be preconditioned.
	* This is often the actual system matrix of a linear sytem.
	*/
	MatrixType* _matrix;

	/**
	 * Store the name or the prefix for the current solver.
	 * Useful when setting options from command-line or file
	 */
	std::string _name;

	/**
	* Enum statitng with type of preconditioner to use.
	*/
	PreconditionerType _preconditioner_type;

	/**
	* Flag indicating if the data structures have been initialized.
	*/
	bool _is_initialized;

	/**
	* Preconditioner context
	*/
	PC _pc;

	/**
	* Petsc Matrix that's been pulled out of the _matrix object.
	* This happens during init...
	*/
	Mat _mat;
	
	
	/**
	* This function is called by PETSc to initialize the preconditioner.
	* ctx will hold the Preconditioner.
	*/
	static PetscErrorCode __karma_petsc_preconditioner_setup (PC pc);

	/**
	* This function is called by PETSc to acctually apply the preconditioner.
	* ctx will hold the Preconditioner.
	*/
	static PetscErrorCode __karma_petsc_preconditioner_apply(PC pc, Vec x, Vec y);

};


// inline functions for PetscNativePreconditioner


template<typename T>
inline
PetscNativePreconditioner<T>::PetscNativePreconditioner () :
	ISolver<PetscNativePreconditioner<T> >(),
	_lin_operator(NULL),
	_matrix(NULL),
	_preconditioner_type (ICC_PRECOND),
	_is_initialized      (false)
{
}



template<typename T>
inline
PetscNativePreconditioner<T>::~PetscNativePreconditioner ()
{
	this->clear ();
}


template<typename T>
inline
void PetscNativePreconditioner<T>::set_operator(KARMALinearOperator& linop)
{
	_lin_operator = (&linop) ;
}


template<typename T>
inline
KARMALinearOperator& PetscNativePreconditioner<T>::get_operator() const
{
	if (_lin_operator == NULL)
		karma_error() ;
	
	return *_lin_operator ;
}


template<typename T>
inline
std::string& PetscNativePreconditioner<T>::name()
{
	return _name ;
}


template<typename T>
inline
const std::string& PetscNativePreconditioner<T>::name() const
{
	return _name ;
}


template<typename T>
inline
void PetscNativePreconditioner<T>::set_matrix(MatrixType& mat)
{
	//If the matrix is changing then we (probably) need to reinitialize.
	_is_initialized = false;
	_matrix = &mat;
}


#endif // #ifdef __petsc_native_preconditioner_h__
