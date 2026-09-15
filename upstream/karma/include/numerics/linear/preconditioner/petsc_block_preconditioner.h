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


#ifndef __petsc_block_preconditioner_h__
#define __petsc_block_preconditioner_h__

#ifdef USE_PETSC_BLOCK_EXTENSIONS

// C++ includes

// Local includes
#include "enum_solver_package.h"
#include "enum_preconditioner_type.h"

#include "ivector.h"
#include "imatrix.h"

/**
 * Petsc include files.
 */
#include "petscext_pc.h"

// Forward declarations
class KARMALinearOperator;


/**
 * This class provides an interface to the preconditioners available in Petsc.
 * 
 * In the below comments P is the matrix to be preconditioned with Apply()
 * performing the equivalent of the matrix vector product P^-1 x.  This
 * can also be thought of as (usually approximately) solving for Py=x.
 *
 * Original Derek Gaston for libMesh
 *
 * Modifications by @author Vijay Mahadevan, 2009 for KARMA
 */
template<typename T>
class PetscBlockPreconditioner : public IKARMA<PetscBlockPreconditioner<T> >
{
public:

	/**
	*  Constructor. Initializes PetscBlockPreconditioner data structures
	*/
	PetscBlockPreconditioner ();

	/**
	* Destructor.
	*/
	virtual ~PetscBlockPreconditioner ();

	/**
	* Computes the preconditioned vector "y" based on input "x".
	* Usually by solving Py=x to get the action of P^-1 x.
	*/
	void apply(const IVector<VectorTraits<T, KARMA::BLOCK> >& x, IVector<VectorTraits<T, KARMA::BLOCK> >& y);

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
	void set_matrix(IMatrix<MatrixTraits<T, KARMA::BLOCK> >& mat);

	/**
	* Returns the type of preconditioner to use.
	*/
	PreconditionerType type () const
	{
		return _preconditioner_type;
	}

	/**
	 * Set the name for the current solver.
	 */
	void set_name(const std::string& name) ;

	/**
	* Sets the type of preconditioner to use.
	*/
	void set_type (const PreconditionerType pct);

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
	static void set_type (const PreconditionerType& preconditioner_type, PC & pc);
	
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
	IMatrix<MatrixTraits<T, KARMA::BLOCK> >* _matrix;

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


// inline functions for PetscBlockPreconditioner


template<typename T>
inline
PetscBlockPreconditioner<T>::PetscBlockPreconditioner () :
	_lin_operator(NULL),
	_matrix(NULL),
	_preconditioner_type (BJACOBI_PRECOND),
	_is_initialized      (false)
{
}



template<typename T>
inline
PetscBlockPreconditioner<T>::~PetscBlockPreconditioner ()
{
	this->clear ();
}


template<typename T>
inline
void PetscBlockPreconditioner<T>::set_operator(KARMALinearOperator& linop)
{
	_lin_operator = dynamic_cast<KARMALinearOperator>(&linop) ;

	// check if our Operator was really a Linear operator
	karma_assert(_lin_operator != NULL) ;
}



template<typename T>
inline
void PetscBlockPreconditioner<T>::set_name(const std::string& name)
{
	_name = name ;
}

template<typename T>
inline
KARMALinearOperator& PetscBlockPreconditioner<T>::get_operator() const
{
	if (_lin_operator == NULL)
		karma_error() ;
	
	return *_lin_operator ;
}


template<typename T>
inline
void PetscBlockPreconditioner<T>::set_matrix(IMatrix<MatrixTraits<T, KARMA::BLOCK> >& mat)
{
	//If the matrix is changing then we (probably) need to reinitialize.
	_is_initialized = false;
	_matrix = &mat;
}


template<typename T>
inline
void PetscBlockPreconditioner<T>::set_type (const PreconditionerType pct)
{
	//If the preconditioner type changes we (probably) need to reinitialize.
	_is_initialized = false;
	_preconditioner_type = pct;
}

#endif // #ifdef USE_PETSC_BLOCK_EXTENSIONS

#endif // #ifdef __petsc_block_preconditioner_h__
