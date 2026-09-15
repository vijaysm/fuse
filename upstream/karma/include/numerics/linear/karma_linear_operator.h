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


#ifndef __karma_linear_operator_h__
#define __karma_linear_operator_h__

#include "karma_linear_operations.h"
#include "karma_operator.h"

/**
 * This class implements the LinearOperator derived from KARMAOperator.
 * Essentially a linear system can be completely manipulated if a matvec, 
 * solve/matrix operations are provided by user.
 * This class serves to abstract out the kind of Matrix being used and
 * the use of KSP, PC objects from the user, leaving only the assembly
 * routines (if mat-free) to be implemented.
 *
 * Note: If matrix_function is provided, no need for either matvec or solve.
 * 
 * @author Vijay M, 2009
 */


// ------------------------------------------------------------
// KARMALinearOperator class definition
class KARMALinearOperator : public KARMAOperator
{
public:

  /**
   *  Constructor. Initializes and calls Parent KARMAOperator
   */
  KARMALinearOperator ();
   
  /**
   *  Copy constructor. Initializes and calls Parent KARMAOperator
   *  and copies the temporary id's from the supplied object.
   */
  KARMALinearOperator (const KARMALinearOperator& dummy) ;
   
  /**
   *  Copy constructor. Initializes and calls Parent KARMAOperator
   *  and copies the temporary id's from the supplied object.
   */
  KARMALinearOperator& operator = (const KARMALinearOperator& dummy) ;
   
  /**
   * Destoys all the function pointers allocated.
   */
  virtual ~KARMALinearOperator ();
  
  void attach_matvec_function(matopr_function m) ;
  
  void attach_solve_function(matopr_function m) ;
  
  void attach_matrix_function(matrix_function m) ;
  
  /**
   * Function that computes the residual \p R(X) of the nonlinear system
   * at the input iterate \p X.
   */
  PetscErrorCode matvec (const KARMARealVector& X, KARMARealVector& R,
			const KARMAParameters& varparams, KARMAParameters* out_varparams=NULL) const ;

  /**
   * Function that computes the Jacobian \p J(X) or its approximation of the nonlinear system
   * at the input iterate \p X.
   */
  PetscErrorCode matrix (const KARMARealVector& X, KARMARealMatrix& J,
			const KARMAParameters& varparams, KARMAParameters* out_varparams=NULL) const ;

  /**
   * Function that inverts the matrix \p J(X) if provided or perform matrix-free
   * solve and multiply onto a given vector \p X.
   * Requires that both residual and jacobian function if inverting matrix
   * Requires that residual function only if the solve is done matrix-free
   */
  PetscErrorCode solve (const KARMARealVector& X, KARMARealVector& Y,
			const KARMAParameters& varparams, KARMAParameters* out_varparams=NULL) const ;
	
   void  get_raw_function_pointer(const KARMA::KARMALinearOperations& ops, matopr_function *funptr) ;

   void  get_raw_function_pointer(const KARMA::KARMALinearOperations& ops, matrix_function *funptr) ;
	
	bool is_matvec() const ;
	
	bool is_matrix() const ;
	
	bool is_solve() const ;

   virtual bool is_ready()  const;
   
protected:

	/**
	*  Constructor. Initializes Solver data structures
	*/
	KARMALinearOperator (unsigned int num_matvec_type, unsigned int num_matrix_type);

	const unsigned int _matvec_id, _solve_id, _matrix_id ;
	
	unsigned int _temp_matvec_id, _temp_solve_id, _temp_matrix_id ;
	
};

// Inline routines for KARMALinearOperator


inline
KARMALinearOperator::~KARMALinearOperator ()
{
}


inline
void KARMALinearOperator::attach_matvec_function(matopr_function f)
{
	karma_assert(f != NULL) ;
	
	_temp_matvec_id = _matvec_id ;
	_matopr_fnptrs[_matvec_id] = f ;
}


inline
void KARMALinearOperator::attach_solve_function(matopr_function f)
{
	karma_assert(f != NULL) ;
	
	_temp_solve_id = _solve_id ;
	_matopr_fnptrs[_solve_id] = f ;
}


inline
void KARMALinearOperator::attach_matrix_function(matrix_function f)
{
	karma_assert(f != NULL) ;
	
	_temp_matrix_id = _matrix_id ;
	_matrix_fnptrs[_matrix_id] = f ;
}

inline
bool KARMALinearOperator::is_matvec() const
{
	return (_temp_matvec_id==_matvec_id) ;
}

inline
bool KARMALinearOperator::is_matrix() const
{
	return (_temp_matrix_id==_matrix_id) ;
}

inline
bool KARMALinearOperator::is_solve() const
{
	return (_temp_solve_id==_solve_id) ;
}

inline
		PetscErrorCode KARMALinearOperator::matvec (const KARMARealVector& X, KARMARealVector& Y,
			const KARMAParameters& varparams, KARMAParameters* out_varparams) const
{
	karma_assert(_matvec_id < _n_matopr_type) ;
	
	return this->call(_matvec_id, X, Y, varparams, out_varparams) ;
}

inline
		PetscErrorCode KARMALinearOperator::matrix (const KARMARealVector& X, KARMARealMatrix& J,
			const KARMAParameters& varparams, KARMAParameters* out_varparams) const
{
	karma_assert(_matrix_id < _n_matrix_type) ;
	
	return this->call(_matrix_id, X, J, varparams, out_varparams) ;
}

inline
		PetscErrorCode KARMALinearOperator::solve (const KARMARealVector& X, KARMARealVector& Y,
			const KARMAParameters& varparams, KARMAParameters* out_varparams) const
{
	karma_assert(_solve_id < _n_matopr_type) ;
	
	return this->call(_solve_id, X, Y, varparams, out_varparams) ;
}

inline
void KARMALinearOperator::get_raw_function_pointer(const KARMA::KARMALinearOperations& ops, matrix_function *funptr)
{
	int op_num = static_cast<int>(ops) ;
	
	switch(op_num)
	{
		case 2:
			funptr = &_matrix_fnptrs[0] ;
		default:
			funptr = NULL ;
	}
}


inline
void KARMALinearOperator::get_raw_function_pointer(const KARMA::KARMALinearOperations& ops, matopr_function *funptr)
{
	int op_num = static_cast<int>(ops) ;
	
	switch(op_num)
	{
		case 0:
			funptr = &_matopr_fnptrs[0] ;
		case 1:
			funptr = &_matopr_fnptrs[1] ;
		default:
			funptr = NULL ;
	}
}


inline
bool KARMALinearOperator::is_ready() const
{
	return (is_matvec() || is_matrix()) ;
}


#endif	// define __karma_linear_operator_h__
