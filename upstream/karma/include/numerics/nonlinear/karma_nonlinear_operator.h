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


#ifndef __karma_nonlinear_operator_h__
#define __karma_nonlinear_operator_h__

#include "karma_nonlinear_operations.h"
#include "karma_linear_operator.h"

/**
 * This class implements the NonlinearOperator derived from KARMAOperator.
 * Essentially a Nonlinear system can be completely manipulated if the residual, 
 * a jacobian LinearOperator are provided by user.
 *
 * Note: If matrix_function is provided, no need for either matvec or solve.
 * 
 * @author Vijay M, 2009
 */

class KARMANonlinearOperator : public KARMALinearOperator
{
public:

  /**
   *  Constructor. Initializes Solver data structures
   */
  KARMANonlinearOperator ();
    
  /**
   *  Copy constructor. Initializes and calls Parent KARMAOperator
   *  and copies the temporary id's from the supplied object.
   */
  KARMANonlinearOperator (const KARMANonlinearOperator& dummy) ;
   
  /**
   *  Assignment operator. Copies the data from supplied KARMAOperator
   *  object.
   */
  KARMANonlinearOperator& operator = (const KARMANonlinearOperator& dummy) ;
   
  /**
   * Virtual destructor.
   */
  virtual ~KARMANonlinearOperator ();
	
  void attach_residual_function(matopr_function m) ;
	
  /**
   * Function that computes the residual \p R(X) of the nonlinear system
   * at the input iterate \p X.
   */
  PetscErrorCode residual (const KARMARealVector& X, KARMARealVector& R,
				const KARMAParameters& varparams, KARMAParameters* out_varparams=NULL) const ;
   
   virtual bool is_ready() const ;
   
   bool is_residual() const ;
   
   void  get_raw_function_pointer(const KARMA::KARMANonlinearOperations& ops, matopr_function *funptr) ;

   void  get_raw_function_pointer(const KARMA::KARMANonlinearOperations& ops, matrix_function *funptr) ;


protected:

	const unsigned int _residual_id ;
	
	unsigned int _temp_residual_id ;
	
};


// Inline functions for KARMANonlinearOperator

inline
KARMANonlinearOperator::~KARMANonlinearOperator ()
{
}

inline
PetscErrorCode KARMANonlinearOperator::residual (const KARMARealVector& X, KARMARealVector& R,
		const KARMAParameters& varparams, KARMAParameters* out_varparams) const
{
	karma_assert(_residual_id < _n_matopr_type) ;
	
	return this->call(_residual_id, X, R, varparams, out_varparams) ;
}

inline
bool KARMANonlinearOperator::is_residual() const
{
	return (_temp_residual_id == _residual_id) ;
}

inline
bool KARMANonlinearOperator::is_ready() const
{
	return (is_residual() && KARMANonlinearOperator::is_ready()) ;
}

inline
void  KARMANonlinearOperator::get_raw_function_pointer(const KARMA::KARMANonlinearOperations& ops, matopr_function *funptr)
{
	int op_num = static_cast<int>(ops) ;
	
	switch(op_num)
	{
		case 0:
			funptr = &_matopr_fnptrs[0] ;
		case 1:
			funptr = &_matopr_fnptrs[1] ;
		case 3:
			funptr = &_matopr_fnptrs[2] ;
		default:
			funptr = NULL ;
	}
}

inline
void  KARMANonlinearOperator::get_raw_function_pointer(const KARMA::KARMANonlinearOperations& ops, matrix_function *funptr)
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

#endif	// define __karma_nonlinear_operator_h__
