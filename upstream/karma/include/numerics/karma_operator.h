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


#ifndef __karma_operator_h__
#define __karma_operator_h__

// C++ includes
#include <vector>

#include "karma.h"
#include "karma_vector.h"
#include "karma_matrix.h"

// Forward declarations
class KARMAParameters ;

typedef PetscErrorCode (*matopr_function)(const KARMARealVector& X, KARMARealVector& R, const KARMAParameters& varparams, KARMAParameters* out_varparams);

typedef PetscErrorCode (*matrix_function)(const KARMARealVector& X, KARMARealMatrix& J, const KARMAParameters& varparams, KARMAParameters* out_varparams);

/**
 * This class implements the KARMAOperator which serves as the abstract base for
 * both the Linear and Nonlinear Type operators used in KARMA. 
 * The Operator defined here represents a complete set of tools to operate on or
 * manipulate a linear/nonlinear system of equations and solve it to obtain the solution.
 * The operators store the interface routines necessary to perform this.
 * In theory, KARMAOperator is generic and can hold all operations that obey the definition of
 * matopr_function and matrix_function.
 *
 * @author Vijay M, 2009
 */

class KARMAOperator
{
public:

	/**
	* Destructor.
	*/
	virtual ~KARMAOperator ();

	/**
   * Function that checks if the current operator is ready for use
   */
	virtual bool is_ready() const = 0 ;
	
	void set_params(KARMAParameters& params) ;

	const KARMAParameters& get_params() const ;

protected:

	/**
	*  Constructor. Initializes Operator data structures
	*/
	KARMAOperator (unsigned int num_matopr_type, unsigned int num_matrix_type);

	/**
	*  Copy Constructor. Initializes Operator data structures from given object.
	*/
	KARMAOperator (const KARMAOperator& dummy);

	/**
	* Function that computes the residual \p R(X) of the nonlinear system
	* at the input iterate \p X.
	*/
	PetscErrorCode call (unsigned int func_id, const KARMARealVector& X, KARMARealVector& R,
			const KARMAParameters& varparams, KARMAParameters* out_varparams=NULL) const ;
	
	/**
	* Function that computes the Jacobian \p J(X) or its approximation of the nonlinear system
	* at the input iterate \p X.
	*/
	PetscErrorCode call (unsigned int func_id, const KARMARealVector& X, KARMARealMatrix& J,
			const KARMAParameters& varparams, KARMAParameters* out_varparams=NULL) const ;

	// Protected member attributes
	unsigned int _n_operations ;
	unsigned int _n_matopr_type ;
	unsigned int _n_matrix_type ;
	
	std::vector<matopr_function> _matopr_fnptrs ;
	std::vector<matrix_function> _matrix_fnptrs ;
	
	KARMAParameters* _const_input_params ;
	
};

// inline functions for KARMAOperator

inline
void KARMAOperator::set_params(KARMAParameters& params)
{
	_const_input_params = &params ;
}

inline
const KARMAParameters& KARMAOperator::get_params() const
{
	if (_const_input_params == NULL)
		karma_error() ;
	
	return *_const_input_params ;
}

inline
PetscErrorCode KARMAOperator::call (unsigned int func_id, const KARMARealVector& X, KARMARealVector& R,
									const KARMAParameters& varparams, KARMAParameters* out_varparams) const
{
	karma_assert(func_id < _n_operations) ;
	karma_assert(func_id < _n_matopr_type) ;
	
	return (*_matopr_fnptrs[func_id])(X, R, varparams, out_varparams) ;
}

inline
PetscErrorCode KARMAOperator::call (unsigned int func_id, const KARMARealVector& X, KARMARealMatrix& J,
									const KARMAParameters& varparams, KARMAParameters* out_varparams) const
{
	karma_assert(func_id < _n_operations) ;
	karma_assert(func_id < _n_matrix_type) ;
	
	return (*_matrix_fnptrs[func_id])(X, J, varparams, out_varparams) ;
}

#endif	// define __karma_operator_h__
