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
#include "karma_error.h"
#include "karma_linear_operator.h"


KARMALinearOperator::KARMALinearOperator ()
		: KARMAOperator(2, 1),
		_matvec_id(0),
		_solve_id(1),
		_matrix_id(0),
		_temp_matvec_id(255), 
		_temp_solve_id(255),
		_temp_matrix_id(255)
{
	
} 


KARMALinearOperator::KARMALinearOperator (const KARMALinearOperator& dummy)
	: KARMAOperator(dummy),
		_matvec_id(0),
		_solve_id(1),
		_matrix_id(0),
		_temp_matvec_id(dummy._temp_matvec_id),
		_temp_solve_id(dummy._temp_solve_id),
		_temp_matrix_id(dummy._temp_matrix_id)
{
}


KARMALinearOperator& KARMALinearOperator::operator = (const KARMALinearOperator& dummy)
{
	_temp_matvec_id = dummy._temp_matvec_id ;
	_temp_solve_id = dummy._temp_solve_id ;
	_temp_matrix_id = dummy._temp_matrix_id ;
	
	_matopr_fnptrs.clear() ;	_matopr_fnptrs.resize(dummy._matopr_fnptrs.size()) ;
	_matrix_fnptrs.clear() ;	_matrix_fnptrs.resize(dummy._matrix_fnptrs.size()) ;
	
	for (unsigned int i=0; i<_matopr_fnptrs.size(); i++)
		_matopr_fnptrs[i] = dummy._matopr_fnptrs[i] ;
	
	for (unsigned int i=0; i<_matrix_fnptrs.size(); i++)
		_matrix_fnptrs[i] = dummy._matrix_fnptrs[i] ;
	
	return *this ;
}


KARMALinearOperator::KARMALinearOperator (unsigned int num_matopr_type, unsigned int num_matrix_type)
		: KARMAOperator(num_matopr_type, num_matrix_type),
		_matvec_id(0),
		_solve_id(1),
		_matrix_id(0),
		_temp_matvec_id(255),
		_temp_solve_id(255),
		_temp_matrix_id(255)
{
	// We need atleast 1 function: More specifically the matvec function pointer.
	karma_assert(num_matopr_type >= 1) ;
}

