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



#include "karma_error.h"
#include "karma_nonlinear_operator.h"



KARMANonlinearOperator::KARMANonlinearOperator ()
		: KARMALinearOperator(3, 1),
		_residual_id(2),
		_temp_residual_id(255)
{
	
}


KARMANonlinearOperator::KARMANonlinearOperator (const KARMANonlinearOperator& dummy)
	: KARMALinearOperator(dummy),
		_residual_id(2),
		_temp_residual_id(dummy._temp_residual_id)
{ 
}


KARMANonlinearOperator& KARMANonlinearOperator::operator = (const KARMANonlinearOperator& dummy)
{
	_temp_matvec_id = dummy._temp_matvec_id ;
	_temp_solve_id = dummy._temp_solve_id ;
	_temp_matrix_id = dummy._temp_matrix_id ;
	_temp_residual_id = dummy._temp_residual_id ;
	
	_matopr_fnptrs.clear() ;	_matopr_fnptrs.resize(dummy._matopr_fnptrs.size()) ;
	_matrix_fnptrs.clear() ;	_matrix_fnptrs.resize(dummy._matrix_fnptrs.size()) ;
	
	for (unsigned int i=0; i<_matopr_fnptrs.size(); i++)
		_matopr_fnptrs[i] = dummy._matopr_fnptrs[i] ;
	
	for (unsigned int i=0; i<_matrix_fnptrs.size(); i++)
		_matrix_fnptrs[i] = dummy._matrix_fnptrs[i] ;
	
	return *this ;
}


void KARMANonlinearOperator::attach_residual_function(matopr_function f)
{
	karma_assert(f != NULL) ;
	
	_temp_residual_id = _residual_id ;
	_matopr_fnptrs[_residual_id] = f ;
}

