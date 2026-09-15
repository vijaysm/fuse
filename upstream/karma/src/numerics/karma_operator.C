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


#include "karma_operator.h"
#include "karma_parameters.h"


KARMAOperator::KARMAOperator (unsigned int num_matopr_type, unsigned int num_matrix_type) :
		_n_operations(num_matopr_type+num_matrix_type),
		_n_matopr_type(num_matopr_type),
		_n_matrix_type(num_matrix_type)
{	
	_matopr_fnptrs.resize(num_matopr_type) ;
	_matrix_fnptrs.resize(num_matrix_type) ;
}

KARMAOperator::KARMAOperator (const KARMAOperator& dummy) :
		_n_operations(dummy._n_operations),
		_n_matopr_type(dummy._n_matopr_type),
		_n_matrix_type(dummy._n_matrix_type)
{
	_matopr_fnptrs.resize(_n_matopr_type) ;
	_matrix_fnptrs.resize(_n_matrix_type) ;
	
	for (unsigned int i=0; i<_matopr_fnptrs.size(); i++)
		_matopr_fnptrs[i] = dummy._matopr_fnptrs[i] ;
	
	for (unsigned int i=0; i<_matrix_fnptrs.size(); i++)
		_matrix_fnptrs[i] = dummy._matrix_fnptrs[i] ;
}


KARMAOperator::~KARMAOperator ()
{
	_matopr_fnptrs.clear() ;
	_matrix_fnptrs.clear() ;
	
	_n_operations = _n_matopr_type = _n_matrix_type = 0 ;
}
