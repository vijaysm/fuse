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


#include "karma_boundary_condition.h"
#include "enum_utils.h"

KARMABoundaryCondition::KARMABoundaryCondition (KARMA::PHYSICS::BCType type, bool time_dependent_bc) : 
		_type(type),
		_normal(NULL),
		_is_time_dependent_bcs(time_dependent_bc)
{
	switch(_type)
	{
		case KARMA::PHYSICS::Dirichlet:
			if (_is_time_dependent_bcs)
				_values.resize(2) ;		// u[x0] = g_d, u[x1] = dg_d/dt
			else
				_values.resize(1) ;		// u[x0] = g_d
			break ;
		case KARMA::PHYSICS::Neumann:
			_normal = new FaceValues() ;	// u.n = g_n
      _values.resize(1) ; // prescribed outward normal flux g
			break ;
		case KARMA::PHYSICS::Robin:
			// a*u.n + b*u = c
			_values.resize(3) ;
			// values[0] = a ; values[1] = b; and values[2] = c 
			_values[0] = 0.0 ;	_values[1] = 0.0 ;	_values[2] = 0.0 ;
			break ;
		default:
			// KARMA::PHYSICS::User = Custom B.C implementation
			// This is just the container; User knows how to handle this data !
			// We will not allocate this but hope that user will call values()
			// and allocate, assign accordingly.
			_values.resize(0) ;
			break ;
	}
}

KARMABoundaryCondition::KARMABoundaryCondition (const KARMABoundaryCondition& dummy)
  : _type(dummy._type),
    _normal(NULL),
    _is_time_dependent_bcs(dummy._is_time_dependent_bcs)
{
	// resize and copy data
	_values.resize(dummy._values.size()) ;
	for (unsigned int i=0; i<_values.size(); i++)
		_values[i] = dummy._values[i] ;
	
	// copy values from normal
	if (dummy._normal != NULL)
	{
		if (_normal == NULL)
			_normal = new FaceValues(*dummy._normal) ;
	}
}
 

KARMABoundaryCondition& KARMABoundaryCondition::operator = (const KARMABoundaryCondition& dummy)
{
  if (this == &dummy)
    return *this ;

  _type = dummy._type ;
  _is_time_dependent_bcs = dummy._is_time_dependent_bcs ;
	
	// resize and copy data
	_values.resize(dummy._values.size()) ;
	for (unsigned int i=0; i<_values.size(); i++)
		_values[i] = dummy._values[i] ;
	
	// copy values from normal
  delete _normal ;
  _normal = dummy._normal != NULL ? new FaceValues(*dummy._normal) : NULL ;

	return *this ;
}

KARMABoundaryCondition::~KARMABoundaryCondition ()
{
	_values.clear() ;
	
	if (_normal != NULL)
		delete _normal ;
	_normal = NULL ;
}

