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
#include "karma_parameters.h"
#include "karma_physics_base_property_manager.h"

KARMAPhysicsBasePropertyManager::KARMAPhysicsBasePropertyManager () :
		_exact_solution(NULL),
		_exact_gradient(NULL)
{
}

KARMAPhysicsBasePropertyManager::KARMAPhysicsBasePropertyManager (const KARMAPhysicsBasePropertyManager& dummy)
	: 	_exact_solution(dummy._exact_solution),
		_exact_gradient(dummy._exact_gradient)
{
	const_property_iterator it = dummy._prop_fnptrs.begin() ;
	for (; it != dummy._prop_fnptrs.end(); it++)
	{
		_prop_fnptrs.insert(std::pair<std::string, parameterized_property_function>(it->first, it->second)) ;
	}
}

KARMAPhysicsBasePropertyManager& KARMAPhysicsBasePropertyManager::operator = (const KARMAPhysicsBasePropertyManager& dummy)
{
	_exact_solution = dummy._exact_solution ;
	_exact_gradient = dummy._exact_gradient ;
	
	_prop_fnptrs.clear () ;
	
	const_property_iterator it = dummy._prop_fnptrs.begin() ;
	for (; it != dummy._prop_fnptrs.end(); it++)
	{
		_prop_fnptrs.insert(std::pair<std::string, parameterized_property_function>(it->first, it->second)) ;
	}
	
	return *this ;
}

KARMAPhysicsBasePropertyManager::~KARMAPhysicsBasePropertyManager ()
{
	_prop_fnptrs.clear() ;
	
	_exact_solution = NULL ;
	_exact_gradient = NULL ;
}

