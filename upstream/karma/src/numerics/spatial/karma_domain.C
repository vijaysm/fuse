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


#include "karma_domain.h"
#include <cassert>

  /**
   *  Constructor. Initializes Solver data structures
   */
KARMADomain::KARMADomain (unsigned int dim, DomainShape type, double _LX, double _LY, double _LZ, bool _isSym)
{
	Type = type ;
	
	// currently only cubic shapes are supported since i use build_cube function to generate mesh
	// in future when a gneral unstructured mesh is gennerated, this restriction will be lifted off
	assert(Type != Hexagon || Type != Unstructured) ;
	
	Dim = dim ;
	LX = _LX ;
	LY = _LY ;
	LZ = _LZ ;
	IsSymmetricDomain = _isSym ;
	IsSymmetricDomainX = _isSym ;
	IsSymmetricDomainY = _isSym ;
	IsSymmetricDomainZ = _isSym ;
	
}

  /**
   *  Constructor. Initializes Solver data structures
   */
KARMADomain::KARMADomain (unsigned int dim, DomainShape type, double _LX, double _LY, double _LZ, bool _isSymX, bool _isSymY, bool _isSymZ)
{
	Type = type ;
	
	// currently only cubic shapes are supported since i use build_cube function to generate mesh
	// in future when a gneral unstructured mesh is gennerated, this restriction will be lifted off
	assert(Type != Hexagon || Type != Unstructured) ;
	
	Dim = dim ;
	LX = _LX ;
	LY = _LY ;
	LZ = _LZ ;
	IsSymmetricDomain = _isSymX & _isSymY & _isSymZ ;
	IsSymmetricDomainX = _isSymX ;
	IsSymmetricDomainY = _isSymY ;
	IsSymmetricDomainZ = _isSymZ ;
	
}


  /**
   *  Copy Constructor
   */
KARMADomain::KARMADomain (KARMADomain &_dom)
{
	Type = _dom.Type ;
	
	// currently only cubic shapes are supported since i use build_cube function to generate mesh
	// in future when a gneral unstructured mesh is gennerated, this restriction will be lifted off
	assert(Type != Hexagon || Type != Unstructured) ;
	
	Dim = _dom.Dim ;
	LX = _dom.LX ;
	LY = _dom.LY ;
	LZ = _dom.LZ ;
	IsSymmetricDomain = _dom.IsSymmetricDomain ;
	IsSymmetricDomainX = _dom.IsSymmetricDomainX ;
	IsSymmetricDomainY = _dom.IsSymmetricDomainY ;
	IsSymmetricDomainZ = _dom.IsSymmetricDomainZ ;
	
}


KARMADomain::~KARMADomain ()
{
	
}
