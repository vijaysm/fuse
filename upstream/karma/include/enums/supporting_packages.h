// $Id: supporting_packages.h 450 2011-09-19 20:55:34Z vijaysm $

// KARMA: K(C)ode for Accident and Reactor Modeling Analysis
// Copyright (C) 2008-2009  Vijay S Mahadevan
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//

#ifndef __supporting_packages_h__
#define __supporting_packages_h__

namespace KARMA
{

	enum SpatialDiscretizationPackage
	{
		LIBMESH=0,	// default
		DEALII=1, 
		HERMES=2, 
		FENICS=3
	};

	enum LAPackage	// Linear Algebra Solver package
	{
		PETSC=0,	// default
		TRILINOS=1, 
		LASPACK=2
	};

}

#endif // #define __supporting_packages_h__
