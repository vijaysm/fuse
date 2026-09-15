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


#include "karma.h"
#include "rosenbrock.h"

void Rosenbrock::init ()
{
	// first initialize the base class data
	RK::init() ;
	
	// now let us allocate the Gamma array
	G = new Real*[_primary_stages] ;
	for(int istage = 0; istage < _primary_stages; istage++)
	{
		G[istage] = new Real[_primary_stages] ;
	}
}

void Rosenbrock::clear ()
{
	// first free the base class data
	RK::clear() ;
}

Rosenbrock::Rosenbrock(unsigned int nstages, unsigned int norder, KARMA::TEMPORAL::RKType type) :
		RK(nstages, norder, type, KARMA::TEMPORAL::ROSENBROCK)
{	
	// now allocate the data
	init () ;	
}

Rosenbrock::Rosenbrock(unsigned int nstages, unsigned int norder, unsigned int nsec_order, KARMA::TEMPORAL::RKType type) :
		RK(nstages, norder, nsec_order, type, KARMA::TEMPORAL::ROSENBROCKEmbedded)
{	
	// now allocate the data
	init () ;
}
	
Rosenbrock::~Rosenbrock() 
{
	// let us clear the allocated memory
	clear () ;
	
	// now let us de-allocate the Gamma array
	for(unsigned int istage = 0; istage < _primary_stages; istage++)
	{
		delete [] G[istage] ;
	}
	
	delete [] G ;
}
