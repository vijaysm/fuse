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
#include "rk.h"
// #include "forward_euler.h"
// #include "backward_euler.h"
// #include "crank_nicolson.h"
// #include "dopri45.h"
// #include "sdirk22.h"
// #include "sdirk23.h"
// #include "sdirk33.h"
// #include "sdirk34.h"
// #include "sdirk45.h"
// #include "trbe.h"
// #include "sdirk332.h"
// #include "sdirk543.h"
// #include "sdirk643.h"
// #include "kutta44.h"
// #include "kutta22.h"
// #include "ros22.h"
// #include "grk4a.h"
// #include "grk4t.h"
// #include "grk4sn.h"
// #include "ros3p.h"
// #include "rodasp.h"

void RK::init ()
{
	karma_assert(_order >= _secondary_order) ;
	
	B = new Real[_primary_stages] ;
	C = new Real[_primary_stages] ;
	E = new Real[_primary_stages] ;
	
	A = new Real*[_primary_stages] ;
	AInv = new Real*[_primary_stages] ;
	for(unsigned int istage = 0; istage < _primary_stages; istage++)
	{
		A[istage] = new Real[_primary_stages] ;
		AInv[istage] = new Real[_primary_stages] ;
	}
	
	_is_A_invertible = false ;
}

void RK::clear ()
{
}

RK::RK(unsigned int nstages, unsigned int norder, KARMA::TEMPORAL::RKType type, KARMA::TEMPORAL::RKCategory cat) :
		KARMATSBase(norder), _primary_stages(nstages), _secondary_order(norder), _type(type), _cat(cat)
{
	init () ;
}

RK::RK(unsigned int nstages, unsigned int norder, unsigned int nsec_order, KARMA::TEMPORAL::RKType type, KARMA::TEMPORAL::RKCategory cat) :
		KARMATSBase(norder), _primary_stages(nstages), _secondary_order(nsec_order), _type(type), _cat(cat)
{
	init () ;
}
	
RK::~RK() 
{
	clear () ;
	
	delete [] B ;
	delete [] C ;
	delete [] E ;

	for(unsigned int istage = 0; istage < _primary_stages; istage++)
	{
		delete [] A[istage] ;
		delete [] AInv[istage] ;
	}
	
	delete A ;	A = NULL ;// delete A ;
	delete AInv ;	AInv = NULL ;// delete AInv ;
}

// template <typename DerivedRKType>
RK* RK::build(KARMA::TEMPORAL::RKType type)
// RK* RK::build()
{
	return RKMethodTraits<enum_traits<KARMA::TEMPORAL::RKType>::N>::build(type) ;
	// return static_cast<RK *>(new DerivedRKType()) ;
	// RK* obj ;
	
	// for (unsigned int index=0; index < enum_traits<KARMA::TEMPORAL::RKType>::N; index++)
	// {
		// if (RKMethodTraits<index>::Type == type)
		// {
			// obj = new RKMethodTraits<index>::MethodType() ;
			// break ;
		// }
	// }
	
	// if (type == KARMA::TEMPORAL::FE)
		// obj = static_cast<RK *>(new FE()) ;
	// else if (type == KARMA::TEMPORAL::BE)
		// obj = static_cast<RK *>(new BE()) ;
	// else if(type == KARMA::TEMPORAL::CN)
		// obj = static_cast<RK *>(new CN()) ;
	// else if(type == KARMA::TEMPORAL::DOPRI45)
		// obj = static_cast<RK *>(new DOPRI45()) ;
	// else if(type == KARMA::TEMPORAL::SDIRK22)
		// obj = static_cast<RK *>(new SDIRK22()) ;
	// else if(type == KARMA::TEMPORAL::SDIRK23)
		// obj = static_cast<RK *>(new SDIRK23()) ;
	// else if(type == KARMA::TEMPORAL::SDIRK33)
		// obj = static_cast<RK *>(new SDIRK33()) ;
	// else if(type == KARMA::TEMPORAL::SDIRK34)
		// obj = static_cast<RK *>(new SDIRK34()) ;
	// else if(type == KARMA::TEMPORAL::SDIRK45)
		// obj = static_cast<RK *>(new SDIRK45()) ;
	// else if(type == KARMA::TEMPORAL::TRBE)
		// obj = static_cast<RK *>(new TRBE()) ;
	// else if(type == KARMA::TEMPORAL::SDIRK332)
		// obj = static_cast<RK *>(new SDIRK332()) ;
	// else if(type == KARMA::TEMPORAL::SDIRK543)
		// obj = static_cast<RK *>(new SDIRK543()) ;
	// else if(type == KARMA::TEMPORAL::SDIRK643)
		// obj = static_cast<RK *>(new SDIRK643()) ;
	// else if(type == KARMA::TEMPORAL::KUTTA22)
		// obj = static_cast<RK *>(new KUTTA22()) ;
	// else if(type == KARMA::TEMPORAL::KUTTA44)
		// obj = static_cast<RK *>(new KUTTA44()) ;
	// else if(type == KARMA::TEMPORAL::ROS22)
		// obj = static_cast<RK *>(new ROS22()) ;
	// else if(type == KARMA::TEMPORAL::GRK4A)
		// obj = static_cast<RK *>(new GRK4A()) ;
	// else if(type == KARMA::TEMPORAL::GRK4T)
		// obj = static_cast<RK *>(new GRK4T()) ;
	// else if(type == KARMA::TEMPORAL::GRK4Sn)
		// obj = static_cast<RK *>(new GRK4Sn()) ;
	// else if(type == KARMA::TEMPORAL::ROS3P)
		// obj = static_cast<RK *>(new ROS3P()) ;
	// else if(type == KARMA::TEMPORAL::RODASP)
		// obj = static_cast<RK *>(new RODASP()) ;
	// else
		// obj = NULL ;
	
	// return obj ;
}

void RK::print()
{
	std::cout << "\nNow printing coefficients for SDIRK643\n" ;
	for(int i=0; i<_primary_stages; i++)
	{
		std::cout << "\n" ;
		for(int j=0; j<_primary_stages; j++)
			std::cout << A[i][j] << ", " ;
	}
	std::cout << "\nB Coefficients\n" ;
	for(int i=0; i<_primary_stages; i++)
		std::cout << B[i] << ", " ;
	std::cout << "\nC Coefficients\n" ;
	for(int i=0; i<_primary_stages; i++)
		std::cout << C[i] << ", " ;
	std::cout << "\n" ;
}
