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


#include "sdirk22.h"

SDIRK22::SDIRK22() :
			RK(2, 2, KARMA::TEMPORAL::SDIRK22, KARMA::TEMPORAL::SDIRK)
{
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
		{
			A[i][j] = 0.0 ;
			AInv[i][j] = 0.0 ;
		}
		
	// double gamma = 1.0 - 1.0/sqrt(2.0) ;
	// // double gamma = 0.25 ;
	
	// C[0] = gamma ; C[1] = 1.0-gamma ;

	// B[0] = 0.5 ; B[1] = 0.5 ;
		
	// A[0][0] = gamma ;
	// // A[1][0] = 2.0/(1.0+2*gamma) ; A[1][1] = gamma ;
	// A[1][0] = (1.0-2*gamma) ; A[1][1] = gamma ;
	
	// _is_A_invertible = true ;
	// AInv[0][0] = 1.0/gamma ;
	// AInv[1][0] = (-1.0+2*gamma)/gamma/gamma ; AInv[1][1] = 1.0/gamma ;
	
	// This is the optimal L-stable SDIRK22 method
	// Interesting to note that this is also IM-BDF2
	// \cite{Wenjun_Ying}: Composite backward differentiation formula: an extension of the TR-BDF2 scheme
	// gamma is the quadratic polynomial of gamma^2 - 2*gamma + 0.5 = 0
	// So gamma is 1.0 +/- 1.0/std::sqrt(2)
	double gamma = 1.0 - 1.0/sqrt(2.0) ;
	
	C[0] = gamma ; C[1] = 1.0 ;

	B[0] = 1.0-gamma ; B[1] = gamma ;

	A[0][0] = gamma ;
	// A[1][0] = 2.0/(1.0+2*gamma) ; A[1][1] = gamma ;
	A[1][0] = (1.0-gamma) ; A[1][1] = gamma ;
	
	_is_A_invertible = true ;
	AInv[0][0] = 1.0/gamma ;
	AInv[1][0] = (-1.0+2*gamma)/gamma/gamma ; AInv[1][1] = 1.0/gamma ;
}

SDIRK22::~SDIRK22()
{
}


