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


#include "sdirk33.h"

#include "karma.h"
	
SDIRK33::SDIRK33() :
			RK(3, 3, KARMA::TEMPORAL::SDIRK33, KARMA::TEMPORAL::SDIRK)
{
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
		{
			A[i][j] = 0.0 ;
			AInv[i][j] = 0.0 ;
		}
	
	// Method: Butcher Numerical methods for ODE; p-245
	// Method: Alexander - Diagonally Implicit Runge-Kutta Methods for Stiff O.D.E.'s
	// A, L-stable method 
	// gamma is the root of x^3 - 3*x^2 + 1.5*x - 1/6 = 0  where gamma \in (1/6,1/2)
	double gamma =  0.43586652150845899941601945119356 ;
	
	C[0] = gamma ; C[1] = 0.5*(1.0+gamma) ; C[2] = 1.0 ;

	B[0] = -0.25*(6.*gamma*gamma-16.*gamma+1.0) ; B[1] = 0.25*(6.*gamma*gamma-20.*gamma+5.0) ; B[2] = gamma ;

	A[0][0] = gamma ;
	A[1][0] = 0.5*(1.0-gamma) ; A[1][1] = gamma ;
	A[2][0] = -0.25*(6.*gamma*gamma-16.*gamma+1.0) ; A[2][1] = 0.25*(6.*gamma*gamma-20.*gamma+5.0) ; A[2][2] = gamma ;
	
	// An alternate version that is A-stable but not L-stable
	// This has lower truncation error estimates
	// double gamma = 0.146446609406726 ;
	
	// C[0] = gamma ; C[1] = 0.5 ; C[2] = 1.0-gamma ;

	// B[0] = 1.0/3 ; B[1] = 1.0/3 ; B[2] = 1.0/3 ;

	// A[0][0] = gamma ;
	// A[1][0] = 0.353553390593274 ; A[1][1] = gamma ;
	// A[2][0] = 0.353553390593274 ; A[2][1] = 0.353553390593274 ; A[2][2] = gamma ;

}


SDIRK33::~SDIRK33()
{
}

