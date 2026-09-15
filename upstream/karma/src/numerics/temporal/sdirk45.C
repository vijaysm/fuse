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


#include "sdirk45.h"

SDIRK45::SDIRK45() :
			RK(4, 5, KARMA::TEMPORAL::SDIRK45, KARMA::TEMPORAL::SDIRK)
{
	// Method: Butcher Numerical methods for ODE; p-212
	// 4 stage 5th order method with explicit first stage
	C[0] = 0 ; C[1] = 0.25 ; C[2] = 0.7 ; C[3] = 1.0 ;

	B[0] = 1.0/14.0 ; B[1] = 32.0/81.0 ; B[2] = 250.0/567.0 ; B[3] = 5.0/54.0 ;

	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
			A[i][j] = 0.0 ;
	
	A[0][0] = 0. ;
	A[1][0] = 1.0/8.0 ; A[1][1] = 1.0/8.0 ;
	A[2][0] = -0.01 ; A[2][1] = 14.0/25.0 ; A[2][2] = 3.0/20.0 ;
	A[3][0] = 2.0/7.0 ; A[3][1] = 0.0 ; A[3][2] = 5.0/7.0 ; A[3][3] = 0.0 ;
	
}


SDIRK45::~SDIRK45()
{
}

