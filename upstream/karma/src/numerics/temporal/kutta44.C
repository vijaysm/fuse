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


#include "kutta44.h"

KUTTA44::KUTTA44() :
			RK(4, 4, KARMA::TEMPORAL::KUTTA44, KARMA::TEMPORAL::ERK)
{	
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
			A[i][j] = 0.0 ;
	
	// Kutta's method based on 3/8th Quadrature Rule
	C[0] = 0.0 ; C[1] = 1.0/3.0 ; C[2] = 2.0/3.0 ; C[3] = 1.0 ;

	B[0] = 1.0/8.0 ; B[1] = 3.0/8.0 ; B[2] = 3.0/8.0 ; B[3] = 1.0/8.0 ;

	A[1][0] = 1.0/3 ;
	A[2][0] = -1.0/3.0 ; A[2][1] = 1.0 ;
	A[3][0] = 1.0 ; A[3][1] = -1.0 ; A[3][2] = 1.0 ;
}

KUTTA44::~KUTTA44()
{
}


