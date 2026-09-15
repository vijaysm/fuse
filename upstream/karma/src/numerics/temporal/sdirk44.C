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


#include "sdirk44.h"

SDIRK44::SDIRK44() :
			RK(4, 4, KARMA::TEMPORAL::SDIRK44, KARMA::TEMPORAL::SDIRK)
{
	// Fourth order L-stable and stiffly accurate scheme.
	C[0] = 0.5 ; C[1] = 2.0/3.0 ; C[2] = 0.5 ; C[3] = 1.0 ;

	B[0] = 1.5 ; B[1] = -1.5 ; B[2] = 0.5 ; B[3] = 0.5 ;

	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
			A[i][j] = 0.0 ;
	
	A[0][0] = 0.5 ;
	A[1][0] = 1.0/6.0 ; A[1][1] = 0.5 ;
	A[2][0] = -0.5 ; A[2][1] = 0.5 ; A[2][2] = 0.5 ;
	A[3][0] = 1.5 ; A[3][1] = -1.5 ; A[3][2] = 0.5 ; A[3][3] = 0.5 ;
	
}


SDIRK44::~SDIRK44()
{
}

