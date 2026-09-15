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


#include "sdirk543.h"

SDIRK543::SDIRK543() :
			RK(5, 4, 3, KARMA::TEMPORAL::SDIRK543, KARMA::TEMPORAL::SDIRKEmbedded)
{
	double gamma = 0.25 ;
	
	// Embedded SDIRK4 scheme by Hairer: SDIRK4(3): 5 stages, 4th and 3rd order embedded
	C[0] = 0.25 ; C[1] = 0.75 ; C[2] = 11.0/20 ; C[3] = 0.5 ; C[4] = 1.0 ;

	B[0] = 25.0/24.0 ; B[1] = -49.0/48.0 ; B[2] = 125.0/16.0 ; B[3] = -85.0/12.0 ; B[4] = gamma ;
	
	E[0] = -3.0/16 ; E[1] = -27.0/32 ; E[2] = 25.0/32 ; E[3] = 0 ; E[4] = gamma ;

	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
			A[i][j] = 0.0 ;
	
	A[0][0] = gamma ;
	A[1][0] = 0.5 ; A[1][1] = gamma ;
	A[2][0] = 17.0/50.0 ; A[2][1] = -1.0/25.0 ; A[2][2] = gamma ;
	A[3][0] = 371.0/1360.0 ; A[3][1] = -137.0/2720.0 ; A[3][2] = 15.0/544.0 ; A[3][3] = gamma ;
	A[4][0] = 25.0/24.0 ; A[4][1] = -49.0/48.0 ; A[4][2] = 125.0/16.0 ; A[4][3] = -85.0/12.0 ;  A[4][4] = gamma ;
}

SDIRK543::~SDIRK543()
{
}
