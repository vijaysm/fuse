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


#include "dopri45.h"

DOPRI45::DOPRI45() :
			RK(7, 5, KARMA::TEMPORAL::DOPRI45, KARMA::TEMPORAL::ERKEmbedded)
{	
	E[0] = 71.0/57600.0 ;	E[1] = 0.0 ;	E[2] = -71.0/16695.0 ; E[3] = 71.0/1920.0;
	E[4] = -17253.0/339200.0 ; E[5] = 22.0/525.0; E[6] = -1.0/40.0 ;

	C[0] = 0.0 ; C[1] = 1.0/5 ; C[2] = 3.0/10 ; C[3] = 4.0/5 ;
	C[4] = 8.0/9 ; C[5] = 1.0 ; C[6] = 1.0 ;

	B[0] = 35.0/384.0 ; B[1] = 0.0 ; B[2] = 500.0/1113.0 ;
	B[3] = 125.0/192.0 ; B[4] = -2187.0/6784.0 ; B[5] = 11.0/84.0 ; B[6] = 0.0 ;

	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
			A[i][j] = 0.0 ;
	
	A[1][0] = 0.2 ;
	A[2][0] = 3.0/40.0 ; A[2][1] = 9.0/40.0 ;
	A[3][0] = 44.0/45.0 ; A[3][1] = -56.0/15.0 ; A[3][2] = 32.0/9.0 ;
	A[4][0] = 19372.0/6561.0 ; A[4][1] = -25360.0/2187.0 ;
	A[4][2] = 64448.0/6561.0;  A[4][3] = -212.0/729.0 ;
	A[5][0] = 9017.0/3168.0 ; A[5][1] = -355.0/33.0 ; A[5][2] = 46732.0/5247.0 ;
	A[5][3] = 49.0/176.0 ; A[5][4] = -5103.0/18656.0 ;
	A[6][0] = 35.0/384.0 ; A[6][1] = 0.0 ; A[6][2] = 500.0/1113.0 ;
	A[6][3] = 125.0/192.0 ; A[6][4] = -2187.0/6784.0 ; A[6][5] = 11.0/84.0 ;
	
}

DOPRI45::~DOPRI45()
{
}

