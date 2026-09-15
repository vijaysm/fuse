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


#include "sdirk332.h"

SDIRK332::SDIRK332() :
			RK(3, 3, 2, KARMA::TEMPORAL::SDIRK332, KARMA::TEMPORAL::SDIRKEmbedded)
{
	double gamma = 5.0/6.0 ;
	
	// Embedded SDIRK scheme : 3 stages, embedded orders 3 and 2
	// Ref: Runge�Kutta research in Trondheim by A. Kv�rn� ,  S. P. N�rsett ,  B. Owren 
	C[0] = gamma ; C[1] = 24.0/108.0 ; C[2] = 1.0-gamma ;

	B[0] = 26.0/61 ; B[1] = 324.0/671.0 ; B[2] = 1.0/11.0 ;
	
	E[0] = B[0] - 25.0/61.0 ; E[1] = B[1] - 36.0/61.0 ; E[2] = B[2]  ;

	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
			A[i][j] = 0.0 ;
	
	A[0][0] = gamma ;
	A[1][0] = -61.0/108.0 ; A[1][1] = gamma ;
	A[2][0] = -23.0/183.0 ; A[2][1] = -33.0/61.0 ; A[2][2] = gamma ;
}

SDIRK332::~SDIRK332()
{
}

