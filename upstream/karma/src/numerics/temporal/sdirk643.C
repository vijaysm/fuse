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


#include "sdirk643.h"

SDIRK643::SDIRK643() :
			RK(6, 4, 3, KARMA::TEMPORAL::SDIRK643, KARMA::TEMPORAL::SDIRKEmbedded)
			// stages, hi order, low order 
{
	double gamma = 0.25 ;
	
	// Embedded ESDIRK scheme : ARK4(3)6L[2]SA�ESDIRK: 6 stages, 4th and 3rd order
	// Ref: Additive Runge�Kutta schemes for convection�diffusion�reaction equations, Christopher A. Kennedy, Mark H. Carpenter
	C[0] = 0.0 ; C[1] = 0.5 ; C[2] = 83.0/250.0 ; C[3] = 31.0/50.0 ; C[4] = 17.0/20.0 ; C[5] = 1.0 ;

	B[0] = 82889.0/524892.0 ; B[1] = 0.0 ; B[2] = 15625.0/83664.0 ; B[3] = 69875.0/102672.0 ; B[4] = -2260.0/8211.0 ; B[5] = 0.25 ;
	
	E[0] = B[0] - 4586570599.0/29645900160.0 ; E[1] = B[1] - 0.0 ; E[2] = B[2] - 178811875.0/945068544.0 ; E[3] = B[3] - 814220225.0/1159782912.0 ; E[4] = B[4] + 3700637.0/11593932.0 ; E[5] = B[5] - 61727.0/225920.0 ;

	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
			A[i][j] = 0.0 ;
	
	A[1][0] = gamma ; A[1][1] = gamma ;
	A[2][0] = 8611.0/62500.0 ; A[2][1] = -1743.0/31250.0 ; A[2][2] = gamma ;
	A[3][0] = 5012029.0/34652500.0 ; A[3][1] = -654441.0/2922500.0 ; A[3][2] = 174375.0/388108.0 ; A[3][3] = gamma ;
	A[4][0] = 15267082809.0/155376265600.0 ; A[4][1] = -71443401.0/120774400.0 ; A[4][2] = 730878875.0/902184768.0 ; A[4][3] = 2285395.0/8070912.0; A[4][4] = gamma ;
	A[5][0] = 82889.0/524892.0 ; A[5][1] = 0.0 ; A[5][2] = 15625.0/83664.0 ; A[5][3] = 69875.0/102672.0 ; A[5][4] = -2260.0/8211.0 ; A[5][5] = gamma ;
	
}

SDIRK643::~SDIRK643()
{
}

