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


#include "grk4t.h"

GRK4T::GRK4T() : Rosenbrock(4, 4, 3, KARMA::TEMPORAL::GRK4T)
{
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
		{
			A[i][j] = 0.0 ;
			G[i][j] = 0.0 ;
		}
	
	/*
	This is the L-stable GRK method by Kaps and Rentrop.
	Method = GRK4Tn2 
	Number of stages = 4 
	High/low orders  = 4, 3 
		Gamma value =   2.3100000000000001E-001 
		Alpha vector:
		  0.0000000000000000E+000
		  4.6200000000000002E-001
		  8.8020833333299997E-001
		  8.8020833333299997E-001
		A matrix (strictly lower triangular):
		  row 2 :   2.0000000000000000E+000 
		  row 3 :   4.5247082073599998E+000   4.1635287885999999E+000 
		  row 4 :   4.5247082073599998E+000   4.1635287885999999E+000   0.0000000000000000E+000 
		Gamma hat vector:
		  5.3361000000000006E-002
		 -9.1544532507119997E-003
		  1.2722993504274901E-001
		 -1.2786077435670000E-002
		C hat matrix (strictly lower triangular):
		  row 2 :  -1.1715570032558702E+000   0.0000000000000000E+000 
		  row 3 :   1.3906552803181502E+000   3.6902408159463003E-002   0.0000000000000000E+000 
		  row 4 :  -4.2881537591438701E-001  -1.9647429782418899E+000  -4.8142135642062006E-001   0.0000000000000000E+000 
		d vector for the higher order solution:
		  3.9575037466300000E+000
		  4.6248923883600002E+000
		  6.1747726387299995E-001
		  1.2826129452680000E+000
		d prime vector for the lower order solution:
		  6.2596591495500000E+000
		  7.6985268737499997E+000
		 -2.5580353792900001E-001
		  0.0000000000000000E+000
	*/
	double gamma = 0.231 ;
	
	C[0] = 0. ; C[1] = 4.6200000000000002E-001 ; C[2] = 8.8020833333299997E-001 ; C[3] = 8.8020833333299997E-001 ;

	B[0] = 3.9575037466300000E+000 ; B[1] = 4.6248923883600002E+000 ; B[2] = 6.1747726387299995E-001; B[3] = 1.2826129452680000E+000 ;
	E[0] = B[0] - 6.2596591495500000E+000 ; E[1] = B[1] - 7.6985268737499997E+000 ; E[2] = B[2] + 2.5580353792900001E-001 ; E[3] = B[3] ;
	
	G[0][0] = gamma;
	G[1][0] = -1.1715570032558702E+000 ;	G[1][1] = gamma;
	G[2][0] = 1.3906552803181502E+000 ;	G[2][1] = 3.6902408159463003E-002;	G[2][2] = gamma;
	G[3][0] = -4.2881537591438701E-001 ;	G[3][1] = -1.9647429782418899E+000; G[3][2] = -4.8142135642062006E-001; G[3][3] = gamma;
	
	A[1][0] = 2.0000000000000000E+000   ;
	A[2][0] = 4.5247082073599998E+000  ; A[2][1] = 4.1635287885999999E+000 ;
	A[3][0] = 4.5247082073599998E+000  ; A[3][1] = 4.1635287885999999E+000 ; A[3][2] = 0.0 ; 
	
}

GRK4T::~GRK4T()
{
}


