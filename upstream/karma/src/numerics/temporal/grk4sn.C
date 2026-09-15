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


#include "grk4sn.h"

GRK4Sn::GRK4Sn() : Rosenbrock(4, 4, 3, KARMA::TEMPORAL::GRK4Sn)
{
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
		{
			A[i][j] = 0.0 ;
			G[i][j] = 0.0 ;
		}
	
	/*
	This is the A-stable GRK method by Shampine.
	Method =  GRK4Sn 
	Number of stages = 4 
	High/low orders  = 4, 3 
		Gamma value =   5.0000000000000000E-001 
		Alpha vector:
		  0.0000000000000000E+000
		  1.0000000000000000E+000
		  5.9999999999999998E-001
		  5.9999999999999998E-001
		A matrix (strictly lower triangular):
		  row 2 :   2.0000000000000000E+000 
		  row 3 :   1.9199999999999999E+000   2.3999999999999999E-001 
		  row 4 :   1.9199999999999999E+000   2.3999999999999999E-001   0.0000000000000000E+000 
		Gamma hat vector:
		  2.5000000000000000E-001
		 -7.5000000000000000E-001
		  1.2100000000000000E+000
		  5.8000000000000003E-002
		C hat matrix (strictly lower triangular):
		  row 2 :  -4.0000000000000000E+000   0.0000000000000000E+000 
		  row 3 :   7.4400000000000004E+000   1.2000000000000000E+000   0.0000000000000000E+000 
		  row 4 :  -4.4800000000000001E-001  -2.1600000000000000E-001  -2.0000000000000001E-001   0.0000000000000000E+000 
		d vector for the higher order solution:
		  2.1111111111111112E+000
		  5.0000000000000000E-001
		  2.3148148148148148E-001
		  1.1574074074074074E+000
		d prime vector for the lower order solution:
		  1.7962962962962963E+000
		  3.0555555555555558E-001
		  2.3148148148148148E-001
		  0.0000000000000000E+000
	*/
	double gamma = 5.0000000000000000E-001 ;
	
	C[0] = 0. ; C[1] = 1.0000000000000000E+000 ; C[2] = 5.9999999999999998E-001 ; C[3] = 5.9999999999999998E-001 ;

	B[0] = 2.1111111111111112E+000 ; B[1] = 5.0000000000000000E-001 ; B[2] = 2.3148148148148148E-001; B[3] = 1.1574074074074074E+000 ;
	E[0] = B[0] - 1.7962962962962963E+000 ; E[1] = B[1] - 3.0555555555555558E-001 ; E[2] = B[2] - 2.3148148148148148E-001 ; E[3] = B[3] ;

	G[0][0] = gamma;
	G[1][0] = -4.0000000000000000E+000 ;	G[1][1] = gamma;
	G[2][0] = 7.4400000000000004E+000 ;	G[2][1] = 1.2000000000000000E+000;	G[2][2] = gamma;
	G[3][0] = -4.4800000000000001E-001 ;	G[3][1] = -2.1600000000000000E-001; G[3][2] = -2.0000000000000001E-001; G[3][3] = gamma;
	
	A[1][0] = 2.0000000000000000E+000  ;
	A[2][0] = 1.9199999999999999E+000  ; A[2][1] = 2.3999999999999999E-001  ;
	A[3][0] = 1.9199999999999999E+000  ; A[3][1] = 2.3999999999999999E-001  ; A[3][2] = 0.0 ; 
	
}

GRK4Sn::~GRK4Sn()
{
}


