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


#include "grk4a.h"

GRK4A::GRK4A() : Rosenbrock(4, 4, 3, KARMA::TEMPORAL::GRK4A)
{
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
		{
			A[i][j] = 0.0 ;
			G[i][j] = 0.0 ;
		}
	
	/*
	This is the A-stable GRK method by Kaps and Rentrop.
	Method =  GRK4A
	Number of stages = 4 
	High/low orders  = 4, 4 
		Gamma value =   3.9500000000000002E-001 
		Alpha vector:
		  0.0000000000000000E+000
		  4.3800000000000000E-001
		  8.6999999999950006E-001
		  8.6999999999950006E-001
		A matrix (strictly lower triangular):
		  row 2 :   1.1088607594936706E+000 
		  row 3 :   2.3770852619819602E+000   1.8501149888987331E-001 
		  row 4 :   2.3770852619819602E+000   1.8501149888987331E-001  -3.1045862459902379E-017 
		Gamma hat vector:
		  1.5602500000000002E-001
		 -1.4720559621618001E-001
		  2.6185326351170012E-002
		  1.7146740502115951E-001
		C hat matrix (strictly lower triangular):
		  row 2 :  -1.9434744189468354E+000  -1.7541523789077474E-016 
		  row 3 :   4.1695753098949334E-001   1.3239678207291139E+000  -1.7541523789077474E-016 
		  row 4 :   1.5195132577836896E+000   1.3537081503017017E+000  -8.5415149525822787E-001   0.0000000000000000E+000 
		d vector for the higher order solution:
		  1.8456832404082615E+000
		  1.3697968943635339E-001
		  7.1290977832951119E-001
		  6.3291139240506333E-001
		d prime vector for the lower order solution:
		  1.8456832404082615E+000
		  1.3697968943635339E-001
		  7.1290977832951119E-001
		  6.3291139240506333E-001
	*/
	double gamma = 3.95E-001 ;
	
	C[0] = 0. ; C[1] = 4.3800000000000000E-001 ; C[2] = 8.6999999999950006E-001 ; C[3] = 8.6999999999950006E-001 ;

	B[0] = 1.8456832404082615E+000 ; B[1] = 1.3697968943635339E-001 ; B[2] = 7.1290977832951119E-001; B[3] = 6.3291139240506333E-001 ;
	E[0] = B[0] - 1.8456832404082615E+000 ; E[1] = B[1] - 1.3697968943635339E-001 ; E[2] = B[2] - 7.1290977832951119E-001 ; E[3] = B[3] - 6.3291139240506333E-001 ;
	
	G[0][0] = gamma;
	G[1][0] = -1.9434744189468354E+000 ;	G[1][1] = gamma;
	G[2][0] = 4.1695753098949334E-001 ;	G[2][1] = 1.3239678207291139E+000;	G[2][2] = gamma;
	G[3][0] = 1.5195132577836896E+000 ;	G[3][1] = 1.3537081503017017E+000; G[3][2] = -8.5415149525822787E-001; G[3][3] = gamma;
	
	A[1][0] = 1.1088607594936706E+000  ;
	A[2][0] = 2.3770852619819602E+000  ; A[2][1] = 1.8501149888987331E-001 ;
	A[3][0] = 2.3770852619819602E+000  ; A[3][1] = 1.8501149888987331E-001 ; A[3][2] = 0.0 ; 
	
}

GRK4A::~GRK4A()
{
}


