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


#include "ros3p.h"

ROS3P::ROS3P() : Rosenbrock(3, 3, 2, KARMA::TEMPORAL::ROS3P)
{
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
		{
			A[i][j] = 0.0 ;
			G[i][j] = 0.0 ;
		}
	
	/* 
	This is the 3rd order L-stable Rosenbrock method for Parabolic problems.
	Method =   ROS3P 
	Number of stages = 3 
	High/low orders  = 3, 3 
		Gamma value =   7.8867513459481287E-001 
		Alpha vector:
		  0.0000000000000000E+000
		  1.0000000000000000E+000
		  1.0000000000000000E+000
		A matrix (strictly lower triangular):
		  row 2 :   1.2679491924311230E+000 
		  row 3 :   1.2679491924311230E+000   0.0000000000000000E+000 
		Gamma hat vector:
		  6.2200846792814612E-001
		 -1.6666666666666660E-001
		 -8.4967936855888626E-001
		C hat matrix (strictly lower triangular):
		  row 2 :  -1.2679491924311224E+000   0.0000000000000000E+000 
		  row 3 :  -2.7320508075688776E+000  -1.3660254037844384E+000   0.0000000000000000E+000 
		d vector for the higher order solution:
		  2.0000000000000000E+000
		  5.7735026918962584E-001
		  4.2264973081037421E-001
		d prime vector for the lower order solution:
		  2.1132486540518709E+000
		  1.0000000000000000E+000
		  4.2264973081037421E-001
	*/
	double gamma = 7.8867513459481287E-001 ;
	
	C[0] = 0. ; C[1] = 1.0000000000000000E+000 ; C[2] = 1.0000000000000000E+000 ;

	B[0] = 2.0000000000000000E+000 ; B[1] = 5.7735026918962584E-001 ; B[2] = 4.2264973081037421E-001;

	G[0][0] = gamma;
	G[1][0] = -1.2679491924311224E+000 ;	G[1][1] = gamma;
	G[2][0] = -2.7320508075688776E+000 ;	G[2][1] = -1.3660254037844384E+000;	G[2][2] = gamma;
	
	A[1][0] = 1.2679491924311230E+000  ;
	A[2][0] = 1.2679491924311230E+000  ; A[2][1] = 0.0  ;
	
	E[0] = B[0] - 2.1132486540518709E+000 ; E[1] = B[1] - 1.0000000000000000E+000 ; E[2] = B[2] - 4.2264973081037421E-001 ;

}

ROS3P::~ROS3P()
{
}


