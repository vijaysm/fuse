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


#include "ros22.h"

ROS22::ROS22() : Rosenbrock(2, 2, KARMA::TEMPORAL::ROS22)
{
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
		{
			A[i][j] = 0.0 ;
			G[i][j] = 0.0 ;
		}
	
	/*
	This is the optimal L-stable ROS22 method with modified coefficients
	Method =    ROS22
	Number of stages = 2 
	High/low orders  = 2, 2 
		Gamma value =   2.9289321881345254E-001 
		Alpha vector:
		  0.0000000000000000E+000
		  1.0000000000000000E+000
		A matrix (strictly lower triangular):
		  row 2 :   3.4142135623730945E+000 
		Gamma hat vector:
		  8.5786437626904993E-002
		 -8.5786437626904993E-002
		C hat matrix (strictly lower triangular):
		  row 2 :  -2.0000000000000000E+000   0.0000000000000000E+000 
		d vector for the higher order solution:
		  5.1213203435596419E+000
		  1.7071067811865472E+000
		d prime vector for the lower order solution:
		  5.1213203435596419E+000
		  1.7071067811865472E+000
	*/
	double gamma = 2.9289321881345254E-001 ;
	
	C[0] = 0. ; C[1] = 1.0 ;

	B[0] = 5.1213203435596419E+000 ; B[1] = 1.7071067811865472E+000 ;

	G[0][0] = gamma;
	G[1][0] = -2.0 ;	G[1][1] = gamma;
	
	A[1][0] = 3.4142135623730945E+000  ;

}

ROS22::~ROS22()
{
}


