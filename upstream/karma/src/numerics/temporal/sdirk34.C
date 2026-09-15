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


#include "sdirk34.h"

SDIRK34::SDIRK34() :
			RK(3, 4, KARMA::TEMPORAL::SDIRK34, KARMA::TEMPORAL::SDIRK)
{
	// gamma is a solution of x^3-1.5*x^2+.5*x-1/24=0
	// 1.06857902130163
	// 0.12888640051572
	// 0.302534578182651
	
	//  'SDIRK34' // 3-stage, 4th order singly diagonally implicit (SSP)
	//  s=3; r=1.7588;
	//  g=0.5*(1-cos(pi/18)/sqrt(3)-sin(pi/18));
	//  q=(0.5-g)^2;
	//  A=[g     0    0
	//      0.5-g g    0
	//      2*g  1-4*g g];
	//  b=[1/(24*q) 1-1/(12*q) 1/(24*q)]';
	//  c=sum(A,2);

	double gamma = 0.12888640051572 ;
	
	C[0] = gamma ; C[1] = 0.5 ; C[2] = 1.0 - gamma ;
	
	double q = 1.0/((0.5-gamma)*(0.5-gamma)) ;

	B[0] = q/24.0 ;
	B[1] = 1.0-q/12.0 ;
	B[2] = q /24.0; 

	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
			A[i][j] = 0.0 ;
	
	A[0][0] = gamma ;
	A[1][0] = 0.5-gamma ; A[1][1] = gamma ;
	A[2][0] = 2.0*gamma ; A[2][1] = 1.0-4.0*gamma ;	A[2][2] = gamma ;
	
	_is_A_invertible = true ;
	AInv[0][0] = 1.0/gamma ;
	AInv[1][0] = 0.5*(-1.0+2.0*gamma)/gamma/gamma ; AInv[1][1] = 1.0/gamma ;
	AInv[2][0] = 0.5*(1.0-6.0*gamma+4.0*gamma*gamma)/gamma/gamma/gamma ; AInv[2][1] = (-1.0+4.0*gamma)/gamma/gamma ; AInv[2][2] = 1.0/gamma ;
}

SDIRK34::~SDIRK34()
{
}

