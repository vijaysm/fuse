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


#include "rodasp.h"

RODASP::RODASP() : Rosenbrock(6, 4, 3, KARMA::TEMPORAL::RODASP)
{
	for(int i=0; i<_primary_stages; i++)
		for(int j=0; j<_primary_stages; j++)
		{
			A[i][j] = 0.0 ;
			G[i][j] = 0.0 ;
		}
	
	/*
	This is the 4th order, L-stable Rosenbrock method for Parabolic problems and DAEs.
	Method =  RODASP 
	Number of stages = 6 
	High/low orders  = 4, 3 
		Gamma value =   2.5000000000000000E-001 
		Alpha vector:
		  0.0000000000000000E+000
		  7.4999999999999989E-001
		  2.1000000000000005E-001
		  6.2999999999999989E-001
		  9.9999999999999833E-001
		  9.9999999999999978E-001
		A matrix (strictly lower triangular):
		  row 2 :   3.0000000000000000E+000 
		  row 3 :   1.8310367934867591E+000   4.9551839674337950E-001 
		  row 4 :   2.3043765826926692E+000  -5.2492752457430007E-002  -1.1767987618327820E+000 
		  row 5 :  -7.1704549624230252E+000  -4.7416366714817864E+000  -1.6310026313309709E+001  -1.0620040441114009E+000 
		  row 6 :  -7.1704549624230252E+000  -4.7416366714817846E+000  -1.6310026313309709E+001  -1.0620040441114009E+000   1.0000000000000000E+000 
		Gamma hat vector:
		  6.2500000000000000E-002
		 -1.2499999999999997E-001
		 -5.8759999999999602E-003
		 -9.0499999999999782E-003
		  5.6804062321946791E-016
		 -4.9960036108132044E-016
		C hat matrix (strictly lower triangular):
		  row 2 :  -3.0000000000000000E+000   0.0000000000000000E+000 
		  row 3 :  -2.1979487934867588E+000  -5.5196639674337955E-001   0.0000000000000000E+000 
		  row 4 :   2.7044826421428825E+000   1.6950676528570665E+000   4.8837148616060251E+000   0.0000000000000000E+000 
		  row 5 :   8.5477375168741929E+000   3.8741778843149075E+000   1.3686902189910326E+001   3.5400134803713348E+000   0.0000000000000000E+000 
		  row 6 :   8.6565145773263321E+000   3.8252124402861827E+000   1.4249888946656668E+001   4.6020175244827373E+000  -1.4285714285714293E+000   0.0000000000000000E+000 
		d vector for the higher order solution:
		 -7.1704549624230260E+000
		 -4.7416366714817864E+000
		 -1.6310026313309709E+001
		 -1.0620040441114009E+000
		  1.0000000000000000E+000
		  1.0000000000000000E+000
		d prime vector for the lower order solution:
		 -7.1704549624230252E+000
		 -4.7416366714817846E+000
		 -1.6310026313309709E+001
		 -1.0620040441114009E+000
		  1.0000000000000000E+000
		  0.0000000000000000E+000
	*/
	double gamma = 0.25  ;
	
	C[0] = 0. ; C[1] = 7.5E-001 ; C[2] = 2.1E-001 ; C[3] = 6.3E-001; C[4] = 1.0; C[5] = 1.0 ;

	B[0] = -7.1704549624230260E+000 ; B[1] = -4.7416366714817864E+000 ; B[2] = -1.6310026313309709E+001; B[3] = -1.0620040441114009E+000 ; B[4] = 1.0000000000000000E+000; B[5] = 1.0000000000000000E+000 ;

	G[0][0] = gamma;
	G[1][0] = -3.0000000000000000E+000 ;	G[1][1] = gamma;
	G[2][0] = -2.1979487934867588E+000 ;	G[2][1] = -5.5196639674337955E-001;	G[2][2] = gamma;
	G[3][0] = 2.7044826421428825E+000  ; 	G[3][1] = 1.6950676528570665E+000  ; G[3][2] = 4.8837148616060251E+000; G[3][3] = gamma ;
	G[4][0] = 8.5477375168741929E+000  ; 	G[4][1] = 3.8741778843149075E+000   ; G[4][2] = 1.3686902189910326E+001  ; G[4][3] = 3.5400134803713348E+000 ;	G[4][4] = gamma ;
	G[5][0] = 8.6565145773263321E+000   ; G[5][1] = 3.8252124402861827E+000   ; G[5][2] = 1.4249888946656668E+001   ; G[5][3] = 4.6020175244827373E+000; G[5][4] = -1.4285714285714293E+000 ;	G[5][5] = gamma ;
	
	A[1][0] = 3.0000000000000000E+000  ;
	A[2][0] = 1.8310367934867591E+000  ; A[2][1] = 4.9551839674337950E-001  ;
	A[3][0] = 2.3043765826926692E+000  ; A[3][1] = -5.2492752457430007E-002  ; A[3][2] = -1.1767987618327820E+000 ;
	A[4][0] = -7.1704549624230252E+000  ; A[4][1] = -4.7416366714817864E+000  ; A[4][2] = -1.6310026313309709E+001  ; A[4][3] = -1.0620040441114009E+000; 
	A[5][0] = -7.1704549624230252E+000  ; A[5][1] = -4.7416366714817846E+000  ; A[5][2] = -1.6310026313309709E+001  ; A[5][3] = -1.0620040441114009E+000   ; A[5][4] = 1.0000000000000000E+000 ;
	
	E[0] = 0.0 ; E[1] = 0.0 ; E[2] = 0.0 ; E[3] = 0.0 ; E[4] = 0.0 ; E[5] = B[5] ;

}

RODASP::~RODASP()
{
}


