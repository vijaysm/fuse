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


#include "crank_nicolson.h"

CN::CN() : 
#ifdef IMPLICIT_MIDPOINT
	RK(2, 1, KARMA::TEMPORAL::CN, KARMA::TEMPORAL::FIRKNonSingular)
#else
	RK(2, 2, KARMA::TEMPORAL::CN, KARMA::TEMPORAL::FIRKSingular)
#endif
{
#ifdef IMPLICIT_MIDPOINT
	// implicit midpoint rule
	A[0][0] = 0.5 ;

	B[0] = 1.0 ;
	C[0] = 0.5 ;
	
	_is_A_invertible = true ;
	AInv[0][0] = 2.0 ;
#else
	// crank nicolson rule
	A[0][0] = 0.0 ;		A[0][1] = 0.0 ;
	A[1][0] = 0.5 ;		A[1][1] = 0.5 ;

	B[0] = 0.5 ;		B[1] = 0.5 ;
	C[0] = 0.0 ;		C[1] = 1.0 ;
	
	_is_A_invertible = false ;
#endif
}

CN::~CN ()
{
	
}

