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


#ifndef __runge_kutta_schemes_h__
#define __runge_kutta_schemes_h__

#include "enum_utils.h"

namespace KARMA
{
	namespace TEMPORAL
	{
#define RK_CATEGORIES \
    X(ERK, 0)		\
	X(DIRK, 1)		\
	X(SDIRK, 2)		\
	X(ESDIRK, 3)	\
	X(FIRKNonSingular, 4)	\
	X(FIRKSingular, 5)	\
	X(ROSENBROCK, 6)	\
	X(ERKEmbedded, 7)	\
	X(DIRKEmbedded, 8)	\
	X(SDIRKEmbedded, 9)	\
	X(ROSENBROCKEmbedded, 10)

#define RK_TYPES \
	X(FE, 0) \
	X(KUTTA22, 1) \
	X(KUTTA44, 2) \
	X(DOPRI45, 3) \
	X(BE, 4) \
	X(CN, 5) \
	X(SDIRK22, 6) \
	X(SDIRK23, 7) \
	X(SDIRK34, 8) \
	X(SDIRK332, 9) \
	X(SDIRK543, 10) \
	X(SDIRK643, 11) \
	X(TRBE, 12) \
	X(SDIRK33, 13) \
	X(SDIRK45, 14) \
	X(ROS22, 15) \
	X(GRK4A, 16) \
	X(GRK4T, 17) \
	X(GRK4Sn, 18) \
	X(ROS3P, 19) \
	X(RODASP, 20) \
	X(SDIRK44, 21)

#define X(a,b) a=b,

	/**
	 *  0: Explicit RK
	 *  1: Diagonally Implicit RK
	 *  2: Singly Diagonally Implicit RK
	 *  3: Explicit Singly Diagonally Implicit RK (1st stage explicit)
	 *  4: Fully Implicit RK with det(A) != 0
	 *  5: Fully Implicit RK with det(A) == 0
	 *  6: Rosenbrock family of methods with Newton-type linearization
	 *  7: Embedded (a higher and lower order) schemes of ERK type
	 *  8: Embedded (a higher and lower order) schemes of DIRK type
	 *  9: Embedded (a higher and lower order) schemes of SDIRK type
	 *  10: Embedded (a higher and lower order) schemes of Rosenbrock type
	 */
	enum RKCategory { RK_CATEGORIES };
	
	enum RKType { RK_TYPES };
	
#undef X

	}

}

template<>
struct enum_traits<KARMA::TEMPORAL::RKCategory>  {
	static const std::string enum_type ;
	static const unsigned int N = 11;	// number of enum values
	static const bool sorted = true;	// are they sorted ? (value-wise)
	static const int enumerators[];		// integer representations
	static const std::string names[] ;				// string versions
};
	
template<>
struct enum_traits<KARMA::TEMPORAL::RKType>  {
	static const std::string enum_type ;
	static const unsigned int N = 22;	// number of enum values
	static const bool sorted = true;	// are they sorted ? (value-wise)
	static const int enumerators[];		// integer representations
	static const std::string names[] ;				// string versions
};

#endif // #define __runge_kutta_schemes_h__
