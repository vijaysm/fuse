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


#ifndef __physics_properties_h__
#define __physics_properties_h__

#include "enum_utils.h"

namespace KARMA
{
	namespace PHYSICS
	{

#define PHYPROPERTIES \
    X(Diffusion, 0) \
    X(Advection, 1) \
    X(Reaction, 2) \
    X(Source, 3) \
    X(UserProperty, 4)

#define X(a,b) a=b,

		enum PropertyType { PHYPROPERTIES };
	
#undef X
	}
}

template<>
struct enum_traits<KARMA::PHYSICS::PropertyType>  {
	static const std::string enum_type ;
	static const unsigned int N = 5;	// number of enum values
	static const bool sorted = true;	// are they sorted ? (value-wise)
	static const int enumerators[];		// integer representations
	static const std::string names[] ;		// string versions
};

#endif // #define __physics_properties_h__

