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


#ifndef __sdirk332_h__
#define __sdirk332_h__

#include "rk.h"

// Forward declarations
class ImplicitDIRKIntegrator ;
class EmbeddedDIRKIntegrator ;

class SDIRK332 : public RK
{
public:

	SDIRK332() ;
	
	virtual ~SDIRK332() ;

	// type of the temporal integration scheme
	virtual const std::string name () ;

	// is the method A-stable : is the method unconditionally stable ?
	virtual bool is_A_stable() const ;

	// is the method L-stable : does the method damp modes well ?
	virtual bool is_L_stable() const ;

	// is the method S-stable : does it work for stiff problems ?
	virtual bool is_S_stable() const ;

	// is the last stage solution the same as first stage in the subsequent step
	virtual bool is_FSAL() const ;

};

// Inline methods for SDIRK332


// is the method A-stable : is the method unconditionally stable ?
inline
bool SDIRK332::is_A_stable() const 
{
	return true ;
}

// is the method L-stable : does the method damp modes well ?
inline
bool SDIRK332::is_L_stable() const 
{
	return true ;
}

// is the method S-stable : does it work for stiff problems ?
inline
bool SDIRK332::is_S_stable() const 
{
	return true ;
}

inline
bool SDIRK332::is_FSAL() const
{
	return false ;
}

inline
const std::string SDIRK332::name ()
{
    return "SDIRK3(3,2)" ;
}


template<>
struct RKMethodTraits<9>	// Enum SDIRK332 = 9
{
	const static KARMA::TEMPORAL::RKType Type=KARMA::TEMPORAL::SDIRK332 ;
	typedef SDIRK332 MethodType ;
	typedef ImplicitDIRKIntegrator StaticIntegratorType ;
	typedef EmbeddedDIRKIntegrator AdaptiveIntegratorType ;
	
	static RK* build(KARMA::TEMPORAL::RKType type)
	{
		return (type == Type ? new MethodType() : RKMethodTraits<Type-1>::build(type) ) ;
	}
};



#endif // define __sdirk332_h__
