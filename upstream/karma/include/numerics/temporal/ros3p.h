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


#ifndef __ros3p_h__
#define __ros3p_h__

#include "rosenbrock.h"

// Forward declarations
class RosenbrockIntegrator ;

class ROS3P : public Rosenbrock
{
public:

	ROS3P() ;
	
	virtual ~ROS3P() ;

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

// Inline methods for ROS3P

// is the method A-stable : is the method unconditionally stable ?
inline
bool ROS3P::is_A_stable() const 
{
	return true ;
}

// is the method L-stable : does the method damp modes well ?
inline
bool ROS3P::is_L_stable() const 
{
	return true ;
}

// is the method S-stable : does it work for stiff problems ?
inline
bool ROS3P::is_S_stable() const 
{
	return true ;
}

inline
bool ROS3P::is_FSAL() const
{
	return false ;
}

inline
const std::string ROS3P::name ()
{
    return "ROSP3(3,2)" ;
}


template<>
struct RKMethodTraits<19>	// Enum ROS3P = 19
{
	const static KARMA::TEMPORAL::RKType Type=KARMA::TEMPORAL::ROS3P ;
	typedef ROS3P MethodType ;
	typedef RosenbrockIntegrator StaticIntegratorType ;
	typedef RosenbrockIntegrator AdaptiveIntegratorType ;
	
	static RK* build(KARMA::TEMPORAL::RKType type)
	{
		return (type == Type ? new MethodType() : RKMethodTraits<Type-1>::build(type) ) ;
	}
};


#endif // define __ros3p_h__
