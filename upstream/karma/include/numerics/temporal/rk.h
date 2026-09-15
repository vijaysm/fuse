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


#ifndef __rk_h__
#define __rk_h__

#include "karma_ts_base.h"
#include "runge_kutta_schemes.h"

class RK : public KARMATSBase
{
public:

	virtual ~RK() ;
		    
	// type of the temporal integration scheme
	virtual const std::string name () = 0 ;

	// is the method A-stable : is the method unconditionally stable ?
	virtual bool is_A_stable() const = 0 ;

	// is the method L-stable : does the method damp modes well ?
	virtual bool is_L_stable() const = 0 ;

	// is the method S-stable : does it work for stiff problems ?
	virtual bool is_S_stable() const = 0 ;

	// is the last stage solution the same as first stage in the subsequent step
	virtual bool is_FSAL() const = 0 ;

	// template <typename DerivedRKType>
	static RK* build(KARMA::TEMPORAL::RKType ) ;

	double** Acoeff () const ;
	
	double** AInvcoeff() const ;

	double* Bcoeff () const ;
	
	double* Ccoeff () const ;

	double* Ecoeff () const ;
	
	// get number of stages
	unsigned int n_stages() const ;
	
	unsigned int n_secondary_order () const ;
	
	bool is_A_invertible () const ;

	KARMA::TEMPORAL::RKType rk_type() const ;

	KARMA::TEMPORAL::RKCategory rk_category() const ;

	virtual void init () ;

	virtual void clear () ;

	virtual void print () ;

protected:
	
	RK(unsigned int nstages, unsigned int norder, KARMA::TEMPORAL::RKType type, KARMA::TEMPORAL::RKCategory cat) ;
	
	RK(unsigned int nstages, unsigned int norder, unsigned int nsec_order, KARMA::TEMPORAL::RKType type, KARMA::TEMPORAL::RKCategory cat) ;
	
	
// Member variables

	// number of stages for RK method
	unsigned int _primary_stages ;
	
	// if embedded scheme, this is the secondary order if coefficients with B_sec is used.
	// NOTE: E[i] = B[i] - B_sec[i]
	unsigned int _secondary_order ;
	
	// type of RK method : BE, CN, SDIRK22 etc...
	KARMA::TEMPORAL::RKType _type ;
	
	// category of RK method : ERK, DIRK, SDIRK etc...
	KARMA::TEMPORAL::RKCategory _cat ;
	
	bool _is_A_invertible ;

	// RK Butcher coefficient matrix
	Real **A, **AInv;
	
	// RK Butcher coefficients for solution
	Real *B;
	
	// RK Butcher coefficients for time advancement
	Real *C;

	// RK Butcher coefficients for time advancement
	Real *E;

};


// Inline methods for RK

inline
unsigned int RK::n_stages() const 
{
	return _primary_stages ;
}

inline
unsigned int RK::n_secondary_order() const 
{
	return _secondary_order ;
}

inline
double** RK::Acoeff () const 
{
	return A ;
}

inline
bool RK::is_A_invertible () const 
{
	return _is_A_invertible ;
}

inline
double** RK::AInvcoeff () const 
{
	return AInv ;
}

inline
double* RK::Bcoeff () const 
{
	return B ;
}

inline
double* RK::Ccoeff () const 
{
	return C ;
}

inline
double* RK::Ecoeff () const 
{
	return E ;
}

inline
KARMA::TEMPORAL::RKType RK::rk_type() const
{
	return _type ;
}

inline
KARMA::TEMPORAL::RKCategory RK::rk_category() const
{
	return _cat ;
}

// Forward declaration
class ExplicitRKIntegrator ;

// define a temporal method trait
template<int rk_type>
struct RKMethodTraits
{
	typedef ExplicitRKIntegrator StaticIntegratorType ;
	typedef ExplicitRKIntegrator AdaptiveIntegratorType ;
	static RK* build(KARMA::TEMPORAL::RKType type)
	{
		// the default implementation is return NOTHING.
		// template specialize for different methods
		return NULL ;
	}
};

#endif // define __rk_h__
