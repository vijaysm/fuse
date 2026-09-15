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


#ifndef __karma_ts_base_h__
#define __karma_ts_base_h__

#include "ikarma.h"

class KARMATSBase : public IKARMA<KARMATSBase>
{
public:

	virtual ~KARMATSBase() ;
    
    // get order of accuracy of scheme
	unsigned int n_order() ;
    
	// type of the temporal integration scheme
	virtual const std::string name () = 0 ;

	// is the method A-stable : is the method unconditionally stable ?
	virtual bool is_A_stable() const = 0 ;

	// is the method L-stable : does the method damp modes well ?
	virtual bool is_L_stable() const = 0 ;

	// is the method S-stable : does it work for stiff problems ?
	virtual bool is_S_stable() const = 0 ;

	virtual void init () ;

	virtual void clear () ;
    
protected:
	
	KARMATSBase(int norder) ;

// Member variables

	unsigned int _order ;

};


inline
unsigned int KARMATSBase::n_order ()
{
	return _order ;
}


#endif // define __karma_ts_base_h__
