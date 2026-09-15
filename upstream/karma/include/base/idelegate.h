// $Id: idelegate.h 450 2011-09-19 20:55:34Z vijaysm $

// KARMA: K(C)ode for Accident and Reactor Modeling Analysis
// Copyright (C) 2008-2009  Vijay S Mahadevan
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//


#ifndef __idelegate_h__
#define __idelegate_h__

/**
 * Abstract Delegate or Functor base class.
 * Only requires the operator() handle.
 */
class IDelegate
{
public:
    virtual void operator()() const { };
    virtual ~IDelegate()
    {}
};


/**
 * A templated concrete implementation of IDelegate.
 * NOTE: The function pointer has no arguments (N=0).
 */
template<typename T, unsigned int N=0>
class Delegate : public IDelegate
{
public:
  typedef void (T::*F)();

  Delegate( T& t, F f ) : _object(&t), _function(f) { }
  void operator()() const { (_object->*_function)(); }

private:
  T* _object;
  F  _function;
};

template<typename T>
Delegate<T> make_delegate( T& t, void (T::*f) () )
{
  return Delegate<T>( t, f );
}

#endif	// __idelegate_h__
