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


#ifndef __static_vector_h__
#define __static_vector_h__

#include "typename_traits.h"
#include <iostream>


namespace KARMA
{

//== StaticVector : Based on FLENS FixedSizeArray1D implementation==
template <typename T, unsigned int N>
class StaticVector
{
public:

	typedef T ElementType;

	StaticVector();

	StaticVector(const StaticVector<T,N> &rhs);

	~StaticVector();

	// -- operators --------------------------------------------------------

	StaticVector<T, N>
	operator=(const StaticVector<T, N> &rhs);

	const T & operator()(unsigned int index) const;

	T& operator()(unsigned int index);

    // -- methods ----------------------------------------------------------

	unsigned int start() const;

	unsigned int end() const;

	unsigned int size() const;

	unsigned int stride() const;

	const T* data() const;

	void print(std::ostream& ) const ;

	T * data();

private:
	
	T _data[N];
};


template <typename T, unsigned int N>
struct TypeInfoTraits<StaticVector<T,N> >
{
	typedef StaticVector<T,N> Impl;
	typedef T ElementType;

	static const unsigned int size = N;
	static const unsigned int stride = 1;
	static const unsigned int start = 0;
	static const unsigned int end = N-1;
};

}

#endif // __static_vector_h__
