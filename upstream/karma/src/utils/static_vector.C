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


#include "karma.h"
#include "karma_error.h"
#include "static_vector.h"

using namespace KARMA ;

//== StaticVector ==========================================================

// Low-Level BLAS: Fixed Dimensions - Level 1 copy routine
template <unsigned int N, typename TX, unsigned int incX, typename TY, unsigned int incY>
void blas1_copy(const TX *x, TY *y)
{
	for (int i=0, j=0; i<N; i+=incX, j+=incY)
	{
		y[j] = x[i];
	}
}


template <typename T, unsigned int N>
StaticVector<T,N>::StaticVector()
{
}

template <typename T, unsigned int N>
StaticVector<T,N>::StaticVector(const StaticVector<T,N> &rhs)
{
	blas1_copy<N,T,1,T,1>(rhs._data, _data);
}

template <typename T, unsigned int N>
StaticVector<T,N>::~StaticVector()
{
}

// -- operators --------------------------------------------------------


template <typename T, unsigned int N>
StaticVector<T, N> StaticVector<T,N>::operator=(const StaticVector<T, N> &rhs)
{
	blas1_copy<N,T,1,T,1>(rhs._data, _data);
	
	return *this;
}

template <typename T, unsigned int N>
const T& StaticVector<T,N>::operator()(unsigned int index) const
{
	karma_assert(index>=0);
	karma_assert(index<N);
	
	return _data[index];
}

template <typename T, unsigned int N>
T& StaticVector<T,N>::operator()(unsigned int index)
{
	karma_assert(index>=0);
	karma_assert(index<N);
	
	return _data[index];
}

// -- methods ----------------------------------------------------------

template <typename T, unsigned int N>
unsigned int StaticVector<T,N>::start() const
{
	return TypeInfoTraits<StaticVector<T,N> >::start;
}

template <typename T, unsigned int N>
unsigned int StaticVector<T,N>::end() const
{
	return TypeInfoTraits<StaticVector<T,N> >::end;
}

template <typename T, unsigned int N>
unsigned int StaticVector<T,N>::size() const
{
	return TypeInfoTraits<StaticVector<T,N> >::size;
}

template <typename T, unsigned int N>
unsigned int StaticVector<T,N>::stride() const
{
	return TypeInfoTraits<StaticVector<T,N> >::stride;
}

template <typename T, unsigned int N>
const T* StaticVector<T,N>::data() const
{
	return _data;
}

template <typename T, unsigned int N>
T* StaticVector<T,N>::data()
{
	return _data;
}


template <typename T, unsigned int N>
void StaticVector<T,N>::print (std::ostream &out) const
{
	for (unsigned int i=0; i<N; ++i)
	{
		out << _data[i] << " ";
	}
	out << std::endl;
}



//------------------------------------------------------------------
// Explicit instantiations
//
// These must appear in the namespace enclosing the template; the
// using-directive above is not sufficient.
namespace KARMA
{
	template class StaticVector<double, 3>;
}
