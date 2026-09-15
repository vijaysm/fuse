// $Id: icollection.C 450 2011-09-19 20:55:34Z vijaysm $

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


// C++ includes
#include "icollection.h"


//------------------------------------------
// Explicit instantiations
template class Collection1D<unsigned int>;
template class Collection1D<double>;
//------------------------------------------

template <typename T>
std::ostream& operator<<(std::ostream &out, const Collection1D<double> &x) ;
std::ostream& operator<<(std::ostream &out, const Collection1D<unsigned int> &x) ;

struct temptest
{
public:
	void operator()()
	{
		const unsigned int N = 10;
		ICollection<unsigned int> a(N) ;

		unsigned int* ad = a.data() ;
		for (unsigned int i=0; i < N; i++)
			ad[i] = i ;

		std::cout << "\nPrinting a vector.\n" ;
		std::cout << a ;
	}
};

/*
template <typename T>
std::ostream &
operator<<(std::ostream &out, const Collection1D<T> &x)
{
	// pretty print the vector
	out << "\nSize of Vector = " << x.size() ;
	out << "\ni\tX(i)\n" ;
	
	const T* vector = x.data() ;
	for (unsigned int i=0; i < x.size(); i++)
	{
		out << "\n" << i << "\t" << vector[i] ;
	}
	out << "\n" ;
}


template <typename T>
std::ostream &
operator<<(std::ostream &out, const Collection2D<T> &x)
{
	// pretty print the vector
	std::pair<unsigned int, unsigned int> dims = x.size() ;
	out << "\nSize of Matrix = [" << dims.first << ", " << dims.second << "]" ;
	out << "\ni\tX(i)\n" ;
	
	const Collection1D<T>* matrix = x.data() ;
	for (unsigned int i=0; i < dims.first; i++)
	{
		// technically, we could do this but it is too verbose.
		// out << "\n" << i << "\t" << vector[i] ;
		out << "\n" ;
		const T* row = matrix->data() ;
		for (unsigned int j=0; j < dims.second; j++)
			out << vector[i] << "\t" ;
	}
	out << "\n" ;
}
*/
