// $Id$

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


#ifndef __icollection1_h__
#define __icollection1_h__

// C++ includes
#include <iostream>
#include <vector>
#include <map>

#include "karma.h"

/**
 * This class provides a templated static collection.  The size
 * is provided at compile time and helper methods implemented in
 * the class provide std::vector like interface.
 * Inspired primarily by boost::ICollection.
 */



//== ICollection ==========================================================

template <typename T>
class ICollection
{
public:
	typedef T ElementType;

	explicit
	ICollection(unsigned int N);

	ICollection(const ICollection<T> &rhs) ;
	
	virtual ~ICollection() ;

	// -- assignment operator ---------------
	ICollection<T>& operator=(const ICollection<T> &rhs) ;

	// -- methods ------------------------
	const T* data() const ;

	T* data() ;

protected:

	// default: 1-dimensional array
	T* _data;
	
	const unsigned int _length ;
	
};

// inline methods for ICollection

template <typename T>
inline
ICollection<T>::ICollection(unsigned int N) :
		_length(N)
{
	_data = new T[N] ;
}

template <typename T>
inline
ICollection<T>::ICollection(const ICollection<T> &rhs) :
		_length (rhs._length)
{
	for (unsigned int i=0; i < _length; i++)
		_data[i] = rhs._data[i] ;
}

template <typename T>
inline
ICollection<T>::~ICollection()
{
	delete [] _data ;
}


template <typename T>
inline
const T* ICollection<T>::data() const
{
	return _data ;
}

template <typename T>
inline
T* ICollection<T>::data()
{
	return _data ;
}

template <typename T>
inline
ICollection<T>& ICollection<T>::operator=(const ICollection<T> &rhs)
{
	// we can only copy between arrays of same size
	karma_assert(_length == rhs._length) ;

	for (unsigned int i=0; i < _length; i++)
		_data[i] = rhs._data[i] ;
	
	return *this ;
}



//== Collection1D ==========================================================

template <typename T>
class Collection1D : public ICollection<T>
{
public:
	
	typedef ICollection<T> Parent ;
	
	explicit
	Collection1D(unsigned int N) : ICollection<T>(N)
	{	}

	Collection1D(const ICollection<T> &rhs) : ICollection<T>(rhs)
	{	}

	virtual ~Collection1D()
	{	}
	
	unsigned int size() const
	{	return Parent::_length ; }

	const T& operator()(unsigned int index) const
	{	return Parent::_data[index];	}

	T& operator()(unsigned int index)
	{	return Parent::_data[index];	}

};


//== Collection2D ==========================================================

template <typename T>
class Collection2D : public ICollection<Collection1D<T> >
{
public:
	
	typedef ICollection<Collection1D<T> > Parent ;

	explicit
	Collection2D(unsigned int M, unsigned int N) : ICollection<Collection1D<T> >(M)
	{
		
	}

	// not entirely sure if passing the buck to base class is the way to go ;)
	Collection2D(const Collection2D<T> &rhs) : ICollection<Collection1D<T> >(rhs)
	{	}

	virtual ~Collection2D()
	{	}

	// -- operators --------------------------------------------------------

	const T & operator()(int row, int col) const
	{
		return Parent::_data[row]()(col) ;
	}

	T & operator()(int row, int col)
	{
		return Parent::_data[row]()(col) ;
	}
	
	// -- methods ----------------------------------------------------------
	std::pair<unsigned int, unsigned int> size() const
	{
		return std::pair<unsigned int, unsigned int>(Parent::_length,_col_length) ;
	}

private: 
	
	unsigned int _col_length ;
	
};


//== IO ========================================================================

template <typename T>
std::ostream& operator<<(std::ostream &out, const Collection1D<T> &x)
{
	// pretty print the vector
	out << "\nSize of Vector = " << x.size() ;
	out << "\ni\tX(i)\n" ;
	
	const T* vec = x.data() ;
	for (unsigned int i=0; i < x.size(); i++)
	{
		out << "\n" << i << "\t" << vec[i] ;
	}
	out << "\n" ;
}


template <typename T>
std::ostream& operator<<(std::ostream &out, const Collection2D<T> &x)
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
			out << row[i] << "\t" ;
	}
	out << "\n" ;
}



// ------------------------------------------------------------
// ICollection class definition

// template<typename T, unsigned int N>
// class ICollection {
// public:
// 	// types
// 	typedef T                                     value_type;
// 	typedef T*                                    iterator;
// 	typedef const T*                              const_iterator;
// 	typedef std::reverse_iterator<iterator>       reverse_iterator;
// 	typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
// 	typedef T&                                    reference;
// 	typedef const T&                              const_reference;
// 	typedef std::size_t                           unsigned int;
// 	typedef std::ptrdiff_t                        difference_type;
// 	
// 	// static constants
// 	static const unsigned int static_size = N;
// 
// 	// construct/copy/destruct
// 	template<typename U> ICollection& operator=(const ICollection<U, N>&);
// 
// 	// iterator support
// 	iterator begin();
// 	const_iterator begin() const;
// 	iterator end();
// 	const_iterator end() const;
// 
// 	// reverse iterator support
// 	reverse_iterator rbegin();
// 	const_reverse_iterator rbegin() const;
// 	reverse_iterator rend();
// 	const_reverse_iterator rend() const;
// 
// 	// capacity
// 	unsigned int size();
// 	bool empty();
// 	unsigned int max_size();
// 
// 	// element access
// 	reference operator[](unsigned int);
// 	const_reference operator[](unsigned int) const;
// 	reference at(unsigned int);
// 	const_reference at(unsigned int) const;
// 	reference front();
// 	const_reference front() const;
// 	reference back();
// 	const_reference back() const;
// 	const T* data() const;
// 	T* c_array();
// 
// 	// modifiers
// 	void swap(ICollection<T, N>&);
// 	void assign(const T&);
// 	T elements[N];
// };
// 
// // specialized algorithms
// template<typename T, std::size_t N> void swap(ICollection<T, N>&, ICollection<T, N>&);
// 
// // comparisons
// template<typename T, std::size_t N>
// bool operator==(const ICollection<T, N>&, const ICollection<T, N>&);
// template<typename T, std::size_t N>
// bool operator!=(const ICollection<T, N>&, const ICollection<T, N>&);
// 
// inline
// template<typename T, unsigned int N>

#endif	// __icollection1_h__
