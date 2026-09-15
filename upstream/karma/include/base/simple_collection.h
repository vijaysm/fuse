// $Id: simple_collection.h 450 2011-09-19 20:55:34Z vijaysm $

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


#ifndef __icollection_h__
#define __icollection_h__

#include <vector>

namespace KARMA
{

template <class T>
class ICollection
{
public:

    ICollection(unsigned int nsize) ;

    void add(const unsigned int i, T* BaseItem) ;

    //Remove a specific Item from the collection
    void remove(int iKey) ;

    void clear() ; //Clear the collection

    //Return the number of items in collection
    unsigned int size() ;

    //Operator Returning a reference to T
    T* operator [](unsigned int iKey) ;

protected:
	//The Vector container that will hold the collection of Items
	std::vector<T*> _collection;
};

// inline methods for ICollection

template<typename T>
inline
ICollection<T>::ICollection(unsigned int nsize)
{
    _collection.resize(nsize) ;
    for (unsigned int i=0; i < nsize; i++)
        _collection[i] = NULL ;
}

template<typename T>
inline
void ICollection<T>::add(const unsigned int i, T* BaseItem)
{
    //Add the item to the container
    _collection[i] = BaseItem ;
}

//Remove a specific Item from the collection
template<typename T>
inline
void ICollection<T>::remove(int iKey)
{
        //Remove the Item using the vector’s erase function
    _collection.erase(iKey);
}

template<typename T>
inline
void ICollection<T>::clear() //Clear the collection
{
    _collection.clear();
}

//Return the number of items in collection
template<typename T>
inline
unsigned int ICollection<T>::size()
{
    return _collection.size();
}

//Operator Returning a reference to TBase
template<typename T>
inline
T* ICollection<T>::operator [](unsigned int iKey)
{
    return _collection[iKey];
}

}

#endif	// __icollection_h__
