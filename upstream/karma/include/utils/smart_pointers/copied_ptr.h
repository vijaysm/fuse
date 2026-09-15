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

#ifndef __copied_ptr_h__
#define __copied_ptr_h__


/**
 * copied_ptr - simple copy-on-create/assign pointer.
 *
 * Note: If the actual object pointed to belongs to a derived class,
 * then copied_ptr will not create a copy of the derived class object,
 * but a new base class object.
 * 
 * If you want to use a polymorphic copy-on-assign pointer, use
 * cloned_ptr.
 *
 * When assigning p = q, where both p and q are copied_ptr's, it creates
 * a new copy of the object pointed by p, and have q point to this copy.
 *
 * NOTE: This smart pointer is recommended for usage in situations where
 * the object size is small and when assignment needs cloning by default.
 * It also expects the object to implement a copy constructor.
 */

template <class X>
class copied_ptr
{
public:
    typedef X element_type;

    explicit copied_ptr(X* p = 0) throw()
			: _raw_ptr(p)
	{}
	
    ~copied_ptr()
	{
		delete _raw_ptr;
	}
	
    copied_ptr(const copied_ptr& r)
	{
		copy(r.get());
	}

	copied_ptr& operator=(const copied_ptr& r)
    {
        if (this != &r)
		{
            delete _raw_ptr;
            copy(r);
        }
        return *this;
    }

    X& operator*()  const throw()
	{
		return *_raw_ptr;
	}

	X* operator->()
	{
		return _raw_ptr;
	}

	X* get()
	{
		return _raw_ptr;
	}

private:

	// Store the reference to the true object
	X* _raw_ptr;
	
	// helper method to copy the passed pointer
	void copy(const copied_ptr& r)
	{
		_raw_ptr = r._raw_ptr ? new X(*r._raw_ptr) : 0;
	}
};

#endif // __copied_ptr_h__
