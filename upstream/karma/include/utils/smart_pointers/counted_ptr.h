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

#ifndef __counted_ptr_h__
#define __counted_ptr_h__

/* For ANSI-challenged compilers, you may want to #define
 * NO_MEMBER_TEMPLATES or explicit */
/**
 * counted_ptr - simple reference counted pointer.
 *
 * The is a non-intrusive implementation that allocates an additional
 * int and pointer for every counted object.
 */

template <class X>
class counted_ptr
{
public:
    typedef X element_type;

    explicit counted_ptr(X* p = 0) // allocate a new counter
        : oCounter(0)
	{
		if (p)
			oCounter = new counter(p);
	}
	
    ~counted_ptr()
    {
		release();
	}
	
    counted_ptr(const counted_ptr& r) throw()
    {
		acquire(r.oCounter);
	}

	counted_ptr& operator=(const counted_ptr& r)
    {
        if (this != &r) {
            release();
            acquire(r.oCounter);
        }
        return *this;
    }

    template <class Y> friend class counted_ptr<Y>;
	template <class Y> counted_ptr(const counted_ptr<Y>& r) throw()
    {
		acquire(r.oCounter);
	}

	template <class Y> counted_ptr& operator=(const counted_ptr<Y>& r)
    {
        if (this != &r) {
            release();
            acquire(r.oCounter);
        }
        return *this;
    }

    X& operator*()  const throw()   {return *oCounter->ptr;}

	X* operator->()
	{
		return oCounter->ptr;
	}

	X* get()
	{
		return oCounter ? oCounter->ptr : 0;
	}

	bool unique()   const throw()
    {
		return (oCounter ? oCounter->count == 1 : true);
	}

private:

    struct counter
	{
        counter(X* p = 0, unsigned c = 1) : ptr(p), count(c)
		{}
		
        X*          ptr;
        unsigned    count;
		
    }* oCounter;

    void acquire(counter* c) throw()
    {
		// increment the count
        oCounter = c;
        if (c) ++c->count;
    }

    void release()
    {
		// decrement the count, delete if it is 0
        if (oCounter) {
            if (--oCounter->count == 0) {
                delete oCounter->ptr;
                delete oCounter;
            }
            oCounter = 0;
        }
    }
};

#endif // __counted_ptr_h__
