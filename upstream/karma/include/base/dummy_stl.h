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

/*!@file dummySTL.h A dummy STL definition for doxygen
  <B>DO NOT INCLUDE THIS FILE!!!</B> */

//! The Standard Template Library (STL)
/*! This is not the real STL documentation, you can find that at the
  <A HREF="http://www.sgi.com/tech/stl/">Silicon Graphics STL site</A>. */
namespace std
{
  //! The STL vector container class. Use with \code#include <vector>\endcode
  /*! See <A HREF="http://www.sgi.com/tech/stl/Vector.html">SGI's
    documentation</A> of std::vector. */
  template<class T, class Alloc>
  class vector { public: T data; Alloc allocator; };

  //! The STL deque container class. Use with \code#include <deque>\endcode
  /*! See <A HREF="http://www.sgi.com/tech/stl/Deque.html">SGI's
    documentation</A> of std::deque. */
  template<class T, class Alloc>
  class deque { public: T data; Alloc allocator; };

  //! The STL list container class. Use with \code#include <list>\endcode
  /*! See <A HREF="http://www.sgi.com/tech/stl/List.html">SGI's
    documentation</A> of std::list. */
  template<class T, class Alloc>
  class list { public: T data; Alloc allocator; };

  //! The STL slist container class. Use with \code#include <slist>\endcode
  /*! See <A HREF="http://www.sgi.com/tech/stl/Slist.html">SGI's
    documentation</A> of std::slist. */
  template<class T, class Alloc>
  class slist { public: T data; Alloc allocator; };

  //! The STL set container class. Use with \code#include <set>\endcode
  /*! See <A HREF="http://www.sgi.com/tech/stl/set.html">SGI's
    documentation</A> of std::set. */
  template<class Key, class Compare, class Alloc>
  class set { public: Key keys; Compare comparator; Alloc allocator; };

  //! The STL map container class. Use with \code#include <map>\endcode
  /*! See <A HREF="http://www.sgi.com/tech/stl/Map.html">SGI's
    documentation</A> of std::map. */
  template<class Key, class Data, class Compare, class Alloc>
  class map { public: Key keys; Data data; Compare comparator;
    Alloc allocator; };

  //! The STL multiset container class. Use with \code#include <set>\endcode
  /*! See <A HREF="http://www.sgi.com/tech/stl/multiset.html">SGI's
    documentation</A> of std::multiset. */
  template<class Key, class Compare, class Alloc>
  class multiset { public: Key keys; Compare comparator; Alloc allocator; };

  //! The STL multimap container class. Use with \code#include <map>\endcode
  /*! See <A HREF="http://www.sgi.com/tech/stl/Multimap.html">SGI's
    documentation</A> of std::multimap. */
  template<class Key, class Data, class Compare, class Alloc>
  class multimap { public: Key keys; Data data; Compare comparator;
    Alloc allocator; };

  //! The STL string class. Use with \code#include <string>\endcode
  /*! std::string is actually std::basic_string<char>.
    See <A HREF="http://www.sgi.com/tech/stl/basic_string.html">SGI's
    documentation</A> of std::basic_string. */
  class string { };
}

// ######################################################################
/* End:  */
