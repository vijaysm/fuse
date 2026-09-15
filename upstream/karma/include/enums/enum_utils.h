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


#ifndef __enum_utils_h__
#define __enum_utils_h__

#include "karma.h"
#include "karma_error.h"

// c++ includes
#include <string>
#include <algorithm>

template <typename T>
struct enum_traits {};

namespace KARMA
{
	namespace Utility
	{
		
		template<typename T>
		inline
		T int_to_enum(int v)
		{
			typedef enum_traits<T> traits;
			const int *first = traits::enumerators;	// get the beginning of the enumerator list
			const int *last = traits::enumerators + traits::N ;	// get the end of the enumerator list
			if (traits::sorted)
			{
				// if the enumerators are sorted at compile time,
				// use this to speed-up the search
				if (std::binary_search(first, last, v))
					return T(v);
			}
			else if (std::find(first, last, v) != last)	// else, lets use the default STL search
			{
				return T(v);
			}
			
			karma_exception("Enum does not exist.") ;
		}

		template<typename T>
		inline
		T string_to_enum(const std::string& v)
		{
			typedef enum_traits<T> traits;
			const std::string *first = traits::names;	// get the beginning of the enumerator list
			const std::string *last = traits::names + traits::N ;	// get the end of the enumerator list
			
			int index = std::find(first, last, v) - first ;
			
			if (index < traits::N)
				return T(index) ;
			else
				karma_exception("Enum does not exist.") ;
		}


		template<typename T>
		inline
		int enum_to_int(T v)
		{
			return static_cast<int>(v) ;
		}

		template <typename T>
		inline
		const std::string& enum_to_string (const T e)
		{
			typedef enum_traits<T> traits;
			return traits::names[static_cast<int>(e)] ;
		}
		
	}

}

#endif	// __enum_utils_h__
