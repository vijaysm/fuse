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


#ifndef __string_utils_h__
#define __string_utils_h__

#include <string>

// template method to convert arbitrary types to 
// std::string representation
template<typename Type>
inline std::string to_string(const Type data)
{
	std::ostringstream ost;
	ost << data;
	return ost.str();
}

// specialization for char*
template<>
inline std::string to_string(const char* data)
{
	return std::string(data);
}

// specialization for std::string
template<>
inline std::string to_string(const std::string& data)
{
	return std::string(data);
}


// stream output operator overloads
template<typename T>
		std::string& operator<<(std::string& s, T i)
{
	return s.append(to_string<T>(i));
}

#endif	// #define __string_utils_h__
