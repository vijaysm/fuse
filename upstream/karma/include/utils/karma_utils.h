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


#ifndef __karma_utils_h__
#define __karma_utils_h__

#include <string>
#include <cstdlib>    // For srand()
#include <ctime>    // For time()
#include <cmath>
#include <limits>	// For UINT_MAX
#include <vector>
#include <algorithm>
#include <map>
#include <numeric>
#include <iostream>
#include <sstream>

#define DEFAULT_MAX std::numeric_limits<unsigned int>::max()
#define DEFAULT_MIN std::numeric_limits<unsigned int>::min()


template<typename X>
struct type_to_string ;
		
#define KARMA_DECL_SLFT(X,Y) \
 template<>                 \
 struct type_to_string<X>	\
{											\
     static const char* to_string()			\
{ return Y; }							\
};

#define TYPE_STRING_LITERAL(X) \
    type_to_string<X>::to_string()

// Basic type specification to string literal
KARMA_DECL_SLFT(char, "char");
KARMA_DECL_SLFT(unsigned char, "unsigned char");
KARMA_DECL_SLFT(short int, "short int");
KARMA_DECL_SLFT(short unsigned int, "short unsigned int");
KARMA_DECL_SLFT(int, "int");
KARMA_DECL_SLFT(unsigned int, "unsigned int");
KARMA_DECL_SLFT(long, "long");
KARMA_DECL_SLFT(unsigned long, "unsigned long");
KARMA_DECL_SLFT(float, "float");
KARMA_DECL_SLFT(double, "double");
KARMA_DECL_SLFT(long double, "long double");


namespace KARMA
{
	std::vector<double> error_measure(int dim, std::string solFile, std::string refFile, int varnum, bool verbose) ;

	template <typename T>
	inline
	T min(T a, T b)
	{
		return ((a < b) ? a : b) ;
	}


	template <typename T>
	inline
	T max(T a, T b)
	{
		return ((a > b) ? a : b) ;
	}


	template <typename T>
	inline
	T pow(T a, T b)
	{
		return std::pow(a, b) ;
	}

	// An elegant method to compute the factorial of
	// an integer number at compile time
	template <int N>
	struct factorial
	{
		enum { value = N * factorial<N-1>::value };
	};
	
	template <>
	struct factorial<1>
	{
		enum { value = 1 };
	};

	//              n!
	// C(n,r) = ---------
	//          k! (n-k)!
	template <int n, int r>
	struct nCr
	{
		enum { RET = factorial<n>::RET/(factorial<r>::RET*factorial<n-r>::RET) };
	};

	//            n!
	// P(n,r) = ------
	//            r!
	template <int n, int r>
	struct nPr
	{
		enum { RET = factorial<n>::RET/factorial<r>::RET };
	};

	template <typename T>
	inline
	signed short int sgn(T a)
	{
		return (a < 0) ? -1 : 1 ;
	}

	template <typename T>
	inline
	T vector_sum(std::vector<T>& a)
	{ 
		// Array can be a pointer or an iterator
		T sum = 0 ;
		for (int i = a.begin(); i!=a.end(); i++) 
			sum += a[i];
		return sum;
	}
	
	template <class Container>
	inline
	void reciprocal (Container& c)
	{
		// std::bind1st was removed in C++17.
		typedef typename Container::value_type value_type;
		std::transform(c.begin(),c.end(),c.begin(),
			[](const value_type& v) { return value_type(1) / v; });
	}
	
	// Initialize the random number generator 
	inline
	void initialize_rng()
	{
		srand(time(0));
	}
	
	// Create a random number (unsigned int)
	inline
	unsigned int create_random_id()
	{
		return rand() % DEFAULT_MAX + DEFAULT_MIN ;
	}
	
	// Print out a map.
	inline std::ostream&
	operator<< (std::ostream &out, const std::map<double,double> &m)
	{
		for (std::map<double,double>::const_iterator it = m.begin (); it != m.end (); ++it)
		    std::cout << (*it).first << "\t" << (*it).second << "\n";

		std::cout.flush() ;
		return out;
	}

	// Convert a string representation to a number
	template <typename T>
	inline
	T string_to_number ( const std::string &text, T default_value = T() )
	{
		std::stringstream ss;
		for ( std::string::const_iterator i=text.begin(); i!=text.end(); ++i )
			if ( isdigit(*i) || *i=='e' || *i=='-' || *i=='+' || *i=='.' )
				ss << *i;
		T result;
		return ss >> result ? result : default_value;
	}
	
}
 
#endif // #define __karma_utils_h__
