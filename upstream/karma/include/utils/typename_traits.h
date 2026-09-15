// $Id: typename_traits.h 450 2011-09-19 20:55:34Z vijaysm $
//
// KARMA: K(c)ode for Analysis of Reactor and other Multiphysics Applications.
// 				Copyright (C) 2007 Vijay S. Mahadevan
//				Copyright (C) 2004 Sandia Corporation
//
// This library is free software; you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as
// published by the Free Software Foundation; version 2.1 of the
// License.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
// USA
//
// This is a modified version of Teuchos_TypeNameTraits.hpp from Trilinos.


#ifndef __typename_traits_h__
#define __typename_traits_h__

/*! \file TypeNameTraits.h
 \brief Defines basic traits returning the
    name of a type in a portable and readable way.
*/

#if defined(HAVE_GCC_ABI_DEMANGLE)
#include <cxxabi.h>
#endif

// C++ includes
#include <vector>
#include <string>
#include <complex>
#include <typeinfo>

namespace KARMA
{
	/** \brief Demangle a C++ name if valid.
	*
	* The name must have come from <tt>typeid(...).name()</tt> in order to be
	* valid name to pass to this function.
	*/
	std::string demangle_name( const std::string &mangledName );
	
	/** \brief Template function for returning the demangled name of an object.
	*/
	template<typename T>
	std::string type_name( const T &t )
	{
		return demangle_name(typeid(t).name());
	}

  /** \brief Default traits class that just returns
   * <tt>typeid(T).name()</tt>.
   */
	template<typename T>
	class TypeNameTraits
	{
	public:
		static std::string name()
		{
			return demangle_name(typeid(T).name());
		}
	};

#define TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(TYPE) \
  template<> \
  class TypeNameTraits<TYPE> \
  { \
  public: \
    static std::string name() { return (#TYPE); } \
  } \
	
	TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(bool);
	TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(char);
	TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(int);
	TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(short int);
	TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(long int);
	TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(float);
	TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(double);

	template<typename T>
	class TypeNameTraits<T*>
	{
	public:
		static std::string name() { return TypeNameTraits<T>::name() + "*"; }
	};
	
	template<>
	class TypeNameTraits<std::string>
	{
	public:
		static std::string name() { return "string"; }
	};

	template<typename T>
	class TypeNameTraits<std::vector<T> >
	{
	public:
		static std::string name() { return "std::vector<"+TypeNameTraits<T>::name()
			+">"; }
	};
	
	template<typename T>
	class TypeNameTraits<std::complex<T> >
	{
	public:
		static std::string name() { return "std::complex<"+TypeNameTraits<T>::name()
			+">"; }
	};

	/** \brief Default traits class that stores information about the type T.
	* <tt>typeid(T).name()</tt>.
	*/
	template <typename T>
	struct TypeInfoTraits
	{
		typedef T Impl;
		typedef typename Impl::ElementType ElementType;
	};

#define DefaultTypeInfoTraits(Type, T) class Type; \
                                 \
                                 template <> \
                                 struct TypeInfoTraits<Type> \
{ \
                                     typedef Type Impl; \
                                     typedef T ElementType; \
};



} // namespace KARMA

#endif // __typename_traits_h__
