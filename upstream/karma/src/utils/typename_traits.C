// $Id: typename_traits.C 450 2011-09-19 20:55:34Z vijaysm $
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
// This is a modified version of Teuchos_TypeNameTraits.cpp from Trilinos.

#include "typename_traits.h"
#include "karma.h"

// Define this if you want to force name demangling 
//#define KARMA_HAVE_DEMANGLE

#if defined(HAVE_GCC_ABI_DEMANGLE) && defined(KARMA_HAVE_DEMANGLE)
#include <cxxabi.h>
#endif


std::string KARMA::demangle_name( const std::string &mangledName )
{
#if defined(HAVE_GCC_ABI_DEMANGLE) && defined(KARMA_HAVE_DEMANGLE)
  int status;
  char *_demangledName = abi::__cxa_demangle(mangledName.c_str(),0,0,&status);
  if (status != 0 || 0==_demangledName) {
#ifdef NDEBUG
    std::string nullstr("NULL");
    const char* demangle_output = _demangledName ? _demangledName
      : nullstr.c_str();
    KARMA_THROW( LogicError( "Error, name demangling with g++ has been enabled but the function "
      "abi::__cxa_demangle(" << mangledName << ") returned returnVal = "
      << demangle_output << " and status = " << status << "." );
#endif
    if (_demangledName != NULL)
      free(_demangledName);
    return ( mangledName + "<demangle-failed>" );
  }
  const std::string demangledName(_demangledName);
  free(_demangledName); // We have to free this before we return!
  return demangledName;
#else
  return mangledName;
#endif
}
