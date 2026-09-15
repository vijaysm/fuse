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


#ifndef __karma_exceptions_h__
#define __karma_exceptions_h__

#include "karma_config.h"

#include <cstdlib>
#include <stdexcept>
#include <string>

// The exception TYPES are declared unconditionally, and only the behaviour of
// KARMA_THROW is conditional.
//
// They used to sit inside `#ifdef KARMA_ENABLE_EXCEPTIONS`, which made the
// exceptions-disabled configuration fail to compile: 645 references to these
// names across include/ and src/ are unconditional, because a call site that
// wants to report an error names the error it is reporting whether or not the
// build will throw it. Guarding the declarations meant the option could never
// have worked, and guarding 645 call sites instead would be absurd. A type
// that is only ever constructed inside KARMA_THROW costs nothing when
// KARMA_THROW does not construct it.
namespace KARMA {

  /**
   * A class to represent the internal "this should never happen"
   * errors, to be thrown by "karma_error();"
   */
  class LogicError : public std::logic_error
    {
    public:
      LogicError() : std::logic_error( "ERROR in KARMA internal logic" ) {}
	  
      LogicError(std::string& msg) : std::logic_error( msg ) {}
    };


  /**
   * A class to stub for features that should be in KARMA, but
   * haven't been written yet, to be thrown by
   * "karma_not_implemented();"
   */
  class NotImplemented : public std::logic_error
    {
    public:
      NotImplemented() : std::logic_error( "ERROR: not implemented !" ) {}
    };

  /**
   * A class that specifies that either the reference or the pointer supplied
   * for a KARMA routine is not what is expected. This error is thrown by
   * "karma_invalid_args();"
   */
  class InvalidArgument : public std::logic_error
    {
    public:
      InvalidArgument(const std::string& detailed_error) : std::logic_error( "ERROR: invalid argument passed to routine ! " + detailed_error ) {}
    };

  /**
	 * A class that specifies that a pointer that we are trying to access is NULL.
	 * This is a lot more generic than the \p DynamicCastFailure error which 
	 * is specific to casting failures alone.
   */
	class NullPointer : public std::logic_error
	{
		public:
			NullPointer(const std::string& detailed_error) : std::logic_error( "ERROR: null pointer ! " + detailed_error ) {}
	};


  /**
	 * A class representing that a dynamic cast failed to produce expected output.
   */
	class DynamicCastFailure:  public std::runtime_error
	{
		public:
			DynamicCastFailure() : std::runtime_error( "Failed dynamic cast!" ) {}
	};

  /**
   * A class representing a failed attempt by the library to open a
   * file (or construct an fstream, etc), to be thrown by
   * "karma_file_error(filename);" For ease of debugging, "filename"
   * should include any (absolute or relative or implicit) pathname
   * that was part of the failed open.
   */
  class FileError : public std::runtime_error
    {
    public:
      FileError(const std::string& filename) : std::runtime_error( "ERROR accessing file: " + filename ) {}
    };


}

#ifdef KARMA_ENABLE_EXCEPTIONS

#define KARMA_THROW(e) do { throw e; } while (0)

#else

// Exceptions disabled: fail fast instead.
//
// `decltype` takes an unevaluated operand, so the exception is never
// constructed - no string is built and no allocation happens on the abort
// path. It is still a reference to `e` as far as the compiler is concerned,
// which is the point: a parameter that exists only to build the exception
// message would otherwise be unused in this configuration and -Werror would
// reject the whole translation unit. The expression is also still
// type-checked, so a KARMA_THROW naming a nonexistent exception fails to
// compile in both configurations rather than only in the throwing one.
//
// `decltype` rather than a bare `sizeof`: the operand of `sizeof` may be a
// type-id, so `sizeof(LogicError())` parses as sizeof applied to the FUNCTION
// type "no arguments, returning LogicError" and is rejected. The operand of
// `decltype` is always an expression, which is what every call site writes.
#define KARMA_THROW(e) do { (void) sizeof(decltype(e)); std::abort(); } while (0)

#endif // KARMA_ENABLE_EXCEPTIONS

#endif // __karma_exceptions_h__
