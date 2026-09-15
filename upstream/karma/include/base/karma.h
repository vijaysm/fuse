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


#ifndef __karma_h__
#define __karma_h__

#include "karma_io_buffers.h"

#include "petsc.h"

#include "ikarma.h"
#include "karma_options.h"
#include "karma_base.h"
#include "karma_constants.h"
#include "karma_exceptions.h"

// C++ includes
#include <vector>

// Forward declarations
class GetPot ;

#ifdef KARMA_HAVE_LIBMESH
#include "perf_log.h"
#include "auto_ptr.h"
#endif

// --------------------------------------------------------

namespace KARMA
{
	class KARMAPackage
	{
	public:
	   /**
	   * @initializes the KARMA context and interface singleton object to be used in the current simulation. Also, initializes libMesh, PETSC and SLEPC, MPI objects to be used in the physics package.
	   */
		KARMAPackage (int argc, char **argv, const std::string& s_option_file="") ;
		
	   /**
	   * @destroys the objects allocated in KARMA initialize.
	   */
		~KARMAPackage () ;
		
		unsigned int get_option(const std::string& option, unsigned int default_opt=0) ;
		
		int get_option(const std::string& option, int default_opt=0) ;
		
		double get_option(const std::string& option, double default_opt=0.0) ;
		
		std::string get_option(const std::string& option, std::string default_opt="") ;
		
		bool search_option(const std::string& option) ;
		
		
	private:
	
		AutoPtr<GetPot> _options_cmdline_db ;
		AutoPtr<GetPot> _options_file_db ;
		
		bool search_option(const std::string& option, bool& is_cmd_line) ;
			
	};
	
	// AutoPtr<KARMAFILELog> FileLogger ;
	// AutoPtr<KARMAConsoleLog> ConsoleLogger ;
	
	extern bool verbose ;
	
#ifdef KARMA_HAVE_LIBMESH
	/**
	* A \p PerfLog object to log performance.  If the library is configured
	* with \p --enable-perflog then it will log key functions.
	*/
	extern libMesh::PerfLog perflog ; 
	
#endif

// Let's define a couple output streams - these will default
// to cout/cerr, but KARMAInit (or the user) can also set them to
// something more sophisticated.
//
// We use a proxy class rather than references so they can be
// reseated at runtime.

// typedef cout_proxy out ;
// typedef cerr_proxy err ;

}

// Macros for performance logging.  This allows us
// to add performance monitors to the code without
// impacting performance when performance logging
// is disabled.
#ifdef KARMA_HAVE_LIBMESH

#ifdef LIBMESH_ENABLE_PERFORMANCE_LOGGING

#define KARMA_LOG_PUSH(a,b)   { KARMA::perflog.push(a,b); }
#define KARMA_LOG_POP(a,b)    { KARMA::perflog.pop(a,b); }

#else

#define KARMA_LOG_PUSH(a,b)   {}
#define KARMA_LOG_POP(a,b)    {}

#endif	// LIBMESH_ENABLE_PERFORMANCE_LOGGING

#else

#define KARMA_LOG_PUSH(a,b)   {}
#define KARMA_LOG_POP(a,b)    {}

#endif	// KARMA_HAVE_LIBMESH

#endif // #define __karma_h__
