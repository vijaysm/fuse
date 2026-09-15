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

#include "karma_log.h"
#include "karma_print_trace.h"

// C includes
#include <unistd.h>

// C++ includes
#include <fstream>
#include <sstream>

/* Most of these Macros and Functions have been taken from Libmesh.
 * Change these as needed for KARMA.
 */

// These are useful macros that behave like functions in the code.
// If you want to make sure you are accessing a section of code just
// stick a karma_here(); in it, for example

#define karma_here()     do { KARMA_LOG(ERROR) << "[" << KARMA::processor_id << "] " << __FILE__ << ", line " << __LINE__ << ", compiled " << __DATE__ << " at " << __TIME__ << std::endl; } while (0)

// the karma_stop() macro will stop the code until a SIGCONT signal
// is recieved.  This is useful, for example, when determining the
// memory used by a given operation.  A karma_stop() could be
// instered before and after a questionable operation and the delta
// memory can be obtained from a ps or top.  This macro only works for
// serial cases.
#ifdef KARMA_HAVE_CSIGNAL
#  include <csignal>
#  define karma_stop()     do { if (KARMA::n_procs() == 1) { karma_here(); KARMA_LOG(ERROR) << "Stopping process " << getpid() << "..." << std::endl; std::raise(SIGSTOP); std::cout << "Continuing process " << getpid() << "..." << std::endl; } } while(0)
#else
#  define karma_stop()     do { if (KARMA::n_procs() == 1) { karma_here(); KARMA_LOG(ERROR) << "WARNING:  karma_stop() does not work without the <csignal> header file!" << std::endl; } } while(0)
#endif

// The karma_write_traceout() macro writes stack trace files, if
// that feature has been configured.
#define karma_write_traceout()   do { std::stringstream outname; outname << "traceout_" << KARMA::processor_id << '_' << getpid() << ".txt"; std::ofstream traceout(outname.str().c_str()); KARMA::print_trace(traceout); } while(0)

// The karma_error() macro prints a message and throws a LogicError
// exception
//
// The karma_not_implemented() macro prints a message and throws a
// NotImplemented exception
//
// The karma_file_error(const std::string& filename) macro prints a message
// and throws a FileError exception
#define karma_error()    do { karma_write_traceout(); karma_here(); KARMA_THROW(KARMA::LogicError()); } while(0)
#define karma_exception(msg)    do { KARMA_LOG(ERROR) << msg; karma_write_traceout(); karma_here(); KARMA_THROW(KARMA::LogicError()); } while(0)
#define karma_invalid_args(msg)    do { KARMA_LOG(ERROR) << msg; karma_write_traceout(); karma_here(); KARMA_THROW(KARMA::InvalidArgument(msg)); } while(0)
#define karma_not_implemented()    do { karma_write_traceout(); karma_here(); KARMA_THROW(KARMA::NotImplemented()); } while(0)
#define karma_file_error(filename)    do { karma_write_traceout(); karma_here(); KARMA_THROW(KARMA::FileError(filename)); } while(0)

// The karma_assert() macro acts like C's assert(), but throws a 
// karma_error() (including stack trace, etc) instead of just exiting
#ifdef NDEBUG
#define karma_assert(asserted) 
#else
#define karma_assert(asserted)  do { if (!(asserted)) { karma_exception( "Assertion `" #asserted "' failed.\n" ); } } while(0)
#endif

// The karma_do_once macro helps us avoid redundant repeated
// repetitions of the same warning messages
#undef karma_do_once
#define karma_do_once(do_this) do { static bool did_this_already = false; if (!did_this_already) { did_this_already = true; do_this; } } while (0)


// The karma_experimental macro warns that you are using
// bleeding-edge code
#undef karma_experimental
#define karma_experimental() karma_do_once(KARMA_LOG(NONE) << "*** Warning, This code is untested, experimental, or likely to see future API changes: " << __FILE__ << ", line " << __LINE__ << ", compiled " << __DATE__ << " at " << __TIME__ << " ***" << std::endl;)


// The karma_deprecated macro warns that you are using obsolete code
// that will be removed in future versions.
#undef karma_deprecated
#define karma_deprecated() karma_do_once(KARMA_LOG(NONE) << "*** Warning, This code is deprecated, and likely to be removed in future library versions! " << __FILE__ << ", line " << __LINE__ << ", compiled " << __DATE__ << " at " << __TIME__ << " ***" << std::endl;)

