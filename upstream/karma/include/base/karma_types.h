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


#ifndef __karma_types_h__
#define __karma_types_h__

// C++ includes
#include <complex>
#include <cstdint>
#include <memory>

#ifdef KARMA_HAVE_MOAB
#include "moab/Types.hpp"
#include "moab/EntityHandle.hpp"
#endif

/**
 * Scalar, handle and ownership types for KARMA.
 *
 * These were previously supplied by libMesh (\p libmesh_common.h for
 * \p Real / \p Number / \p Complex / \p TOLERANCE, and \p auto_ptr.h for
 * \p AutoPtr).  They are defined here so that KARMA owns its own scalar
 * type system and can be built without libMesh.
 *
 * Every name here is a type alias rather than a preprocessor macro, so
 * that these types obey scope and can be forward-declared, qualified and
 * shadowed like any other.  Precision is selected by the build system
 * through the KARMA_DEFAULT_*_PRECISION feature flags, which are the only
 * macros involved.
 */

// The type to use for real numbers.  The build system selects the working
// precision by defining exactly one of the KARMA_DEFAULT_*_PRECISION
// feature flags; double is the default and the only combination currently
// exercised.
#if defined (KARMA_DEFAULT_SINGLE_PRECISION)
using Real = float;
#elif defined (KARMA_DEFAULT_TRIPLE_PRECISION)
using Real = long double;
#else
using Real = double;
#endif

// The type to use for complex numbers.
using Complex = std::complex<Real>;
using COMPLEX = std::complex<Real>;

// The value type carried by vectors and matrices.  This is either Real or
// Complex depending on how the library was configured; PETSc must be built
// with a matching --with-scalar-type for the complex case.
#if defined (KARMA_USE_COMPLEX_NUMBERS)
using Number = Complex;
#else
using Number = Real;
#endif

// Define a corresponding tolerance.  This is what should be considered
// "good enough" when doing floating point comparisons; for example,
// v == 0 is written as std::abs(v) < TOLERANCE.
#if defined (KARMA_DEFAULT_SINGLE_PRECISION)
static const Real TOLERANCE = 2.5e-3;
#elif defined (KARMA_DEFAULT_TRIPLE_PRECISION)
static const Real TOLERANCE = 1.e-8;
#else
static const Real TOLERANCE = 1.e-6;
#endif

/**
 * \p AutoPtr is the exclusive-ownership smart pointer used throughout
 * KARMA.  It was libMesh's pre-C++11 \p std::auto_ptr workalike; it is now
 * an alias for \p std::unique_ptr, which has the same ownership semantics
 * with correct move behaviour.
 *
 * Two consequences of the switch are worth knowing when touching old code:
 * copy-initialisation from an lvalue no longer compiles (use \p std::move),
 * and returning a temporary still works unchanged.  A survey of the tree at
 * the time of the change found no by-value parameters and no lvalue
 * assignments, so no call sites required edits.
 *
 * New code should prefer \p std::unique_ptr directly.
 */
template <typename T>
using AutoPtr = std::unique_ptr<T>;

namespace KARMA
{
	// The scalar types are intentionally visible at global scope as well,
	// since the bulk of the library predates the KARMA namespace and refers
	// to them unqualified.
	using ::Complex;
	using ::Number;
	using ::Real;

	/**
	 * Entity identifier widths.
	 *
	 * \p GlobalId numbers entities across all ranks and is the primary key
	 * for degree-of-freedom ordering; \p LocalId numbers them within a rank,
	 * including ghosts.
	 *
	 * Both are 32-bit today, and that is a constraint of the stack rather
	 * than a preference: \p PetscInt is \p int in this build (PETSc was
	 * configured without \p --with-64-bit-indices), and DMMoab reads the id
	 * tag as \p int.  Storing a 64-bit id in the tag and handing it to DMMoab
	 * yields silently wrong DoF numbers, not an error.
	 *
	 * Widening therefore takes two coordinated steps: rebuild PETSc with
	 * 64-bit indices, then change \p GlobalId here.  Everything else reads
	 * these aliases, so no other file needs editing.  The 32-bit ceiling is
	 * roughly two billion degrees of freedom.
	 */
	using GlobalId = std::int32_t;
	using LocalId  = std::int32_t;

#ifdef KARMA_HAVE_MOAB
	/**
	 * Mesh entity and tag handles.  KARMA's spatial layer is backed by MOAB,
	 * so these alias MOAB's own types rather than wrapping them.
	 */
	using EntityHandle = moab::EntityHandle;
	using Tag          = moab::Tag;
#endif
}

#endif // __karma_types_h__
