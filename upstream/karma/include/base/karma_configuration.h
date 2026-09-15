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

#ifndef __karma_defines_h__
#define __karma_defines_h__

#include "karma_config.h"

#ifdef KARMA_HAVE_PETSC
#include "petscversion.h"
/**
 * These macros are defined to make testing for PETSc versions within
 * the library code easy. Usage for these macros:
 * (i) PETSC_VERSION_LT: preserve backward compatibility
 * 	   a) To satisfy a signature for all versions less than 'x.y.z',
 *        #if PETSC_VERSION_LT(x,y,z), do stuff #endif 
 * (ii) PETSC_VERSION_GTE: new functionality
 * 	   a) To satisfy a signature for all versions greater than or equal to 'x.y.z',
 *        #if PETSC_VERSION_GTE(x,y,z), do stuff #endif 
 */
#define PETSC_VERSION_LESS_THAN(major,minor,subminor) \
	((KARMA_DETECTED_PETSC_VERSION_MAJOR * 10000 + \
	KARMA_DETECTED_PETSC_VERSION_MINOR * 100 + \
	KARMA_DETECTED_PETSC_VERSION_SUBMINOR) \
	<  \
	(major)*10000 + (minor)*100 + (subminor))

#define PETSC_VERSION_GREATER_THAN(major,minor,subminor) \
	((KARMA_DETECTED_PETSC_VERSION_MAJOR * 10000 + \
	KARMA_DETECTED_PETSC_VERSION_MINOR * 100 + \
	KARMA_DETECTED_PETSC_VERSION_SUBMINOR) \
	>  \
	(major)*10000 + (minor)*100 + (subminor))

#define PETSC_VERSION_GREATER_THAN_EQUAL_TO(major,minor,subminor) \
	((KARMA_DETECTED_PETSC_VERSION_MAJOR * 10000 + \
	KARMA_DETECTED_PETSC_VERSION_MINOR * 100 + \
	KARMA_DETECTED_PETSC_VERSION_SUBMINOR) \
	>=  \
	(major)*10000 + (minor)*100 + (subminor))

#endif


#ifdef KARMA_HAVE_SLEPC
/**
 * These macros are defined to make testing for SLEPc versions within
 * the library code easy. Usage for these macros:
 * (i) SLEPC_VERSION_LT: preserve backward compatibility
 * 	   a) To satisfy a signature for all versions less than 'x.y.z',
 *        #if SLEPC_VERSION_LT(x,y,z), do stuff #endif 
 * (ii) SLEPC_VERSION_GTE: new functionality
 * 	   a) To satisfy a signature for all versions greater than or equal to 'x.y.z',
 *        #if SLEPC_VERSION_GTE(x,y,z), do stuff #endif 
 */
#define SLEPC_VERSION_LT(major,minor,subminor) \
	((KARMA_DETECTED_SLEPC_VERSION_MAJOR * 10000 + \
	KARMA_DETECTED_SLEPC_VERSION_MINOR * 100 + \
	KARMA_DETECTED_SLEPC_VERSION_SUBMINOR) \
	<  \
	(major)*10000 + (minor)*100 + (subminor))

#define SLEPC_VERSION_GTE(major,minor,subminor) \
	((KARMA_DETECTED_SLEPC_VERSION_MAJOR * 10000 + \
	KARMA_DETECTED_SLEPC_VERSION_MINOR * 100 + \
	KARMA_DETECTED_SLEPC_VERSION_SUBMINOR) \
	>=  \
	(major)*10000 + (minor)*100 + (subminor))

#endif



#endif	// __karma_defines_h__
