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


#ifndef __karma_vector_h__
#define __karma_vector_h__


/**
 * This is simply a proxy class.
 * We will use the object KARMAVector uniformly
 * throughout the library when we want to manipulate
 * vectors representing discrete unknowns.
 * Specializations based on the block-form can be
 * made use of by configuring with PetscExt.
*/
#include "karma_config.h"
#include "petsc_native_vector.h"

typedef PetscNativeVector<PetscScalar> KARMARealVector ;

#ifdef USE_PETSC_BLOCK_EXTENSIONS

#include "petsc_block_vector.h" 

typedef PetscBlockVector<PetscScalar> KARMAVector ;

#else

typedef PetscNativeVector<PetscScalar> KARMAVector ;

#endif

#endif // #define __karma_vector_h__

