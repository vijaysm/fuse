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


#ifndef __petsc_vec_utils_h__
#define __petsc_vec_utils_h__

#include "petscvec.h"

namespace KARMA
{

	PetscErrorCode VecAdd(Vec x, Vec y) ;

	PetscErrorCode VecSubtract(Vec x, Vec y) ;

	PetscErrorCode copy_vector(Vec , Vec , bool ) ;
	
	PetscErrorCode UpdateParent(Vec , const Vec , std::vector<unsigned int>& ) ;
	
	PetscErrorCode UpdateChild(const Vec , Vec , std::vector<unsigned int>& ) ;
}
 
#endif // #define __petsc_vec_utils_h__
