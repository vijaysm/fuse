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


#include "karma.h"
#include "karma_error.h"
#include "utility.h"
#include "petsc_vec_utils.h"

/* @brief
 * both x and y vectors need to be distributed
 * in the same way;
 * collective on Vec
 * x = x + y
*/
#undef __FUNCT__
#define __FUNCT__ "KARMA::VecAdd"
PetscErrorCode KARMA::VecAdd(Vec x, Vec y)
{
	PetscScalar *xx, *yy ;
	PetscInt start, end ;	
	PetscInt mlenx ;

	// check if both local and global sizes match
	{
		PetscInt lenx, mleny, leny ;
		VecGetSize(x,&lenx);
		VecGetLocalSize(x,&mlenx);
	
		VecGetSize(y,&leny);
		VecGetLocalSize(y,&mleny);
	
		if ((lenx != leny) || (mlenx != mleny))
			PetscFunctionReturn(-1) ;
	}

	VecGetOwnershipRange(x,&start,&end);
	
	VecGetArray(x, &xx) ;
	VecGetArray(y, &yy) ;
	
	for(int i=0; i < mlenx; i++)
	{
		xx[i + start] += yy[i + start] ;
	}

	VecRestoreArray(x, &xx) ;
	VecRestoreArray(y, &yy) ;

	PetscFunctionReturn(0) ;
}

/* @brief
 * both x and y vectors need to be distributed
 * in the same way;
 * collective on Vec
 * x = x - y
*/
#undef __FUNCT__
#define __FUNCT__ "KARMA::VecSubtract"
PetscErrorCode KARMA::VecSubtract(Vec x, Vec y)
{
	PetscScalar *xx, *yy ;
	PetscInt start, end ;	
	PetscInt mlenx ;

	// check if both local and global sizes match
	{
		PetscInt lenx, mleny, leny ;
		VecGetSize(x,&lenx);
		VecGetLocalSize(x,&mlenx);
	
		VecGetSize(y,&leny);
		VecGetLocalSize(y,&mleny);
	
		if ((lenx != leny) || (mlenx != mleny))
			PetscFunctionReturn(-1) ;
	}

	VecGetOwnershipRange(x,&start,&end);
	
	VecGetArray(x, &xx) ;
	VecGetArray(y, &yy) ;
	
	for(int i=0; i < mlenx; i++)
	{
		xx[i + start] -= yy[i + start] ;
	}

	VecRestoreArray(x, &xx) ;
	VecRestoreArray(y, &yy) ;

	PetscFunctionReturn(0) ;
}


/* @brief
* a routine to copy Vectors of un-equal sizes
* for library developers only since this can lead to very serious memory over-writes
* and corrupt the program memory leading to seg-faults;
* If you are a user, use at your own risk. 
*/
#undef __FUNCT__
#define __FUNCT__ "KARMA::copy_vector"
PetscErrorCode KARMA::copy_vector(Vec xin, Vec xout, bool is_input_smaller)
{
	PetscInt ldim, low_in, high_in ;
	PetscInt low_out, high_out ;
	VecGetOwnershipRange(xin,&low_in,&high_in);
	VecGetOwnershipRange(xout,&low_out,&high_out);

	karma_assert(low_in == low_out) ;
	
	PetscErrorCode ierr;
	
	if (is_input_smaller)
		ierr = VecGetLocalSize(xin,&ldim);
	else
		ierr = VecGetLocalSize(xout,&ldim);
	CHKERRQ(ierr);

	PetscScalar *xarr, *yarr ;
	
	ierr = VecGetArray(xin, &xarr) ;CHKERRQ(ierr);
	ierr = VecGetArray(xout, &yarr) ;CHKERRQ(ierr);
	
	for (int i=0; i<ldim; i++)
	{
		yarr[i] = xarr[i] ;
	}

	ierr = VecRestoreArray(xin, &xarr) ;CHKERRQ(ierr);
	ierr = VecRestoreArray(xout, &yarr) ;CHKERRQ(ierr);
	
	return ierr ;
}


/* @brief
 * Update the Parent vector with elements from Child at the appropriate 
 * locations specified by parent_rows.
 * collective on Vec
*/
#undef __FUNCT__
#define __FUNCT__ "KARMA::UpdateParent"
PetscErrorCode KARMA::UpdateParent(Vec parent, const Vec child, std::vector<unsigned int>& parent_rows)
{
	// PETSc data structures
	IS parent_is, child_is;
	VecScatter scatter;
	PetscErrorCode ierr = 0;
	
	// Use iota to fill an array with entries [0,1,2,3,4,...parent_rows.size()]
	// std::vector<int> idx(parent_rows.size());
	// Utility::iota (idx.begin(), idx.end(), 0);

	// Construct index sets
	ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 parent_rows.size(),
			 (int*) &parent_rows[0],
			 PETSC_OWN_POINTER,
			 &parent_is); CHKERRQ(ierr);

	// ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 // idx.size(),
			 // (int*) &idx[0],
			 // &child_is); CHKERRQ(ierr);

	// Construct the scatter object
	ierr = VecScatterCreate(child,
			  PETSC_NULL,
			  parent,
			  parent_is,
			  &scatter); CHKERRQ(ierr);
			  
	// Now scatter from Child to parent vector
	ierr = VecScatterBegin(scatter,
			 child,
			 parent,
			 INSERT_VALUES,
			 SCATTER_FORWARD); CHKERRQ(ierr);

	ierr = VecScatterEnd(scatter,
			   child,
			   parent,
			   INSERT_VALUES,
			   SCATTER_FORWARD); CHKERRQ(ierr);

	// Clean up 
	ierr = ISDestroy(&parent_is);       CHKERRABORT(KARMA_COMM_WORLD,ierr);
	// ierr = ISDestroy(child_is);    CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = VecScatterDestroy(&scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr); 

	PetscFunctionReturn(0) ;
}

/* @brief
 * Update the Child vector with elements from Parent using locations specified 
 * in parent_rows.
 * collective on Vec
*/
#undef __FUNCT__
#define __FUNCT__ "KARMA::UpdateChild"
PetscErrorCode KARMA::UpdateChild(const Vec parent, Vec child, std::vector<unsigned int>& parent_rows)
{
	// PETSc data structures
	IS parent_is, child_is;
	VecScatter scatter;
	PetscErrorCode ierr = 0;
	
	// Use iota to fill an array with entries [0,1,2,3,4,...parent_rows.size()]
	// std::vector<int> idx(parent_rows.size());
	// Utility::iota (idx.begin(), idx.end(), 0);

	// Construct index sets
	ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 parent_rows.size(),
			 (int*) &parent_rows[0],
			 PETSC_OWN_POINTER,
			 &parent_is); CHKERRQ(ierr);

	// ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			// idx.size(),
			 // (int*) &idx[0],
			 // &child_is); CHKERRQ(ierr);

	// Construct the scatter object
	ierr = VecScatterCreate(parent,
			  parent_is,
			  child,
			  PETSC_NULL, // child_is,
			  &scatter); CHKERRQ(ierr);
			  
	// Now scatter from Child to parent vector
	ierr = VecScatterBegin(scatter,
			 parent,
			 child,
			 INSERT_VALUES,
			 SCATTER_FORWARD); CHKERRQ(ierr);

	ierr = VecScatterEnd(scatter,
			   parent,
			   child,
			   INSERT_VALUES,
			   SCATTER_FORWARD); CHKERRQ(ierr);

	// Clean up 
	ierr = ISDestroy(&parent_is);       CHKERRABORT(KARMA_COMM_WORLD,ierr);
	// ierr = ISDestroy(child_is);    CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = VecScatterDestroy(&scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr); 

	PetscFunctionReturn(0) ;
}

