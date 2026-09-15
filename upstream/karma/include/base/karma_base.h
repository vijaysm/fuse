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


#ifndef __karma_base_h__
#define __karma_base_h__

#include "karma_config.h"

// #include "memtrack.h"
#include "petsc.h"

//////////////////////////////////////////////////////////////////
//			   START CUSTOM TYPE DEFINITIONS					//
//////////////////////////////////////////////////////////////////

/**
 * Regular integer. Remember if PETSc was configured with Int64,
 * usage of this type in unwanted places can blow up the memory.
 */
using Int = PetscInt;

/**
 * Standard unsigned integer.
 */
using UInt = unsigned int;

/**
 * Should we do the same for PetscTruth ? It is only an
 * enum now but the mailing list discussions have noted
 * that this might become bool.
 * NOTE: Wait till PETSc makes the change.
 */
// using Bool = PetscTruth;

/**
 * Alias for the PETSc type that represents a real number
 * version of PetscScalar.
 */
using Scalar = PetscScalar;

//////////////////////////////////////////////////////////////////
//				END CUSTOM TYPE DEFINITIONS						//
//////////////////////////////////////////////////////////////////

// Scalar types (Real, Number, Complex, TOLERANCE) and AutoPtr.  These
// used to arrive transitively from libMesh; KARMA now owns them.
#include "karma_types.h"

/**
   * Store the MPI Communicator to be used in KARMA.
*/
extern MPI_Comm  KARMA_COMM_WORLD ;
extern MPI_Comm  KARMA_COMM_SELF ;

// TODO: Cache the nprocs and rank variables
namespace KARMA
{
	/**
	* @returns the number of processors used in the current simulation.
	*/
	extern int num_processors ;
	
	/**
	* @returns the index of the local processor.
	*/
	extern int processor_id ;
	
}


#endif // #define __karma_base_h__
