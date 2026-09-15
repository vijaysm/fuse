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


#ifndef __karma_context_h__
#define __karma_context_h__

// petsc related includes
#include "petsc.h"

#include "karma_vector.h"
#include "karma_matrix.h"

// libmesh includes
#include "reference_counted_object.h"

class KARMAContext : public ReferenceCountedObject<KARMAContext>
{
public:

	KARMAContext() ;

	virtual ~KARMAContext() ;

#ifdef USE_PETSC_BLOCK_EXTENSIONS
	PetscErrorCode init (int nphysics) ;
	
	PetscErrorCode reinit (int nphysics) ;
#else
	PetscErrorCode init (int ldofs, int gdofs, const std::vector<unsigned int>& nnz, const std::vector<unsigned int>& onz) ;
	
	PetscErrorCode reinit (int ldofs, int gdofs, const std::vector<unsigned int>& nnz, const std::vector<unsigned int>& onz) ;
#endif

	PetscErrorCode clear() ;
	
	PetscErrorCode step_pre_process ();
	
	PetscErrorCode step_post_process () ;
	
	void print_info () ;
	
	unsigned int& set_preconditioner_id() ;

	unsigned int get_preconditioner_id() const ;

	bool is_jacobian_matrix_free() const ;

	bool is_preconditioner_matrix_free() const ;
    
	bool is_lumped_mass_matrix() const ;
    
	void set_options () ;
	
	// member data
	
	const PetscMPIInt rank;   	// rank of processor
	const PetscMPIInt size;   	// size of communicator

	PetscScalar time;		// Current time for which solution is sought
	PetscScalar delt;		// Current time step used for the temporal solution calculation

	// Current global solution and residual vector for all physics
	KARMAVector *Solution, *Residual ;

	// Work vectors of the same size as solution and residual.
	KARMAVector *Work_1, *Work_2 ;
	
	// Jacobian and Preconditioner matrix for the coupled physics
	KARMAMatrix *Jacobian, *Preconditioner ;

private:
	
	Vec _x, _res ;			// raw solution and residual vectors
	
	Vec _work1, _work2 ;		// raw work vectors
	
	Mat _jacobian, _preconditioner ;			// raw Jacobian and preconditioenr matrix references
	
	PetscBool _ksp_ew ;	// flag to specify using inexact newton iteration

	unsigned int _preconditioner_id ;
	bool _initialized ;

	bool _is_lumped ;
	bool _jacobian_mfree ; 			// Flag to specify whether to create Jacobian matrices
	bool _preconditioner_mfree ; 	// Flag to specify whether to use matrices for preconditioner

	// declare friends
	friend class KARMAInterface ;
};

inline
unsigned int& KARMAContext::set_preconditioner_id()
{
	return _preconditioner_id ;
}

inline
unsigned int KARMAContext::get_preconditioner_id() const
{
	return _preconditioner_id ;
}

inline
bool KARMAContext::is_jacobian_matrix_free() const
{
	return _jacobian_mfree ;
}

inline
bool KARMAContext::is_preconditioner_matrix_free() const
{
	return _preconditioner_mfree ;
}

inline
bool KARMAContext::is_lumped_mass_matrix() const
{
	return _is_lumped ;
}

#endif	// #define __karma_context_h__
