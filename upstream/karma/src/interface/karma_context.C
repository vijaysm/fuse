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
#include "karma_context.h"
#include "karma_vector.h"

#undef __FUNCT__
#define __FUNCT__ "KARMAContext::constructor"
KARMAContext::KARMAContext () :
	rank(KARMA::processor_id),
	size(KARMA::num_processors),
	Solution(NULL),
	Residual(NULL),
	Work_1(NULL),
	Work_2(NULL),
	_jacobian_mfree(false),
	_preconditioner_id(0),	// by default, always use the jacobian matrix provided as preconditioner
	_preconditioner_mfree(false),
	_initialized(false)
{
	// let the user set the right values before calculations begin.
	time = 0.0 ;	delt = 0.0 ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAContext::destructor"
KARMAContext::~KARMAContext ()
{
	clear () ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAContext::set_options"
void KARMAContext::set_options ()
{
	// Find out whether we are doing matrix-free so that we can avoid allocation for matrices
	PetscBool flg ;
	PetscErrorCode ierr = 0 ;

	// Find out if it is a completely matrix-free method
	ierr = PetscOptionsHasName(PETSC_NULL,"-main_snes_mf", &flg);CHKERRABORT(KARMA_COMM_WORLD, ierr);
	
	if (!flg)
	{
		// Find out if we are using the matrix for the preconditioner but jacobian is calculated using matrix-free method
		ierr = PetscOptionsHasName(PETSC_NULL,"-main_snes_mf_operator", &flg);CHKERRABORT(KARMA_COMM_WORLD, ierr);
		if (!flg)
			_jacobian_mfree = false ;
		else
			_jacobian_mfree = true ;
			
		_preconditioner_mfree = false ;
	}
	else
	{
		_jacobian_mfree = true ;		_preconditioner_mfree = true ;
	}
	
	ierr = PetscOptionsHasName(PETSC_NULL,"-no_ew", &flg);CHKERRABORT(KARMA_COMM_WORLD, ierr);
	if (flg)
		_ksp_ew = PETSC_FALSE ;
	else
		_ksp_ew= PETSC_TRUE ;
	
	ierr = PetscOptionsHasName(PETSC_NULL,"-lumped_mass", &flg);CHKERRABORT(KARMA_COMM_WORLD, ierr);
	if (flg)
		_is_lumped = true ;
	else
		_is_lumped= false ;
	
	ierr = PetscOptionsGetInt("", "-preconditioner_id", (PetscInt*)&_preconditioner_id, &flg);CHKERRABORT(KARMA_COMM_WORLD, ierr);
	
}

#undef __FUNCT__
#define __FUNCT__ "KARMAContext::print_info"
void KARMAContext::print_info ()
{
	PetscPrintf(KARMA_COMM_WORLD,"------------------------------\n") ;
	PetscPrintf(KARMA_COMM_WORLD,"Current RunContext Information") ;
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------\n") ;
	
	// Need to print more information about the solution, residual, work vectors and 
	// jacobian, preconditioner matrices along with their memory allocated.
	// If this is block vector and matrix, print how they are arranged.
}

#undef __FUNCT__
#define __FUNCT__ "KARMAContext::step_pre_process"
PetscErrorCode KARMAContext::step_pre_process ()
{
	// nothing to do
	return 0 ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAContext::step_post_process"
PetscErrorCode KARMAContext::step_post_process ()
{
	PetscErrorCode 	ierr=0;
	
	// may be help in finding the truncation error in time or space ; 
	// whatever in the scope of the context is possible
	
	// for example, could output solution fields to file
	/*
	if (iosoln && istep%10 == 0)
			{
				perf_main.start_event ("print output");
			  // After solving the system write the solution
				GMVIO (mesh).write_equation_systems ("results.gmv", equation_systems);
				//VTKIO (mesh).write_equation_systems ("results.vtk", equation_systems);
				perf_main.stop_event ("print output");
			}
			*/
	
	return ierr ;
}

#ifndef USE_PETSC_BLOCK_EXTENSIONS


#undef __FUNCT__
#define __FUNCT__ "KARMAContext::reinit"
PetscErrorCode KARMAContext::reinit (int ldofs, int gdofs, const std::vector<unsigned int>& nnz, const std::vector<unsigned int>& onz)
{
	PetscErrorCode 	ierr;
	
	this->clear() ;
	
	if (!_initialized)
	{
		Solution = new KARMAVector(gdofs, ldofs) ;
		_x = Solution->vec() ;
		ierr = VecDuplicate(_x, &_res) ;CHKERRQ(ierr);
		
		ierr = VecDuplicate(_x, &_work1) ;	CHKERRQ(ierr);
		ierr = VecDuplicate(_x, &_work2) ;	CHKERRQ(ierr);

		Work_1 = new KARMAVector (_work1) ;
		Work_2 = new KARMAVector (_work2) ;
		
		if (_jacobian_mfree)
		{
			ierr = MatCreateShell(KARMA_COMM_WORLD,ldofs,ldofs,gdofs,gdofs,0, &_jacobian);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatSetFromOptions(_jacobian) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
		else
		{
			ierr = MatCreateAIJ(KARMA_COMM_WORLD, ldofs, ldofs, gdofs, gdofs, 
									PETSC_NULL, (int*) &nnz[0], PETSC_NULL, (int*) &onz[0], &_jacobian) ;	CHKERRQ(ierr);
			ierr = MatSetFromOptions(_jacobian);CHKERRQ(ierr);
		}
		
		if (_preconditioner_mfree)
		{
			ierr = MatCreateShell(KARMA_COMM_WORLD,ldofs,ldofs,gdofs,gdofs,0, &_preconditioner);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatSetFromOptions(_preconditioner) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		} 
		else
		{
			if (_jacobian_mfree)
			{
				// SNES_MF_OPERATOR
				ierr = MatCreateAIJ(KARMA_COMM_WORLD, ldofs, ldofs, gdofs, gdofs, 
											PETSC_NULL, (int*) &nnz[0], PETSC_NULL, (int*) &onz[0], &_preconditioner) ;	CHKERRQ(ierr);
				ierr = MatSetFromOptions(_preconditioner);CHKERRQ(ierr);
			}
			else
			{
				// Re-use Jacobian's matrix ; No need to recreate it again
				_preconditioner = _jacobian ;
			}
		}
		
		Residual = new KARMAVector (_res) ;

		_initialized = true ;
	}
	
	PetscFunctionReturn(0) ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAContext::init"
PetscErrorCode KARMAContext::init (int ldofs, int gdofs, const std::vector<unsigned int>& nnz, const std::vector<unsigned int>& onz)
{
	PetscErrorCode 	ierr;
	
	if (!_initialized)
	{
		Solution = new KARMAVector(gdofs, ldofs) ;
		_x = Solution->vec() ;
		
		ierr = VecDuplicate(_x, &_res) ;CHKERRQ(ierr);
		Residual = new KARMAVector (_res) ;
		
		ierr = VecDuplicate(_x, &_work1) ;	CHKERRQ(ierr);
		ierr = VecDuplicate(_x, &_work2) ;	CHKERRQ(ierr);

		Work_1 = new KARMAVector (_work1) ;
		Work_2 = new KARMAVector (_work2) ;
		
		if (_preconditioner_mfree)
		{
			ierr = MatCreateShell(KARMA_COMM_WORLD,ldofs,ldofs,gdofs,gdofs,0, &_preconditioner);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatSetOptionsPrefix(_preconditioner, "prec_") ;
			ierr = MatSetFromOptions(_preconditioner) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		} 
		else
		{
			// SNES_MF_OPERATOR
			ierr = MatCreateAIJ(KARMA_COMM_WORLD, ldofs, ldofs, gdofs, gdofs, 
										PETSC_NULL, (int*) &nnz[0], PETSC_NULL, (int*) &onz[0], &_preconditioner) ;	CHKERRQ(ierr);
			ierr = MatSetOptionsPrefix(_preconditioner, "prec_") ;
			ierr = MatSetFromOptions(_preconditioner);CHKERRQ(ierr);
			ierr = MatZeroEntries(_preconditioner); CHKERRQ(ierr);
		}
		
		if (_jacobian_mfree)
		{
			ierr = MatCreateShell(KARMA_COMM_WORLD,ldofs,ldofs,gdofs,gdofs,0, &_jacobian);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatSetOptionsPrefix(_jacobian, "jac_") ;
			ierr = MatSetFromOptions(_jacobian) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
		else
		{
			// We already allocated for the preconditioner; Let's just re-use the same reference.
			_jacobian = _preconditioner ;

			// ierr = MatCreateMPIAIJ(KARMA_COMM_WORLD, ldofs, ldofs, gdofs, gdofs, 
									// PETSC_NULL, (int*) &nnz[0], PETSC_NULL, (int*) &onz[0], &_jacobian) ;	CHKERRQ(ierr);
			// ierr = MatSetOptionsPrefix(_jacobian, "jac_") ;
			// ierr = MatSetFromOptions(_jacobian);CHKERRQ(ierr);
			// ierr = MatZeroEntries(_jacobian); CHKERRQ(ierr);
		}

		_initialized = true ;
	}
	
	PetscFunctionReturn(0) ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAContext::clear"
PetscErrorCode KARMAContext::clear ()
{
	PetscErrorCode 	ierr;
	
	if (_initialized)
	{	
		// delete the vectors that are wrappers
		delete Residual ;
		delete Solution ;

		delete Work_1 ;
		delete Work_2 ;
	
		ierr = MatDestroy(&_preconditioner) ;	CHKERRQ(ierr);
		_preconditioner = PETSC_NULL ;
		
		if(_jacobian_mfree)
		// if (_jacobian != PETSC_NULL)
		{
			ierr = MatDestroy(&_jacobian) ;	CHKERRQ(ierr);
		}
		_jacobian = PETSC_NULL ;
		
		ierr = VecDestroy(&_res) ;	CHKERRQ(ierr);		
		ierr = VecDestroy(&_work1) ;	CHKERRQ(ierr);
		ierr = VecDestroy(&_work2) ;	CHKERRQ(ierr);
		
		_initialized = false ;
	}
	
	PetscFunctionReturn(0) ;
}

#else


#undef __FUNCT__
#define __FUNCT__ "KARMAContext::clear"
PetscErrorCode KARMAContext::init (int nphysics)
{
	PetscErrorCode 	ierr;
	
	if (!_initialized)
	{
		KARMA_LOG(logDEBUG1) << "creating block vectors now." ;
		
		ierr = VecCreate( KARMA_COMM_WORLD, &_x );	CHKERRQ(ierr);
		ierr = VecSetSizes( _x, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = VecSetType( _x, "block" );	CHKERRQ(ierr);
		
		KARMA_LOG(logDEBUG1) << "duplicating block vectors now." ;
		ierr = VecCreate( KARMA_COMM_WORLD, &_res );	CHKERRQ(ierr);
		ierr = VecSetSizes( _res, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = VecSetType( _res, "block" );	CHKERRQ(ierr);
		
		ierr = VecCreate( KARMA_COMM_WORLD, &_work1 );	CHKERRQ(ierr);
		ierr = VecSetSizes( _work1, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = VecSetType( _work1, "block" );	CHKERRQ(ierr);
		
		ierr = VecCreate( KARMA_COMM_WORLD, &_work2 );	CHKERRQ(ierr);
		ierr = VecSetSizes( _work2, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = VecSetType( _work2, "block" );	CHKERRQ(ierr);
		

		KARMA_LOG(logDEBUG1) << "creating wrappers for vectors now." ;
		Solution = new KARMABlockVector(nphysics) ;
		Residual = new KARMABlockVector (nphysics) ;
		Work_1 = new KARMABlockVector (nphysics) ;
		Work_2 = new KARMABlockVector (nphysics) ;
		
		KARMA_LOG(logDEBUG1) << "creating block matrices now: jacobian" ;
		ierr = MatCreate(KARMA_COMM_WORLD, &_jacobian) ;	CHKERRQ(ierr);
		ierr = MatSetSizes( _jacobian, nphysics, nphysics, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = MatSetType( _jacobian, "block" );	CHKERRQ(ierr);
		
		KARMA_LOG(logDEBUG1) << "creating block matrices now: preconditioner" ;
		ierr = MatCreate(KARMA_COMM_WORLD, &_preconditioner) ;	CHKERRQ(ierr);
		ierr = MatSetSizes( _preconditioner, nphysics, nphysics, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = MatSetType( _preconditioner, "block" );	CHKERRQ(ierr);
		
		// // need to delay the initialization for the block matrices
		// if (_jacobian_mfree)
		// {
			// // this is incorrect. we need the dofs of each physics. need some alternate way
			// // or actually create a block matrix with linear operators at 
			// ierr = MatCreateShell(KARMA_COMM_WORLD,nphysics,nphysics,nphysics,nphysics,0, &_jacobian);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			// ierr = MatSetFromOptions(_jacobian) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		// }
		// else
		// {
			// KARMA_LOG(logDEBUG1) << "creating block matrices now: jacobian" ;
			// ierr = MatCreate(KARMA_COMM_WORLD, &_jacobian) ;	CHKERRQ(ierr);
			// ierr = MatSetSizes( _jacobian, nphysics, nphysics, nphysics, nphysics );	CHKERRQ(ierr);
			// ierr = MatSetType( _jacobian, "block" );	CHKERRQ(ierr);
		// }
		
		// if (_preconditioner_mfree)
		// {
			// ierr = MatCreateShell(KARMA_COMM_WORLD,nphysics,nphysics,nphysics,nphysics,0, &_preconditioner);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			// ierr = MatSetFromOptions(_preconditioner) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		// } 
		// else
		// {
			// if (_jacobian_mfree)
			// {
				// // SNES_MF_OPERATOR
				// KARMA_LOG(logDEBUG1) << "creating block matrices now: preconditioner" ;
				// ierr = MatCreate(KARMA_COMM_WORLD, &_preconditioner) ;	CHKERRQ(ierr);
				// ierr = MatSetSizes( _preconditioner, nphysics, nphysics, nphysics, nphysics );	CHKERRQ(ierr);
				// ierr = MatSetType( _preconditioner, "block" );	CHKERRQ(ierr);
				
			// }
			// else
			// {
				// // Re-use Jacobian's matrix ; No need to recreate it again
				// _preconditioner = _jacobian ;
			// }
		// }
		
		_initialized = true ;
	}
	
	PetscFunctionReturn(0) ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAContext::clear"
PetscErrorCode KARMAContext::reinit (int nphysics)
{
	PetscErrorCode 	ierr;
	
	if (_initialized)
		this->clear() ;
	
	if (!_initialized)
	{
		KARMA_LOG(logDEBUG1) << "creating block vectors now." ;
		
		ierr = VecCreate( KARMA_COMM_WORLD, &_x );	CHKERRQ(ierr);
		ierr = VecSetSizes( _x, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = VecSetType( _x, "block" );	CHKERRQ(ierr);
		
		KARMA_LOG(logDEBUG1) << "duplicating block vectors now." ;
		ierr = VecCreate( KARMA_COMM_WORLD, &_res );	CHKERRQ(ierr);
		ierr = VecSetSizes( _res, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = VecSetType( _res, "block" );	CHKERRQ(ierr);
		
		ierr = VecCreate( KARMA_COMM_WORLD, &_work1 );	CHKERRQ(ierr);
		ierr = VecSetSizes( _work1, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = VecSetType( _work1, "block" );	CHKERRQ(ierr);
		
		ierr = VecCreate( KARMA_COMM_WORLD, &_work2 );	CHKERRQ(ierr);
		ierr = VecSetSizes( _work2, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = VecSetType( _work2, "block" );	CHKERRQ(ierr);
		

		KARMA_LOG(logDEBUG1) << "creating wrappers for vectors now." ;
		Solution = new KARMABlockVector(nphysics) ;
		Residual = new KARMABlockVector (nphysics) ;
		Work_1 = new KARMABlockVector (nphysics) ;
		Work_2 = new KARMABlockVector (nphysics) ;
		
		KARMA_LOG(logDEBUG1) << "creating block matrices now: jacobian" ;
		ierr = MatCreate(KARMA_COMM_WORLD, &_jacobian) ;	CHKERRQ(ierr);
		ierr = MatSetSizes( _jacobian, nphysics, nphysics, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = MatSetType( _jacobian, "block" );	CHKERRQ(ierr);
		
		KARMA_LOG(logDEBUG1) << "creating block matrices now: preconditioner" ;
		ierr = MatCreate(KARMA_COMM_WORLD, &_preconditioner) ;	CHKERRQ(ierr);
		ierr = MatSetSizes( _preconditioner, nphysics, nphysics, nphysics, nphysics );	CHKERRQ(ierr);
		ierr = MatSetType( _preconditioner, "block" );	CHKERRQ(ierr);
		
		_initialized = true ;
	}
	
	PetscFunctionReturn(0) ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAContext::clear"
PetscErrorCode KARMAContext::clear ()
{
	PetscErrorCode 	ierr;
	
	if (_initialized)
	{	
		// delete the vectors that are wrappers
		delete Residual ;
		delete Solution ;

		delete Work_1 ;
		delete Work_2 ;
	
		ierr = MatDestroy(_jacobian) ;	CHKERRQ(ierr);	_jacobian = PETSC_NULL ;
		
		if(_jacobian_mfree || _preconditioner_mfree)
		{
			ierr = MatDestroy(_preconditioner) ;	CHKERRQ(ierr);
		}
		_preconditioner = PETSC_NULL ;
		
		ierr = MatDestroy(_jacobian) ;	CHKERRQ(ierr);	_jacobian = PETSC_NULL ;
		ierr = MatDestroy(_preconditioner) ;	CHKERRQ(ierr);	_preconditioner = PETSC_NULL ;
		
		ierr = VecDestroy(_x) ;	CHKERRQ(ierr);		
		ierr = VecDestroy(_res) ;	CHKERRQ(ierr);		
		ierr = VecDestroy(_work1) ;	CHKERRQ(ierr);
		ierr = VecDestroy(_work2) ;	CHKERRQ(ierr);
		
		_initialized = false ;
	
	}
	
	PetscFunctionReturn(0) ;
}

#endif
