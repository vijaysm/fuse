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


#include "karma_context.h"

#ifdef USE_PETSC_BLOCK_EXTENSIONS

#include "karma.h"
#include "karma_interface.h"
#include "karma_wrappers.h"
#include "karma_vector.h"
#include "karma_matrix.h"

#include "karma_block_vector.h"
#include "karma_block_matrix.h"
#include "karma_linear_solver.h"
#include "karma_preconditioner.h"

#include "dof_map.h"
#include "karma_physics_base.h"
#include "karma_monitor.h"

#include "petscext.h"
#include "petscext_vec.h"
#include "petscext_mat.h"
#include "petscext_ksp.h"
#include "petscext_pc.h"
#include "petscext_snes.h"


#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::init"	
PetscErrorCode KARMAInterface::init ()
{
	PetscErrorCode 	ierr=0;
	
	if (interface_instantiated && !_initialized)
	{
		PetscTruth flg ;
		
		// create a new context and store in _CTX object
		ierr = create_SNES_object() ;CHKERRQ(ierr);
		
		// initialize the global context
		_CTX->init (_physics.size()) ;
		
		// initialize all the physics objects
		{
			const const_physics_iterator end = _physics.end();
			
			unsigned int physics_index = 0 ;
			// first initialize all physics
			for (const_physics_iterator pos = _physics.begin(); pos != end; ++pos)
			{
				KARMAPhysicsBase* physics = pos->second ; 
				
				physics->init() ;
				
				KARMAVector& phy_soln = physics->get_solution() ;
				KARMAVector& phy_res = physics->get_residual() ;
				KARMAVector& phy_temp1 = physics->get_temporary_vector() ;
				KARMAVector& phy_temp2 = physics->get_old_solution() ;
				
				ierr = VecBlockSetValue( _CTX->_x, physics_index, phy_soln.vec(), INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
				ierr = VecBlockSetValue( _CTX->_res, physics_index, phy_res.vec(), INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
				ierr = VecBlockSetValue( _CTX->_work1, physics_index, phy_temp1.vec(), INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
				ierr = VecBlockSetValue( _CTX->_work2, physics_index, phy_temp2.vec(), INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
				
				physics_index++ ;
			}

			physics_index = 0 ;
			// now initialize the block vectors and matrices correctly
			for (const_physics_iterator pos = _physics.begin(); pos != end; ++pos)
			{
				KARMAPhysicsBase* physics = pos->second ; 
				
				// Now that all physics components are inited, let us update the 
				// coupled physics contexts.
				physics->init_coupled_physics() ;
				
				unsigned int jphysics_index = 0 ;
				for (const_physics_iterator iter_pos = _physics.begin(); iter_pos != end; ++iter_pos)
				{
					KARMAMatrix* phy_jac = NULL ;
					if (physics_index == jphysics_index)
						phy_jac = &physics->get_jacobian() ;
					else
					{
						phy_jac = physics->get_coupled_physics_matrix(iter_pos->second->system_type()) ;
					}
					
					if (phy_jac != NULL)
					{
						ierr = MatBlockSetValue( _CTX->_jacobian, physics_index, jphysics_index, phy_jac->mat(), DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
						ierr = MatBlockSetValue( _CTX->_preconditioner, physics_index, jphysics_index, phy_jac->mat(), DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
					}
					jphysics_index++ ;
				}
				
				physics_index++ ;
			}
			
			/* Assemble */
			ierr = VecAssemblyBegin( _CTX->_x );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyEnd( _CTX->_x );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyBegin( _CTX->_res );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyEnd( _CTX->_res );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyBegin( _CTX->_work1 );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyEnd( _CTX->_work1 );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyBegin( _CTX->_work2 );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyEnd( _CTX->_work2 );	CHKERRABORT(KARMA_COMM_WORLD,ierr);

			ierr = MatAssemblyBegin( _CTX->_jacobian, MAT_FINAL_ASSEMBLY );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatAssemblyEnd( _CTX->_jacobian, MAT_FINAL_ASSEMBLY );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatAssemblyBegin( _CTX->_preconditioner, MAT_FINAL_ASSEMBLY );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatAssemblyEnd( _CTX->_preconditioner, MAT_FINAL_ASSEMBLY );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
		
		// set the wrapper residual function pointer in SNES context
		ierr = SNESSetFunction(_snes, _CTX->_res, KARMA::karma_wrapper_temporal_residual, PETSC_NULL);CHKERRQ(ierr);

		// check to see if user has set jac_fd to be used. this creates the finite differenced
		// versison of the jacobian and can be very expensive. Use this only for debug purposes.
		ierr = PetscOptionsHasName(PETSC_NULL,"-jac_fd", &flg);CHKERRQ(ierr);
		
		if (flg)
		{
			MatFDColoring  fdcoloring;
			ISColoring     iscoloring;
			
			// Create the data structure that SNESDefaultComputeJacobianColor() uses
			// to compute the actual Jacobians via finite differences.
			MatFDColoringCreate(_CTX->_jacobian, iscoloring, &fdcoloring);
			MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))KARMA::karma_wrapper_temporal_residual, PETSC_NULL);
			MatFDColoringSetFromOptions(fdcoloring);
			
			// Tell SNES to use the routine SNESDefaultComputeJacobianColor()
			// to compute Jacobians.
			SNESSetJacobian(_snes, _CTX->_jacobian, _CTX->_preconditioner, SNESDefaultComputeJacobianColor, fdcoloring);
			ISColoringDestroy(iscoloring);
		}
		else
		{
			// if (_CTX->is_jacobian_matrix_free () && _CTX->is_preconditioner_matrix_free ())
			// {
				// // if jac and prec matrix are shell, then just set Jacobian to the dummy wrapper
				// // NOTE: Not sure if this really helps. Can probably remove this in the future and instead
				// // check for shell matrix in the wrapper and return without doing anything if true
				// ierr = SNESSetJacobian(_snes, _CTX->_jacobian, _CTX->_jacobian, KARMA::karma_wrapper_temporal_jacobian_dummy, PETSC_NULL); CHKERRQ(ierr);
			// }
			// else
			// {
				ierr = SNESSetJacobian(_snes, _CTX->_jacobian, _CTX->_preconditioner, KARMA::karma_wrapper_temporal_jacobian, PETSC_NULL); CHKERRQ(ierr);
			// }
		}
		
		// by default, just set the custom KARMA monitor for the nonlinear solver
		ierr = SNESMonitorSet(_snes, KARMA::KARMASNESMonitor, 0, 0);CHKERRQ(ierr);

		// let us use Inexact Newton method by default; this is the Eisenstat-Walker scheme
		// where the linear tolerance changes based on nonlinear residual norm
		ierr = SNESKSPSetUseEW(_snes, _CTX->_ksp_ew) ;CHKERRQ(ierr);
		
		// now since there are going to be krylov solvers used for each physics, use
		// the Flexible GMREs method as the default KSP for coupled physics context
		ierr = KSPSetType(_ksp, KSPFGMRES) ;CHKERRQ(ierr);
		ierr = KSPGMRESSetRestart(_ksp, 50) ;CHKERRQ(ierr);
		ierr = KSPGMRESSetOrthogonalization(_ksp, KSPGMRESModifiedGramSchmidtOrthogonalization) ;CHKERRQ(ierr);
		
		// set the right preconditioning by default, unless user wants left from command-line
		ierr = KSPSetPreconditionerSide(_ksp, PC_RIGHT) ;
		
		// if user wants to optimize the initial guess by solving preconditioner first, set knoll to true.
		ierr = PetscOptionsHasName(PETSC_NULL,"-knoll", &flg);CHKERRQ(ierr);
		if (flg)
		{
			ierr = KSPSetInitialGuessKnoll( _ksp, PETSC_TRUE );CHKERRQ(ierr);
		}
		
		// set "main" as the prefix for all the nonlinear, linear solver options
		ierr = SNESAppendOptionsPrefix(_snes, "main_") ;

		// create and set the type of our block-pc 
		{
			ierr = PCSetOperators( _pc, _CTX->_jacobian, _CTX->_jacobian, SAME_PRECONDITIONER );
			ierr = PCSetType( _pc, "block" ); CHKERRQ(ierr);
			
			const_physics_iterator       pos = _physics.begin();
			const const_physics_iterator end = _physics.end();
			
			unsigned int physics_index = 0 ;
			PetscInt nlocal,first ;
	
			/* Configure sub ksp's on the block pc */
			for (; pos != end; ++pos)
			{
				KARMAPhysicsBase* physics = pos->second ;
				
				// We only have block-diagonal matrix now.
				
				// Q11 will be an iterative solve
				KSP sub_ksp ;
				ierr = PCBlockGetSubKSP( _pc, physics_index, &sub_ksp ); CHKERRQ(ierr);
				PC sub_pc ;
				ierr = KSPGetPC( sub_ksp, &sub_pc ); CHKERRQ(ierr);
				const std::string physics_name = physics->system_type() ;
				std::stringstream sstr ;
				sstr << physics_name << "_" ;
				ierr = KSPSetOptionsPrefix(sub_ksp, sstr.str().c_str()) ; CHKERRQ(ierr);
				
				if (!physics_name.compare("DiffusionReactionSystem") || 
					!physics_name.compare("ThermalConductionSystem"))
				{
					ierr = KSPSetType( sub_ksp, KSPCG); CHKERRQ(ierr);
					if (KARMA::num_procs() > 1)
					{
						// ierr = PCSetType( sub_pc, PCHYPRE); CHKERRQ(ierr);
						// ierr = PCHYPRESetType( sub_pc, "pilut" ); CHKERRQ(ierr);
						
						ierr = PCSetType( sub_pc, PCBJACOBI); CHKERRQ(ierr);

						KSP* sub_subksp ;
						// Extract the array of KSP contexts for the local blocks
						ierr = PCBJacobiGetSubKSP(sub_pc,&nlocal,&first,&sub_subksp); CHKERRQ(ierr);
						
						/*
						   Loop over the local blocks, setting various KSP options
						   for each block.  
						*/
						for (unsigned int i=first; i<first+nlocal; i++)
						{
							PC sub_subpc ;
							ierr = KSPGetPC(sub_subksp[i],&sub_subpc);CHKERRQ(ierr);
							
							ierr = PCSetType(sub_subpc,PCICC);CHKERRQ(ierr);
							ierr = KSPSetType(sub_subksp[i],KSPCG);CHKERRQ(ierr);
							ierr = KSPSetTolerances(sub_subksp[i],1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
							ierr = KSPSetFromOptions(sub_subksp[i]) ;CHKERRQ(ierr);
						}
						
					}
					else
					{
						ierr = PCSetType( sub_pc, PCICC ); CHKERRQ(ierr);
					}
				}
				else
				{
					ierr = KSPSetType( sub_ksp, KSPGMRES); CHKERRQ(ierr);
					
					if (!physics_name.compare("AdvectionReactionSystem") || 
							!physics_name.compare("CoolantFlow_1D_System"))
					{
						// this is a 1-D system. We do not expect the size of system to be big.
						// so use block-jacobi overall and use LU for individual processors.
						if (KARMA::num_procs() > 1)
						{
							// ierr = PCSetType( sub_pc, PCHYPRE); CHKERRQ(ierr);
							// ierr = PCHYPRESetType( sub_pc, "euclid" ); CHKERRQ(ierr);
							
							ierr = PCSetType( sub_pc, PCBJACOBI); CHKERRQ(ierr);

							KSP* sub_subksp ;
							// Extract the array of KSP contexts for the local blocks
							ierr = PCBJacobiGetSubKSP(sub_pc,&nlocal,&first,&sub_subksp); CHKERRQ(ierr);
							
							/*
							   Loop over the local blocks, setting various KSP options
							   for each block.  
							*/
							for (unsigned int i=0; i<nlocal; i++)
							{
								PC sub_subpc ;
								ierr = KSPGetPC(sub_subksp[i],&sub_subpc);CHKERRQ(ierr);
								
								ierr = PCSetType(sub_subpc, PCLU);CHKERRQ(ierr);
								ierr = KSPSetType(sub_subksp[i], KSPRICHARDSON);CHKERRQ(ierr);
								ierr = KSPSetTolerances(sub_subksp[i],1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
							}
						}
						else
						{
							ierr = PCSetType( sub_pc, PCLU ); CHKERRQ(ierr);
						}
					}
					else
					{
						if (KARMA::num_procs() > 1)
						{
							// ierr = PCSetType( sub_pc, PCHYPRE); CHKERRQ(ierr);
							// ierr = PCHYPRESetType( sub_pc, "pilut" ); CHKERRQ(ierr);
							
							ierr = PCSetType( sub_pc, PCBJACOBI); CHKERRQ(ierr);

							KSP* sub_subksp ;
							// Extract the array of KSP contexts for the local blocks
							ierr = PCBJacobiGetSubKSP(sub_pc,&nlocal,&first,&sub_subksp); CHKERRQ(ierr);
							
							/*
							   Loop over the local blocks, setting various KSP options
							   for each block.  
							*/
							for (unsigned int i=0; i<nlocal; i++)
							{
								PC sub_subpc ;
								ierr = KSPGetPC(sub_subksp[i],&sub_subpc);CHKERRQ(ierr);
								
								ierr = PCSetType(sub_subpc, PCILU);CHKERRQ(ierr);
								ierr = KSPSetType(sub_subksp[i], KSPGMRES);CHKERRQ(ierr);
								ierr = KSPSetTolerances(sub_subksp[i],1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
							}
						}
						else
						{
							ierr = PCSetType( sub_pc, PCILU ); CHKERRQ(ierr);
						}
					}
					
				}
				
				KSPSetFromOptions( sub_ksp );

				physics_index++ ;
			}
		}
		
		// override all options by user-specified args from command-line
		ierr = SNESSetFromOptions(_snes); CHKERRQ(ierr);
		
		ierr = PetscOptionsHasName(PETSC_NULL,"-main_snes_ls", &flg);CHKERRQ(ierr);
		if (!flg)
		{
			// use the basic newton method unless user specifies otherwise from command line
			ierr = SNESLineSearchSet(_snes, SNESLineSearchNo, PETSC_NULL) ;CHKERRQ(ierr);
		}
	
		// TODO: since each physics can have more than one preconditioner, use this option in future
		// to specify the index of the preconditioner to be used as the default
		// For example: if physics has linearized jacobian, block jacobian, point jacobi then
		// -user_precond 2 will use point jacobi as preconditioner with -snes_mf_operator !!
		// check if user wants to use custom preconditioning
		ierr = PetscOptionsHasName(PETSC_NULL,"-user_precond", &flg);CHKERRQ(ierr);
		
		if (flg)
		{
			ierr = PCSetType(_pc, PCSHELL) ;
			PCShellSetContext(_pc, (void*)this) ;
			PCShellSetApply(_pc, (PetscErrorCode (*)(PC,Vec,Vec))KARMA::karma_wrapper_petsc_shell_preconditioner_solve) ;
		}
		
		// get the default parameters used for convergence testing
		ierr = SNESGetTolerances(_snes, &_abstol, &_rtol, &_stol, &_maxit, &_maxf);CHKERRQ(ierr);
		// this information can be printed with the option -snes_view
		
		_params.set<Real> ("time") = 0.0 ;
		
		_initialized = true ;
	}
	
	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::reinit"	
PetscErrorCode KARMAInterface::reinit ()
{
	PetscErrorCode 	ierr=0;
	
	if (interface_instantiated)
	{
		PetscTruth flg ;
		
		// create a new context and store in _CTX object
		ierr = create_SNES_object() ;CHKERRQ(ierr);
		
		// initialize the global context
		_CTX->reinit (_physics.size()) ;
		
		unsigned int physics_index = 0 ;
		// initialize all the physics objects
		{
			const_physics_iterator       pos = _physics.begin();
			const const_physics_iterator end = _physics.end();
			
			// first re-initialize all physics
			for (; pos != end; ++pos)
			{
				KARMAPhysicsBase* physics = pos->second ; 
				
				physics->reinit() ;

				KARMAVector& phy_soln = physics->get_solution() ;
				KARMAVector& phy_res = physics->get_residual() ;
				KARMAVector& phy_temp1 = physics->get_temporary_vector() ;
				KARMAVector& phy_temp2 = physics->get_old_solution() ;
				
				ierr = VecBlockSetValue( _CTX->_x, physics_index, phy_soln.vec(), INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
				ierr = VecBlockSetValue( _CTX->_res, physics_index, phy_res.vec(), INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
				ierr = VecBlockSetValue( _CTX->_work1, physics_index, phy_temp1.vec(), INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
				ierr = VecBlockSetValue( _CTX->_work2, physics_index, phy_temp2.vec(), INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
				
				unsigned int jphysics_index = 0 ;
				for (const_physics_iterator iter_pos = _physics.begin(); iter_pos != end; ++iter_pos)
				{
					KARMAMatrix* phy_jac = NULL ;
					if (physics_index == jphysics_index)
						phy_jac = &physics->get_jacobian() ;
					else
					{
						phy_jac = physics->get_coupled_physics_matrix(iter_pos->second->system_type()) ;
					}
					
					if (phy_jac != NULL)
					{
						phy_jac->close() ;
						ierr = MatBlockSetValue( _CTX->_jacobian, physics_index, jphysics_index, phy_jac->mat(), DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
						ierr = MatBlockSetValue( _CTX->_preconditioner, physics_index, jphysics_index, phy_jac->mat(), DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
					}
					jphysics_index++ ;
				}
				
				physics_index++ ;
			}
			
			/* Assemble */
			ierr = VecAssemblyBegin( _CTX->_x );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyEnd( _CTX->_x );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyBegin( _CTX->_res );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyEnd( _CTX->_res );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyBegin( _CTX->_work1 );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = VecAssemblyEnd( _CTX->_work1 );	CHKERRABORT(KARMA_COMM_WORLD,ierr);

			ierr = MatAssemblyBegin( _CTX->_jacobian, MAT_FINAL_ASSEMBLY );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatAssemblyEnd( _CTX->_jacobian, MAT_FINAL_ASSEMBLY );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatAssemblyBegin( _CTX->_preconditioner, MAT_FINAL_ASSEMBLY );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatAssemblyEnd( _CTX->_preconditioner, MAT_FINAL_ASSEMBLY );	CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
	
	}
	
	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_physics_temporary_vectors"
PetscErrorCode KARMAInterface::synchronize_physics_temporary_vectors (const KARMABlockVector& XX)
{
	KARMA_LOG_PUSH("synchronize_physics_temporary_vectors", "Interface");
	
	PetscFunctionReturn(0);
	
	PetscErrorCode ierr = 0 ;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	unsigned int physics_index = 0 ;
	
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAVector& sol = pos->second->get_temporary_vector() ;
		
		// get the reference to the vector corresponding to this physics
		const KARMAVector* block_ref = XX.get_block_vector(physics_index) ;	CHKERRQ(ierr) ;
		
		ierr = VecCopy(block_ref->vec(), sol.vec()) ;	CHKERRQ(ierr) ;
		
		sol.close() ;

		physics_index++ ;
	}

	KARMA_LOG_POP("synchronize_physics_temporary_vectors", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_physics_solution(PetscVector)"
PetscErrorCode KARMAInterface::synchronize_physics_solution (const KARMABlockVector& X)
{
	// Take the current solution from X and copy to 
	// each of the system's corresponding solution vector
	
	KARMA_LOG_PUSH("synchronize_physics_solution", "Interface");
	
	PetscFunctionReturn(0);
	
	PetscErrorCode ierr = 0 ;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	unsigned int physics_index = 0 ;
	
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAVector& sol = pos->second->get_solution() ;
		
		// get the reference to the vector corresponding to this physics
		const KARMAVector* block_ref = X.get_block_vector(physics_index) ;	CHKERRQ(ierr) ;
		
		ierr = VecCopy(block_ref->vec(), sol.vec()) ;	CHKERRQ(ierr) ;
		
		sol.close() ;

		physics_index++ ;
	}

	KARMA_LOG_POP("synchronize_physics_solution", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_interface_solution(PetscVector)"
PetscErrorCode KARMAInterface::synchronize_interface_solution (KARMABlockVector& X)
{
	// Take the current solution from each physics and copy to 
	// solution vector X supplied as a parameter.
	
	KARMA_LOG_PUSH("synchronize_interface_solution", "Interface");

	PetscErrorCode ierr = 0 ;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	unsigned int physics_index = 0 ;
	
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAVector& sol = pos->second->get_solution() ;
		
		// get the reference to the vector corresponding to this physics
		KARMAVector* block_ref = X.get_block_vector(physics_index) ;	CHKERRQ(ierr) ;
		
		ierr = VecCopy(sol.vec(), block_ref->vec()) ;	CHKERRQ(ierr) ;

		physics_index++ ;
	}

	X.close() ;
	
	KARMA_LOG_POP("synchronize_interface_solution", "Interface");

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_physics_residual(PetscVector)"
PetscErrorCode KARMAInterface::synchronize_physics_residual (const KARMABlockVector& R)
{
	KARMA_LOG_PUSH("synchronize_physics_residual", "Interface");

	PetscErrorCode ierr = 0 ;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	unsigned int physics_index = 0 ;
	
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAVector& sol = pos->second->get_residual() ;
		
		// get the reference to the vector corresponding to this physics
		const KARMAVector* block_ref = R.get_block_vector(physics_index) ;	CHKERRQ(ierr) ;
		
		ierr = VecCopy(block_ref->vec(), sol.vec()) ;	CHKERRQ(ierr) ;
		
		sol.close() ;

		physics_index++ ;
	}

	KARMA_LOG_POP("synchronize_physics_residual", "Interface");

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_interface_residual(PetscVector)"
PetscErrorCode KARMAInterface::synchronize_interface_residual (KARMABlockVector& R)
{
	KARMA_LOG_PUSH("synchronize_interface_residual", "Interface");

	PetscErrorCode ierr = 0 ;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	unsigned int physics_index = 0 ;
	
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAVector& sol = pos->second->get_residual() ;
		
		// get the reference to the vector corresponding to this physics
		KARMAVector* block_ref = R.get_block_vector(physics_index) ;	CHKERRQ(ierr) ;
		
		ierr = VecCopy(sol.vec(), block_ref->vec()) ;	CHKERRQ(ierr) ;

		physics_index++ ;
	}

	R.close() ;
	
	KARMA_LOG_POP("synchronize_interface_residual", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_interface_jacobian"
PetscErrorCode KARMAInterface::synchronize_interface_jacobian (unsigned int physics_index, KARMAPhysicsBase* physics, KARMABlockMatrix& J)
{
	KARMA_LOG_PUSH("synchronize_interface_jacobian", "Interface");

	// In the block version, let us just set the block matrix to the value of the 
	// Jacobian matrix from the given system
	J.set_block_matrix(physics_index, physics_index, &physics->get_jacobian()) ;
	// ierr = MatBlockSetValue( P.mat(), physics_index, physics_index, phy_jac.mat(), DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );	CHKERRQ(ierr) ;

	KARMA_LOG_POP("synchronize_interface_jacobian", "Interface");

	PetscFunctionReturn(0);
}

#endif
