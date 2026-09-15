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

#ifndef USE_PETSC_BLOCK_EXTENSIONS

#include "karma.h"
#include "karma_interface.h"
#include "karma_wrappers.h"

#include "karma_linear_operations.h"
#include "karma_nonlinear_operations.h"

#include "karma_utils.h"
#include "karma_monitor.h"

#include "isystem.h"

#include "utility.h"
#include "petsc_vec_utils.h"

// Define the DofMap, which handles degree of freedom indexing.
#include "dof_map.h"

#include "karma_vector.h"
#include "karma_matrix.h"
#include "karma_linear_solver.h"
#include "karma_physics_base.h"


#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::init"	
PetscErrorCode KARMAInterface::init ()
{
	PetscErrorCode 	ierr=0;
	
	if (interface_instantiated && !_initialized)
	{
		PetscBool flg ;
		
		// create a new context and store in _CTX object
		ierr = create_SNES_object() ;CHKERRQ(ierr);
		
		std::vector<unsigned int> nnz, noz ;
		_gdofs = 0 ;	_ldofs = 0 ;
		
		// initialize all the physics objects
		{
			const const_physics_iterator end = _physics.end();
			
			// first initialize all physics
			for (const_physics_iterator pos = _physics.begin(); pos != end; ++pos)
			{
				KARMAPhysicsBase* physics = pos->second ; 
				
				physics->init() ;
				
				_gdofs += physics->get_current_system().n_dofs() ;
				_ldofs += physics->get_current_system().n_local_dofs() ;
				
				_dof_stride.insert(dof_pair(physics->id(), _gdofs)) ;
				_ldof_stride.insert(dof_pair(physics->id(), _ldofs)) ;
			}
	
			nnz.resize(_ldofs) ;	noz.resize(_ldofs) ;
			unsigned int index = 0 ;
			// now initialize the block vectors and matrices correctly
			for (const_physics_iterator pos = _physics.begin(); pos != end; ++pos)
			{
				KARMAPhysicsBase* physics = pos->second ;
			
				// Now that all physics components are inited, let us update the 
				// coupled physics contexts.
				physics->init_coupled_physics() ;

				// Now update the global n_nz and n_oz vectors
				// const DofMap& dofmap = physics->get_dof_map() ;
				
				const std::vector<unsigned int>& phy_nnz = physics->get_n_nz() ;
				const std::vector<unsigned int>& phy_noz = physics->get_n_oz() ;
				
				// KARMA_LOG(logINFO) << "Physics: " <<  physics->system_type() << "\tDOFS=" << pos->second->get_current_system().n_dofs() 
				//			<< "\tNNZ length = " << phy_nnz.size() << "\tNOZ length = " << phy_noz.size() ;
				
				// check if we reserved enough memory.
				karma_assert(phy_nnz.size()+index <= _ldof_stride[physics->id()]) ;
				
				// copy the nnz and noz vectors to the global array
				for (unsigned int ist=0; ist < phy_nnz.size(); ist++)
				{
					nnz[index+ist] = phy_nnz[ist] ;
					noz[index+ist] = phy_noz[ist] ;
				}
				
				// KARMA_LOG(logINFO) << "Inteface: Nonzeros structure" ;
				// for (unsigned int i=0; i < phy_nnz.size(); i++)
					// KARMA_LOG(logINFO) << "Inteface: Nonzeros[" << i << "] = " << phy_nnz[i] << "\t" << phy_noz[i] ;
				
				// update the offset index
				index += phy_nnz.size() ;

				// // in case our mesh has hanging node or other constraints, lets
				// // make sure dofmap captures that.
				// for (unsigned int isys=0; isys != pos->second->n_systems(); isys++)
				// {
					// System& sys = pos->second->get_system(isys) ;
					// DofMap& dofmap = sys.get_dof_map() ;
					// dofmap.create_dof_constraints(sys.get_mesh()) ;
				// }
			}
		}
		
		// initialize the global context
		_CTX->init (_ldofs, _gdofs, nnz, noz) ;
	
		if (_CTX->is_jacobian_matrix_free ())
		{
			ierr = MatShellSetOperation(_CTX->_jacobian, MATOP_MULT,(void(*)())KARMA::karma_wrapper_petsc_shell_jacobian_matvec);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatShellSetOperation(_CTX->_jacobian, MATOP_SOLVE,(void(*)())KARMA::karma_wrapper_petsc_shell_jacobian_solve);CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}

		if (_CTX->is_preconditioner_matrix_free ())
		{
			ierr = MatShellSetOperation(_CTX->_preconditioner, MATOP_MULT,(void(*)())KARMA::karma_wrapper_petsc_shell_preconditioner_matvec);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatShellSetOperation(_CTX->_preconditioner, MATOP_SOLVE,(void(*)())KARMA::karma_wrapper_petsc_shell_preconditioner_solve);CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
		
		
		// TODO: Find the structure of all solution vectors and distribute a big - coupled residual vector accordingly;
		// Length of this vector = total cumulative unknowns in all physics
		// Might also have to store an Index set to help out on this
		
		ierr = SNESSetFunction(_snes, _CTX->_res, KARMA::karma_wrapper_temporal_residual, PETSC_NULL);CHKERRQ(ierr);
		
		// check to see if user has set jac_fd to be used. this creates the finite differenced
		// versison of the jacobian and can be very expensive. Use this only for debug purposes.
		ierr = PetscOptionsHasName(PETSC_NULL,"-jac_fd", &flg);CHKERRQ(ierr);
		
		if (flg)
		{
			MatFDColoring  fdcoloring;
			ISColoring     iscoloring;

			/*
				Create the data structure that SNESDefaultComputeJacobianColor() uses
				to compute the actual Jacobians via finite differences.
			*/
			MatFDColoringCreate(_CTX->_preconditioner, iscoloring, &fdcoloring);
			MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))KARMA::karma_wrapper_temporal_residual, PETSC_NULL);
			MatFDColoringSetFromOptions(fdcoloring);
			/*
				 Tell SNES to use the routine SNESDefaultComputeJacobianColor()
				to compute Jacobians.
			*/
			SNESSetJacobian(_snes, _CTX->_jacobian, _CTX->_preconditioner, SNESDefaultComputeJacobianColor, fdcoloring);
			ISColoringDestroy(&iscoloring);
		}
		else
		{
			// if (_CTX->is_jacobian_matrix_free () && _CTX->is_preconditioner_matrix_free () && false)
			// {
				// // if jac and prec matrix are shell, then call the dummy wrapper 
				// ierr = SNESSetJacobian(_snes, _CTX->_jacobian, _CTX->_preconditioner, KARMA::karma_wrapper_temporal_jacobian_dummy, PETSC_NULL); CHKERRQ(ierr);
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
		ierr = KSPSetPCSide(_ksp, PC_RIGHT) ;
		
		// if user wants to optimize the initial guess by solving preconditioner first, set knoll to true.
		ierr = PetscOptionsHasName(PETSC_NULL,"-knoll", &flg);CHKERRQ(ierr);
		if (flg)
		{
			ierr = KSPSetInitialGuessKnoll( _ksp, PETSC_TRUE );CHKERRQ(ierr);
		}
		
		// set "main" as the prefix for all the nonlinear, linear solver options
		ierr = SNESAppendOptionsPrefix(_snes, "main_") ;
		
		// set the preconditioner type based on the context
		if (KARMA::num_processors > 1 && !(_CTX->is_jacobian_matrix_free () && _CTX->is_preconditioner_matrix_free ()))
		{
			// In parallel, let the default preconditioner be HYPRE
			ierr = PCSetType( _pc, PCHYPRE); CHKERRQ(ierr);
			ierr = PCHYPRESetType(_pc, "boomeramg"); CHKERRQ(ierr);
			// use a maximum of 5 iterations
			ierr = PetscOptionsSetValue("-main_pc_hypre_boomeramg_max_iter","5"); CHKERRQ(ierr);
			// use a fixed number of iterations
			ierr = PetscOptionsSetValue("-main_pc_hypre_boomeramg_rtol","0"); CHKERRQ(ierr);
			// use sor for relaxing up and down during MG cycles
			// ierr = PetscOptionsSetValue("-main_pc_hypre_boomeramg_relax_type_all","sor"); CHKERRQ(ierr);
		}
		else
		{
			// In serial mode, let the default preconditioner be ILU
			ierr = PCSetType( _pc, PCILU ); CHKERRQ(ierr);
		}
		
		// create and set the type of our block-pc based on our own rules.
		// the user can override these by specifying options from command-line.
		{
			char pc_type[10] ;
			bool use_bjacobi = true ;
			
			ierr = PetscOptionsGetString(PETSC_NULL,"-main_pc_type", pc_type, 10, &flg);CHKERRQ(ierr);
			if (flg)
			{
				if (!strcmp(pc_type, "asm"))
					use_bjacobi = false ;
				else if (!strcmp(pc_type, "bjacobi"))
					use_bjacobi = true ;
				else
					flg = PETSC_FALSE ;
			}
			if (flg)
			{
				// add the sub-pc type to options db
				ierr = PetscOptionsSetValue("-main_sub_ksp_type","richardson"); CHKERRQ(ierr);
				ierr = PetscOptionsSetValue("-main_sub_pc_type","ilu"); CHKERRQ(ierr);
				
				// if (use_bjacobi)
				// {
					// ierr = PCSetType( _pc, PCBJACOBI); CHKERRQ(ierr);
				// }
				// else
				// {
					// ierr = PCSetType( _pc, PCASM); CHKERRQ(ierr);
				// }

				// // Call KSPSetUp() to set the block Jacobi data structures (including
				// // creation of an internal KSP context for each block).
				// ierr = KSPSetUp(_ksp) ; CHKERRQ(ierr);
				
				// PetscInt nlocal,first ;
				
				// KSP* sub_ksp ;
				// // Extract the array of KSP contexts for the local blocks
				// if (use_bjacobi)
				// {
					// ierr = PCBJacobiGetSubKSP(_pc,&nlocal,&first,&sub_ksp); CHKERRQ(ierr);
				// }
				// else
				// {
					// ierr = PCASMGetSubKSP(_pc,&nlocal,&first,&sub_ksp); CHKERRQ(ierr);
				// }
				
				// // Loop over the local blocks, setting various KSP options
				// // for each block.
				// for (int i=0; i<nlocal; i++)
				// {
					// PC sub_pc ;
					// ierr = KSPGetPC(sub_ksp[i],&sub_pc);CHKERRQ(ierr);
					
					// ierr = KSPSetType(sub_ksp[i],KSPGMRES);CHKERRQ(ierr);
					// ierr = PCSetType(sub_pc,PCILU);CHKERRQ(ierr);
					// ierr = KSPSetTolerances(sub_ksp[i],1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
					// // ierr = KSPSetFromOptions(sub_ksp[i]) ;CHKERRQ(ierr);
				// }
				
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
		PetscBool flg ;
		
		_gdofs = 0 ;	_ldofs = 0 ;
		std::vector<unsigned int> nnz, noz ;
		_dof_stride.clear() ;
		_ldof_stride.clear() ;

		// initialize all the physics objects
		{
			const_physics_iterator       pos = _physics.begin();
			const const_physics_iterator end = _physics.end();
	
			for (; pos != end; ++pos)
			{
				pos->second->reinit() ;
				_gdofs += pos->second->get_current_system().n_dofs() ;
				_ldofs += pos->second->get_current_system().n_local_dofs() ;

				_dof_stride.insert(dof_pair(pos->second->id(), _gdofs)) ;
				_ldof_stride.insert(dof_pair(pos->second->id(), _ldofs)) ;
				
			}
		}

		// initialize the global context
		_CTX->reinit (_ldofs, _gdofs, nnz, noz) ;
		
		if (_CTX->is_jacobian_matrix_free ())
		{
			ierr = MatShellSetOperation(_CTX->_jacobian, MATOP_MULT,(void(*)())KARMA::karma_wrapper_petsc_shell_jacobian_matvec);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatShellSetOperation(_CTX->_jacobian, MATOP_SOLVE,(void(*)())KARMA::karma_wrapper_petsc_shell_jacobian_solve);CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}

		if (_CTX->is_preconditioner_matrix_free ())
		{
			ierr = MatShellSetOperation(_CTX->_preconditioner, MATOP_MULT,(void(*)())KARMA::karma_wrapper_petsc_shell_preconditioner_matvec);CHKERRABORT(KARMA_COMM_WORLD,ierr);
			ierr = MatShellSetOperation(_CTX->_preconditioner, MATOP_SOLVE,(void(*)())KARMA::karma_wrapper_petsc_shell_preconditioner_solve);CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
		
		
		// TODO: Find the structure of all solution vectors and distribute a big - coupled residual vector accordingly;
		// Length of this vector = total cumulative unknowns in all physics
		// Might also have to store an Index set to help out on this
		ierr = SNESSetFunction(_snes, _CTX->_res, KARMA::karma_wrapper_temporal_residual, PETSC_NULL);CHKERRQ(ierr);
		
		// check to see if user has set jac_fd to be used. this creates the finite differenced
		// versison of the jacobian and can be very expensive. Use this only for debug purposes.
		ierr = PetscOptionsHasName(PETSC_NULL,"-jac_fd", &flg);CHKERRQ(ierr);
		if (flg)
		{
			MatFDColoring  fdcoloring;
			ISColoring     iscoloring;

			/*
				Create the data structure that SNESDefaultComputeJacobianColor() uses
				to compute the actual Jacobians via finite differences.
			*/
			MatFDColoringCreate(_CTX->_preconditioner, iscoloring, &fdcoloring);
			MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))KARMA::karma_wrapper_temporal_residual, PETSC_NULL);
			MatFDColoringSetFromOptions(fdcoloring);
			/*
				 Tell SNES to use the routine SNESDefaultComputeJacobianColor()
				to compute Jacobians.
			*/
			SNESSetJacobian(_snes, _CTX->_jacobian, _CTX->_preconditioner, SNESDefaultComputeJacobianColor, fdcoloring);
			ISColoringDestroy(&iscoloring);
		}
		else
		{
			if (_CTX->is_jacobian_matrix_free () && _CTX->is_preconditioner_matrix_free () && false)
			{
				// if jac and prec matrix are shell, then call the dummy wrapper
				ierr = SNESSetJacobian(_snes, _CTX->_jacobian, _CTX->_preconditioner, KARMA::karma_wrapper_temporal_jacobian_dummy, PETSC_NULL); CHKERRQ(ierr);
			}
			else
			{
				ierr = SNESSetJacobian(_snes, _CTX->_jacobian, _CTX->_preconditioner, KARMA::karma_wrapper_temporal_jacobian, PETSC_NULL); CHKERRQ(ierr);
			}
		}
		
		_initialized = true ;
	}
	
	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::localize"
template <typename T>
void localize (KARMARealVector& v_global, KARMARealVector& v_local)
{
	karma_assert (v_local.local_size() == v_global.size());

	int ierr = 0;
	const int n = v_global.size();

	IS is;
	VecScatter scatter;

	// Create idx, idx[i] = i;
	std::vector<int> idx(n); Utility::iota (idx.begin(), idx.end(), 0);

	// Create the index set & scatter object
	ierr = ISCreateGeneral(KARMA_COMM_WORLD, n, &idx[0], PETSC_USE_POINTER, &is);
		 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = VecScatterCreate(v_global.vec(), is,
			  v_local.vec(), is,
			  &scatter);
		 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Perform the scatter
	// API argument order change in PETSc 2.3.3
	ierr = VecScatterBegin(scatter, v_global.vec(), v_local.vec(),
			 INSERT_VALUES, SCATTER_FORWARD);
		 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = VecScatterEnd  (scatter, v_global.vec(), v_local.vec(),
						 INSERT_VALUES, SCATTER_FORWARD);
		 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Clean up
	ierr = ISDestroy (&is);	 CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = VecScatterDestroy(&scatter);	 CHKERRABORT(KARMA_COMM_WORLD,ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::create_subvector"
void create_subvector(KARMAVector* parentvector, KARMAVector* petsc_subvector,
				      const std::vector<unsigned int>& rows,
					  unsigned int startindex=0)
{
	// PETSc data structures
	IS parent_is, subvector_is;
	VecScatter scatter;
	int ierr = 0;

	// Make sure the passed int subvector is really a PetscVector
	karma_assert(petsc_subvector != NULL);

	Vec pvector = parentvector->vec() ;
	Vec subvector = petsc_subvector->vec() ;

	// If the petsc_subvector is already initialized, we assume that the
	// user has already allocated the *correct* amount of space for it.
	// If not, we use the appropriate PETSc routines to initialize it.
	if (!petsc_subvector->initialized())
	{
		// Call the init function on our petsc_subvector
		petsc_subvector->init(rows.size(), PETSC_DECIDE, true) ;
	}
  
	// Use iota to fill an array with entries [0,1,2,3,4,...rows.size()]
	std::vector<int> idx(rows.size());
	Utility::iota (idx.begin(), idx.end(), startindex);

	// Construct index sets
	ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 rows.size(),
			 (int*) &rows[0],
			 PETSC_USE_POINTER,
			 &parent_is); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 rows.size(),
			 (int*) &idx[0],
			 PETSC_USE_POINTER,
			 &subvector_is); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Construct the scatter object
	ierr = VecScatterCreate(pvector,
			  parent_is,
			  subvector,
			  subvector_is,
			  &scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Actually perform the scatter
#if PETSC_VERSION_LESS_THAN(2,3,3)
	ierr = VecScatterBegin(pvector,
			 subvector,
			 INSERT_VALUES,
			 SCATTER_FORWARD,
			 scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = VecScatterEnd(pvector,
			   subvector,
			   INSERT_VALUES,
			   SCATTER_FORWARD,
			   scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr);
#else
	// API argument order change in PETSc 2.3.3
	ierr = VecScatterBegin(scatter,
			 pvector,
			 subvector,
			 INSERT_VALUES,
			 SCATTER_FORWARD); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = VecScatterEnd(scatter,
			   pvector,
			   subvector,
			   INSERT_VALUES,
			   SCATTER_FORWARD); CHKERRABORT(KARMA_COMM_WORLD,ierr);
#endif
  
	// Clean up 
	ierr = ISDestroy(&parent_is);       CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = ISDestroy(&subvector_is);    CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = VecScatterDestroy(&scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr); 

}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::fill_parent"
void fill_parent(KARMAVector* parentvector, KARMAVector* petsc_subvector,
					  unsigned int startindex=0)
{
	// PETSc data structures
	IS parent_is, subvector_is;
	VecScatter scatter;
	int ierr = 0;

	// Make sure the passed int subvector is really a PetscVector
	karma_assert(petsc_subvector != NULL);

	Vec pvector = parentvector->vec() ;
	Vec subvector = petsc_subvector->vec() ;
	unsigned int subvector_size = petsc_subvector->size() ;

	// If the petsc_subvector is already initialized, we assume that the
	// user has already allocated the *correct* amount of space for it.
	// If not, we use the appropriate PETSc routines to initialize it.
	if (!petsc_subvector->initialized())
	{
		// Call the init function on our petsc_subvector
		petsc_subvector->init(subvector_size, PETSC_DECIDE, true) ;
	}

	// Use iota to fill an array with entries [0,1,2,3,4,...subvector_size]
	std::vector<int> idx(subvector_size), rows(subvector_size);
	Utility::iota (idx.begin(), idx.end(), startindex);
	Utility::iota (rows.begin(), rows.end(), 0);

	// Construct index sets
	ierr = ISCreateGeneral(KARMA_COMM_WORLD,
		subvector_size,
		(int*) &idx[0],
		 PETSC_USE_POINTER,
		&parent_is); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = ISCreateGeneral(KARMA_COMM_WORLD,
		subvector_size,
		(int*) &rows[0],
		 PETSC_USE_POINTER,
		&subvector_is); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Construct the scatter object
	ierr = VecScatterCreate(	subvector,
		subvector_is,
		pvector,
		parent_is,
		&scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Actually perform the scatter
#if PETSC_VERSION_LESS_THAN(2,3,3)
	ierr = VecScatterBegin(subvector,
		pvector,
		INSERT_VALUES,
		SCATTER_FORWARD,
		scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = VecScatterEnd(subvector,
		pvector,
		INSERT_VALUES,
		SCATTER_FORWARD,
		scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr);
#else
	// API argument order change in PETSc 2.3.3
	ierr = VecScatterBegin(scatter,
		subvector,
		pvector,
		INSERT_VALUES,
		SCATTER_FORWARD); CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = VecScatterEnd(scatter,
		subvector,
		pvector,
		INSERT_VALUES,
		SCATTER_FORWARD); CHKERRABORT(KARMA_COMM_WORLD,ierr);
#endif

	// Clean up 
	ierr = ISDestroy(&parent_is);       CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = ISDestroy(&subvector_is);    CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = VecScatterDestroy(&scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr); 
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_physics_temporary_vectors"
PetscErrorCode KARMAInterface::synchronize_physics_temporary_vectors (const KARMAVector& X)
{
	KARMA_LOG_PUSH("synchronize_physics_temporary_vectors", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	int physics_dof_start = 0 ;
	
	std::vector<unsigned int> rows(0) ;
	
	// Loop over each physics and synchronize the temporary vectors based on given global vector
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics temporary vector
		KARMAVector& sol = pos->second->get_temporary_vector() ;
		
		rows.resize(sol.size()) ;
		Utility::iota (rows.begin(), rows.end(), physics_dof_start);
		
		X.create_subvector(sol, rows) ;

		Parallel::barrier() ;

		sol.close() ;
	
		physics_dof_start += sol.size() ;
	}
	
	rows.clear() ;	

	KARMA_LOG_POP("synchronize_physics_temporary_vectors", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_physics_solution(PetscVector)"
PetscErrorCode KARMAInterface::synchronize_physics_solution (const KARMAVector& X)
{
	KARMA_LOG_PUSH("synchronize_physics_solution", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	int physics_dof_start = 0 ;

	std::vector<unsigned int> rows(0) ;

	// Loop over each physics and synchronize the temporary vectors based on given global vector
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics solution vector
		KARMAVector& sol = pos->second->get_solution() ;
		
		rows.resize(sol.size()) ;
		Utility::iota (rows.begin(), rows.end(), physics_dof_start);
		X.create_subvector(sol, rows) ;

		Parallel::barrier() ;

		sol.close() ;
		
		pos->second->update() ;
		
		physics_dof_start += sol.size() ;
	}
	
	rows.clear() ;
	
	KARMA_LOG_POP("synchronize_physics_solution", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_interface_solution(PetscVector)"
PetscErrorCode KARMAInterface::synchronize_interface_solution (KARMAVector& X)
{
	KARMA_LOG_PUSH("synchronize_interface_solution", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	int physics_dof_start = 0 ;

	std::vector<unsigned int> rows(0) ;

	// Loop over each physics and synchronize the interface vector with physics solution vectors
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics solution 
		KARMAVector& sol = pos->second->get_solution() ;
		
		rows.resize(sol.local_size()) ;
		Utility::iota (rows.begin(), rows.end(), physics_dof_start+sol.first_local_index());
		
		fill_parent(&X, &sol, physics_dof_start) ;
		
		Parallel::barrier() ;

		physics_dof_start += sol.size() ;
	}

	X.close() ;
	
	rows.clear() ;
	
	KARMA_LOG_POP("synchronize_interface_solution", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_physics_residual(PetscVector)"
PetscErrorCode KARMAInterface::synchronize_physics_residual (const KARMAVector& R)
{
	KARMA_LOG_PUSH("synchronize_physics_residual", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	int physics_dof_start = 0 ;

	std::vector<unsigned int> rows(0) ;
	
	// Loop over each physics and synchronize the residual vectors based on given global vector
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics residual vector
		KARMAVector& sol = pos->second->get_residual() ;
		
		rows.resize(sol.size()) ;
		Utility::iota (rows.begin(), rows.end(), physics_dof_start);
		
		R.create_subvector(sol, rows) ;

		Parallel::barrier() ;

		sol.close() ;

		physics_dof_start += sol.size() ;
	}
	
	rows.clear() ;

	KARMA_LOG_POP("synchronize_physics_residual", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_interface_residual(PetscVector)"
PetscErrorCode KARMAInterface::synchronize_interface_residual (KARMAVector& R)
{
	KARMA_LOG_PUSH("synchronize_interface_residual", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	int physics_dof_start = 0 ;

	std::vector<unsigned int> rows(0) ;

	// Loop over each physics and synchronize the interface vector with physics residual vectors
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics residual vector 
		KARMAVector& sol = pos->second->get_residual() ;
		
		rows.resize(sol.local_size()) ;
		Utility::iota (rows.begin(), rows.end(), physics_dof_start+sol.first_local_index());
		
		fill_parent(&R, &sol, physics_dof_start) ;
		
		Parallel::barrier() ;

		physics_dof_start += sol.size() ;
		
	}

	R.close() ;
	
	rows.clear() ;
	
	KARMA_LOG_POP("synchronize_interface_residual", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_interface_jacobian"
PetscErrorCode KARMAInterface::synchronize_interface_jacobian (unsigned int physics_index, KARMAPhysicsBase* physics, KARMAMatrix& J)
{
	KARMA_LOG_PUSH("synchronize_interface_jacobian", "Interface");

	PetscErrorCode ierr ;

	KARMAMatrix& phyjac = physics->get_jacobian () ;
	
	PetscInt ncols;
	const PetscInt* cols;
	const PetscScalar *vals ;
	
	for (unsigned int irows=phyjac.row_start(); irows < phyjac.row_stop(); irows++)
	{
		// First, get the matrix row entries for the physics. 
		ierr = MatGetRow(phyjac.mat(), irows, &ncols, &cols, &vals) ;CHKERRQ(ierr);
		
		std::vector<int> idx(ncols);
		for (int icols=0; icols < ncols; icols++)
			idx[icols] = physics_index+cols[icols] ;
		
		int rownum = physics_index+irows ;
		ierr = MatSetValues(J.mat(), 1, &rownum, ncols, &idx[0], &vals[0], INSERT_VALUES) ;CHKERRQ(ierr);
		
		ierr = MatRestoreRow(phyjac.mat(), irows, &ncols, &cols, &vals) ;CHKERRQ(ierr);
	}

	KARMA_LOG_POP("synchronize_interface_jacobian", "Interface");

	PetscFunctionReturn(ierr);
}



// #undef __FUNCT__
// #define __FUNCT__ "KARMAInterface::jacobian_solve"
// PetscErrorCode KARMAInterface::jacobian_solve (const KARMAVector& pX, KARMAVector& pY)
// {
	// // this routine is not yet complete
	// // specialize between SS and temporal jacobians
	// karma_error() ;

	// KARMA_LOG_PUSH("jacobian_solve", "Interface");

	// const double& dtime = _CTX->time ;
	// const_physics_iterator       pos = _physics.begin();
	// const const_physics_iterator end = _physics.end();

	// PetscErrorCode ierr ;

	// _params.clear() ;

	// // this->enforce_constraints(pX) ;

	// this->synchronize_physics_temporary_vectors(pX) ;

	// // Interface should store the dof_indicies of any given physics in a global vector;
	// // that way before calling the residual function, it could project solution from other physics and pass onto current physics.
	// for (; pos != end; ++pos)
	// {
		// // Get a reference to the physics system we are solving
		// KARMAPhysicsBase* physics = pos->second ;
		// KARMASystem& implicit_system = physics->get_current_system() ;

		// const std::vector<unsigned int>& send_list = implicit_system.get_dof_map().get_send_list ();

		// KARMAVector& phy_temp = physics->get_temporary_vector() ;
		// KARMAVector& phy_res = physics->get_residual() ;

		// _params.set<KARMAPhysicsBase*>(PHYSICS) = physics ;
		// _params.set<Real>(TIME) = dtime ;

		// // TODO: check if there is a solve method available.
		// // 		If YES, then call it and be done.
		// //		If NO,
		// //			Check if jacobian matrix is created explicitly through a routine.
		// //				If YES, create a KSP context and sovle the sucker.
		// //				If NO, create a shell matrix and define this matvec operator, solve.
		// if (physics->jac_operator->is_solve())
		// {
			// ierr = physics->jac_operator->solve (phy_temp, phy_res, _params);CHKERRQ(ierr);
		// }
		// else
		// {
			// // bool matcreated = true ;
			// Mat _jac ;
			// // KSP _jacksp ;
			// // PC _jacpc ;
			// int locdofs = phy_res.local_size() ;
			// int numdofs = phy_res.size() ;

			// if (physics->jac_operator->is_matrix())
			// {
				// KARMAVector& phy_sol = physics->get_local_solution() ;
				// physics->get_solution().localize(phy_sol, send_list) ;

				// ierr = MatCreateMPIAIJ(KARMA_COMM_WORLD,locdofs,locdofs,numdofs,numdofs,1,NULL,0,NULL, &_jac);CHKERRQ(ierr);
				// ierr = MatSetFromOptions(_jac) ;CHKERRQ(ierr);

				// KARMAMatrix prec_mat (_jac) ;

				// physics->jac_operator->matrix(phy_sol, prec_mat, _params) ;
			// }
		// }
	// }

	// this->synchronize_interface_residual(pY) ;

	// KARMA_LOG_POP("jacobian_solve", "Interface");

	// PetscFunctionReturn(ierr);
// }


// #undef __FUNCT__
// #define __FUNCT__ "KARMAInterface::preconditioner_solve"
// PetscErrorCode KARMAInterface::preconditioner_solve (const KARMAVector& pX, KARMAVector& pY )
// {
	// KARMA_LOG_PUSH("preconditioner_solve", "Interface");

	// // this routine is not yet complete
	// // specialize between SS and temporal jacobians
	
	// PetscErrorCode ierr = 0 ;

	// const double dtime = _CTX->time ;
	// const unsigned int prec_id = _CTX->get_preconditioner_id () ;

	// // first find the SS preconditioner
	// if (this->is_transient())
	// {
		// // Transient Preconditioner Solve: pY = inv( M - delt * aii * Prec )  * pX

		// const_physics_iterator       pos = _physics.begin();
		// const const_physics_iterator end = _physics.end();

		// _params.clear() ;

		// this->synchronize_physics_temporary_vectors(pX) ;

		// for (; pos != end; ++pos)
		// {
			// // Get a reference to the physics system we are solving
			// KARMAPhysicsBase* physics = pos->second ;
			// KARMASystem& implicit_system = physics->get_current_system() ;

			// const std::vector<unsigned int>& send_list = implicit_system.get_dof_map().get_send_list ();

			// KARMAVector& phy_temp = physics->get_temporary_vector() ;
			// KARMAVector& phy_res = physics->get_residual() ;

			// // KARMAVector& phy_sol = physics->get_local_solution() ;
			// // physics->get_solution().localize(phy_sol, send_list) ;

			// _params.set<KARMAPhysicsBase*>(PHYSICS) = physics ;
			// _params.set<Real>(TIME) = dtime ;

			// KARMALinearOperator* optr = NULL ;

			// if (prec_id == 0)
			// {
				// // use jacobian matrix as preconditioner
				// optr = physics->jac_operator ;
			// }
			// else
			// {
				// // use the user-implemented version of preconditioner
				// optr = physics->prec_operator[prec_id-1] ;
			// }

			// // KARMA_LOG(logINFO) << "Preconditioner: IsSolve=" << optr->is_solve() << ", IsMatVec=" << optr->is_matvec() << ", IsMatrix=" << optr->is_matrix() ;

			// // TODO: Check if preconditioner matrix is created explicitly through a routine.
			// //			If YES, create a KSP context and sovle the sucker.
			// //			If NO, create a shell matrix and define this matvec operator, solve.
			// Mat _prec ;
			// KSP _precksp ;
			// PC _precpc ;
			// int locdofs = phy_res.local_size() ;
			// int numdofs = phy_res.size() ;

			// ierr = KSPCreate(KARMA_COMM_WORLD, &_precksp) ;CHKERRQ(ierr);
			// KSPAppendOptionsPrefix(_precksp, "prec_") ;

			// // we are just assuming that each individual block in unsymmetric
			// // if solving symmetric problem, override this option from command-line to use CG
			// ierr = KSPSetType(_precksp, KSPGMRES) ;
			// ierr = KSPGetPC(_precksp, &_precpc) ;CHKERRQ(ierr);

			// ierr = MatCreateShell(KARMA_COMM_WORLD,locdofs,locdofs,numdofs,numdofs,(void*)physics, &_prec);CHKERRQ(ierr);
			// ierr = MatSetFromOptions(_prec) ;CHKERRQ(ierr);

			// // KARMA_LOG(logINFO) << "Preconditioner: check this implementation before using." ;
			// // karma_error() ;

			// bool matcreated = true ;
			// if (optr->is_matrix())
			// {
				// KARMAMatrix& prec_mat = physics->get_jacobian()  ;

				// KARMAVector& phy_sol = physics->get_local_solution() ;
				// physics->get_solution().localize(phy_sol, send_list) ;

				// KARMA_LOG(logINFO) << "Preconditioner: Solving the assembled preconditioner matrix." ;

				// PetscBool assembled ;
				// ierr = MatAssembled(prec_mat.mat(), &assembled) ;CHKERRQ(ierr);
				// // if the matrix is unassembled, then call the matrix construction routine
				// if (assembled == PETSC_FALSE)
				// {
					// ierr = optr->matrix(phy_sol, prec_mat, _params) ;CHKERRQ(ierr);

					// TemporalIntegrator* integrator = NULL ;
					// ierr = SNESGetApplicationContext(_snes, (void **)&integrator) ;CHKERRQ(ierr);

					// double **A = integrator->Acoeff () ;

					// int currstage = integrator->current_stage() ;
					// ierr = MatAYPX(prec_mat.mat(), -_CTX->delt*A[currstage][currstage], physics->get_mass_matrix()->mat(), SUBSET_NONZERO_PATTERN) ;
					// prec_mat.close() ;
				// }

				// ierr = MatDuplicate(prec_mat.mat(), MAT_COPY_VALUES, &_prec); CHKERRQ(ierr);

				// // ierr = MatGetRowSum(physics->get_mass_matrix()->mat(), phy_work.vec()) ;CHKERRQ(ierr);
				// // ierr = MatDiagonalSet(_prec, phy_work.vec(), ADD_VALUES) ;CHKERRQ(ierr);
				// ierr = MatAssemblyBegin(_prec, MAT_FINAL_ASSEMBLY);
				// ierr = MatAssemblyEnd(_prec, MAT_FINAL_ASSEMBLY);

				// // a matrix is available = use ICC as default
				// ierr = PCSetType(_precpc, PCLU) ;CHKERRQ(ierr);

			// }
			// else if (optr->is_matvec())
			// {
				// matopr_function *precvec_fn = NULL ;
				// const KARMA::KARMALinearOperations opmatvec = KARMA::MatVec ;
				// optr->get_raw_function_pointer(opmatvec, precvec_fn) ;

				// // Check whether the preconditioner's matrix-vector operation is defined
				// // If not, this is a badly defined LinearOperator and the solve cannot be performed.
				// karma_assert(precvec_fn != NULL);

				// KARMA_LOG(logINFO) << "Preconditioner: Solving by repeatedly applying matvec for preconditioner operator." ;
				// KARMA_LOG(logWARNING) << "Preconditioner: Matrix solve by matvec will work, but chances are that it is computationally expensive." ;

				// // Set the matvec operation for the preconditioenr
				// ierr = MatShellSetOperation(_prec, MATOP_MULT,(void(*)())(precvec_fn));CHKERRQ(ierr);

				// // no matrix = no preconditioner for this solve
				// ierr = PCSetType(_precpc, PCNONE) ;

			// }
			// else
			// {
				// PetscPrintf(KARMA_COMM_WORLD, "\nNeither solve, matrix or matvec handles are provided.\n");
				// PetscPrintf(KARMA_COMM_WORLD, "\nPreconditioner solve is impossible for Physics = %s", physics->system_type().c_str()) ;

				// karma_error() ;
			// }

			// // set the matrix and preconditioner operators
			// ierr = KSPSetOperators(_precksp, _prec, _prec, SAME_NONZERO_PATTERN) ;CHKERRQ(ierr);

			// // allow user to control the options from command line: use prefix = "-prec_"
			// ierr = KSPSetFromOptions(_precksp) ;

			// ierr = KSPSolve(_precksp, phy_temp.vec(), phy_res.vec()) ;CHKERRQ(ierr);

			// ierr = KSPDestroy(_precksp) ;CHKERRQ(ierr);
			// ierr = MatDestroy(_prec) ;CHKERRQ(ierr);

			// _prec = PETSC_NULL ;

		// }

		// this->synchronize_interface_residual(pY) ;

	// }
	// else
	// {
		// // SS Preconditioner Solve: pY = inv( Prec )  * pX

		// const_physics_iterator       pos = _physics.begin();
		// const const_physics_iterator end = _physics.end();

		// _params.clear() ;

		// this->synchronize_physics_temporary_vectors(pX) ;

		// for (; pos != end; ++pos)
		// {
			// // Get a reference to the physics system we are solving
			// KARMAPhysicsBase* physics = pos->second ;
			// KARMASystem& implicit_system = physics->get_current_system() ;

			// const std::vector<unsigned int>& send_list = implicit_system.get_dof_map().get_send_list ();

			// KARMAVector& phy_temp = physics->get_temporary_vector() ;
			// KARMAVector& phy_res = physics->get_residual() ;

			// _params.set<KARMAPhysicsBase*>(PHYSICS) = physics ;
			// _params.set<Real>(TIME) = dtime ;

			// KARMALinearOperator* optr = NULL ;

			// if (prec_id == 0)
			// {
				// // use jacobian matrix as preconditioner
				// optr = dynamic_cast<KARMALinearOperator*>(physics->jac_operator) ;
			// }
			// else
			// {
				// // use the user-implemented version of preconditioner
				// optr = physics->prec_operator[prec_id-1] ;
			// }

			// //  KARMA_LOG(logINFO) << "Preconditioner: IsSolve=" << optr->is_solve() << ", IsMatVec=" << optr->is_matvec() << ", IsMatrix=" << optr->is_matrix() ;

			// // TODO: check if there is a solve method available.
			// // 		If YES, then call it and be done.
			// //		If NO,
			// //			Check if jacobian matrix is created explicitly through a routine.
			// //				If YES, create a KSP context and sovle the sucker.
			// //				If NO, create a shell matrix and define this matvec operator, solve.
			// if (optr->is_solve())
			// {
				// // if (KARMA::verbose)
					// // PetscPrintf(KARMA_COMM_WORLD, "\nUser provided preconditioner : Directly calling is_solve for preconditioner operator.") ;

				// // The preconditioner, if linearized, is about the prescribed solution passed here.
				// // Typically, this would be the solution after last newton iteration update
				// ierr = optr->solve (phy_temp, phy_res, _params) ;CHKERRQ(ierr);
			// }
			// else
			// {
				// bool matcreated = true ;
				// Mat _prec ;
				// KSP _precksp ;
				// PC _precpc ;
				// int locdofs = phy_res.local_size() ;
				// int numdofs = phy_res.size() ;

				// ierr = KSPCreate(KARMA_COMM_WORLD, &_precksp) ;CHKERRQ(ierr);
				// KSPAppendOptionsPrefix(_precksp, "prec_") ;

				// // we are just assuming that each individual block in unsymmetric
				// // if solving symmetric problem, override this option from command-line to use CG
				// ierr = KSPSetType(_precksp, KSPGMRES) ;
				// ierr = KSPGetPC(_precksp, &_precpc) ;CHKERRQ(ierr);

				// if (optr->is_matrix())
				// {
					// // ierr = MatCreateMPIAIJ(KARMA_COMM_WORLD,locdofs,locdofs,numdofs,numdofs,1,NULL,0,NULL, &_prec);CHKERRQ(ierr);
					// // ierr = MatSetFromOptions(_prec) ;CHKERRQ(ierr);

					// KARMAMatrix& prec_mat = physics->get_jacobian()  ;

					// KARMAVector& phy_sol = physics->get_local_solution() ;
					// physics->get_solution().localize(phy_sol, send_list) ;

					// // KARMA_LOG(logINFO) << "Preconditioner: Solving the assembled preconditioner matrix." ;

					// PetscBool assembled ;
					// ierr = MatAssembled(prec_mat.mat(), &assembled) ;CHKERRQ(ierr);
					// // if the matrix is unassembled, then call the matrix construction routine
					// if (assembled == PETSC_FALSE)
					// {
						// ierr = optr->matrix(phy_sol, prec_mat, _params) ;CHKERRQ(ierr);
					// }

					// // a matrix is available = use ICC as default
					// ierr = PCSetType(_precpc, PCILU) ;CHKERRQ(ierr);

					// _prec = prec_mat.mat() ;
					// matcreated = false ;
					
				// }
				// else if (optr->is_matvec())
				// {
					// ierr = MatCreateShell(KARMA_COMM_WORLD,locdofs,locdofs,numdofs,numdofs,(void*)(&_params), &_prec);CHKERRQ(ierr);
					// ierr = MatSetFromOptions(_prec) ;CHKERRQ(ierr);

// // 					matopr_function *precvec_fn = NULL ;
// // 					const KARMA::KARMALinearOperations opmatvec = KARMA::MatVec ;
// // 					optr->get_raw_function_pointer(opmatvec, precvec_fn) ;

					// // Check whether the preconditioner's matrix-vector operation is defined
					// // If not, this is a badly defined LinearOperator and the solve cannot be performed.
// // 					karma_assert(precvec_fn != NULL);
					
					// _params.set<KARMALinearOperator*>("operator") = optr ;
					// _params.set<KARMA::KARMALinearOperations>("operation") = KARMA::MatVec ;
					
					// // ierr = MatShellSetContext(_prec, (void*)(&_params)) ;CHKERRQ(ierr);

					// // KARMA_LOG(logINFO) << "Preconditioner: Solving by repeatedly applying matvec for preconditioner operator. " << (precvec_fn==NULL) ;
					// // KARMA_LOG(logWARNING) << "Preconditioner: Matrix solve by matvec will work, but chances are that it is computationally expensive." ;

					// // Set the matvec operation for the preconditioenr
					// // ierr = MatShellSetOperation(_prec, MATOP_MULT,(void(*)())(precvec_fn));CHKERRQ(ierr);
					// ierr = MatShellSetOperation(_prec, MATOP_MULT,(void(*)())(KARMA::karma_wrapper_petsc_shell_linear_operations));CHKERRQ(ierr);
					
					// // ierr = MatSolve(_prec, phy_temp.vec(), phy_res.vec()) ;CHKERRQ(ierr);

					// // no matrix = no preconditioner for this solve
					// ierr = PCSetType(_precpc, PCNONE) ;

				// }
				// else
				// {
					// PetscPrintf(KARMA_COMM_WORLD, "\nNeither solve, matrix or matvec handles are provided.\n");
					// PetscPrintf(KARMA_COMM_WORLD, "\nPreconditioner solve is impossible for Physics = %s", physics->system_type().c_str()) ;

					// karma_error() ;
				// }
				
				// // set the matrix and preconditioner operators
				// ierr = KSPSetOperators(_precksp, _prec, _prec, SAME_NONZERO_PATTERN) ;CHKERRQ(ierr);

				// // allow user to control the options from command line: use prefix = "-prec_"
				// ierr = KSPSetFromOptions(_precksp) ;

				// ierr = KSPSolve(_precksp, phy_temp.vec(), phy_res.vec()) ;CHKERRQ(ierr);

				// ierr = KSPDestroy(_precksp) ;CHKERRQ(ierr);

				// if (matcreated)
				// {
					// ierr = MatDestroy(_prec) ;CHKERRQ(ierr);
				// }

				// _prec = PETSC_NULL ;
			// }
		// }

		// this->synchronize_interface_residual(pY) ;

	// }

	// KARMA_LOG_POP("preconditioner_solve", "Interface");

	// PetscFunctionReturn(ierr);
// }


#endif
