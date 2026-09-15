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


#include "petscsnes.h"

#include "karma.h"
#include "karma_vector.h"
#include "karma_matrix.h"
#include "isystem.h"
#include "karma_context.h"
#include "karma_interface.h"
#include "karma_physics_base.h"
#include "karma_linear_solver.h"
#include "karma_preconditioner.h"
#include "karma_utils.h"

#include "karma_linear_operations.h"
#include "karma_nonlinear_operations.h"


#include "utility.h"
#include "petsc_vec_utils.h"

// #ifdef USE_PETSC_BLOCK_EXTENSIONS
// #include "petscext.h"
// #include "petscext_vec.h"
// #include "petscext_mat.h"
// #include "petscext_ksp.h"
// #include "petscext_pc.h"
// #include "petscext_snes.h"
// #endif

// Define the DofMap, which handles degree of freedom indexing.
#include "dof_map.h"

#define PHYSICS "physics"

// initialize our static private variables so that singleton pattern executes seamlessly !

// first set the private reference to NULL.
KARMAInterface* KARMAInterface::__global_KARMA_interface = NULL ;
// set the instantiated to false
bool KARMAInterface::interface_instantiated = false ;

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::get_interface"
KARMAInterface* KARMAInterface::get_interface()
{
	// if interface has not been instantiated yet, then
	if(! interface_instantiated)
	{
		// create the Interface singleton object and store the reference
		__global_KARMA_interface = new KARMAInterface();

		interface_instantiated = true;
	}

	// return the singleton object reference
	return __global_KARMA_interface;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::get_context"
KARMAContext* KARMAInterface::get_context ()
{
	// return the run context for our singleton interface
	return (KARMAInterface::get_interface()->_CTX) ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::constructor"
KARMAInterface::KARMAInterface()
	:	_maxit(500), _maxf(5000),
		_abstol(1e-10), _rtol(1e-7), _stol(1e-12)
{
	// create the run context
	_CTX = new KARMAContext () ;

	// if using petsc block extensions, register all the objects.
// #ifdef USE_PETSC_BLOCK_EXTENSIONS
	// // register all the extensions to Petsc
	// PetscExtVecRegisterAll();
	// PetscExtMatRegisterAll();
	// PetscExtKSPRegisterAll();
	// PetscExtPCRegisterAll();
	// PetscExtSNESRegisterAll();
// #endif

	_initialized = false ;
	_prepared_for_transient = false ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::create_SNES_object"
PetscErrorCode KARMAInterface::create_SNES_object()
{
	PetscErrorCode 	ierr = 0;

	// Create nonlinear solver context
	ierr = SNESCreate(KARMA_COMM_WORLD, &_snes);CHKERRQ(ierr);

	ierr = SNESSetTolerances(_snes, _abstol, _rtol, _stol, _maxit, _maxf);CHKERRQ(ierr);

	ierr = SNESGetKSP(_snes, &_ksp);	CHKERRQ(ierr);

	ierr = KSPGetPC(_ksp, &_pc);	CHKERRQ(ierr);

	_CTX->set_options () ;

	// Wait to set command-line options till we init()
	// ierr = SNESSetFromOptions(_snes) ;CHKERRQ(ierr);

	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::destructor"
KARMAInterface::~KARMAInterface()
{
	this->clear () ;

	interface_instantiated = false;

	delete _CTX ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::print_info"
void KARMAInterface::print_info ()
{
	PetscPrintf(KARMA_COMM_WORLD,"\n---------------------\n") ;
	PetscPrintf(KARMA_COMM_WORLD,"Default Nonlinear-Solver tolerance information") ;
	PetscPrintf(KARMA_COMM_WORLD,"\n---------------------\n") ;
	PetscPrintf(KARMA_COMM_WORLD,"\natol=%G, rtol=%G, stol=%G, maxit=%D, maxf=%D\n", _abstol, _rtol, _stol, _maxit, _maxf);

	PetscPrintf(KARMA_COMM_WORLD,"\n-------------------\n") ;
	PetscPrintf(KARMA_COMM_WORLD,"Physics Information") ;
	PetscPrintf(KARMA_COMM_WORLD,"\n-------------------\n") ;

	// loop over all the physics objects and print their information
	{
		const_physics_iterator       pos = _physics.begin();
		const const_physics_iterator end = _physics.end();

		for (; pos != end; ++pos)
		{
			PetscPrintf(KARMA_COMM_WORLD,"\n") ;
			pos->second->print_info() ;
		}
	}

	// print the run information on context
	// _CTX->print_info () ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::prepare_for_transient"
void KARMAInterface::prepare_for_transient()
{
	KARMA_LOG_PUSH("prepare_for_transient", "Interface") ;

	// loop over all the physics objects and call prepare_for_transient for each
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	for (; pos != end; ++pos)
	{
		pos->second->prepare_for_transient() ;
	}

	_prepared_for_transient = true ;
	
	KARMA_LOG_POP("prepare_for_transient", "Interface") ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::set_shell_matrix_context"
PetscErrorCode KARMAInterface::set_shell_matrix_context(void *ctx)
{
	PetscErrorCode ierr ;

#if PETSC_VERSION_LESS_THAN(3,0,0)
	MatType mtype_jac ;
	ierr = MatGetType(_CTX->_jacobian, &mtype_jac) ;
#else
	const MatType mtype_jac ;
	ierr = MatGetType(_CTX->_jacobian, &mtype_jac) ;
#endif
	if (!strcmp(mtype_jac, "shell"))
	{
		ierr = MatShellSetContext(_CTX->_jacobian, ctx);CHKERRQ(ierr);
	}

#if PETSC_VERSION_LESS_THAN(3,0,0)
	MatType mtype_prec ;
	ierr = MatGetType(_CTX->_preconditioner, &mtype_prec) ;
#else
	const MatType mtype_prec ;
	ierr = MatGetType(_CTX->_preconditioner, &mtype_prec) ;
#endif
	if (!strcmp(mtype_prec, "shell"))
	{
		ierr = MatShellSetContext(_CTX->_preconditioner, ctx);CHKERRQ(ierr);
	}

	PetscFunctionReturn(0) ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::clear"
PetscErrorCode KARMAInterface::clear ()
{
	PetscErrorCode 	ierr;

	if (interface_instantiated && _initialized)
	{
		// clear the systems.  We must delete them since we newed them!

		_dof_stride.clear() ;
		_ldof_stride.clear() ;

		// clear all the physics objects
		{
			const_physics_iterator       pos = _physics.begin();
			const const_physics_iterator end = _physics.end();

			for (; pos != end; ++pos)
			{
				KARMAPhysicsBase* _phy = pos->second ;
				if (_phy->is_initialized())
				{
					_phy->clear() ;
					delete _phy ;
				}
				else
					_phy = NULL ;
			}

			_physics.clear ();
		}

		_CTX->clear () ;

		ierr = SNESDestroy(&_snes);	CHKERRQ(ierr);

		interface_instantiated = false ;

		_initialized = false ;
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::stage_preprocess"
void KARMAInterface::stage_preprocess ()
{
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		physics->stage_preprocess () ;
	}
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::stage_postprocess"
void KARMAInterface::stage_postprocess ()
{
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		physics->stage_postprocess () ;
	}
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::step_preprocess"
void KARMAInterface::step_preprocess ()
{
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		physics->step_preprocess () ;
	}
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::step_postprocess"
void KARMAInterface::step_postprocess ()
{
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		physics->step_postprocess () ;
	}
}


#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::form_initial_guess"
PetscErrorCode KARMAInterface::form_initial_guess ()
{
	PetscErrorCode ierr ;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		ierr = physics->form_initial_guess (physics->get_solution()) ;CHKERRQ(ierr);
		
		physics->update() ;
	}

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::form_initial_guess"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::form_initial_guess (KARMABlockVector& xsoln)
#else
PetscErrorCode KARMAInterface::form_initial_guess (KARMAVector& xsoln)
#endif
{
	PetscErrorCode ierr ;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	ierr = this->synchronize_physics_solution (xsoln) ;CHKERRQ(ierr);

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		ierr = physics->form_initial_guess(physics->get_solution()) ;CHKERRQ(ierr);
	}

	ierr = this->synchronize_interface_solution (xsoln) ;CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::compute_dynamical_timescales"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::compute_dynamical_timescales (KARMABlockVector& old_sol, KARMABlockVector& curr_sol, 
				Real delt, std::vector<Real>& dyn_time)
#else
PetscErrorCode KARMAInterface::compute_dynamical_timescales (KARMAVector& old_sol, KARMAVector& curr_sol, 
				Real delt, std::vector<Real>& dyn_time)
#endif
{
	// compute the dynamical time scales for each physics;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	PetscErrorCode ierr = 0 ;

	// Work_1 = u_{n}/u_{n+1}
	// ierr = VecPointwiseDivide(_CTX->Work_1->vec(), old_sol.vec(), curr_sol.vec()) ;	CHKERRQ(ierr);
	// Work_1 = 1.0 - u_{n}/u_{n+1}
	// ierr = VecShift(_CTX->Work_1, -1.0) ;	CHKERRQ(ierr);
	// _CTX->Work_1->close() ;

	// now synchronize the (rate of solution change) to all physics
	ierr = this->synchronize_physics_solution (curr_sol) ;CHKERRQ(ierr);
	ierr = this->synchronize_physics_temporary_vectors (old_sol) ;CHKERRQ(ierr);

	PetscInt iphy = 0 ;
	PetscReal dynmax ;

	dyn_time.resize(_physics.size()) ;

	// Now that the required data is supplied to the physics, update each physics's time scales
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		// KARMAVector& work_1 = system->get_temporary_vector() ;
		// ierr = VecNorm(work_1.vec(), NORM_INFINITY, &dynmax) ;CHKERRQ(ierr);

		// Work_1 = u_{n}/u_{n+1}
		KARMAVector& phy_oldsol = physics->get_temporary_vector() ;
		KARMAVector& phy_sol = physics->get_solution() ;

		ierr = VecMaxPointwiseDivide(phy_sol.vec(), phy_oldsol.vec(), &dynmax) ;CHKERRQ(ierr);

		// update the time scale: delt / max| (u_{n+1} - u_{n})/delt * 1/u_{n+1} |
		// dyn_time[iphy++] = delt / fabs(1.0 - dynmax) ;
		// alternately, evaluate time scale at t_{n+1/2}: delt / max| (u_{n+1} - u_{n})/delt * 1/u_{n+1/2} |
		dyn_time[iphy++] = 0.5 * delt * (1.0 + dynmax) / fabs(1.0 - dynmax) ;

		KARMA_LOG(INFO) << physics->system_type() << "\t Model 1: " << delt / fabs(1.0 - dynmax) << "\t Model 2 : " << 0.5 * delt * (1.0 + dynmax) / fabs(1.0 - dynmax) ;
	}

	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::get_current_residual"
PetscErrorCode KARMAInterface::get_current_residual ()
{
	PetscErrorCode ierr ;

	ierr = this->system_residual(*_CTX->Solution, *_CTX->Residual) ;CHKERRQ(ierr);

	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::get_current_residual"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::get_current_residual (KARMABlockVector& res)
#else
PetscErrorCode KARMAInterface::get_current_residual (KARMAVector& res)
#endif
{
	PetscErrorCode ierr ;

	ierr = this->get_current_residual() ;CHKERRQ(ierr);

	PetscBool flg ;
	ierr = VecEqual(res.vec(), _CTX->_res, &flg) ;CHKERRQ(ierr);

	// if these vectors are different, then copy the residual to given vector.
	if (!flg)
	{
		ierr = VecCopy(_CTX->_res, res.vec()) ;CHKERRQ(ierr);
		res.close() ;
	}

	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::solve_nonlinear_system"
PetscErrorCode KARMAInterface::solve_nonlinear_system ()
{
	PetscErrorCode ierr = solve_nonlinear_system(*_CTX->Solution) ;

	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::solve_nonlinear_system"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::solve_nonlinear_system (KARMABlockVector& soln)
#else
PetscErrorCode KARMAInterface::solve_nonlinear_system (KARMAVector& soln)
#endif
{
	PetscErrorCode ierr = 0 ;

	ierr = PetscPrintf(KARMA_COMM_WORLD,"\n--------------------------------------------------------------");CHKERRQ(ierr);
	ierr = PetscPrintf(KARMA_COMM_WORLD,"\n|  Solving the nonlinear problem for discrete-exact solution  |") ;
	ierr = PetscPrintf(KARMA_COMM_WORLD,"\n--------------------------------------------------------------\n\n");CHKERRQ(ierr);

	// Solve the initial S.S problem
	ierr = SNESSolve(_snes, PETSC_NULL, soln.vec()); CHKERRQ(ierr);

	this->synchronize_physics_solution() ;

	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::read_solution"
PetscErrorCode KARMAInterface::read_solution (std::string& file_prefix)
{
	KARMA_LOG_PUSH("read_solution", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	PetscPrintf(KARMA_COMM_WORLD, "\nReading solutions with Prefix = %s\n", file_prefix.c_str()) ;

	// loop over each physics and write the solution for each in its own file along with given prefix
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		physics->read (file_prefix) ;
	}

	KARMA_LOG_POP("read_solution", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::write_solution"
PetscErrorCode KARMAInterface::write_solution (std::string& file_prefix, IOPackage ioType)
{
	KARMA_LOG_PUSH("write_solution", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	PetscPrintf(KARMA_COMM_WORLD, "\nPrinting solutions with Prefix = %s\n", file_prefix.c_str()) ;

	// loop over each physics and write the solution for each in its own file along with given prefix
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

        physics->write (file_prefix, ioType) ;
	}

	KARMA_LOG_POP("write_solution", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::is_physics_converged"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
bool KARMAInterface::is_physics_converged(KARMABlockVector& res, PetscReal atol, PetscReal rtol, std::string& pname)
#else
bool KARMAInterface::is_physics_converged(KARMAVector& res, PetscReal atol, PetscReal rtol, std::string& pname)
#endif
{
	// Ask if each of the physics have their own criteria to determine if the physics is converged.
	// This is a good idea if the solution vector is not the only criteria and maybe some derived
	// variable needs to be checked for positivity, and convergence before deciding this.
	PetscErrorCode ierr = 0 ;

	bool result = true ;
	pname = "" ;

	KARMA_LOG_PUSH("is_physics_converged", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	ierr = this->synchronize_physics_residual (res) ;CHKERRQ(ierr);

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		result = (result && physics->is_physics_converged(atol, rtol)) ;

		if (!result)
		{
			pname = physics->system_type() ;
			break ;
		}
	}

	res.clear() ;

	KARMA_LOG_POP("is_physics_converged", "Interface");

	return result ;
}

// #undef __FUNCT__
// #define __FUNCT__ "KARMAInterface::predict_solution"
// PetscErrorCode KARMAInterface::predict_solution (KARMAVector& xsoln)
// {
	// // Essentially do an explicit, 2nd order explicit prediction;
	// // Very useful to minimize the nonlinear iteration by providing a good
	// // guess for a new time step calculation.
	// PetscErrorCode ierr = 0 ;

	// KARMA_LOG_PUSH("predict_solution", "Interface");

	// const_physics_iterator       pos = _physics.begin();
	// const const_physics_iterator end = _physics.end();

	// ierr = this->synchronize_physics_solution (xsoln) ;CHKERRQ(ierr);

	// for (; pos != end; ++pos)
	// {
		// // Get a reference to the physics system we are solving
		// KARMAPhysicsBase* system = pos->second ;

		// ierr = system->predict_solution() ;CHKERRQ(ierr);
	// }

	// ierr = this->synchronize_interface_solution (xsoln) ;CHKERRQ(ierr);

	// KARMA_LOG_POP("predict_solution", "Interface");

	// return ierr ;
// }


#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_physics_solution"
PetscErrorCode KARMAInterface::synchronize_physics_solution ()
{
	PetscErrorCode ierr ;

	ierr = synchronize_physics_solution (*_CTX->Solution) ;CHKERRQ(ierr);

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_interface_solution"
PetscErrorCode KARMAInterface::synchronize_interface_solution ()
{
	PetscErrorCode ierr ;

	ierr = synchronize_interface_solution (*_CTX->Solution) ;CHKERRQ(ierr);

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_physics_residual"
PetscErrorCode KARMAInterface::synchronize_physics_residual ()
{
	PetscErrorCode ierr ;

	ierr = synchronize_physics_residual (*_CTX->Residual) ;

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::synchronize_interface_residual"
PetscErrorCode KARMAInterface::synchronize_interface_residual ()
{
	PetscErrorCode ierr ;

	ierr = synchronize_interface_residual (*_CTX->Residual) ;

	PetscFunctionReturn(ierr);
}


#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::system_residual"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::system_residual (const KARMABlockVector& pX, KARMABlockVector& pR)
#else
PetscErrorCode KARMAInterface::system_residual (const KARMAVector& pX, KARMAVector& pR)
#endif
{
	KARMA_LOG_PUSH("system_residual", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	PetscErrorCode ierr ;

	_params.clear() ;
	_params.set<Real>(TIME) = _CTX->time ;
	_params.set<Real>(DELT) = _CTX->delt ;

	this->synchronize_physics_solution(pX) ;

	// loop over all the physics and compute the residuals and assemble in to interface residual
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		KARMAVector& phy_res = physics->get_residual() ;
		KARMAVector& phy_sol = physics->get_local_solution() ;

		_params.set<KARMAPhysicsBase*>(PHYSICS) = physics ;

		// call the residual function for the physics
		ierr = physics->jac_operator->residual (phy_sol, phy_res, _params);CHKERRQ(ierr);

		physics = NULL ;
	}

	this->synchronize_interface_residual(pR) ;

 	// PetscPrintf(KARMA_COMM_WORLD, "\n\nglobal residual\n\n") ;
  	//pR.print_global(std::cout) ;
  	//std::cin.get() ;

	KARMA_LOG_POP("system_residual", "Interface");

	PetscFunctionReturn(ierr);
}

// For now, implement only the block jacobi version of it; later, use the idea of perturbing
// only the other physics solution and calling the residual function to get the off-diagonal blocks.
// Same concept can be used for preconditioner matrix
#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::jacobian_matrix"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::jacobian_matrix (const KARMABlockVector& X, KARMABlockMatrix&  J)
#else
PetscErrorCode KARMAInterface::jacobian_matrix (const KARMAVector& X, KARMAMatrix&  J)
#endif
{
	KARMA_LOG_PUSH("jacobian_matrix", "Interface");

	// J.close() ;
	// PetscFunctionReturn(0);
	
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();
	
	PetscErrorCode ierr ;

	_params.clear() ;
	_params.set<Real>(TIME) = _CTX->time ;
	_params.set<Real>(DELT) = _CTX->delt ;

	this->synchronize_physics_solution(X) ;
	
	unsigned int physics_index = 0 ;

	// Interface should store the dof_indicies of any given physics in a global vector;
	// that way before calling the residual function, it could project solution from other physics and pass onto current physics.
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		physics->update() ;

		KARMAMatrix& phy_jac = physics->get_jacobian() ;
		
		KARMAVector& phy_sol = physics->get_local_solution() ;
		
		_params.set<KARMAPhysicsBase*>("physics") = physics ;
		
		ierr = physics->jac_operator->matrix (phy_sol, phy_jac, _params);CHKERRQ(ierr);
		phy_jac.close() ;
		
		// Go through the dof_map and loop through the elements and store the dof_indices
		// Use this information to selectively update the values in the current processor rows
		this->synchronize_interface_jacobian (physics_index, physics, J) ;

		physics_index++ ;
		physics_index += phy_sol.size() ;
		
	}

	J.close() ;

	KARMA_LOG_POP("jacobian_matrix", "Interface");

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::mass_solve"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::mass_solve (const KARMABlockVector& pX, KARMABlockVector& pY)
#else
PetscErrorCode KARMAInterface::mass_solve (const KARMAVector& pX, KARMAVector& pY)
#endif
{
	KARMA_LOG_PUSH("mass_solve", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	PetscErrorCode ierr = 0 ;

	_params.clear() ;
	_params.set<Real>(TIME) = _CTX->time ;

	this->synchronize_physics_temporary_vectors(pX) ;

	KARMAParameters out_params ;

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		// obtain the references to temporary_vector and residual
		KARMAVector& phy_sol = physics->get_temporary_vector() ;
		KARMAVector& phy_res = physics->get_residual() ;

		// if a pointer to the matrix-vector operation is not present:
		// by default, we will assume the Identity matrix for the mass matrix of physics system.
		// but do inform the user that there is no Solve operator.
		if (!physics->mass_operator->is_solve() && !physics->mass_operator->is_matrix())
		{
			// KARMA_LOG(logWARNING) << "Mass-Solve operator not found. Assuming Identity." ;
			ierr = VecCopy(phy_sol.vec(), phy_res.vec()) ;CHKERRQ(ierr);
			continue ;
		}

		// Parameter to use Mass-Lumping
		_params.set<bool>("is_lumped") = _CTX->is_lumped_mass_matrix() ;
		if (physics->is_mass_matrix_dirty())
		{
			// Set the reference for the physics being solved for.
			_params.set<KARMAPhysicsBase*>(PHYSICS) = physics ;

			// This parameter, when set to true will create the matrix and
			// store it in current physics, so that it can be re-used !
			// This should provide considerable CPU time gain for time-independent Mass matrices.
			_params.set<bool>("compute_matrix") = physics->is_mass_matrix_dirty() ;

			// If we called with, create_mass_matrix for mass_multiply,
			// it would be wise to just re-use that to solve the matrix !
			// This should provide considerable CPU time gain for time-independent Mass matrices.
			out_params.set<KARMARealMatrix*>("mass_matrix") = &physics->get_mass_matrix() ;

			// Let us call the actual Mass solver for the physics, which will recompute Mass and solve
			ierr = physics->mass_operator->solve (phy_sol, phy_res, _params, &out_params) ;CHKERRQ(ierr);

			// Since we computed the matrix once, we do not need to recompute this again.
			// NOTE: If mass matrix is time dependent, this needs to be true.
			physics->mass_matrix_dirty(false) ;
		}
		else
		{
			KARMARealMatrix& Mass = physics->get_mass_matrix() ;

			if (_CTX->is_lumped_mass_matrix())
			{
				ierr = MatGetRowSum(Mass.mat(), phy_res.vec()) ;CHKERRQ(ierr);
				phy_res.close() ;

				// The lumped mass matrix is a diagonal matrix stored in mY.
				// hence inverting it is just dividing the vectors pointwise.
				ierr = VecPointwiseDivide(phy_res.vec(), phy_sol.vec(), phy_res.vec()) ;CHKERRQ(ierr);
				phy_res.close() ;
			}
			else
			{
				// get the KSP context
				KARMALinearSolver* solver = physics->get_mass_solver() ;

				// solve the mass matrix
				std::pair<unsigned int, Real> result = solver->solve(Mass, Mass, phy_res, phy_sol, 1e-13, 100) ;
			}
		}
	}

	this->synchronize_interface_residual(pY) ;

	KARMA_LOG_POP("mass_solve", "Interface");

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::mass_multiply"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::mass_multiply (const KARMABlockVector& pX, KARMABlockVector& pY)
#else
PetscErrorCode KARMAInterface::mass_multiply (const KARMAVector& pX, KARMAVector& pY)
#endif
{
	KARMA_LOG_PUSH("mass_multiply", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	PetscErrorCode ierr ;

	_params.clear() ;
	_params.set<Real>(TIME) = _CTX->time ;

	this->synchronize_physics_temporary_vectors(pX) ;

	KARMAParameters out_params ;

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		// this might not work if there are multiple physics since X and Xold contains solution for all physics
		// Temporaxry since i have only one physics :
		KARMAVector& phy_sol = physics->get_temporary_vector() ;
		KARMAVector& phy_res = physics->get_residual() ;

		// a pointer to the matrix-vector operation is not present.
		// by default, we will assume the Identity matrix for the Implicit system.
		// Hence, do nothing and copy given vector to output vector
		if (!physics->mass_operator->is_matvec() && !physics->mass_operator->is_matrix())
		{
			// TODO: Just see if the mass-matrix exists and that it is non-zero.
			// Maybe the user set only a solve method on the operator ??
			// Then we can just mulitply by the vector and move on.
			// KARMA_LOG(WARNING) << "Mass-Multiply operator not found. Assuming Identity." ;
			// karma_error() ;
			ierr = VecCopy(phy_sol.vec(), phy_res.vec()) ;CHKERRQ(ierr);
			continue ;
		}

		// Parameter to use Mass-Lumping: If lumping is used for solve, also use for multiplies.
		_params.set<bool>("is_lumped") = _CTX->is_lumped_mass_matrix() ;

		// Since we computed the matrix once, we do not need to recompute this again.
		// NOTE: If mass matrix is time dependent, this needs to be true.
		if (physics->is_mass_matrix_dirty())
		{
			// Set the reference for the physics being solved for.
			_params.set<KARMAPhysicsBase*>(PHYSICS) = physics ;

			// This parameter, when set to true will create the matrix and
			// store it in current physics, so that it can be re-used !
			// This should provide considerable CPU time gain for time-independent Mass matrices.
			_params.set<bool>("compute_matrix") = physics->is_mass_matrix_dirty() ;

			// If we called with, create_mass_matrix for mass_multiply,
			// it would be wise to just re-use that to solve the matrix !
			// This should provide considerable CPU time gain for time-independent Mass matrices.
			out_params.set<KARMARealMatrix*>("mass_matrix") = &physics->get_mass_matrix() ;

			// Let us call the actual Mass solver for the physics, which will recompute Mass and multiply
			ierr = physics->mass_operator->matvec (phy_sol, phy_res, _params, &out_params);CHKERRQ(ierr);

			physics->mass_matrix_dirty(false) ;
		}
		else
		{
			if (_CTX->is_lumped_mass_matrix())
			{
				// we already have the mass matrix; just perform a mat-vec operation.
				Mat massmat = physics->get_mass_matrix().mat() ;
				ierr = MatGetRowSum(massmat, phy_res.vec()) ;CHKERRQ(ierr);
				phy_res.close() ;
				ierr = VecPointwiseMult(phy_res.vec(), phy_res.vec(), phy_sol.vec()) ;CHKERRQ(ierr);
				phy_res.vec() ;
			}
			else
			{
				// we already have the mass matrix; just perform a mat-vec operation.
				Mat massmat = physics->get_mass_matrix().mat() ;
				ierr = MatMult(massmat, phy_sol.vec(), phy_res.vec()) ;CHKERRQ(ierr);
				phy_res.close() ;
			}
		}
	}

	this->synchronize_interface_residual(pY) ;

	KARMA_LOG_POP("mass_multiply", "Interface");

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::mass_matrix_subtract"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::mass_matrix_subtract(KARMABlockMatrix& P, Real const_multiplier)
#else
PetscErrorCode KARMAInterface::mass_matrix_subtract(KARMAMatrix& P, const Real const_multiplier)
#endif
{
	KARMA_LOG_PUSH("mass_matrix_subtract", "Interface");

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();
	
	PetscErrorCode ierr ;

	int physics_index = 0 ;

	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* system = pos->second ;

		KARMAMatrix& phy_jac = system->get_jacobian() ;
		
		KARMAMatrix& phy_mass = system->get_mass_matrix() ;
		
		// check if each physics has stored its mass-matrix
		// TODO: Can;t just shift the jacobian by 1; Need to shift by mass matrix still..
		// Do something alone the lines of 
		// 			ierr = MatGetRowSum(Mass.mat(), tempvec) ;CHKERRQ(ierr);
		// Next shift only the diagonal by the vector
		//			MatDiagonalSet(phy_jac.mat(), tempvec, ADD_VALUES)
		if (_CTX->is_lumped_mass_matrix())
		//if (phy_mass == NULL)
		{
			// NULL mass matrix => assume identity
			ierr = MatScale(phy_jac.mat(), const_multiplier) ;CHKERRQ(ierr);
			ierr = MatShift(phy_jac.mat(), 1.0) ;CHKERRQ(ierr);
		}
		else
		{
			ierr = MatAYPX(phy_jac.mat(), const_multiplier, phy_mass.mat(), SUBSET_NONZERO_PATTERN) ;CHKERRQ(ierr);
		}
		
		phy_jac.close() ;
		
		this->synchronize_interface_jacobian (physics_index, system, P) ;
		
#ifdef USE_PETSC_BLOCK_EXTENSIONS
		physics_index++ ;
#else
		physics_index += system->size() ;
#endif

	}

	P.close() ;

	KARMA_LOG_POP("mass_matrix_subtract", "Interface");

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::jacobian_matvec"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::jacobian_matvec (const KARMABlockVector& pX, KARMABlockVector& pY)
#else
PetscErrorCode KARMAInterface::jacobian_matvec (const KARMAVector& pX, KARMAVector& pY)
#endif
{
	KARMA_LOG_PUSH("jacobian_matvec", "Interface");

	// this routine is not yet complete
	// specialize between SS and temporal jacobians
	
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	PetscErrorCode ierr ;

	_params.clear() ;
	_params.set<Real>(TIME) = _CTX->time ;
	_params.set<Real>(DELT) = _CTX->delt ;

	this->synchronize_physics_temporary_vectors(pX) ;

	// Interface should store the dof_indicies of any given physics in a global vector;
	// that way before calling the residual function, it could project solution from other physics and pass onto current physics.
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;
		ISystem& implicit_system = physics->get_current_system() ;

		if (!implicit_system.sparsity().is_ready())
			KARMA_LOG(ERROR) << "User sparsity pattern is not ready. Parallel runs won't work." ;
		
		const std::vector<unsigned int>& send_list = *implicit_system.sparsity().send_list ;

		KARMAVector& phy_temp = physics->get_temporary_vector() ;
		KARMAVector& phy_res = physics->get_residual() ;

		_params.set<KARMAPhysicsBase*>(PHYSICS) = physics ;

		// TODO: check if there is a matvec method available.
		// 		If YES, then call it and be done.
		//		If NO,
		//			Check if jacobian matrix is created explicitly through a routine.
		//				If YES, use MatMult and store result in a subvector of pY
		//				If NO, throw an exception saying this operation cannot be performed
		if (physics->jac_operator->is_matvec())
		{
			ierr = physics->jac_operator->matvec (phy_temp, phy_res, _params);CHKERRQ(ierr);
		}
		else if (physics->jac_operator->is_matrix())
		{
			KARMAMatrix& jac = physics->get_jacobian() ;

			KARMAVector& phy_sol = physics->get_local_solution() ;
			physics->get_solution().localize(phy_sol, send_list) ;

			// should we actually recompute the jacobian ? Or just use the data already
			// computed in jac ?
			ierr = physics->jac_operator->matrix (phy_sol, jac, _params);CHKERRQ(ierr);

			ierr = MatMult(jac.mat(), phy_temp.vec(), phy_res.vec()) ;CHKERRQ(ierr);
		}
		else
		{
			PetscPrintf(KARMA_COMM_WORLD, "\nJacobian operator does not implement MATVEC operation. ABORTING ...\n") ;
			karma_error() ;
		}
	}

	this->synchronize_interface_residual(pY) ;

	KARMA_LOG_POP("jacobian_matvec", "Interface");

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::jacobian_solve"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::jacobian_solve (const KARMABlockVector& pX, KARMABlockVector& pY)
#else
PetscErrorCode KARMAInterface::jacobian_solve (const KARMAVector& pX, KARMAVector& pY)
#endif
{
	// this routine is not yet complete
	// specialize between SS and temporal jacobians
	karma_error() ;
}

// Build the preconditioner matrix just like the jacobian matrix; 
#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::preconditioner_matrix"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::preconditioner_matrix (const KARMABlockVector& , KARMABlockMatrix& )
#else
PetscErrorCode KARMAInterface::preconditioner_matrix (const KARMAVector& , KARMAMatrix& )
#endif
{
	KARMA_LOG_PUSH("preconditioner_matrix", "Interface");

	
	KARMA_LOG_POP("preconditioner_matrix", "Interface");

	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::preconditioner_matvec"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::preconditioner_matvec (const KARMABlockVector& pX, KARMABlockVector& pY)
#else
PetscErrorCode KARMAInterface::preconditioner_matvec (const KARMAVector& pX, KARMAVector& pY)
#endif
{
	KARMA_LOG_PUSH("preconditioner_matvec", "Interface");

	const double& dtime = _CTX->time ;
	const double& prec_id = _CTX->get_preconditioner_id () ;
	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	PetscErrorCode ierr ;

	_params.clear() ;

	this->synchronize_physics_temporary_vectors(pX) ;

	// Interface should store the dof_indicies of any given physics in a global vector;
	// that way before calling the residual function, it could project solution from other physics and pass onto current physics.
	for (; pos != end; ++pos)
	{
		// Get a reference to the physics system we are solving
		KARMAPhysicsBase* physics = pos->second ;

		KARMAVector& phy_temp = physics->get_temporary_vector() ;
		KARMAVector& phy_res = physics->get_residual() ;

		// const std::vector<unsigned int>& send_list = physics->get_dof_map().get_send_list ();
		// KARMAVector& phy_sol = physics->get_local_solution() ;
		// physics->get_solution().localize(phy_sol, send_list) ;

		_params.set<KARMAPhysicsBase*>(PHYSICS) = physics ;
		_params.set<Real>(TIME) = dtime ;

		// TODO: check if there is a matvec method available.
		// 		If YES, then call it and be done.
		//		If NO,
		//			Check if jacobian matrix is created explicitly through a routine.
		//				If YES, use MatMult and store result in a subvector of pY
		//				If NO, throw an exception saying this operation cannot be performed
		ierr = physics->prec_operator[prec_id]->matvec (phy_temp, phy_res, _params) ;CHKERRQ(ierr);
	}

	this->synchronize_interface_residual(pY) ;

	KARMA_LOG_POP("preconditioner_matvec", "Interface");

	PetscFunctionReturn(ierr);
}

#undef __FUNCT__
#define __FUNCT__ "KARMAInterface::preconditioner_solve"
#ifdef USE_PETSC_BLOCK_EXTENSIONS
PetscErrorCode KARMAInterface::preconditioner_solve (const KARMABlockVector& pX, KARMABlockVector& pY )
#else
PetscErrorCode KARMAInterface::preconditioner_solve (const KARMAVector& pX, KARMAVector& pY )
#endif
{
	// this routine is not yet complete
	// specialize between SS and temporal jacobians
	karma_error() ;
}
