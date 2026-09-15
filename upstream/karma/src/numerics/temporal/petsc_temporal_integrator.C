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
#include "karma_interface.h"
#include "petsc_temporal_integrator.h"
#include "karma_monitor.h"

PetscTemporalIntegrator::PetscTemporalIntegrator(RKType rktype, Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit) :
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, true, 0, new PetscReal(1.0e-5), 
							new PetscReal(1.0e-5))
{ 
	// set_adaptive_params((tend-tstart)/5, 100000, 0.9, 0.2, 4.0) ;
}

PetscTemporalIntegrator::PetscTemporalIntegrator(RKType rktype, Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit, int itolerin,
			PetscReal *rtolerin, PetscReal *atolerin) : 
			TemporalIntegrator(rktype, solin, tstart, tend, deltinit, true, itolerin, rtolerin, atolerin)
{
	// set_adaptive_params((tend-tstart)/5, 100000, 0.9, 0.2, 4.0) ;
}

PetscTemporalIntegrator::~PetscTemporalIntegrator()
{
	this->clear () ;	
}


////////// WRAPPER ROUTINES BEGIN /////////////////

static PetscErrorCode __petsc_ts_wrapper_function(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
	/**
	 * PETSc time-stepping routine provides the solution X and the time derivative term Xdot.
	 * Assemble the temporal residual = M*Xdot - F_SS(X,t)
	 */
	return(0);
}


static PetscErrorCode __petsc_ts_wrapper_jacobian(TS ts,PetscReal t,Vec X,Vec Xdot,
					PetscReal a,Mat *A,Mat *B,MatStructure *flag,void *ctx)
{
	/**
	 * PETSc time-stepping routine provides the solution X and the time derivative term Xdot.
	 * Assemble the temporal Jacobian = a*M - J_SS(X,t) \approx a*M - B
	 */
	return(0);
}

static PetscErrorCode __petsc_ts_pre_step (TS )
{
	/**
	 * PETSc time-stepping routine provides the solution X and the time derivative term Xdot.
	 * Assemble the temporal Jacobian = a*M - J_SS(X,t) \approx a*M - B
	 */
	 KARMAInterface::get_interface()->step_preprocess() ;
	return(0);
}

static PetscErrorCode __petsc_ts_post_step (TS )
{
	/**
	 * PETSc time-stepping routine provides the solution X and the time derivative term Xdot.
	 * Assemble the temporal Jacobian = a*M - J_SS(X,t) \approx a*M - B
	 */
	 KARMAInterface::get_interface()->step_postprocess() ;
	 
		// output_current_solution(nstep, naccpt) ;
		
	return(0);
}

/////////// WRAPPER ROUTINES END //////////////////

PetscErrorCode PetscTemporalIntegrator::init ()
{
	PetscErrorCode 	ierr = 0;
	
	if(!_initialized)
	{
		TemporalIntegrator::init () ;
		
		ierr = TSCreate(PETSC_COMM_WORLD,&_ts);	CHKERRQ(ierr);
		// TODO: make this user-definable or from command-line
		ierr = TSSetProblemType(_ts,TS_NONLINEAR);	CHKERRQ(ierr);
		// Set the method that the user wanted
		ierr = TSSetType(_ts, _name.c_str()); 	CHKERRQ(ierr);
		
		KARMAContext *global_ctx = KARMAInterface::get_interface()->get_context() ;
		
		// the most important 2 functions in creating a petsc TS object are below.
		
		/**
		 * the IFunction computes F(t,U,U_t) where F = 0 is the DAE to be solved. 
		 * where 		t 	- time at step/stage being solved
		 *				u 	- state vector
		 * 				u_t - time derivative of state vector
		 *				F 	- function vector
		 *				ctx - [optional] user-defined context for matrix evaluation routine 
		 */
		ierr = TSSetIFunction(_ts, global_ctx->Residual->vec(), __petsc_ts_wrapper_function, this);	CHKERRQ(ierr);
		
		/**
		 * the IJacobian computes G(U) = F(t,U,U0+a*U) where F(t,U,U_t) = 0 is the DAE to be solved. 
		 * where 		t 		- time at step/stage being solved
		 *				u 		- state vector
		 * 				u_t 	- time derivative of state vector
		 *				a 		- shift 
		 *				A 		- Jacobian of G(U) = F(t,U,U0+a*U), equivalent to dF/dU + a*dF/dU_t 
		 *				B 		- preconditioning matrix for A, may be same as A 
		 *				flag 	- flag indicating information about the preconditioner matrix 
		 *						  structure (same as flag in KSPSetOperators()) 
		 *				ctx - [optional] user-defined context for matrix evaluation routine 
		 */
		Mat J, P ;
		void *ctx ;
		ierr = SNESGetJacobian(_snes, &J, &P, PETSC_NULL, &ctx) ;	CHKERRQ(ierr);

		ierr = TSSetIJacobian(_ts, J, P, __petsc_ts_wrapper_jacobian, ctx /*this*/);	CHKERRQ(ierr);
		
		ierr = TSSetDuration(_ts, nmax, time_end);	CHKERRQ(ierr);
		
		// set our output monitor for the time-stepper
		ierr = TSMonitorSet(_ts,&KARMA::KARMATSMonitor,this,PETSC_NULL);	CHKERRQ(ierr);
		
		ierr = TSSetOptionsPrefix(_ts, "integrator_");	CHKERRQ(ierr);
		ierr = TSSetInitialTimeStep(_ts, time, h);	CHKERRQ(ierr);
		ierr = TSSetSolution(_ts, sol);	CHKERRQ(ierr);
		ierr = TSSetPreStep(_ts, __petsc_ts_pre_step) ;	CHKERRQ(ierr);
		ierr = TSSetPostStep(_ts, __petsc_ts_post_step) ;	CHKERRQ(ierr);
		
		SNES *snes_ref ;
		ierr = TSGetSNES(_ts, snes_ref) ;	CHKERRQ(ierr);
		// in case, user is using an explicit temporal scheme, 
		// then snes_ref == PETSC_NULL
		if (snes_ref != PETSC_NULL)
		{
			// there is a memory leak here. destroy the memory that snes_ref 
			// was originally pointing at and then replace with our
			// option filled SNES object.
			//ierr = SNESDestroy(snes_ref) ;	CHKERRQ(ierr);
			// snes_ref = &_snes ;
		}

		// Set runtime options
		ierr = TSSetFromOptions(_ts);	CHKERRQ(ierr);

	}
	
	return ierr ;
}

PetscErrorCode PetscTemporalIntegrator::clear ()
{
	PetscErrorCode 	ierr = 0;
	
	if(_initialized)
	{
		ierr = TSDestroy(&_ts) ;	CHKERRQ(ierr);
		
		TemporalIntegrator::clear () ;
	}
	
	return ierr ;
}


void PetscTemporalIntegrator::print_stats()
{
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------");
	PetscPrintf(KARMA_COMM_WORLD,"\n|   PetscTemporalIntegrator performance    |");
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------\n");
	
	PetscPrintf(KARMA_COMM_WORLD,"\nFinal time := %G", time);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of steps := %D", nstep);
	// PetscPrintf(KARMA_COMM_WORLD,"\nNumber of rejected steps := %D", nrejct);
	// PetscPrintf(KARMA_COMM_WORLD,"\nNumber of function evaluations := %D", nfcn);
	
	PetscPrintf(KARMA_COMM_WORLD,"\n") ;
}

PetscErrorCode PetscTemporalIntegrator::core_integrator()
{
	KARMAContext *global_ctx = KARMAInterface::get_interface()->get_context() ;
	PetscInt num_steps ;
	
	// There is nothing to do here but just let PETSc work its magic.
	PetscErrorCode ierr = TSSolve(_ts, global_ctx->Solution->vec(), &time) ;CHKERRQ(ierr);
	
	ierr = TSGetTimeStepNumber(_ts, &num_steps) ;

	nstep = static_cast<unsigned int>(num_steps) ;

	return ierr ;
}
