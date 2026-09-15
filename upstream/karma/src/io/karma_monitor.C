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
#include "karma_monitor.h"
#include "karma_interface.h"
#include "karma_vector.h"


/**
 * Monitor - Optional user-defined monitoring routine that views the
 * current iterate with a petsc viewer. Set by SNESMonitorSet().
 *
 * Input Parameters:
 * \p snes		- the SNES context
 * \p its 		- iteration number
 * \p fnorm 	- 2-norm function value (may be estimated)
 * \p ctx 		- optional user-defined context for private data for the
 *				  monitor routine, as set by SNESMonitorSet()
 */
#undef __FUNCT__
#define __FUNCT__ "KARMASNESMonitor"
PetscErrorCode KARMA::KARMASNESMonitor(SNES _snes, PetscInt its, PetscReal fnorm, void *)
{
	PetscErrorCode ierr;

	PetscFunctionBegin;
	
	int lits = 0 ;
	ierr = SNESGetLinearSolveIterations(_snes, &lits) ;CHKERRQ(ierr);

	if (its == 0)
		PetscPrintf(KARMA_COMM_WORLD,"\n");
		
	ierr = PetscPrintf(KARMA_COMM_WORLD,"SNES {%D}: \tKSP = %d\t||SNES Residual|| := %G\n", its, lits, fnorm); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/**
 * KARMAKSPMonitor - Optional user-defined monitoring routine that views the
 * current iterate with a petsc viewer. Set by KSPMonitorSet().
 *
 * Input Parameters:
 * \p ksp 	- the KSP context
 * \p its 	- iteration number
 * \p fnorm	- 2-norm function value (may be estimated)
 * \p ctx 	- optional user-defined context for private data for the
 *			  monitor routine, as set by SNESMonitorSet()
 */
#undef __FUNCT__
#define __FUNCT__ "KARMAKSPMonitor"
PetscErrorCode KARMA::KARMAKSPMonitor(KSP , PetscInt its, PetscReal fnorm, void *)
{
	PetscErrorCode ierr = PetscPrintf(KARMA_COMM_WORLD,"\nKSP = %d\t||KSP Residual|| := %G", its, fnorm); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}

/**
 * KARMATSMonitor - Optional user-defined monitoring routine that views the
 * current time step with a petsc viewer. Set by TSMonitorSet().
 *
 * Input Parameters:
 * \p ksp 	- the KSP context
 * \p its 	- iteration number
 * \p fnorm	- 2-norm function value (may be estimated)
 * \p ctx 	- optional user-defined context for private data for the
 *			  monitor routine, as set by SNESMonitorSet()
 */
#undef __FUNCT__
#define __FUNCT__ "KARMATSMonitor"
PetscErrorCode KARMA::KARMATSMonitor(TS ts, PetscInt step, PetscReal ptime, Vec v, void *ctx)
{
	PetscErrorCode ierr = PetscPrintf(KARMA_COMM_WORLD,"\nTS: Step = %d\tTime = %G\t", step, ptime); CHKERRQ(ierr);

	PetscFunctionReturn(0);
}


/*
   KARMASNESPostCheck - Optional user-defined routine that checks the validity of
   candidate steps of a line search method.  Set by SNESLineSearchSetPostCheck().

   Input Parameters:
   _snes - the SNES context
   ctx  - optional user-defined context for private data for the 
          monitor routine, as set by SNESLineSearchSetPostCheck()
   xcurrent - current solution
   y - search direction and length
   x    - the new candidate iterate

   Output Parameters:
   y    - proposed step (search direction and length) (possibly changed)
   x    - current iterate (possibly modified)
   
 */
#undef __FUNCT__
#define __FUNCT__ "KARMASNESPostCheck"
PetscErrorCode KARMA::KARMASNESPostCheck(SNES ,Vec ,Vec ,Vec ,void *,PetscBool *,PetscBool *)
{
	PetscErrorCode ierr=0;
	
	PetscFunctionBegin;
  
	// TODO: Check the current iterate solution (xcurrent) for physical validity
	// For example: Scalar flux, Precursors, Temperature, density cannot be negative

	// Also check the current residual norm for each of the physics and store it in CTX.
	// These individual physics residuals can be checked later at SNESSetConvergenceTest routine.
	
	KARMA_LOG(INFO) << "inside post check for snes .." ;

	PetscFunctionReturn(ierr);
}


/*
   KARMASNESConvergenceTest - Optional user-defined routine that checks the convergence of
   SNES line-search method.  Set by SNESSetConvergenceTest().

   Input Parameters:
	_snes 	- the SNES context
	it 	- current iteration (0 is the first and is before any Newton step)
	cctx 	- [optional] convergence context
	xnorm 	- 2-norm of current iterate
	pnorm 	- 2-norm of current step
	fnorm 	- 2-norm of function

   Output Parameters:
	reason 	- reason for convergence/divergence

 */
#undef __FUNCT__
#define __FUNCT__ "KARMASNESConvergenceTest"
PetscErrorCode KARMA::KARMASNESConvergenceTest (SNES ,PetscInt ,PetscReal ,PetscReal ,PetscReal ,SNESConvergedReason *,void *)
{
	PetscErrorCode ierr=0;
	PetscFunctionBegin;
	
	/*
	Vec f ;
	PetscReal atol, rtol ;

PetscErrorCode KARMA::KARMASNESConvergenceTest (SNES _snes,PetscInt it,PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx)  
	ierr = SNESGetFunction(_snes,&f,0,0); CHKERRQ(ierr);
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	std::string phyname ;

	ierr = SNESGetTolerances(_snes, &atol, &rtol, 0, 0, 0);CHKERRQ(ierr);

	bool isconverged = interface->is_physics_converged(f, atol, rtol, phyname) ;

	if (!isconverged)
		PetscPrintf(KARMA_COMM_WORLD, "\nSNES iteration failed to converge in %s physics.\n", phyname.c_str()) ;

	ierr = SNESDefaultConverged(_snes,it,xnorm,pnorm,fnorm,reason,cctx) ;CHKERRQ(ierr);

	*/
	
	PetscFunctionReturn(ierr);
}

