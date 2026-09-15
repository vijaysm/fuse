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
#include "karma_utils.h"
#include "karma_context.h"
#include "karma_interface.h"
#include "dopri45_integrator.h"
#include "parallel.h"

DOPRI45Integrator::DOPRI45Integrator(Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit) : 
		TemporalIntegrator(DOPRI45, solin, tstart, tend, deltinit, true, 0, new PetscReal(1.0e-5),
							new PetscReal(1.0e-5)), beta(0.08), nstiff(1000)
{ 
	set_adaptive_params((tend-tstart)/5, 100000, 0.9, 0.2, 4.0) ;
}

DOPRI45Integrator::DOPRI45Integrator(Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit, int itolerin, 
			PetscReal *rtolerin, PetscReal *atolerin) : 
		TemporalIntegrator(DOPRI45, solin, tstart, tend, deltinit, true, itolerin, rtolerin, atolerin),
			beta(0.08), nstiff(1000)
{
	set_adaptive_params((tend-tstart)/5, 100000, 0.9, 0.2, 4.0) ;
}

DOPRI45Integrator::DOPRI45Integrator(Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit, int itolerin, 
		PetscReal *rtolerin, PetscReal *atolerin, PetscReal deltmax=0, int nmaxsteps=1E5, PetscReal safetyfac=0.9, 
		PetscReal faclin=.5, PetscReal facrin=2, PetscReal betain=0.0, int nstiffin = 0) : 
		TemporalIntegrator(DOPRI45, solin, tstart, tend, deltinit, true, itolerin, rtolerin, atolerin),
		beta(betain), nstiff(nstiffin)
{
	set_adaptive_params(deltmax, nmaxsteps, safetyfac, faclin, facrin) ;
	
	// beta for step control stabilization
	if (beta == 0.0)
		beta = 0.08;	// recommended by Hairer
	else if (beta < 0.0)
		beta = 0.0;
	else if (beta > 0.2) {
		PetscPrintf(KARMA_COMM_WORLD, "\nCurious input for beta : beta = %G\n", beta) ;
		throw -1;
	}
	
	// nstiff, parameter for stiffness detection
	if (nstiff == 0) nstiff = 1000;
	else if (nstiff < 0) nstiff = nmax + 10;
}

DOPRI45Integrator::~DOPRI45Integrator()
{
	this->clear () ;	
}

PetscErrorCode DOPRI45Integrator::init ()
{
	PetscErrorCode 	ierr = 0;
	
	if(!_initialized)
	{
		TemporalIntegrator::init () ;
		
		ierr = VecDuplicate(sol, &yy1) ;	CHKERRQ(ierr);
		ierr = VecDuplicate(sol, &ysti) ;	CHKERRQ(ierr);
		
		ierr = VecCopy(sol, yy1) ;	CHKERRQ(ierr);
		ierr = VecCopy(sol, ysti) ;	CHKERRQ(ierr);
	}
	
	return ierr ;
}

PetscErrorCode DOPRI45Integrator::clear ()
{
	PetscErrorCode 	ierr = 0;
	
	if(_initialized)
	{
		ierr = VecDestroy(&yy1) ;	CHKERRQ(ierr);
		ierr = VecDestroy(&ysti) ;	CHKERRQ(ierr);
		
		TemporalIntegrator::clear () ;
	}
	
	return ierr ;
}

void DOPRI45Integrator::print_stats()
{
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------");
	PetscPrintf(KARMA_COMM_WORLD,"\n| DOPRI4(5) Integrator performance |");
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------\n");
	
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of steps := %D", nstep);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of accepted steps := %D", naccpt);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of rejected steps := %D", nrejct);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of function evaluations := %D", nfcn);
	
	PetscPrintf(KARMA_COMM_WORLD,"\n") ;
}

PetscErrorCode DOPRI45Integrator::core_integrator()
{
	PetscErrorCode 	ierr;

	int stages = _scheme->n_stages() ;
	
	double posneg = KARMAUtils::sign(1.0, time_end-time);
	double facold = 1.0e-4;
	double expo1 = 0.2 - beta*0.75;
	double facc1 = 1.0/facl;
	double facc2 = 1.0/facr;

	// initial preparations
	double atoli = atoler[0];
	double rtoli = rtoler[0];

	bool last = false;
	double hlamb = 0.0;
	int iasti = 0;

	double **A = _scheme->Acoeff() ;
	double *B = _scheme->Bcoeff() ;
	double *C = _scheme->Ccoeff() ;
	double *E = _scheme->Ecoeff() ;

	Function(time, sol, k[0]);

	KARMAInterface* interface = KARMAInterface::get_interface () ;
	
	hmax = fabs(hmax);

	bool reject = false;
	int nonsti=0;
	hold = h;
	
	// basic integration step
	while (true) {
	
		if (nstep > nmax) {
			KARMA_LOG(ERROR) << "Exit of dopri5 at time = " << time << ", more than nmax = "
				<< nmax << " are needed.\n";
			hold = h;
			return -2;
		}
		if (0.1*fabs(h) <= fabs(time)*uround) {
			KARMA_LOG(ERROR) << "Exit of dopri5 at time = " << time << ", step size too small h = "
				<< h << "\n";
			hold = h;
			return -3;
		}
		if ((time + 1.01*h - time_end)*posneg > 0.0) {
			h = time_end - time;
			last = true;
		}
		nstep++;

		interface->step_preprocess() ;

		for (int istage = 1 ; istage < stages; istage++)
		{
			// the first 6 stages
			ierr = VecCopy(sol, yy1) ;	CHKERRQ(ierr);
			for (int jstage = 0 ; jstage < istage; jstage++)
			{
				if(fabs(A[istage][jstage]) > 1E-14)
				{
					ierr = VecAXPY(yy1, h*A[istage][jstage], k[jstage]) ;	CHKERRQ(ierr);
				}
			}
			
			curr_stage = istage ;
			
			_CTX->time = time+h*C[istage] ;
			
			interface->stage_preprocess() ;

			ierr = Function(time+h*C[istage], yy1, k[istage]);	CHKERRQ(ierr);

			interface->stage_postprocess() ;

		}
		
		// Find out the truncation error of numerical solution
		// store error in k[0]
		ierr = VecScale(k[0], h*E[0]) ;	CHKERRQ(ierr);
		for (int istage = 1 ; istage < stages; istage++)
		{
			ierr = VecAXPY(k[0], h*E[istage], k[istage]) ;	CHKERRQ(ierr);
		}

		// error estimation
		double err = 0.0;
		//double dynstep = 0.0 ;
		
		// For now, consider that absolute and relative tolerance are specified together for all components ;
		// Find the scaled error so that error can be found independent of magnitude
		{
			// find the scaling factor for the soltuion vector
			ierr = VecCopy(yy1, k[1]) ;	CHKERRQ(ierr);
			ierr = VecScale(k[1], rtoli) ;	CHKERRQ(ierr);
			ierr = VecShift(k[1], atoli) ;	CHKERRQ(ierr);
			
			PetscInt n, ls, le, ldim, iglobal ;
			PetscScalar *vk0, *vk1, lnrm ;
			VecGetOwnershipRange(k[1],&ls,&le);
			VecGetSize(k[1], &n) ;
			VecGetLocalSize(k[1],&ldim);
		
			VecGetArray(k[0], &vk0) ;
			VecGetArray(k[1], &vk1) ;
			
			lnrm = 0.0 ;
			
			for (int i=0; i <ldim; i++)
			{
				iglobal = i + ls;
				if (vk1[iglobal] > 1E-15)
					vk1[iglobal] = vk0[iglobal]/vk1[iglobal] ;
				lnrm += vk1[iglobal]*vk1[iglobal] ;
			}
			
			VecRestoreArray(k[0], &vk0) ;
			VecRestoreArray(k[1], &vk1) ;
			err = 0 ;

			// since this is done in parallel, reduce the contribution to compute the L2 norm value  
			MPI_Reduce(&lnrm, &err, 1, MPI_DOUBLE, MPI_SUM, 0, KARMA_COMM_WORLD) ;
			
			err = sqrt(err/n) ;
		}

		ierr = PetscPrintf(KARMA_COMM_WORLD,"\n%D\tDOPRI4(5): Time = %G with DelT = %G and LTE = %G, Old DelT = %G", nstep, time, h, err, hold) ;
		
		// computation of hnew
		double fac11 = KARMA::pow<double>(err, expo1);
		// Lund-stabilization
		double fac = fac11/KARMA::pow<double>(facold,beta);
		// we require facl <= hnew/h <= facr
		fac = KARMA::max<double>(facc2, KARMA::min<double>(facc1, fac/safe));
		double hnew = h/fac;

		if (err <= 1.0)
		{
			// step accepted
			facold = KARMA::max<double>(err, 1.0e-4);
			naccpt++;

			// stiffness detection
			if (!(naccpt % nstiff) || (iasti > 0))
			{
				double stnum = 0.0, stden = 0.0;
				ierr = VecAXPBY(k[5], 1.0, -1.0, k[6]) ;	CHKERRQ(ierr);
				ierr = VecNorm(k[5], NORM_2, &stnum) ;	CHKERRQ(ierr);
				ierr = VecAXPBY(ysti, 1.0, -1.0, yy1) ;	CHKERRQ(ierr);
				ierr = VecNorm(ysti, NORM_2, &stden) ;	CHKERRQ(ierr);
				
				if (stden > 0.0) hlamb = h*sqrt(stnum/stden);
				if (hlamb > 3.25)
				{
					nonsti = 0;
					iasti++;
					if (iasti == 15)
					{
						ierr = PetscPrintf(KARMA_COMM_WORLD,"\nThe problem seems to become stiff at time = %G\n", time) ;
						hold = h;
						return -4;
					}
				}
				else {
					nonsti++;
					if (nonsti == 6) iasti = 0;
				}
			}

			ierr = VecCopy(k[6], k[0]) ;	CHKERRQ(ierr);
			ierr = VecCopy(yy1, sol) ;	CHKERRQ(ierr);
			
			time_old = time;
			time += h;
			hold = h;

			// normal exit
			if (last) {
				hold = hnew;
				return 1;
			}

			if (fabs(hnew) > hmax) hnew = posneg*hmax;
			if (reject) hnew = posneg*KARMA::min<double>(fabs(hnew), fabs(h));
			reject = false;
		}
		else
		{
			// step rejected
			hnew = h/KARMA::min<double>(facc1, fac11/safe);
			hold = h ;
			reject = true;
			if (naccpt >= 1) nrejct++;
			last = false;
			// TODO: If rejected, go back and recalculate solution with smaller
			// step size. we do not want any instability to propagate.
		}
		h = hnew;
		
		// set the context variables
		_CTX->delt = h ;
		_CTX->time = time ;
		
		interface->step_postprocess() ;

		output_current_solution(nstep, naccpt) ;
		
		Parallel::barrier() ;
		
	}

	A = NULL ;
	B = NULL ;
	C = NULL ;
	E = NULL ;

	return ierr ;
}
