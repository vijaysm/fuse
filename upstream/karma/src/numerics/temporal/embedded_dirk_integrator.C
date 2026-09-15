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
#include "karma_wrappers.h"
#include "karma_vector.h"
#include "embedded_dirk_integrator.h"
#include "parallel.h"
#include "petsc_snes_utils.h"

// Standard C++ includes
#include <iostream>

EmbeddedDIRKIntegrator::EmbeddedDIRKIntegrator(RKType rktype, Vec &solin, PetscReal tstart,
		PetscReal tend, PetscReal deltinit) : 
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, true, 0, new PetscReal(1.0e-5), new PetscReal(1.0e-10))
{
	karma_assert (_scheme->rk_category() != ERK || _scheme->rk_category() != ERKEmbedded) ;

	PetscScalar dtmax = 0.01, facrr = 4.0, facll = 0.2, safety = 0.8 ;
	PetscOptionsGetScalar(PETSC_NULL,"-safety",&safety,PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL,"-dtmax",&dtmax,PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL,"-facr",&facrr,PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL,"-facl",&facll,PETSC_NULL);

	this->set_adaptive_params(dtmax, 100000, safety, facll, facrr) ;
}

EmbeddedDIRKIntegrator::EmbeddedDIRKIntegrator(RKType rktype, Vec &solin, PetscReal tstart,
		PetscReal tend, PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) : 
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, true, itolerin, rtolerin, atolerin)
{
	karma_assert (_scheme->rk_category() != ERK || _scheme->rk_category() != ERKEmbedded) ;
	
	PetscScalar dtmax = 0.01, facrr = 4.0, facll = 0.2, safety = 0.8 ;
	PetscOptionsGetScalar(PETSC_NULL,"-safety",&safety,PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL,"-dtmax",&dtmax,PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL,"-facr",&facrr,PETSC_NULL);
	PetscOptionsGetScalar(PETSC_NULL,"-facl",&facll,PETSC_NULL);

	this->set_adaptive_params(dtmax, 100000, safety, facll, facrr) ;
}

EmbeddedDIRKIntegrator::~EmbeddedDIRKIntegrator()
{
	this->clear () ;
} 

PetscErrorCode EmbeddedDIRKIntegrator::init ()
{
	PetscErrorCode 	ierr = 0 ;
	
	if (!_initialized)
	{
		TemporalIntegrator::init () ;
		
		// duplicate the solution structure and create our own copy
		ierr = VecDuplicate(sol, &_curr_sol) ;	CHKERRQ(ierr);
		
		pcurr_sol = new KARMAVector(_curr_sol) ;
		
		// Get the SNES, KSP and PC context
		this->get_petsc_solvers () ;
	}
	
	return ierr ;
}

PetscErrorCode EmbeddedDIRKIntegrator::clear ()
{
	PetscErrorCode 	ierr = 0 ;

	// clear the vectors
	if (_initialized)
	{
		delete pcurr_sol ;
		
		// destroy our duplicate of the solution 
		ierr = VecDestroy(&_curr_sol) ;	CHKERRQ(ierr);
		
		TemporalIntegrator::clear () ;
	}

	return ierr ;
}

void EmbeddedDIRKIntegrator::print_stats()
{
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------");
	PetscPrintf(KARMA_COMM_WORLD,"\n| Embedded %s Integrator performance |", _name.c_str());
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------\n");
	
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of steps := %D", nstep);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of accepted steps := %D", naccpt);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of rejected steps := %D", nrejct);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of function evaluations := %D", nfcn);
	
	PetscPrintf(KARMA_COMM_WORLD,"\n") ;
}

PetscErrorCode EmbeddedDIRKIntegrator::core_integrator()
{
	PetscErrorCode 	ierr;
	SNESConvergedReason newton_convergence_reason;
	PetscInt its ;

	int stages = _scheme->n_stages() ;
	
	Real posneg = KARMAUtils::sign(1.0, time_end-time);
	
	ierr = PetscPrintf(KARMA_COMM_WORLD,"\n\t===Implicit Integration with %s scheme ===\n", _name.c_str()) ;
	
	_CTX->time = time ;
	_CTX->delt = h ;

	// initial preparations
	// Set these to SNES tolerances ; 
	Real atoli = atoler[0];
	Real rtoli = rtoler[0];
	
	bool newton_fail = false ;
	
	// if there is thermal feedback, write the total power as a function of time
	std::string delt_file = "output/delt.txt" ;
	
	if (_CTX->rank == 0)
	{
		std::ofstream _file(delt_file.c_str(), std::ios::out) ;
		if (_file.is_open())
		{
			_file << "Time\tCurrent-Delt\tLocal-Error\tPredicted-Delt\n" ;
			_file.close() ;
		}
	}
	
	Real abstol, rtol, stol ;
	int maxit, maxf ;
	ierr = SNESGetTolerances(_snes, &abstol, &rtol, &stol, &maxit, &maxf);CHKERRQ(ierr);
	ierr = SNESSetTolerances(_snes, atoli, rtoli, stol, maxit, maxf);CHKERRQ(ierr);
	
	double **A = _scheme->Acoeff() ;
	double *B = _scheme->Bcoeff() ;
	double *C = _scheme->Ccoeff() ;
	double *E = _scheme->Ecoeff() ;
	
	// Set the context in SNES so that integrator can be accessed from wrapper function
	ierr = SNESSetApplicationContext(_snes, this) ;CHKERRQ(ierr);

	PetscBool calc_dyn_time ;
	PetscOptionsHasName(PETSC_NULL, "-dyn_time", &calc_dyn_time) ;
	bool calculate_dyn_time = calc_dyn_time ;

	PetscBool is_lte_error_given ;
	PetscReal LTE_tolerance = 1e-6 ;	// default truncation relative error tolerance for solution
	ierr = PetscOptionsGetReal(PETSC_NULL, "-lte_error", &LTE_tolerance, &is_lte_error_given) ;
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	std::vector<std::vector<Real> > dyntime_all ;
	
	bool reject = false;
	Real prev_local_error = 1.0 ;
	Real hold = h ;
	const Real order_power = 1.0 / (KARMA::min<Real>(static_cast<double>(_scheme->n_order()), static_cast<double>(_scheme->n_secondary_order())) + 1.0) ;
	
	// basic integration loop: loop till end of time or when step size is too low.
	while(time < time_end && h > 1e-15)
	{
		nstep++;
		ierr = PetscPrintf(KARMA_COMM_WORLD,"\n%D\t%s : DelT = %G : Performing calculation for step %G -> %G", 
							nstep, _scheme->name().c_str(), h, time, time+h) ;

		interface->step_preprocess() ;
		
		// TODO:  Extend to IRK Singular and NonSingular (RADAU and Lobatto) later
		// NOTE: currently implemented only for DIRK schemes
		for (unsigned int istage = 0 ; istage < stages; istage++)
		{
			curr_stage = istage ;
			
			_CTX->time = time+h*C[istage] ;

			ierr = VecCopy(sol, _curr_sol) ;	CHKERRQ(ierr);
			pcurr_sol->close() ;
			
			interface->stage_preprocess() ;
			
			{
				// Compute Zi = \sum {j=1..i-1} {h*a_{ij}*k_j} ;
				// This is a constant vector subtracted from right hand side
				// for DIRK schemes
				ierr = VecSet(Zi, 0.0) ;	CHKERRQ(ierr);
				for (unsigned int jstage = 0 ; jstage < istage; jstage++)
				{
					if(fabs(A[istage][jstage]) > 1E-12)
					{
						ierr = VecAXPY(Zi, h*A[istage][jstage], k[jstage]) ;	CHKERRQ(ierr);
					} 
				}
				
				ierr = VecAssemblyBegin(Zi);	CHKERRQ(ierr);
				ierr = VecAssemblyEnd(Zi);	CHKERRQ(ierr);
				
				// NOTE: Look at Hairer page: 128 for a better guess. Defintely see a good scope
				// for improvement here...
				
				// _soln_guess = un + Zi
				ierr = VecAXPY(_curr_sol, 1.0, Zi) ;CHKERRQ(ierr);
				pcurr_sol->close() ;
				
				KARMAVector stage_wrapper(Zi) ;
				
				// Zi = M*un + \sum {j=1..i-1} {h*a_{ij}*f(Z_j)} ;
				ierr = interface->mass_multiply (*pcurr_sol, stage_wrapper) ;CHKERRQ(ierr);

			}
			
			// If there is an explicit first stage, then there is no nonlinear solve to be performed.
			if (fabs(C[istage]) > 1E-12)
			{
				// solve the temporal nonlinear system
				// ierr = SNESSolve(_snes, Zi, _curr_sol); CHKERRQ(ierr);
				ierr = SNESSolve(_snes, Zi, _curr_sol); CHKERRQ(ierr);
				
				ierr = SNESGetIterationNumber(_snes, &its);CHKERRQ(ierr);
				
				SNESGetConvergedReason(_snes, &newton_convergence_reason);
				
				// Stop newton iteration if no convergence
				if (newton_convergence_reason < 0)
				{
					ierr = PetscPrintf(KARMA_COMM_WORLD,"\nNewton iteration failed. Reason: ") ;
					
					print_snes_converged_reason(newton_convergence_reason) ;
					
					ierr = PetscPrintf(KARMA_COMM_WORLD,"\nStopping at Time {%G} : Stage{%D} : Newton{%D}", 
										time+h*C[istage], istage, its) ;
					
					newton_fail = true ;
				}
				else
					newton_fail = false ;
			}
			
			// if the nonlinear iteration did not succeed, no point computing rest of step
			// reduce time step and recompute solution
			if (newton_fail)
				break ;
			
			// should also be called for explicit stages.
			{
				// find the SS residual and invert the mass matrix to get the istage function residual.
				ierr = Function(time+h*C[istage], _curr_sol, k[istage]);	CHKERRQ(ierr);
			}

			interface->stage_postprocess() ;
		}
	
		// check if we stopped because newton iteration failed.
		if (newton_fail)
		{
			newton_fail = false ;
			// we have not modified solution.
			// just compute new time step size
			h *= facl ;		// reduce step size by maximum allowable limit
			if (naccpt >= 1) nrejct++;
			
			_CTX->delt = h ;
			continue ;
		}
		
		// find the new solution at the end of the step
		// the weighted numerical solution
		{
			ierr = VecCopy(sol, _curr_sol) ;	CHKERRQ(ierr);
			pcurr_sol->close() ;
			
			// U_{n+1} = U_{n} + delt * sum{i=1..s} (b[i] * k_i)
			for (unsigned int istage = 0 ; istage < stages; istage++)
			{
				if (fabs(B[istage]) > 1E-14)
				{
					ierr = VecAXPY(sol, h*B[istage], k[istage]) ;	CHKERRQ(ierr);
				}
			}
			psol->close() ;
		}
		
		// step accepted
		time += posneg*h;
		
		if (calculate_dyn_time)
		{
			std::vector<Real> dyn_time ;
			
			// query the interface to compute the dynamical time scales ..
			ierr = interface->compute_dynamical_timescales (*pcurr_sol, *psol, h, dyn_time) ;
			dyntime_all.push_back(dyn_time) ;
			dyntime_all[nstep][0] = time ;
			for (int iphy=0; iphy<dyn_time.size(); iphy++)
				dyntime_all[nstep][iphy+1] = dyn_time[iphy] ;
		}
		
		// estimate the new time step
		Real hnew = h ;
		{
			ierr = VecScale(k[0], h*E[0]) ;	CHKERRQ(ierr);
			
			// Estimate the local truncation error
			// E_{n+1} = delt * sum{i=1..s} (E[i] * k_i)
			for (unsigned int istage = 1 ; istage < stages; istage++)
			{
				if (fabs(E[istage]) > 1E-12)
				{
					ierr = VecAXPY(k[0], h*E[istage], k[istage]) ;	CHKERRQ(ierr);
				}
			}
			
			ierr = VecAssemblyBegin(k[0]);	CHKERRQ(ierr);
			ierr = VecAssemblyEnd(k[0]);	CHKERRQ(ierr);
			
			// find the scaled local truncation error
			PetscReal local_error ;
			// ierr = VecMaxPointwiseDivide(k[0], sol, &local_error) ;	CHKERRQ(ierr);
			ierr = VecNorm(k[0], NORM_2, &local_error) ;	CHKERRQ(ierr);
			
			// For debug purposes with the Dahlquist ODE equation;
			// The linear problem should serve as a good test bed for error estimator efficiency.
			// Also could technically use any MMS function to evaluate the difference between exact and estimated errors.
			// KARMA_LOG(INFO) << "Exact error = " << fabs(exp(-_CTX->time)-(*psol)(0)) << " vs estimated error : " << local_error ;
			
			// Hairer: IV: Stiff problems - Page 124
			// Real facerr = std::pow(LTE_tolerance/local_error, order_power) * 
						  // std::pow(prev_local_error/local_error, 0.2)  ;
			Real facerr = std::pow(LTE_tolerance/local_error, order_power)  ;
			Real newdelt = safe * h * facerr ;
			
			// check if local truncation error is more than prescribed tolerance
			if (local_error > LTE_tolerance)
			{
				// step rejected
				hnew = KARMA::max<Real>(h*facl, newdelt);
				
				reject = true;
				if (naccpt >= 1) nrejct++;
				
				time -= posneg*h;
				
				KARMA_LOG(INFO) << "Rejecting step: LTE = " << local_error << "\th_{new} = " << hnew ;
				
				// KARMA_LOG(INFO) << "Rejecting step: LTE = " << local_error << "\th_{new} = " << hnew ;
				// If rejected, go back and recalculate solution with smaller
				// step size. we do not want any instability to propagate.
				
				// copy the original U_{n} solution to start the new step
				ierr = VecCopy(_curr_sol, sol) ;	pcurr_sol->close() ;
			}
			else
			{
				reject = false;
				naccpt++;
			
				// k[0] contains the truncation error estimate as a function of dofs
				// Real local_error = pcurr_sol->l2_norm()/std::sqrt(pcurr_sol->size()) ;
				
				// compute estimate for step size based on PI controller
				
				// Real newdelt = safe * h * std::pow(LTE_tolerance/local_error, order_power) ;
				
				// dont increase by more than facr times the step size
				hnew = KARMA::min<Real> (facr*h, newdelt) ;
				
				prev_local_error = local_error ;
				
				if (time+hnew > time_end)
					hnew = time_end-time ;
			
				// dont increase or decrease the step size more by than the allowed factor 
				if (hnew > h)
					hnew = KARMA::min<Real> (facr*h, hnew) ;
				
				hnew = KARMA::min<Real>(hnew, hmax) ;
				
				// check if this is the last time step
				if (time_end - time < 1e-12)
					hnew = 0.0 ;
				
				KARMA_LOG(INFO) << "Accepting step: LTE = " << local_error << "\th_{new} = " << newdelt ;
					
				hold = h ;
			}
			
			// write out the data to log file
			if (_CTX->rank == 0)
			{
				std::ofstream _file(delt_file.c_str(), std::ios::out | std::ios::app | std::ios::ate) ;
				if (_file.is_open())
				{
					_file << std::setprecision(14) << "\n" << time << "\t" << h << "\t" << local_error << "\t" << hnew ;
					_file.flush() ;
					_file.close() ;
				}
			}
			
			Parallel::barrier() ;
		}
		
		h = hnew ;

		// set the context variables
		_CTX->delt = h ;
		_CTX->time = time ;
		
		interface->synchronize_physics_solution(*psol) ;
		
		interface->step_postprocess() ;

		output_current_solution(nstep, naccpt) ;
		
		Parallel::barrier() ;
		
	}
	
	if (dyntime_all.size() > 0)
	{
		// for now, lets just make this very specific to Neutronics/Conduction coupled problem
		// TODO: need to abstract this out and put it in some kinda physics specific routine
		// PetscPrintf(KARMA_COMM_WORLD,"\nTime\t\tNeutronics\t\tConduction\n") ;
		for (unsigned int istep=0; istep < nstep; istep++)
		{
			// PetscPrintf(KARMA_COMM_WORLD,"\n%G\t%G\t%G", dyntime_all[nstep][0], dyntime_all[nstep][1], dyntime_all[nstep][2]) ;
			dyntime_all[istep].clear() ;
		}
		dyntime_all.clear() ;
	}
	
	ierr = SNESSetTolerances(_snes, abstol, rtol, stol, maxit, maxf); CHKERRQ(ierr);
	
	// replace the reference to the integrator with a NULL pointer
	ierr = SNESSetApplicationContext(_snes, PETSC_NULL) ;CHKERRQ(ierr);
	
	PetscFunctionReturn(0) ;
}

