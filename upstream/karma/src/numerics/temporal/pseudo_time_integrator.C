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
#include "karma_vector.h"
#include "karma_utils.h"
#include "parallel.h"
#include "pseudo_time_integrator.h"
#include "petsc_snes_utils.h"

#include <iostream>

using namespace std ;

PseudoTimeIntegrator::PseudoTimeIntegrator(Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit) : 
		TemporalIntegrator(BE, solin, tstart, tend, deltinit, false, 0, new PetscReal(1.0e-5), new PetscReal(1.0e-10))
{
}

PseudoTimeIntegrator::PseudoTimeIntegrator(Vec &solin, PetscReal tstart, 
		PetscReal tend, PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) : 
		TemporalIntegrator(BE, solin, tstart, tend, deltinit, false, itolerin, rtolerin, atolerin)
{
}

PseudoTimeIntegrator::~PseudoTimeIntegrator()
{
	this->clear () ;
} 

PetscErrorCode PseudoTimeIntegrator::init ()
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

PetscErrorCode PseudoTimeIntegrator::clear ()
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

void PseudoTimeIntegrator::print_stats()
{
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------");
	PetscPrintf(KARMA_COMM_WORLD,"\n| %s Integrator performance |", _name.c_str());
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------\n");
	
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of steps := %D", nstep);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of accepted steps := %D", naccpt);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of rejected steps := %D", nrejct);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of function evaluations := %D", nfcn);
	
	PetscPrintf(KARMA_COMM_WORLD,"\n") ;
}

PetscErrorCode PseudoTimeIntegrator::core_integrator()
{
	PetscErrorCode 	ierr;
	SNESConvergedReason newton_convergence_reason;
	PetscInt its ;

	int stages = _scheme->n_stages() ;
	
	Real posneg = KARMAUtils::sign(1.0, time_end-time);
	
	ierr = PetscPrintf(KARMA_COMM_WORLD,"\n\t===Implicit Integration with Backward-Euler scheme ===\n") ;
	
	_CTX->time = time ;
	_CTX->delt = h ;

	Real hold = h ;
	// initial preparations
	// Set these to SNES tolerances ; 
	Real atoli = atoler[0];
	Real rtoli = rtoler[0];
	
	Real abstol, rtol, stol ;
	int maxit, maxf ;
	
	ierr = SNESGetTolerances(_snes, &abstol, &rtol, &stol, &maxit, &maxf);CHKERRQ(ierr);
	ierr = SNESSetTolerances(_snes, atoli, rtoli, stol, maxit, maxf);CHKERRQ(ierr);
	
	double **A = _scheme->Acoeff() ;
	double *B = _scheme->Bcoeff() ;
	double *C = _scheme->Ccoeff() ;
	
	// Set the context in SNES so that integrator can be accessed from wrapper function
	ierr = SNESSetApplicationContext(_snes, this) ;CHKERRQ(ierr);

	Real prev_dynmax = 1.0 ;
	bool reject = false, newton_fail = false ;
	
	PetscBool calc_dyn_time ;
	PetscOptionsHasName(PETSC_NULL, "-dyn_time", &calc_dyn_time) ;
	bool calculate_dyn_time = calc_dyn_time ;
	
	PetscBool predict_soln_flg ;
	PetscOptionsHasName(PETSC_NULL, "-no_predict", &predict_soln_flg) ;
	bool predict_solution = !predict_soln_flg ;

	// if there is thermal feedback, write the total power as a function of time
	std::string delt_file = "output/delt.txt" ;
	if (adaptive)
	{
		if (_CTX->rank == 0)
		{
			std::ofstream _file(delt_file.c_str(), ios::out) ;
			if (_file.is_open())
			{
				_file << "Time\tCurrent-Delt\tDynamical-Time\tPredicted-Delt\n" ;
				_file.close() ;
			}
		}
		
	}
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	std::vector<std::vector<Real> > dyntime_all ;
	
	// basic integration loop
	for(nstep=0; nstep < nmax; nstep++)
	{
		ierr = PetscPrintf(KARMA_COMM_WORLD,"\n%D\t%s : DelT = %G : Performing calculation for step %G -> %G", 
							nstep, _scheme->name().c_str(), h, time, time+h) ;

		interface->step_preprocess() ;
		
		curr_stage = 0 ;
		
		{
			_CTX->time = time+h ;

			ierr = VecCopy(sol, _curr_sol) ;	CHKERRQ(ierr);
			pcurr_sol->close() ;
			
			interface->stage_preprocess() ;
			
			{
				KARMAVector stage_wrapper(Zi) ;
				
				// Zi = M*un + \sum {j=1..i-1} {h*a_{ij}*f(Z_j)} ;
				ierr = interface->mass_multiply (*pcurr_sol, stage_wrapper) ;CHKERRQ(ierr);
			}
			
			// If there is an explicit first stage, then there is no nonlinear solve to be performed.
			{
				// solve the temporal nonlinear system
				ierr = SNESSolve(_snes, Zi, _curr_sol); CHKERRQ(ierr);
				
				ierr = SNESGetIterationNumber(_snes, &its);CHKERRQ(ierr);
				
				SNESGetConvergedReason(_snes, &newton_convergence_reason);
				
				// Stop newton iteration if no convergence
				if (newton_convergence_reason < 0)
				{
					ierr = PetscPrintf(KARMA_COMM_WORLD,"\nNewton iteration failed. Reason: ") ;
					print_snes_converged_reason(newton_convergence_reason) ;
					
					// ierr = PetscPrintf(KARMA_COMM_WORLD,"\nStopping at Time {%G} : Stage{%D} : Newton{%D}", 
										// time+h*C[istage], istage, its) ;
										
					// newton_fail = true ;
				}
			}
			
			// should also be called for explicit stages.
			// NOTE: if there are explicit stages in the middle, this will 
			// fail since _curr_sol might not be the right solution to use.
			{
				// find the SS residual and invert the mass matrix to get the istage function residual.
				ierr = Function(time+h, _curr_sol, k[0]);	CHKERRQ(ierr);
			}

			interface->stage_postprocess() ;
		}
			
		// find the new solution at the end of the step
		// the weighted numerical solution 
		{
			ierr = VecCopy(sol, _curr_sol) ;	CHKERRQ(ierr);
			pcurr_sol->close() ;
			
			// U_{n+1} = U_{n} + delt * sum{i=1..s} (b[i] * k_i)
			ierr = VecAXPY(sol, h, k[0]) ;	CHKERRQ(ierr);
			psol->close() ;
		}
		
		// step accepted
		time += posneg*h;
		
		// set the context variables
		_CTX->delt = h ;
		_CTX->time = time ;
		
		// if (calculate_dyn_time)
		{
			std::vector<Real> dyn_time ;
			// query the interface to compute the dynamical time scales ..
			ierr = interface->compute_dynamical_timescales (*pcurr_sol, *psol, h, dyn_time) ;
			std::vector<Real> dynamical_scales ;	dynamical_scales.resize(dyn_time.size()+1) ;
			dynamical_scales[0] = time ;
			for (unsigned int iphy=0; iphy<dyn_time.size(); iphy++)
				dynamical_scales[iphy+1] = dyn_time[iphy] ;
			dyntime_all.push_back(dynamical_scales) ;
		}
		
		
		{
			Real hnew = h ;
			
			// dynmax = minimum dynamical time scale
			Real dynmax = 1e45 ;
			for (unsigned int iphy=1; iphy<dyntime_all[nstep].size(); iphy++)
				dynmax = KARMA::min<Real>(dynmax, dyntime_all[nstep][iphy]) ;
		
			// // Hairer: IV: Stiff problems - Page 124
			// Real facerr = std::pow(atoli/local_error, 1.0/(_scheme->n_order())) * 
						  // std::pow(prev_local_error/local_error, 1.0/(_scheme->n_order())) *
						  // h / hold ;
		
			{
				reject = false;
				naccpt++;
				
				// compute new step based on maximum dynamical time scale
				// also include ideas from the PI controller to enhance the prediction control
				hnew = safe * dynmax * std::pow(prev_dynmax/dynmax, 0.2)  * std::pow(h / hold, 0.2)   ;
				
				prev_dynmax = dynmax; 
				
				KARMA_LOG(INFO) << "Accepting step: dynamical_time = " << dynmax << "\t h_{new} = " << hnew ;
				
				if (time+hnew > time_end)
					hnew = time_end-time ;
			
				
				// dont increase or decrease the step size more by than the allowed factor 
				if (hnew > h)
					hnew = KARMA::min<Real> (facr*h, hnew) ;
				else
					hnew = KARMA::max<Real> (facl*h, hnew) ;
				
				hnew = KARMA::min<Real>(hnew, hmax) ;
				
				// check if this is the last time step
				if (time_end - time < 1e-12)
					hnew = 0.0 ;
			}
			
			// write out the data to log file
			if (_CTX->rank == 0)
			{
				std::ofstream _file(delt_file.c_str(), ios::out | ios::app | ios::ate) ;
				if (_file.is_open())
				{
					_file << setprecision(14) << "\n" << time << "\t" << h << "\t" << dynmax << "\t" << hnew ;
					_file.flush() ;
					_file.close() ;
				}
			}
			
			Parallel::barrier() ;
			
			hold = h ;
			h = hnew ;
		}
		
		// set the context variables
		_CTX->delt = h ;
		_CTX->time = time ;
		
		interface->synchronize_physics_solution(*psol) ;
		
		interface->step_postprocess() ;

		output_current_solution(nstep, naccpt) ;
		
		Parallel::barrier() ;
		
		if (h  < 1e-14)
			break ;
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

