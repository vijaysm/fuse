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
#include "karma_wrappers.h"
#include "karma_vector.h"
#include "karma_utils.h"
#include "implicit_dirk_integrator.h"
#include "parallel.h"
#include "petsc_snes_utils.h"

#include <iostream>

using namespace std ;

ImplicitDIRKIntegrator::ImplicitDIRKIntegrator(RKType rktype, Vec &solin, PetscReal tstart,
		PetscReal tend, PetscReal deltinit) : 
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, false, 0, new PetscReal(1.0e-5), new PetscReal(1.0e-10))
{
	karma_assert (_scheme->rk_category() != ERK || _scheme->rk_category() != ERKEmbedded) ;
}

ImplicitDIRKIntegrator::ImplicitDIRKIntegrator(RKType rktype, Vec &solin, PetscReal tstart,
		PetscReal tend, PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) : 
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, false, itolerin, rtolerin, atolerin)
{
	karma_assert (_scheme->rk_category() != ERK || _scheme->rk_category() != ERKEmbedded) ;
}

ImplicitDIRKIntegrator::~ImplicitDIRKIntegrator()
{
	this->clear () ;
} 

PetscErrorCode ImplicitDIRKIntegrator::init ()
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

PetscErrorCode ImplicitDIRKIntegrator::clear ()
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

void ImplicitDIRKIntegrator::print_stats()
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

PetscErrorCode ImplicitDIRKIntegrator::core_integrator()
{
	PetscErrorCode 	ierr;
	SNESConvergedReason newton_convergence_reason;
	PetscInt its ;

	int stages = _scheme->n_stages() ;
	
	Real posneg = KARMAUtils::sign(1.0, time_end-time);
	
	ierr = PetscPrintf(KARMA_COMM_WORLD,"\n\t===Implicit Integration with %s scheme ===\n", _name.c_str()) ;
	
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
	KARMAVector stage_wrapper(Zi) ;
	
	// basic integration loop
	for(nstep=0; nstep < nmax; nstep++)
	{
		ierr = PetscPrintf(KARMA_COMM_WORLD,"\n%D\t%s : DelT = %G : Performing calculation for step %G -> %G", 
							nstep, _scheme->name().c_str(), h, time, time+h) ;

		// if (predict_solution)
		// {
			// // perform an explicit prediction. this can bring the initial guess closer
			// // also remember that it could possibly diverge if the time step size is too large.
			// ierr = VecAXPBY(sol, -1.0, 2.0, _curr_sol) ;	CHKERRQ(ierr);
			// psol->close() ;
		// }
		
		interface->step_preprocess() ;
		
		// NOTE: currently implemented only for DIRK schemes
		for (int istage = 0 ; istage < stages; istage++)
		{
			curr_stage = istage ;
			
			_CTX->time = time+h*C[istage] ;

#ifdef MASS_STAGE_SOLVE
			ierr = VecCopy(sol, _curr_sol) ;	CHKERRQ(ierr);
			pcurr_sol->close() ;
#else
			// ierr = VecSet(_curr_sol, 0.0) ;	CHKERRQ(ierr);
#endif
			interface->stage_preprocess() ;
			
			{
				// Compute Zi = \sum {j=1..i-1} {h*a_{ij}*k_j} ;
				// This is a constant vector subtracted from right hand side
				// for DIRK schemes
				ierr = VecSet(Zi, 0.0) ;	CHKERRQ(ierr);
				for (int jstage = 0 ; jstage < istage; jstage++)
				{
					if(fabs(A[istage][jstage]) > 1E-14)
					{
						ierr = VecAXPY(Zi, h*A[istage][jstage], k[jstage]) ;	CHKERRQ(ierr);
					} 
				}
				stage_wrapper.close() ;
				
#ifdef MASS_STAGE_SOLVE
				// _soln_guess = un + Zi
				ierr = VecAXPY(_curr_sol, 1.0, Zi) ;CHKERRQ(ierr);
				pcurr_sol->close() ;
				
				// Zi = M*un + \sum {j=1..i-1} {h*a_{ij}*f(Z_j)} ;
				ierr = interface->mass_multiply (*pcurr_sol, stage_wrapper) ;CHKERRQ(ierr);
#else
				ierr = VecCopy(Zi, _curr_sol) ;	CHKERRQ(ierr);
				pcurr_sol->close() ;
				
				// Zi = M*un + \sum {j=1..i-1} {h*a_{ij}*f(Z_j)} ;
				// TODO: if this works, then compute it only once per step
				// ierr = interface->mass_multiply (*psol, *_CTX->Work_2) ;CHKERRQ(ierr);
				
				// ierr = VecAXPY(Zi, -1.0, _CTX->Work_2->vec()) ;CHKERRQ(ierr);
				
				
				// _soln_guess = un + Zi
				// ierr = VecAXPY(Zi, 1.0, sol) ;CHKERRQ(ierr);
				// stage_wrapper.close() ;
				// M*Zi = \sum {j=1..i-1} {h*a_{ij}*f(Z_j)} ;
				// ierr = interface->mass_multiply (*pcurr_sol, stage_wrapper) ;CHKERRQ(ierr);
#endif

				// NOTE: Look at Hairer page: 128 for a better guess. Defintely see a good scope
				// for improvement here...
				
				// if (predict_solution)
				// {
					// // perform an explicit prediction. this can bring the initial guess closer
					// // also remember that it could possibly diverge if the time step size is too large.
					// ierr = VecAXPBY(sol, -1.0, 2.0, _curr_sol) ;	CHKERRQ(ierr);
					// psol->close() ;
				// }
				
				// well I got overzealous here. this idea requires that we store the solutions
				// for each stage after the nonlinear solve. since I dont do that, this is 
				// not possible. need to find an  alternate way to get a better guess.
				// if (istage > 2)
				// {
					// // see if we can interpolate the solution and find a better initial guess
					// // this is along the lines of an explicit prediction but take into account
					// // the variation in step size
					// // currently: curr_sol = un + \sum {j=1..i-1} {h*a_{ij}*k_j}
					// // we need curr_sol = un + c_{i}/(1-c_{i-1})*\sum {j=1..i-1} {h*a_{ij}*k_j}
					// // solve a least-square system to determine the optimal initial guess
					// // Hairer - STiff problems - Page 128
					// double **init_mat = new double[istage-1][istage-1] ;
					// double *init_rhs = new double[istage-1] ;
					// double *init_coeff = new double[istage-1] ;
					// for (unsigned int kstage=0; kstage < istage-1; kstage++)
					// {
						// init_mat[0][kstage] = C[kstage] ;
					// }
					// init_rhs[0] = C[istage] ;
					// for (unsigned int jstage=1; jstage < istage-1; jstage++)
					// {
						// for (unsigned int mstage=0; mstage < stages; mstage++)
							// for (unsigned int kstage=0; kstage < istage-1; kstage++)
								// init_mat[jstage][kstage] += 
					// }
				// }
			}
			
			// If there is an explicit first stage, then there is no nonlinear solve to be performed.
			if (fabs(C[istage]) > 1E-12 && fabs(A[istage][istage]) > 1e-12)
			{
				// solve the temporal nonlinear system
				ierr = SNESSolve(_snes, Zi, _curr_sol); CHKERRQ(ierr);
				
				ierr = SNESGetIterationNumber(_snes, &its);CHKERRQ(ierr);
				
				SNESGetConvergedReason(_snes, &newton_convergence_reason);
				
				// Stop newton iteration if no convergence
				if (newton_convergence_reason >= 0 || newton_convergence_reason == -5)
					newton_fail = false ;
				else // (newton_convergence_reason < 0)
				{
					ierr = PetscPrintf(KARMA_COMM_WORLD,"\nNewton iteration failed. Reason: ") ;
					
					print_snes_converged_reason(newton_convergence_reason) ;
					
					ierr = PetscPrintf(KARMA_COMM_WORLD,"\nStopping at Time {%G} : Stage{%D} : Newton{%D}", 
										time+h*C[istage], istage, its) ;
										
					newton_fail = true ;
				}
			}
			
			// if the nonlinear iteration did not succeed, no point computing rest of step
			// reduce time step and recompute solution
			// this is not expected during constant time-stepping since user is  required
			// to know the optimal step size to run the problem.
			if (newton_fail && adaptive)
				break ;
			
			// should also be called for explicit stages.
			// NOTE: if there are explicit stages in the middle, this will 
			// fail since _curr_sol might not be the right solution to use.
#ifdef MASS_STAGE_SOLVE
			{
				// find the SS residual and invert the mass matrix to get the istage function residual.
				ierr = Function(time+h*C[istage], _curr_sol, k[istage]);	CHKERRQ(ierr);
			}
#else
			{
				ierr = VecAXPY(_curr_sol, 1.0, sol) ;	CHKERRQ(ierr);
				pcurr_sol->close() ;
				
				// find the SS residual and invert the mass matrix to get the istage function residual.
				ierr = Function(time+h*C[istage], _curr_sol, k[istage]);	CHKERRQ(ierr);
			}
#endif

			interface->stage_postprocess() ;
		}
	
		// check if we stopped because newton iteration failed.
		if (newton_fail && adaptive)
		{
			newton_fail = false ;
			// we have not modified solution.
			// just compute new time step size
			h *= facl ;		// reduce step size by maximum allowable limit
			
			_CTX->delt = h ;
			KARMA_LOG(INFO) << "Rejecting step: Newton iteration failed.\th_{new} = " << h ;
				
			if (naccpt >= 1) nrejct++;
			continue ;
		}
		
		// find the new solution at the end of the step
		// the weighted numerical solution 
#ifdef MASS_STAGE_SOLVE
		{
			ierr = VecCopy(sol, _curr_sol) ;	CHKERRQ(ierr);
			pcurr_sol->close() ;
			
			// U_{n+1} = U_{n} + delt * sum{i=1..s} (b[i] * k_i)
			for (int istage = 0 ; istage < stages; istage++)
			{
				if (fabs(B[istage]) > 1E-12)
				{
					ierr = VecAXPY(sol, h*B[istage], k[istage]) ;	CHKERRQ(ierr);
				}
			}
			psol->close() ;
		}
#else
		{
			// Need to think this over; I still need to invert the mass matrix here
			// to obtain the solution; Check the equations.
			if (_scheme->is_FSAL())
			{
				// U_{n+1} = U_{n} + delt * sum{i=1..s} M^{-1}*(b[i] * f_i)
				// Since the scheme is FSAL, aij of the last row is itself the right
				// solution and so no need to recompute the solution.
				// ierr = VecAXPY(sol, 1.0, _curr_sol);	CHKERRQ(ierr);
				ierr = VecCopy(_curr_sol, sol);	CHKERRQ(ierr);
			}
			else
			{
				// M*U_{n+1} = M*U_{n} + delt * sum{i=1..s} (b[i] * f_i)
				
				// pcurr_sol = M*U_{n}
				ierr = KARMAInterface::get_interface()->mass_multiply (*psol, *pcurr_sol) ;CHKERRQ(ierr);
				
				// So now: pcurr_sol = M*U_{n} + delt * sum{i=1..s} (b[i] * f_i)
				for (int istage = 0 ; istage < stages; istage++)
				{
					if (fabs(B[istage]) > 1E-12)
					{
						ierr = VecAXPY(_curr_sol, h*B[istage], k[istage]) ;	CHKERRQ(ierr);
					}
				}
				pcurr_sol->close() ;
			
				// Now let us invert the mass matrix finally to obtain
				// U_{n+1} = U_{n} + delt * sum{i=1..s} M^{-1}*(b[i] * f_i)
				ierr = KARMAInterface::get_interface()->mass_solve (*pcurr_sol, *psol) ;CHKERRQ(ierr);
			}
		}
#endif
		
		// step accepted
		time += posneg*h;
		
		// set the context variables
		_CTX->delt = h ;
		_CTX->time = time ;
		
		if (calculate_dyn_time || adaptive)
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
		
		if (adaptive)
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
		
			// if the rate of solution change is high,  the current
			// solution has a lot of error in it. 
			// recompute it with smaller step size
			// if (dynmax > facr)
			// {
				// // step rejected
				// hnew = facr*h ;
				// // hnew = KARMA::min<Real>(h * facl, safe * dynmax * h / hold );
				// reject = true;
				// if (naccpt >= 1) nrejct++;
				
				// KARMA_LOG(INFO) << "Restricting step size: dynamical_time = " << dynmax << "\th_{new} = " << hnew ;
				
				// time -= posneg*h;
				
				// // copy the original U_{n} solution to start the new step
				// ierr = VecCopy(_curr_sol, sol) ;	pcurr_sol->close() ;
			// }
			// else
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
				if (time_end - time < 1e-13)
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
		else
		{
			// constant step size; all are accepted.
			naccpt++;
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

