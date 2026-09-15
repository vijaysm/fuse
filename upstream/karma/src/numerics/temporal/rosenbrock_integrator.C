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
#include "rosenbrock.h"
#include "rosenbrock_integrator.h"
#include "parallel.h"
#include "petsc_ksp_utils.h"

#include <iostream>

using namespace std ;

RosenbrockIntegrator::RosenbrockIntegrator(RKType rktype, Vec &solin, PetscReal tstart,
		PetscReal tend, PetscReal deltinit) : 
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, false, 0, new PetscReal(1.0e-5), new PetscReal(1.0e-10))
{
	dfdt = NULL ;
}

RosenbrockIntegrator::RosenbrockIntegrator(RKType rktype, Vec &solin, PetscReal tstart,
		PetscReal tend, PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) : 
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, false, itolerin, rtolerin, atolerin)
{
	dfdt = NULL ;
}

RosenbrockIntegrator::~RosenbrockIntegrator()
{
	this->clear () ;
} 

PetscErrorCode RosenbrockIntegrator::init ()
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

PetscErrorCode RosenbrockIntegrator::clear ()
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

void RosenbrockIntegrator::print_stats()
{
	PetscPrintf(KARMA_COMM_WORLD,"\n\n------------------------------------");
	PetscPrintf(KARMA_COMM_WORLD,"\n| %s Integrator performance |", _name.c_str());
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------\n");
	
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of steps := %D", nstep);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of accepted steps := %D", naccpt);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of rejected steps := %D", nrejct);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of function evaluations := %D", nfcn);
	
	PetscPrintf(KARMA_COMM_WORLD,"\n") ;
}

PetscErrorCode __karma_form_temporal_jacobian(const KARMAVector& solution, const Real factor, KARMAMatrix& jacobian)
{
	PetscErrorCode ierr ;
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	
	// SS Jacobian or an approximation of it
	// Compute once per stage
	ierr = interface->jacobian_matrix (solution, jacobian);	CHKERRQ(ierr);
	
	// J(transient) = M - delt * aii * J(SS)
	ierr = interface->mass_matrix_subtract (jacobian, factor) ;CHKERRQ(ierr);
	
	return ierr ;
}

PetscErrorCode RosenbrockIntegrator::core_integrator()
{
	PetscErrorCode 	ierr;
	KSPConvergedReason convergence_reason;
	PetscInt its ;
	
	// TODO: this should be const pointer
	Rosenbrock* ros_scheme = dynamic_cast<Rosenbrock*>(_scheme);
	
	unsigned int stages = ros_scheme->n_stages() ;
	
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
	ierr = KSPSetTolerances(_ksp, rtoli, atoli, PETSC_DEFAULT, maxit);CHKERRQ(ierr);
	
	double **A = ros_scheme->Acoeff() ;
	double **G = ros_scheme->Gcoeff() ;
	double *B = ros_scheme->Bcoeff() ;
	double *C = ros_scheme->Ccoeff() ;
	
	std::vector<double> gamma_hat ;
	gamma_hat.resize(ros_scheme->n_stages()) ;
	
	for (unsigned int istage=0; istage < gamma_hat.size(); istage++)
	{
		gamma_hat[istage] = 1.0 ;	// one for the diagonal
		for (unsigned int jstage=0; jstage < istage; jstage++)
			gamma_hat[istage] += G[istage][jstage] ;
		gamma_hat[istage] *= G[istage][istage]*G[istage][istage] ;	// multiply by gamma
	}
	
	// Set the context in SNES so that integrator can be accessed from wrapper function
	ierr = SNESSetApplicationContext(_snes, this) ;CHKERRQ(ierr);

	Real prev_dynmax = 1.0 ;
	bool reject = false, krylov_fail = false ;
	
	const bool is_non_autonomous_term = (dfdt != NULL) ;
	Vec non_autonomous_vec ;
	KARMAVector *p_non_autonomous_vec ;
	// if (is_non_autonomous_term)
	{
		ierr = VecDuplicate(Zi, &non_autonomous_vec) ;CHKERRQ(ierr);
		p_non_autonomous_vec = new KARMAVector(non_autonomous_vec) ;
	}
	
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
	
	Mat jac, prec ;
	ierr = SNESGetJacobian(_snes, &jac, &prec, PETSC_NULL, PETSC_NULL) ;CHKERRQ(ierr);
	
	MatStructure msflag = (jac == prec ? SAME_NONZERO_PATTERN : DIFFERENT_NONZERO_PATTERN) ;
	ierr = KSPSetOperators(_ksp, jac, prec, msflag) ;
	
	KARMAMatrix Jac_global(prec) ;
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	std::vector<std::vector<Real> > dyntime_all ;
	KARMAVector stage_wrapper(Zi) ;
	
	// basic integration loop
	for(nstep=0; nstep < nmax; nstep++)
	{
		ierr = PetscPrintf(KARMA_COMM_WORLD,"\n%D\t%s : DelT = %G : Performing calculation for step %G -> %G", 
							nstep, ros_scheme->name().c_str(), h, time, time+h) ;

		interface->step_preprocess() ;
		
		ierr = __karma_form_temporal_jacobian(*psol, -(h * G[0][0]), Jac_global) ;	CHKERRQ(ierr);
		// ierr = interface->jacobian_matrix (X_global, Jac_global);	CHKERRQ(ierr);
		
		if (is_non_autonomous_term)
		{
			ierr = (*dfdt)(time, *psol, *p_non_autonomous_vec);	CHKERRQ(ierr);
			// ierr = VecScale(non_autonomous_vec, h*gamma_hat[istage]) ;	CHKERRQ(ierr);
		}
		else
		{
			const Real time_perturbation = h*1e-6 ;
			_CTX->time = time ;
			
			// obtain the base residual
			// TODO: check if we can avoid recomputation of this residual
			ierr = interface->system_residual (*psol, *p_non_autonomous_vec);	CHKERRQ(ierr);
			
			_CTX->time = time+time_perturbation ;
			// no analytical form of DFDT. So perturb the residual and get the vector
			ierr = interface->system_residual (*psol, *_CTX->Work_2);	CHKERRQ(ierr);
			
			// now compute DFDT
			ierr = VecAXPBY(non_autonomous_vec, 1.0/time_perturbation, -1.0/time_perturbation, _CTX->Work_2->vec()) ;	CHKERRQ(ierr);
		}
		
		// NOTE: currently implemented only for DIRK schemes
		for (unsigned int istage = 0 ; istage < stages; istage++)
		{
			curr_stage = istage ;
			
			_CTX->time = time+h*C[istage] ;

			// if the diagonal Gamma coefficients are different, recompute the Jaccobian
			if (istage > 0 && fabs(G[istage][istage] - G[istage-1][istage-1]) > 1e-10)
			{
				// TODO: Might be inefficient to recompute the SS jacobian
				// Instead, use the fact that the old Jacobian = M - h*g(i-1,i-1)*J and that
				// we now need Jacobian = M - h*g(i,i)*J.
				ierr = __karma_form_temporal_jacobian(*psol, -(h * G[istage][istage]), Jac_global) ;	CHKERRQ(ierr);
				// ierr = interface->jacobian_matrix (X_global, Jac_global);	CHKERRQ(ierr);
			}
			
			interface->stage_preprocess() ;
			
			{
				// Compute Zi = \sum {j=1..i-1} {h*a_{ij}*k_j} ;
				// This is a constant vector subtracted from right hand side
				// for DIRK schemes
				ierr = VecSet(Zi, 0.0) ;	CHKERRQ(ierr);
				ierr = VecCopy(sol, _curr_sol) ;	CHKERRQ(ierr);

				for (unsigned int jstage = 0 ; jstage < istage; jstage++)
				{
					ierr = VecAXPY(Zi, G[istage][jstage], k[jstage]) ;	CHKERRQ(ierr);
					ierr = VecAXPY(_curr_sol, h*A[istage][jstage], k[jstage]) ;	CHKERRQ(ierr);
				}

				if (istage > 0)
				{
					// perform SS jacobian mat-vec on Zi and store in Work_2
					ierr = interface->mass_multiply (stage_wrapper, *_CTX->Work_2) ;CHKERRQ(ierr);
				}

				// then evaluate the function 'f' at _curr_sol 
				ierr = interface->system_residual (*pcurr_sol, stage_wrapper);	CHKERRQ(ierr);

				if (istage > 0)
				{
					// add the contributions together
					ierr = VecAYPX(Zi, G[istage][istage], _CTX->Work_2->vec()) ;	CHKERRQ(ierr);
				}
				else
				{
					// just scale the vector
					ierr = VecScale(Zi, G[istage][istage]) ;	CHKERRQ(ierr);
				}

				if (is_non_autonomous_term)
				{
					// add the non-autonomous form contribution
					ierr = VecAXPY(Zi, h*gamma_hat[istage], non_autonomous_vec) ;	CHKERRQ(ierr);
				}
			}

			// now solve (M - delt*gamma*J) on Zi (rhs) to get the new stage vector ki
			{
				if (istage > 0)
				{
					// a bad initial guess, but will do for now
					ierr = VecCopy(k[istage-1], k[istage]) ;CHKERRQ(ierr);
				}
				
				// solve the temporal nonlinear system
				ierr = KSPSolve(_ksp, Zi, k[istage]); CHKERRQ(ierr);
		
				ierr = KSPGetIterationNumber(_ksp, &its);CHKERRQ(ierr);
				
				KSPGetConvergedReason(_ksp, &convergence_reason);
				
				// Stop newton iteration if no convergence
				if (convergence_reason >= 0 || convergence_reason == -5)
					krylov_fail = false ;
				else // (convergence_reason < 0)
				{
					ierr = PetscPrintf(KARMA_COMM_WORLD,"\nLinear iteration failed. Reason: ") ;
					
					print_ksp_converged_reason(convergence_reason) ;
					
					ierr = PetscPrintf(KARMA_COMM_WORLD,"\nStopping at Time {%G} : Stage{%D}", 
										time+h*C[istage], istage) ;
										
					krylov_fail = true ;
				}
			}
			
			// do not invert mass afterwards; store in k[istage] and move on to next stage
			
			
			// if the nonlinear iteration did not succeed, no point computing rest of step
			// reduce time step and recompute solution
			// this is not expected during constant time-stepping since user is  required
			// to know the optimal step size to run the problem.
			if (krylov_fail && adaptive)
				break ;
			
			interface->stage_postprocess() ;
		}
	
		// check if we stopped because newton iteration failed.
		if (krylov_fail && adaptive)
		{
			krylov_fail = false ;
			// we have not modified solution.
			// just compute new time step size
			h *= facl ;		// reduce step size by maximum allowable limit
			
			_CTX->delt = h ;
			KARMA_LOG(INFO) << "Rejecting step: Krylov iteration failed.\th_{new} = " << h ;
				
			if (naccpt >= 1) nrejct++;
			continue ;
		}
		
		// find the new solution at the end of the step
		// the weighted numerical solution 
		{
			// store previous solution in _curr_sol in case we want to either do
			// explicit prediction or compute dynamical scales
			ierr = VecCopy(sol, _curr_sol) ;	CHKERRQ(ierr);
			
			// U_{n+1} = U_{n} + delt * sum{i=1..s} (b[i] * k_i)
			for (unsigned int istage = 0 ; istage < stages; istage++)
			{
				if (fabs(B[istage]) > 1E-12)
				{
					ierr = VecAXPY(sol, h*B[istage], k[istage]) ;	CHKERRQ(ierr);
				}
			}
		}
		
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
			// Real facerr = std::pow(atoli/local_error, 1.0/(ros_scheme->n_order())) * 
						  // std::pow(prev_local_error/local_error, 1.0/(ros_scheme->n_order())) *
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
	
	// if (is_non_autonomous_term)
	{
		delete p_non_autonomous_vec ;
		ierr = VecDestroy(&non_autonomous_vec) ;CHKERRQ(ierr);
	}
	
	// replace the reference to the integrator with a NULL pointer
	ierr = SNESSetApplicationContext(_snes, PETSC_NULL) ;CHKERRQ(ierr);
	
	PetscFunctionReturn(0) ;
}

