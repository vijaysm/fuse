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
#include "parallel.h"
#include "explicit_rk_integrator.h"

#include <iostream>

ExplicitRKIntegrator::ExplicitRKIntegrator(RKType rktype, Vec &solin,
		PetscReal tstart, PetscReal tend, PetscReal deltinit) : 
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, false, 0, new PetscReal(1.0e-5), new PetscReal(1.0e-5))
{	
	karma_assert (_scheme->rk_category() == ERK || _scheme->rk_category() == ERKEmbedded) ;
}

ExplicitRKIntegrator::ExplicitRKIntegrator(RKType rktype, Vec &solin, PetscReal tstart,
		PetscReal tend, PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) : 
		TemporalIntegrator(rktype, solin, tstart, tend, deltinit, false, itolerin, rtolerin, atolerin)
{
	karma_assert (_scheme->rk_category() == ERK || _scheme->rk_category() == ERKEmbedded) ;
}

ExplicitRKIntegrator::~ExplicitRKIntegrator()
{
	this->clear () ;
}

PetscErrorCode ExplicitRKIntegrator::init ()
{
	PetscErrorCode 	ierr = 0 ;
	
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

PetscErrorCode ExplicitRKIntegrator::clear ()
{
	PetscErrorCode 	ierr = 0 ;
	
	if(_initialized)
	{
		ierr = VecDestroy(&yy1) ;	CHKERRQ(ierr);
		ierr = VecDestroy(&ysti) ;	CHKERRQ(ierr);
		
		TemporalIntegrator::clear () ;
	}
	
	return ierr ;
}

void ExplicitRKIntegrator::print_stats()
{
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------");
	PetscPrintf(KARMA_COMM_WORLD,"\n| %s Integrator performance |", _scheme->name().c_str());
	PetscPrintf(KARMA_COMM_WORLD,"\n------------------------------------\n");
	
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of steps := %D", nstep);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of accepted steps := %D", naccpt);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of rejected steps := %D", nrejct);
	PetscPrintf(KARMA_COMM_WORLD,"\nNumber of function evaluations := %D", nfcn);
	
	PetscPrintf(KARMA_COMM_WORLD,"\n") ;
}

PetscErrorCode ExplicitRKIntegrator::Function(PetscReal dtime, Vec sol, Vec fval)
{
	return TemporalIntegrator::Function(dtime, sol, fval) ;
}

PetscErrorCode ExplicitRKIntegrator::core_integrator()
{
	PetscErrorCode 	ierr=0;

	int stages = _scheme->n_stages() ;
	
	double posneg = KARMAUtils::sign(1.0, time_end-time);

	ierr = PetscPrintf(KARMA_COMM_WORLD,"\n\t===Explicit Integration with %s scheme ===\n", _name.c_str()) ;
	
	// initial preparations
	double **A = _scheme->Acoeff() ;
	double *B = _scheme->Bcoeff() ;
	double *C = _scheme->Ccoeff() ;
	
	// set the context variables
	_CTX->time = time ;
	_CTX->delt = h ;
	
	KARMAVector *pcurr_sol = new KARMAVector(yy1) ;
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	std::vector<std::vector<Real> > dyntime_all ;
	
	Real prev_dynmax = 1.0 ;
	PetscBool calc_dyn_time ;
	PetscOptionsHasName(PETSC_NULL, "-dyn_time", &calc_dyn_time) ;
	bool calculate_dyn_time = calc_dyn_time ;
	
	PetscBool predict_soln_flg ;
	PetscOptionsHasName(PETSC_NULL, "-predict", &predict_soln_flg) ;
	bool predict_solution = predict_soln_flg ;

	// if there is thermal feedback, write the total power as a function of time
	std::string delt_file = "output/delt.txt" ;
	if (adaptive)
	{
		if (_CTX->rank == 0)
		{
			std::ofstream _file(delt_file.c_str(), std::ios::out) ;
			if (_file.is_open())
			{
				_file << "Time\tCurrent-Delt\tDynamical-Time\tPredicted-Delt\n" ;
				_file.close() ;
			}
		}
		
	}
	
	// basic integration loop
	for(nstep=0; nstep < nmax; nstep++)
	{
		ierr = PetscPrintf(KARMA_COMM_WORLD,"\n%s : DelT = %G : Performing calculation for step %G -> %G\n", _scheme->name().c_str(), h, time, time+h) ;
		
		interface->step_preprocess() ;
		
		for (int istage = 0 ; istage < stages; istage++)
		{
			curr_stage = istage ;
			
			_CTX->time = time+h*C[istage] ;

			interface->stage_preprocess() ;

			ierr = VecCopy(sol, yy1) ;	CHKERRQ(ierr);
			for (int jstage = 0 ; jstage < istage; jstage++)
			{
				if(fabs(A[istage][jstage]) > 1E-12)
				{
					ierr = VecAXPY(yy1, h*A[istage][jstage], k[jstage]) ;	CHKERRQ(ierr);
				}
			}
			pcurr_sol->close() ;
			
			ierr = Function(time+h*C[istage], yy1, k[istage]);	CHKERRQ(ierr);
			
			interface->stage_postprocess() ;
		}
			
		// Find out the weighted true numerical solution 
		ierr = VecCopy(sol, yy1) ;	CHKERRQ(ierr);	// yy1 contains the old solution
		for (int istage = 0 ; istage < stages; istage++)
		{
			if (fabs(B[istage]) > 1E-12)
			{
				ierr = VecAXPY(sol, h*B[istage], k[istage]) ;	CHKERRQ(ierr);
			}
		}
		psol->close() ;
		
		// step accepted
		naccpt++;
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
		
			// comment out for now ; this could be used later if there is a generic way to find stiffness and control it.
			/*
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
			*/
	
			// detect stiffness and then adapt according to dynamical step size
			{
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
				std::ofstream _file(delt_file.c_str(), std::ios::out | std::ios::app | std::ios::ate) ;
				if (_file.is_open())
				{
					_file << std::setprecision(14) << "\n" << time << "\t" << h << "\t" << dynmax << "\t" << hnew ;
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
	
		interface->step_postprocess() ;

		output_current_solution(nstep, naccpt) ;
		
		Parallel::barrier() ;
		
		if (h  < 1e-12)
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
	
	PetscFunctionReturn(ierr) ;
}
