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

#include "rk.h"

// This blows. Since we have our type traits defined in the
// individual source files, unless we include them all,
// the specialized RKMethod traits are unavailable.
// NOTE: Reorganize by creating a traits class and just include that.
#include "forward_euler.h"
#include "kutta22.h"
#include "kutta44.h"
#include "dopri45.h"
#include "backward_euler.h"
#include "crank_nicolson.h"
#include "sdirk22.h"
#include "sdirk23.h"
#include "sdirk34.h"
#include "sdirk332.h"
#include "sdirk543.h"
#include "sdirk643.h"
#include "trbe.h"
#include "sdirk33.h"
#include "sdirk45.h"
#include "ros22.h"
#include "grk4a.h"
#include "grk4t.h"
#include "grk4sn.h"
#include "ros3p.h"
#include "rodasp.h"


#include "temporal_integrator.h"
#include "dopri45_integrator.h"
#include "explicit_rk_integrator.h"
#include "implicit_dirk_integrator.h"
#include "embedded_dirk_integrator.h"
#include "rosenbrock_integrator.h"

#include "karma_interface.h"
#include "karma_context.h"
#include "karma_wrappers.h"

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::constructor"
TemporalIntegrator::TemporalIntegrator(const KARMA::TEMPORAL::RKType rktype, Vec &solin, PetscReal tstart, PetscReal tend,
		PetscReal deltinit, bool isadapted, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) : 
	h(deltinit), time(tstart), time_old(tstart), hold(deltinit), time_end(tend), itoler(itolerin), rtoler(rtolerin), atoler(atolerin), 
	rtolerNULL(false), atolerNULL(false), nfcn(0), njac(0), nmass(0), nstep(0), naccpt(0), nrejct(0), _initialized(false)
{
	KARMAInterface* interface = KARMAInterface::get_interface  () ;
	_CTX = interface->get_context() ;
	
	_scheme = RK::build (rktype) ;
	_name = _scheme->name() ;
	
    // rtoler, the relative tolerance of the integration
	if (!rtoler)
	{
		itoler = 0;
		rtoler = new PetscReal;
		*rtoler = 1.0e-5;
		rtolerNULL = true;
	}
	
	// atoler, the absolute tolerance of the integration
	if (!atoler)
	{
		itoler = 0;
		atoler = new PetscReal;
		*atoler = 1.0e-10;
		atolerNULL = true;
	}
	
	adaptive_initiated = false ;
	PetscBool flg ;
	PetscOptionsHasName(PETSC_NULL, "-adaptive", &flg) ;
	adaptive = (isadapted || flg) ;
	
	// check for other adaptive time-stepping properties
	if (adaptive)
	{
		PetscScalar dtmax = 0.01, facrr = 4.0, facll = 0.2, safety = 0.8 ;
		PetscOptionsGetScalar(PETSC_NULL,"-safety",&safety,PETSC_NULL);
		PetscOptionsGetScalar(PETSC_NULL,"-dtmax",&dtmax,PETSC_NULL);
		PetscOptionsGetScalar(PETSC_NULL,"-facr",&facrr,PETSC_NULL);
		PetscOptionsGetScalar(PETSC_NULL,"-facl",&facll,PETSC_NULL);

		this->set_adaptive_params(dtmax, 100000, safety, facll, facrr) ;
	}

	_io_type = VTK ;
	PetscOptionsHasName(PETSC_NULL, "-io_transient", &flg) ;
	_io_soln = flg ;
	
	if (_io_soln)
	{
		_io_frequency = 20 ;	// by default, output every 20 steps
		// get the frequency for writing out data
		PetscOptionsGetInt(PETSC_NULL, "-io_frequency", &_io_frequency, &flg) ;
	}
	
	VecDuplicate(solin, &sol) ;
	VecCopy(solin, sol) ;
	
	psol = new KARMAVector(sol) ;
	
	// set some other parameters
	if (deltinit < 1E-14)
	{
		h = 1E-14 ;
		hold = h ;
	}
	
	double posneg = KARMAUtils::sign(1.0, time_end-time);
	
	if(posneg > 0)
	{
		nmax = static_cast<int>((tend-tstart)/deltinit) ;
		double test = nmax*deltinit -(tend-tstart);
		KARMA_LOG(DEBUG1) << "steps = " << nmax << "\t" << test << std::endl;
		if(fabs(test) > 1e-12)
		{
			if(test < 0.0)	// miscalculated: num_steps > true_steps 
			{
				nmax = nmax+1;
				double second_test = nmax*deltinit -(tend-tstart);
				if(fabs(second_test)>1e-12)
				{
					KARMA_LOG(DEBUG1) << "Wrong number of steps - num_steps > true_steps :: nmax = " << std::setprecision (15) << nmax << std::endl;
					KARMA_LOG(DEBUG1) << std::setprecision (15) << "FINAL_TIME = " << tend << "\tSTART_TIME = " << tstart << "\tDELT = " << deltinit <<"\tNMAX = "<< nmax << std::endl;
					exit (1);
				}
			}
			
			if(test > 0.0)	// miscalculated: num_steps < true_steps
			{
				nmax = nmax-1;
				double second_test = nmax*deltinit -(tend-tstart);
				if(fabs(second_test)>1e-12)
				{
					KARMA_LOG(DEBUG1) << "Wrong number of steps - num_steps < true_steps :: nmax = " << std::setprecision (15) << nmax << std::endl;
					KARMA_LOG(DEBUG1) << std::setprecision (15) << "FINAL_TIME = " << tend << "\tSTART_TIME = " << tstart << "\tDELT = " << deltinit <<"\tNMAX = "<< nmax << std::endl;
					exit (1);
				}
			}
		}
	}
	else
	{
		nmax = static_cast<int>(-(tend-tstart)/deltinit) ;
	}	

    _initialized = false ;
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::destructor"
TemporalIntegrator::~TemporalIntegrator()
{
	this->clear () ;
	
	delete _scheme ;
	
	delete psol ;
	
	PetscErrorCode ierr = VecDestroy(&sol) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
}

#define INTEGRATOR_BUILD_CASE(N, adaptive_solution) case N:	\
			{	\
			if (adaptive_solution){	\
				AutoPtr<TemporalIntegrator> integrator(new RKMethodTraits<N>::AdaptiveIntegratorType(rktype, solin, tstart, tend, \
					deltinit, itolerin, rtolerin, atolerin));	\
				return integrator ;		\
			}	\
			else {	\
				AutoPtr<TemporalIntegrator> integrator(new RKMethodTraits<N>::StaticIntegratorType(rktype, solin, tstart, tend, 	\
					deltinit, itolerin, rtolerin, atolerin));	\
				return integrator ;	\
			}	\
		}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::build"
AutoPtr<TemporalIntegrator> TemporalIntegrator::build(KARMA::TEMPORAL::RKType rktype, Vec &solin, PetscReal tstart, PetscReal tend,
		PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) 
{
	PetscBool adaptive_solution ;
	PetscOptionsHasName(PETSC_NULL, "-adaptive", &adaptive_solution) ;

	switch(rktype)
	{
		INTEGRATOR_BUILD_CASE(0, adaptive_solution) ;	// KARMA::TEMPORAL::FE
		INTEGRATOR_BUILD_CASE(1, adaptive_solution) ;	// KARMA::TEMPORAL::KUTTA22
		INTEGRATOR_BUILD_CASE(2, adaptive_solution) ;	// KARMA::TEMPORAL::KUTTA44
		
		// unfortunately we need to specialize for DOPRI45 alone since the adaptive
		// integrator has a different argument list. Change ??
		case KARMA::TEMPORAL::DOPRI45:
		{
			if (adaptive_solution)
			{
				AutoPtr<TemporalIntegrator> integrator(new RKMethodTraits<KARMA::TEMPORAL::DOPRI45>::AdaptiveIntegratorType(solin, tstart, tend, 
						deltinit, itolerin, rtolerin, atolerin));
				return integrator ;
			}
			else
			{
				AutoPtr<TemporalIntegrator> integrator(new RKMethodTraits<KARMA::TEMPORAL::DOPRI45>::StaticIntegratorType(rktype, solin, tstart, tend, 
					deltinit, itolerin, rtolerin, atolerin));
				return integrator ;
			}
		}
		
		INTEGRATOR_BUILD_CASE(4, adaptive_solution) ;	// KARMA::TEMPORAL::BE
		INTEGRATOR_BUILD_CASE(5, adaptive_solution) ;	// KARMA::TEMPORAL::CN
		INTEGRATOR_BUILD_CASE(6, adaptive_solution) ;	// KARMA::TEMPORAL::SDIRK22
		INTEGRATOR_BUILD_CASE(7, adaptive_solution) ;	// KARMA::TEMPORAL::SDIRK23
		INTEGRATOR_BUILD_CASE(8, adaptive_solution) ;	// KARMA::TEMPORAL::SDIRK34
		INTEGRATOR_BUILD_CASE(9, adaptive_solution) ;	// KARMA::TEMPORAL::SDIRK332
		INTEGRATOR_BUILD_CASE(10, adaptive_solution) ;	// KARMA::TEMPORAL::SDIRK543
		INTEGRATOR_BUILD_CASE(11, adaptive_solution) ;	// KARMA::TEMPORAL::SDIRK643
		INTEGRATOR_BUILD_CASE(12, adaptive_solution) ;	// KARMA::TEMPORAL::TRBE
		INTEGRATOR_BUILD_CASE(13, adaptive_solution) ;	// KARMA::TEMPORAL::SDIRK33
		INTEGRATOR_BUILD_CASE(14, adaptive_solution) ;	// KARMA::TEMPORAL::SDIRK45
		INTEGRATOR_BUILD_CASE(15, adaptive_solution) ;	// KARMA::TEMPORAL::ROS22
		INTEGRATOR_BUILD_CASE(16, adaptive_solution) ;	// KARMA::TEMPORAL::GRK4A
		INTEGRATOR_BUILD_CASE(17, adaptive_solution) ;	// KARMA::TEMPORAL::GRK4T
		INTEGRATOR_BUILD_CASE(18, adaptive_solution) ;	// KARMA::TEMPORAL::GRK4Sn
		INTEGRATOR_BUILD_CASE(19, adaptive_solution) ;	// KARMA::TEMPORAL::ROS3P
		INTEGRATOR_BUILD_CASE(20, adaptive_solution) ;	// KARMA::TEMPORAL::RODASP
		
		
		
		//case KARMA::TEMPORAL::FE:
		// case KARMA::TEMPORAL::KUTTA22:
		// case KARMA::TEMPORAL::KUTTA44:
		// case KARMA::TEMPORAL::BE:
		// case KARMA::TEMPORAL::CN:
		// case KARMA::TEMPORAL::SDIRK22:
		// case KARMA::TEMPORAL::SDIRK23:
		// case KARMA::TEMPORAL::SDIRK33:
		// case KARMA::TEMPORAL::SDIRK34:
		// case KARMA::TEMPORAL::SDIRK45:
		// case KARMA::TEMPORAL::ROS22:
		// case KARMA::TEMPORAL::GRK4A:
		// case KARMA::TEMPORAL::GRK4T:
		// case KARMA::TEMPORAL::GRK4Sn:
		// case KARMA::TEMPORAL::ROS3P:
		// case KARMA::TEMPORAL::RODASP:
		// case KARMA::TEMPORAL::TRBE:
		// case KARMA::TEMPORAL::SDIRK332:
		// case KARMA::TEMPORAL::SDIRK543:
		// case KARMA::TEMPORAL::SDIRK643:
		// {
			// if (adaptive_solution)
			// {
				// AutoPtr<TemporalIntegrator> integrator(new RKMethodTraits<rktype>::AdaptiveIntegratorType(rktype, solin, tstart, tend, 
					// deltinit, itolerin, rtolerin, atolerin));
				// return integrator ;
			// }
			// else
			// {
				// AutoPtr<TemporalIntegrator> integrator(new RKMethodTraits<rktype>::StaticIntegratorType(rktype, solin, tstart, tend, 
					// deltinit, itolerin, rtolerin, atolerin));
				// return integrator ;
			// }
		// }
		default:
		{
			karma_exception( "Invalid RKType specified. Check inputs." ) ;
		}
	}
}

Vec& build_vector()
{
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::set_adaptive_params"
void TemporalIntegrator::set_adaptive_params(PetscReal deltmax, int nmaxsteps, PetscReal safetyfac, PetscReal faclin, PetscReal facrin)
{
	hmax = deltmax ;
	nmax = nmaxsteps ;
	safe = safetyfac ;
	facl = faclin ;
	facr = facrin ;
	
	// -------- maximal step size : Atleast 5 steps
	if (hmax == 0.0) hmax = (time_end - time)/5 ;
	
	// -------- nmax--maximal number of steps
	if (nmax == 0) nmax = 100000;
	if (nmax <= 0) {
		PetscPrintf(KARMA_COMM_WORLD, "\nWrong input, nmax = %D\n", nmax) ;
		throw -1;
	}
	
	uround = 1e-15 ;
	// -------- uround--smallest number satisfying 1.0 + uround > 1.0
	if (uround == 0.0) uround = 1.0e-16;
	if ((uround <= 1.0e-19) || (uround >= 1.0)) {
		PetscPrintf(KARMA_COMM_WORLD, "\nCoefficients have 20 digits, uround = %E\n", uround) ;
		throw -1;
	}
	
	
	// --------- safe--safety factor in step size prediction
	if (safe < 1e-10) safe = 0.9;
	if ((safe <= 1e-6) || (safe > 1.0)) {
		PetscPrintf(KARMA_COMM_WORLD, "\nCurious input for safety factor, safe = %G\n", safe) ;
		throw -1;
	}
	
	// facl, facr, parameters for step size selection
	if (facl < 1e-10) facl = 0.2;
	if (facr < 1e-10) facr = 4.0;
	
	adaptive_initiated = true ;
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::init"
PetscErrorCode TemporalIntegrator::init ()
{
	PetscErrorCode ierr = 0 ;
	
	if (!_initialized)
	{
		// Zi is the stage solution : 
		//		this could be the result of nonlinear iteration for an implicit stage 
		//		the solution value after one-step calculation in an explicit stage
		ierr = VecDuplicate(sol, &Zi) ;	CHKERRQ(ierr);
		ierr = VecCopy(sol, Zi) ;	CHKERRQ(ierr);

		int nstage = _scheme->n_stages() ;
			
		// Use PETSc's VecDuplicateVecs since we store "k" vectors as an array
		ierr = VecDuplicateVecs(sol, nstage, &k) ;	CHKERRQ(ierr);
		
		// Reset the values of vectors to zero.
		for (int i=0; i<nstage; i++)
		{
			ierr = VecSet(k[i], 0.0) ;	CHKERRQ(ierr);
		}
		
		KARMAInterface* interface = KARMAInterface::get_interface () ;
	
		interface->set_shell_matrix_context((void*)this) ;
		
		PetscBool found ;
		ierr = PetscOptionsGetReal("integrator_", "-rtol", rtoler, &found) ;
		
		ierr = PetscOptionsGetReal("integrator_", "-atol", atoler, &found) ;
		
		// *atoler = KARMAPackage::get_option("-integrator_atol", *atoler) ;
		
		_initialized = true ;
		
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::clear"
PetscErrorCode TemporalIntegrator::clear ()
{	
	PetscErrorCode ierr = 0 ;
	
	if(_initialized)
	{	
		ierr = VecDestroy(&Zi) ;	CHKERRQ(ierr);
				
		int nstage = _scheme->n_stages() ;
		
		// Use PETSc's VecDestroyVecs since we store "k" vectors as an array
		ierr = VecDestroyVecs(nstage, &k) ;	CHKERRQ(ierr);
		
		if (rtolerNULL) delete rtoler;
		if (atolerNULL) delete atoler;
		
		_CTX = NULL ;
	
		_initialized = false ; 
	}
	
	PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::integrate"
PetscErrorCode TemporalIntegrator::integrate(bool stats)
{
	PetscErrorCode icomp = core_integrator();

	if (icomp != 0)
	{
		PetscPrintf(KARMA_COMM_WORLD,"\n\nALERT::") ;
		PetscPrintf(KARMA_COMM_WORLD,"\n====Temporal Integration with %s Method failed====\n\n", _name.c_str()) ;
		return icomp;
	}

	// print final solution
	if (stats)
		print_stats () ;

	return icomp;
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::Function"
PetscErrorCode TemporalIntegrator::Function(PetscReal dtime, Vec fsol, Vec fval)
{
	PetscErrorCode 	ierr;
	
	_CTX->time = dtime ;
	
	// create a wrapper around the output vector
	KARMAVector X(fsol), R(fval);
	
#ifdef MASS_STAGE_SOLVE
	
	// evaluate the SS residual with the current solution
	ierr = KARMA::karma_wrapper_petsc_snes_residual (_snes, fsol, _CTX->Work_2->vec(), NULL) ;	CHKERRQ(ierr);
	
	// invert the mass-matrix to store the function for current stage: necessary for 
	// implicit differential equations
	ierr = KARMAInterface::get_interface()->mass_solve (*_CTX->Work_2, R) ;CHKERRQ(ierr);
	
#else
	// evaluate the SS residual with the current solution
	ierr = interface->system_residual (X, R);	CHKERRQ(ierr);
	
#endif

	this->nfcn++ ;

	return ierr ;
}


#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::get_petsc_solvers"
PetscErrorCode TemporalIntegrator::get_petsc_solvers ()
{
	PetscErrorCode ierr ;
	
	KARMAInterface* interface = KARMAInterface::get_interface () ;
	
	_snes = interface->get_SNES_context () ;
	
	ierr = SNESGetKSP(_snes, &_ksp);CHKERRQ(ierr);
	
	ierr = KSPGetPC(_ksp, &_pc);CHKERRQ(ierr);
	
	interface = NULL ;
	
	PetscFunctionReturn(0) ;
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::get_current_solution"
PetscErrorCode TemporalIntegrator::get_current_solution (Vec &sln) const
{ 
	PetscErrorCode ierr = 0 ;
	
	PetscInt n_in, N_in ;
	VecGetSize(sln, &N_in);
	VecGetLocalSize(sln, &n_in);

	// check if the sizes are the same before copying the data.
	karma_assert(N_in == psol->size()) ;
	karma_assert(n_in == psol->local_size()) ;
	
	// copy the current solution to the vector given
	ierr = VecCopy(sol, sln) ;CHKERRQ(ierr);
	
	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::get_current_solution"
Vec& TemporalIntegrator::get_current_solution ()
{ 
	// just pass the reference to the solution vector
	return sol ;
}

#undef __FUNCT__
#define __FUNCT__ "TemporalIntegrator::output_current_solution"
void TemporalIntegrator::output_current_solution(unsigned int nstep, unsigned int naccpt)
{
	// if output is expected, print every "io_frequency" steps
	if (_io_soln && naccpt%_io_frequency == 0)
	{
		// interface->synchronize_physics_solution() ;
		std::stringstream sstr ;
		sstr << "output/transient_" << nstep ;
		std::string fprefix(sstr.str()) ;
		// Output the solution format in the user specified format
		PetscErrorCode ierr = KARMAInterface::get_interface()->write_solution (fprefix, _io_type);CHKERRABORT(KARMA_COMM_WORLD, ierr);
	}
}
