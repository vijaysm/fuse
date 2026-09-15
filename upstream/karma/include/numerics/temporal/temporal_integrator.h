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


#ifndef __temporal_integrator_h__
#define __temporal_integrator_h__

#include "petscsnes.h"	// replace by karma_nonlinear_solver.h

#include "ikarma.h"
#include "karma_vector.h"
#include "rk.h"

// KARMA includes
#include "enum_utils.h"
#include "io_package.h"

using namespace KARMA::TEMPORAL ;

// Forward declarations
class KARMAContext ;

class TemporalIntegrator : public IKARMA<TemporalIntegrator>
{
public:

	TemporalIntegrator(RKType rktype, Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit, bool isadapted, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) ;

	virtual ~TemporalIntegrator() ;
	
	virtual PetscErrorCode init () ;
	
	virtual PetscErrorCode clear () ;
	
	PetscErrorCode integrate(bool stats=true) ;

	// get number of function evaluations
	int num_fevals() const { return nfcn; }

	// get number of attempted steps
	int num_steps() const { return nstep; }

	// get number of accepted steps
	int num_accepted() const { return naccpt; }

	// get number of rejected steps
	int num_rejected() const { return nrejct; }
	
	PetscReal current_time() const {	return time; }
	
	KARMAContext* get_context() { return _CTX ; } 
	
	double** Acoeff () const { return _scheme->Acoeff() ;	}

	double* Bcoeff ()  const { return _scheme->Bcoeff() ;	}
	
	double* Ccoeff ()  const { return _scheme->Ccoeff() ;	}
				
	double* Ecoeff ()  const { return _scheme->Ecoeff() ;	}
	
	int current_stage() const { return curr_stage ;	}
	
	PetscErrorCode get_current_solution (Vec &sln) const ;

	Vec& get_current_solution () ;

	Vec& current_stage_vector() ;
	
	Vec& stage_vector(int stage) ;
	
	Vec constant_residual() const ;
	
	Vec& build_vector() ;
	
	void increment_function_count() { nfcn++ ; }
	
	void increment_jacobian_count() { njac++ ; }
	
	void increment_mass_count() { nmass++ ; }

	void set_output_format (IOPackage format) ;
	
	// TSSetInitialTimeStep(ts,0.0,dt);
	// TSSetDuration(ts,time_steps_max,time_total_max);
	// void set_initial_step_size (Real dt) ;

	
	void output_current_solution(unsigned int nstep, unsigned int naccpt) ;
	
	void *step_preprocess (Vec solx) ;
	
	void *step_postprocess (Vec solx) ;

	static AutoPtr<TemporalIntegrator> build(std::string srktype, Vec& solin, PetscReal tstart, PetscReal tend, 
		PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) ;
	
	static AutoPtr<TemporalIntegrator> build(RKType rktype, Vec& solin, PetscReal tstart, PetscReal tend,
		PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) ;
	
	virtual std::string name() = 0 ;
	
	
protected:

	// set some adaptive parameters ; need to set this before performing time adaptation based on truncation error
	void set_adaptive_params(PetscReal deltmax, int nmaxsteps, PetscReal safetyfac, PetscReal faclin, PetscReal facrin) ;
	
	virtual void print_stats () = 0 ;

	// CoreIntegrator
	virtual PetscErrorCode core_integrator() = 0 ;
	
	virtual PetscErrorCode Function(PetscReal time, Vec sol, Vec fval) ;
	
	PetscErrorCode get_petsc_solvers () ;

	PetscReal estimate_error();
	
	// Member variables
	
	std::string _name ;

	PetscReal hinit;
	
	// integration step length
	PetscReal h;
	
	// independent variable
	PetscReal time ;
	
	// stores past value of time
	PetscReal time_old;
	
	// stores past value of h
	PetscReal hold;

	// final value
	PetscReal time_end ;

	// switch for rtol and atol; if itol = 0, rtol and atol are scalars
	// if itol = 1, rtol and atol are vectors
	int itoler;
	
	// relative error tolerance
	PetscReal *rtoler;
	
	// absolute error tolerance
	PetscReal *atoler;
	
	// variables that indicate whether rtoler and atoler are NULL on input
	// or not--needed for proper memory management in destructor
	bool rtolerNULL;
	bool atolerNULL;
	
	// output format
	IOPackage _io_type ;

	bool _io_soln ;
	
	PetscInt _io_frequency ;
	
// Derived variables

	// maximal step size
	PetscReal hmax;
	// maximal number of steps
	unsigned int nmax;
	// smallest number satisfying 1.0 + uround > 1.0
	PetscReal uround;
	// safety factor in step size prediction
	PetscReal safe;
	// facl, facr--parameters for step size selection
	PetscReal facl;
	PetscReal facr;

// Counting variables

	// total number of function evaluations
	unsigned int nfcn;
	// total number of jacobian matrix evaluations
	unsigned int njac;
	// total number of mass matrix evaluations
	unsigned int nmass;
	// number of attempted steps
	unsigned int nstep;
	// number of accepted steps
	unsigned int naccpt;
	// number of rejected steps
	unsigned int nrejct;
	
	unsigned int curr_stage ;
	
	// is the scheme adaptive
	bool adaptive ;
	bool adaptive_initiated ;

// Important context and solution information variables
 
	KARMAContext *_CTX ;

	// the latest converged solution
	Vec sol ;
	
	// the latest converged solution with a wrapper
	KARMAVector* psol ;

	// current residual : 
	//	if nonlinear iteration performed usually serves as the nonlinear residual vector 
	// 	if linear iteration performed, usually the RHS vector
	
	// Stage residuals
	Vec *k ;

	// Zi = h * sum( A[i][j] * F(tn + h * cj, Yj), j = 1:s ) ;
	Vec Zi ;
	
	SNES _snes ;
	
	KSP _ksp ;
	
	PC _pc ;
	
	RK *_scheme ;

	bool _initialized ;
	
private:

	// static std::map<TemporalIntegratorType, I> ourRegisterer;

};

// Inline methods for TemporalIntegrator

inline
Vec& TemporalIntegrator::current_stage_vector()
{
	return k[curr_stage] ;
}

inline
Vec& TemporalIntegrator::stage_vector(int stage)
{
	return k[stage] ;
}


inline
Vec TemporalIntegrator::constant_residual() const
{
	return Zi ;
}	

inline
AutoPtr<TemporalIntegrator> TemporalIntegrator::build(std::string srktype, Vec &solin, PetscReal tstart, PetscReal tend, 
	PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin)
{
	return build(KARMA::Utility::string_to_enum<RKType>(srktype), solin, tstart,
					tend, deltinit, itolerin, rtolerin, atolerin) ;
}

inline
void TemporalIntegrator::set_output_format (IOPackage format)
{
	_io_type = format ;
}

namespace KARMAUtils
{
	inline
	double sign(double a, double b)
	{
		return (b > 0.0) ? fabs(a) : -fabs(a);
	} // sign

}



#endif // define __temporal_integrator_h__
