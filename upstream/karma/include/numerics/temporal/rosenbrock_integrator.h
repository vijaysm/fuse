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


#ifndef __rosenbrock_integrator_h__
#define __rosenbrock_integrator_h__

#include "temporal_integrator.h"

class RosenbrockIntegrator : public TemporalIntegrator
{
public:
	
	RosenbrockIntegrator (KARMA::TEMPORAL::RKType rktype, Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit) ;

	RosenbrockIntegrator (KARMA::TEMPORAL::RKType rktype, Vec &solin, PetscReal tstart, PetscReal tend, PetscReal deltinit, int itolerin, PetscReal *rtolerin, PetscReal *atolerin) ;
	
	virtual ~RosenbrockIntegrator () ;
	
	virtual PetscErrorCode init () ;
	
	virtual PetscErrorCode clear () ;
	
	virtual std::string name() ;

	PetscErrorCode (*dfdt)(const Real , const KARMAVector&, KARMAVector&) ;
	
protected:

	// CoreIntegrator
	virtual PetscErrorCode core_integrator() ;
	
	virtual void print_stats () ;

// member variables
	
	// vectors used in integration steps
	Vec _curr_sol;
	
	KARMAVector* pcurr_sol ;
	
};

inline
std::string RosenbrockIntegrator::name()
{
	return "RosenbrockIntegrator" ;
}

#endif // define __rosenbrock_integrator_h__
