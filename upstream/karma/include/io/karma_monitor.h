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


#ifndef __karma_monitor_h__
#define __karma_monitor_h__

#include "petscts.h"

namespace KARMA
{
	PetscErrorCode KARMAKSPMonitor(KSP ksp, PetscInt its, PetscReal fnorm, void *ctx) ;
	
	PetscErrorCode KARMASNESMonitor(SNES snes, PetscInt its, PetscReal fnorm, void *ctx) ;

	PetscErrorCode KARMATSMonitor(TS ts, PetscInt step, PetscReal ptime, Vec v, void *ctx) ;

	PetscErrorCode KARMASNESPostCheck(SNES _snes,Vec xcurrent,Vec y,Vec x,void *ctx,PetscBool *changed_y,PetscBool *changed_x) ;
	
	PetscErrorCode KARMASNESConvergenceTest (SNES _snes,PetscInt it,PetscReal xnorm,PetscReal pnorm,PetscReal fnorm,SNESConvergedReason *reason,void *cctx) ;
}


#endif // #define __karma_monitor_h__
