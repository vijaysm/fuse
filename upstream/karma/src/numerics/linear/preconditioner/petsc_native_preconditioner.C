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


// Local Includes
#include "karma.h"
#include "petsc_native_preconditioner.h"
#include "karma_linear_operator.h"

template<typename T>
void PetscNativePreconditioner<T>::apply(const VectorType& x, VectorType& y)
{
//	this->init() ;
	
	PetscErrorCode ierr ; 
	ierr = PCApply(_pc, x.vec(), y.vec());		CHKERRABORT(KARMA_COMM_WORLD,ierr);
}

template<typename T>
void PetscNativePreconditioner<T>::init ()
{
	if(!this->_matrix)
	{
		KARMA_THROW ("ERROR: No matrix set for PetscNativePreconditioner, but init() called" ) ;
	}

	PetscErrorCode ierr = 0 ;
	//Clear the preconditioner in case it has been created in the past
	if(this->_is_initialized)
	{
		ierr = PCDestroy(&_pc);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	
	//Create the preconditioning object
	ierr = PCCreate(KARMA_COMM_WORLD,&_pc);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	//Set the PCType
	set_type(_preconditioner_type, _pc);

#ifdef LIBMESH_HAVE_PETSC_HYPRE
	if(_preconditioner_type == AMG_PRECOND)
		PCHYPRESetType(_pc, "boomeramg");
#endif

	ierr = PCAppendOptionsPrefix(_pc, _name.c_str()) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);

	_mat = this->_matrix->mat();

	ierr = PCSetOperators(_pc,_mat,_mat,SAME_NONZERO_PATTERN);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	ierr = PCShellSetContext(_pc, this) ;	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	this->_is_initialized = true;
}

template<typename T>
void PetscNativePreconditioner<T>::set_shell_routines(PC pc)
{
	/// set the appropriate wrapper shell routines only if \p pc
	/// is a shell preconditioner
	const PCType type ;
	PCGetType(pc, &type) ;
	if (type == PCSHELL)
	{
		PCShellSetSetUp(pc, __karma_petsc_preconditioner_setup);
		PCShellSetApply(pc, __karma_petsc_preconditioner_apply);
	}
	else
	{
		KARMA_LOG(WARNING) << "Setting shell routine references to a non-shell PC object." ;
		karma_here() ;
	}
}

template<typename T>
void PetscNativePreconditioner<T>::set_type (const PreconditionerType preconditioner_type, PC pc)
{
	int ierr = 0;

	switch (preconditioner_type)
	{
		case IDENTITY_PRECOND:
			ierr = PCSetType (pc, (char*) PCNONE);      CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case CHOLESKY_PRECOND:
			ierr = PCSetType (pc, (char*) PCCHOLESKY);  CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case ICC_PRECOND:
			ierr = PCSetType (pc, (char*) PCICC);       CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case ILU_PRECOND:
			ierr = PCSetType (pc, (char*) PCILU);       CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case LU_PRECOND:
			ierr = PCSetType (pc, (char*) PCLU);        CHKERRABORT(KARMA_COMM_WORLD,ierr); return;
		  
		case ASM_PRECOND:
			ierr = PCSetType (pc, (char*) PCASM);       CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case JACOBI_PRECOND:
			ierr = PCSetType (pc, (char*) PCJACOBI);    CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case BLOCK_JACOBI_PRECOND:
			ierr = PCSetType (pc, (char*) PCBJACOBI);   CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case SOR_PRECOND:
			ierr = PCSetType (pc, (char*) PCSOR);       CHKERRABORT(KARMA_COMM_WORLD,ierr); return;
	
		case EISENSTAT_PRECOND:
			ierr = PCSetType (pc, (char*) PCEISENSTAT); CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case AMG_PRECOND:
			ierr = PCSetType (pc, (char*) PCHYPRE);     CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		case USER_PRECOND:
			ierr = PCSetType (pc, (char*) PCMAT);       CHKERRABORT(KARMA_COMM_WORLD,ierr); return;
		
		case SHELL_PRECOND:
			ierr = PCSetType (pc, (char*) PCSHELL);     CHKERRABORT(KARMA_COMM_WORLD,ierr); return;

		default:
			KARMA_LOG(ERROR)	<< "ERROR:  Unsupported PETSC Preconditioner: "
								<< preconditioner_type       << std::endl
								<< "Continuing with PETSC defaults" << std::endl;
	}
}


template<typename T>
PetscErrorCode PetscNativePreconditioner<T>::__karma_petsc_preconditioner_setup (PC pc)
{
	PetscErrorCode ierr = 0 ;
	
	void *ctx ;
	ierr = PCShellGetContext(pc,(void**)&ctx);	CHKERRQ(ierr);
	
	PetscNativePreconditioner<T>* preconditioner = static_cast<PetscNativePreconditioner<T>*>(ctx);

	preconditioner->init();

	return 0;
}

template<typename T>
PetscErrorCode PetscNativePreconditioner<T>::__karma_petsc_preconditioner_apply (PC pc, Vec x, Vec y)
{
	void *ctx ;
	PCShellGetContext(pc,(void**)&ctx);
	
	PetscNativePreconditioner<T>* preconditioner = static_cast<PetscNativePreconditioner<T>*>(ctx);

	VectorType x_vec(x);
	VectorType y_vec(y);

	// TODO: This looks like a bloody disastrous infinite loop.. check the hierarchy
	// and call stack to see exactly which pc's apply method is called inside
	// "preconditioner->apply" 
	preconditioner->apply(x_vec, y_vec);

	return 0;
}




//------------------------------------------------------------------
// Explicit instantiations
template class PetscNativePreconditioner<PetscScalar>;
