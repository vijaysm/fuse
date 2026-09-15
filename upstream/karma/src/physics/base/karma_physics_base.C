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


// C++ includes
#include <algorithm>

// KARMA includes
#include "karma.h"
#include "karma_context.h"
#include "karma_interface.h"
#include "karma_mesh.h"
#include "isystem.h"
#include "karma_physics_base.h"
#include "karma_physics_bc_manager.h"
#include "karma_linear_solver.h"
#include "karma_preconditioner.h"

// Define the mesh and partitioners
#include "karma_partitioner.h"
#include "coupling_matrix.h"
#include "karma_vtk_io.h"
#include "karma_utils.h"

// libMesh includes
#ifdef KARMA_HAVE_LIBMESH
#include "dof_map.h"
#include "mesh_refinement.h"
#include "enum_xdr_mode.h"
#include "gmv_io.h"
// #include "vtk_io.h"
// #include "vtk_io_vtk_indip.h"
#include "tecplot_io.h"

#include "parallel.h"
#include "utility.h"
#endif



#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::constructor"
KARMAPhysicsBase::KARMAPhysicsBase (KARMAMesh& o_mesh,
							const std::string name,
							const unsigned int id,
							const KARMA::BCs& o_BCs,
							const std::string order,
							const std::string family) :
	// EquationSystems(o_mesh),
	 _mesh      (o_mesh),
	BC(o_BCs),
	_physics_id(id),
	_use_lobatto_points(false),
	_name(name),
	_dim(o_mesh.mesh_dimension()),
	_system (this->add_system(_name)),
	_mass(NULL),
	_mass_solver(NULL),
#ifdef LIBMESH_PROJECTION
	_mesh_function(NULL),
	// _dof_map(_system.get_dof_map()),
#endif
	_order(order),
	_family(family),
	_initialized (false),
	_is_linear_physics(false),
	_parent_handle_matrices(true),
	_prepared_for_transient(false),
	_predict_solution(true),
	_default_quad_order(FIFTH),
	_check_all_elems_for_bcs(false),
	_is_mass_mat_dirty(true),
	_is_time_dependent_bcs(false)
{
	_ctx = KARMAInterface::get_context() ;

	PetscBool predict_soln_flg = PETSC_FALSE ;
	PetscOptionsHasName(PETSC_NULL, "-predict", &predict_soln_flg) ;
	_predict_solution = (predict_soln_flg == PETSC_TRUE) ;

	jac_operator = NULL ;
	mass_operator = NULL ;
	prec_operator.clear() ;

	// ExplicitSystem& _sub = this->add_system<ExplicitSystem>("_sub_system_0") ;
	// _sub.add_variable("materials", CONSTANT, MONOMIAL );
	// _sub_systems.push_back(&_sub) ;

}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::constructor"
KARMAPhysicsBase::KARMAPhysicsBase (KARMAMesh& o_mesh,
							const std::string name,
							const unsigned int id,
							const std::string order,
							const std::string family) :
	// EquationSystems(o_mesh),
	 _mesh      (o_mesh),
	BC(),
	_physics_id(id),
	_use_lobatto_points(false),
	_name(name),
	_dim(o_mesh.mesh_dimension()),
	_system (this->add_system(_name)),
	_mass(NULL),
	_mass_solver(NULL),
#ifdef LIBMESH_PROJECTION
	_mesh_function(NULL),
	// _dof_map(_system.get_dof_map()),
#endif
	_order(order),
	_family(family),
	_initialized (false),
	_is_linear_physics(false),
	_parent_handle_matrices(true),
	_prepared_for_transient(false),
	_predict_solution(true),
	_default_quad_order(FIFTH),
	_check_all_elems_for_bcs(false),
	_is_mass_mat_dirty(true),
	_is_time_dependent_bcs(false)
{
	_ctx = KARMAInterface::get_context() ;

	PetscBool predict_soln_flg = PETSC_FALSE ;
	PetscOptionsHasName(PETSC_NULL, "-predict", &predict_soln_flg) ;
	_predict_solution = (predict_soln_flg == PETSC_TRUE) ;

	jac_operator = NULL ;
	mass_operator = NULL ;
	prec_operator.clear() ;

	// ExplicitSystem& _sub = this->add_system<ExplicitSystem>("_sub_system_0") ;
	// _sub.add_variable("materials", CONSTANT, MONOMIAL );
	// _sub_systems.push_back(&_sub) ;

}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::destructor"
KARMAPhysicsBase::~KARMAPhysicsBase ()
{
	this->clear () ;

	// clear the KARMA context reference
	_ctx = NULL ;
}


// unsigned int KARMAPhysicsBase::material_number (const Elem* ele) const
// {
// 	unsigned int matnum = (ele->subdomain_id()) ;
// 
// 	// 0 is the default material id
// 	return (matnum > 0) ? matnum-1 : matnum ;
// }

// int KARMAPhysicsBase::partition_number (const Elem* ele) const
// {
// 	// store this somewhere else
// 	return static_cast<int>(ele->processor_id()) ;
// }


#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::add_coupled_physics_reference"
bool KARMAPhysicsBase::add_coupled_physics_reference(KARMAPhysicsBase *phy)
{
	return add_coupled_physics_reference(phy, phy->_name) ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::add_coupled_physics_reference"
bool KARMAPhysicsBase::add_coupled_physics_reference(KARMAPhysicsBase *phy, std::string& phyname)
{
	if (phy == NULL)
		return false ;

	_coupled_physics_names.push_back(phyname) ;

	_coupled_physics.push_back(phy) ;

	return true ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::step_preprocess"
void KARMAPhysicsBase::step_preprocess ()
{
	// this->impose_dirichlet_conditions_strongly() ;
	if (_predict_solution)
	{
		KARMARealVector& sol = this->get_solution() ;
		KARMARealVector& tempvec = this->get_temporary_vector() ;

		PetscErrorCode ierr = VecCopy( tempvec.vec(), sol.vec() ) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
		sol.close() ;
	}
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::step_postprocess"
void KARMAPhysicsBase::step_postprocess ()
{
	std::string tempvecname = "_oldsolution_" ;
	KARMARealVector& oldsol = this->get_solution_with_name(tempvecname) ;
	KARMARealVector& sol = this->get_solution() ;
	KARMARealVector& tempvec = this->get_temporary_vector() ;

	if (_is_time_dependent_bcs)
	{
		find_boundary_conditions() ;
	}

	PetscErrorCode ierr = 0 ;
	if (_predict_solution)
	{
		ierr = VecCopy( sol.vec(), tempvec.vec() ) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);

		ierr = VecAXPBY(tempvec.vec(), -1.0, 2.0, oldsol.vec()) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);

		tempvec.close() ;
	}

	// copy the current solution to the old solution.
	ierr = VecCopy( sol.vec(), oldsol.vec() ) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);

	// NOTE: Do not call predict_solution after this call since you
	// will get back sol again ! There is no gain in such a call.

}


#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::stage_preprocess"
void KARMAPhysicsBase::stage_preprocess ()
{
	// if there are time-dependent boundary conditions, then call the user-routine,
	// update the boundary values and cache it in our data-structure for all variables.
	if (_is_time_dependent_bcs)
	{
		find_boundary_conditions() ;
	}
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::stage_postprocess"
void KARMAPhysicsBase::stage_postprocess ()
{
}



#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::read"
PetscErrorCode KARMAPhysicsBase::read (std::string& file_prefix)
{
	// Shoud we morph our system to a libMesh understandable object
	// and then pass it to XDR read/write methods ?
	karma_not_implemented() ;
	
	std::stringstream out;

    // Read the output in machine-independent XDR format
    out << file_prefix << "_" << this->system_type() << ".xdr" ;

    try
    {
        // this->read(out.str(), READ, EquationSystems::READ_HEADER |
                                // EquationSystems::READ_DATA | EquationSystems::READ_ADDITIONAL_DATA) ;
    }
    catch(std::exception e)
    {
        PetscPrintf(KARMA_COMM_WORLD, "\nRestarts available only for XDR files. Check if file is correct for %s.\n", this->system_type().c_str()) ;

        karma_error() ;
    }

    this->update() ;

    out.str("") ;

}


#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::write"
PetscErrorCode KARMAPhysicsBase::write (std::string& file_prefix, libMeshEnums::IOPackage ioType)
{
	std::stringstream out;

    this->update() ;

    out.str("") ;

    if (ioType == GMV)
    {
	// Shoud we morph our system to a libMesh understandable object
	// and then pass it to XDR read/write methods ?
	karma_not_implemented() ;
	
        // out << file_prefix << "_" << this->system_type() << ".gmv" ;

        // GMVIO gmv_io (this->get_mesh()) ;
        // if (!this->is_continuous())
        // {
            // gmv_io.discontinuous() = true;
            // gmv_io.p_levels() = true;
            // gmv_io.write_discontinuous_gmv(out.str(), *this, false);
        // }
        // else
        // {
            // gmv_io.write_equation_systems (out.str(), *this);
        // }
    }
    else if (ioType == VTK)
    {
        out << file_prefix << "_" << this->system_type() << ".vtu" ;
        //VTKIO (system->get_mesh()).write_equation_systems (out.str(), *system);
        // VTKIO_NOVTK (this->get_mesh()).write_equation_systems (out.str(), *this);
		KARMAVTKIO (this->get_mesh()).write_physics_solution (out.str(), *this);
    }
    else if (ioType == TECPLOT)
    {
	
	// Shoud we morph our system to a libMesh understandable object
	// and then pass it to XDR read/write methods ?
	karma_not_implemented() ;
	
        // out << file_prefix << "_" << this->system_type() << ".tec" ;
        // TecplotIO (this->get_mesh(), false).write_equation_systems (out.str(), *this);
    }
    else
    {
		
	// Shoud we morph our system to a libMesh understandable object
	// and then pass it to XDR read/write methods ?
	karma_not_implemented() ;
	
        // Print the output in machine-independent XDR format
        // out << file_prefix << "_" << this->system_type() << ".xdr" ;
        // this->write(out.str(), WRITE, EquationSystems::WRITE_DATA | EquationSystems::WRITE_ADDITIONAL_DATA) ;
    }

}


// return a pointer to the actual vector requested
#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::get_solution_with_name"
KARMARealVector& KARMAPhysicsBase::get_solution_with_name(const std::string& solname)
{
	KARMARealVector* tempvec = NULL ;

	if( _system.have_vector(solname) )
	{
		tempvec = dynamic_cast<KARMARealVector *> (&_system.get_vector(solname)) ;
	}
	else
	{
		if( _sub_systems.size() > 0 )
		{
			for (unsigned int isys=0; isys < _sub_systems.size() ; isys++)
			{
				if(_sub_systems[isys]->have_vector(solname))
				{
					tempvec = dynamic_cast<KARMARealVector *> (&_sub_systems[isys]->get_vector(solname)) ;
					break ;
				}
			}
		}
	}

	if (tempvec == NULL)
	{
		// vector not found
		PetscPrintf(KARMA_COMM_WORLD, "\nVector with name = %s not found\n", solname.c_str()) ;
		karma_error() ;
	}

	return *tempvec ;
}

// return a pointer to the actual vector requested
#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::get_solution_with_name"
KARMARealVector& KARMAPhysicsBase::get_solution_with_name(const std::string& solname) const
{
	KARMARealVector* tempvec = NULL ;

	if( _system.have_vector(solname) )
	{
		tempvec = dynamic_cast<KARMARealVector *> (&_system.get_vector(solname)) ;
	}
	else
	{
		if( _sub_systems.size() > 0 )
		{
			for (unsigned int isys=0; isys < _sub_systems.size() ; isys++)
			{
				if(_sub_systems[isys]->have_vector(solname))
				{
					tempvec = dynamic_cast<KARMARealVector *> (&_sub_systems[isys]->get_vector(solname)) ;
					break ;
				}
			}
		}
	}

	if (tempvec == NULL)
	{
		// vector not found
		PetscPrintf(KARMA_COMM_WORLD, "\nVector with name = %s not found\n", solname.c_str()) ;
		karma_error() ;
	}

	return *tempvec ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::prepare_for_transient"
void KARMAPhysicsBase::prepare_for_transient()
{
	// this is a dummy place-holder. If the derived physics has something that
	// needs to be done, before transient begins, now would be the time to do it.
	_prepared_for_transient = true ;

	PetscErrorCode ierr ;
	if (this->mass_operator != NULL)
		if (this->mass_operator->is_solve() || this->mass_operator->is_matrix() )
		{
			// since we are going into the transient soon, create and store the mass matrix;
			// unless mass matrix is time dependent, this is useful.
			KARMAParameters inparam, outparam ;

			// set a time > 0 so that if there is any transient dependent mass-matrix, it is
			// reflected in the computed matrix.
			inparam.set<Real>("time") = 1.0 ;

			// Set the reference for the physics being solved for.
			inparam.set<KARMAPhysicsBase*>("physics") = this ;

			// Parameter to use Mass-Lumping = true
			// We do not want to spend too much time solving it. Just invert a diagonal and throw
			// away the result !
			inparam.set<bool>("is_lumped") = true ;

			// This parameter, when set to true will create the matrix and
			// store it in current physics, so that it can be re-used !
			// This should provide considerable CPU time gain for time-independent Mass matrices.
			inparam.set<bool>("compute_matrix") = true ;

			// If we called with, create_mass_matrix for mass_multiply,
			// it would be wise to just re-use that to solve the matrix !
			// This should provide considerable CPU time gain for time-independent Mass matrices.
			_mass = dynamic_cast<KARMARealMatrix*>(_system.request_matrix("_mass_")) ;
			outparam.set<KARMARealMatrix*>("mass_matrix") = _mass ;

			KARMARealVector& phy_sol = this->get_solution() ;
			KARMARealVector& phy_res = this->get_residual() ;

			// Let us call the actual Mass solver for the physics, which will recompute Mass and solve
			if (this->mass_operator->is_matrix())
			{
				ierr = this->mass_operator->matrix (phy_sol, *_mass, inparam, &outparam) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
			}
			else if (this->mass_operator->is_solve())
			{
				ierr = this->mass_operator->solve (phy_sol, phy_res, inparam, &outparam) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
			}
			else
			{
				// we will assume Identity as the mass matrix. but technically, you arent supposed to be here !
				KARMA_LOG(ERROR) << "Need either a function for mass matrix creation or solve." ;
				karma_error() ;
			}

			this->mass_matrix_dirty(false) ;

			// Now that the mass matrix is created, create a solver to optimize the mass-solves.
			// and assign the reference to the solver
			_mass_solver = new KARMALinearSolver() ;

			// set a prefix for options
			_mass_solver->name() = "mass_" ;

			// Set the default solver to be GMRES; This can be changed by user if need be.
			_mass_solver->solver_type() = GMRES ;

			const bool use_asm_preconditioner = false ;
			if (KARMA::num_processors > 1)
			{
				if (use_asm_preconditioner)
					_mass_solver->preconditioner_type() = ASM_PRECOND ;
				else
					_mass_solver->preconditioner_type() = BLOCK_JACOBI_PRECOND ;

				// // Set the SSOR preconditioner with a default omega.
				// _mass_solver->set_preconditioner_type(EISENSTAT_PRECOND) ;
				// // This can be changed by user to get better performance.
				// ierr = PCEisenstatSetOmega(_mass_solver->pc(), 1.2) ;
			}
			else
			{
				// Set the default preconditioner for CG to be ICC; This can be changed by user if need be.
				_mass_solver->preconditioner_type() = ILU_PRECOND ;
			}

			// Since our matrix is inited by now, let us init the mass matrix solver
			_mass_solver->init(_mass) ;

			if (KARMA::num_processors > 1)
			{
				PetscInt nlocal,first ;

				KSP mass_ksp = _mass_solver->ksp() ;

				ierr = KSPSetUp(mass_ksp) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);

				PC mass_pc ;
				ierr = KSPGetPC(mass_ksp, &mass_pc) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);

				KSP* sub_ksp ;
				// Extract the array of KSP contexts for the local blocks
				if (use_asm_preconditioner)
				{
					ierr = PCASMGetSubKSP(mass_pc,&nlocal,&first,&sub_ksp); CHKERRABORT(KARMA_COMM_WORLD, ierr);
				}
				else
				{
					ierr = PCBJacobiGetSubKSP(mass_pc,&nlocal,&first,&sub_ksp); CHKERRABORT(KARMA_COMM_WORLD, ierr);
				}
				
				// Loop over the local blocks, setting various KSP options
				// for each block.
				for (int i=0; i<nlocal; i++)
				{
					PC sub_pc ;
					ierr = KSPGetPC(sub_ksp[i],&sub_pc);CHKERRABORT(KARMA_COMM_WORLD, ierr);

					ierr = KSPSetType(sub_ksp[i],KSPGMRES);CHKERRABORT(KARMA_COMM_WORLD, ierr);
					ierr = PCSetType(sub_pc,PCILU);CHKERRABORT(KARMA_COMM_WORLD, ierr);
					ierr = KSPSetTolerances(sub_ksp[i],1.e-8,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRABORT(KARMA_COMM_WORLD, ierr);
					ierr = KSPSetFromOptions(sub_ksp[i]) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
				}
			}

		}

	std::string tempvecname = "_oldsolution_" ;
	KARMARealVector& oldsol = this->get_solution_with_name(tempvecname) ;
	ierr = VecCopy( this->get_solution().vec(), oldsol.vec() ) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
	oldsol.close() ;

}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::compute_dynamical_timescale"
PetscReal KARMAPhysicsBase::compute_dynamical_timescale (PetscReal delt, PetscReal time)
{
	// compute the dynamical time scales for each physics;
	// This could be an optional step based on some user parameter
	// Possibly can use this idea to do the hybrid time stepping scheme
	std::string tempvecname = "_oldsolution_" ;
	KARMARealVector& oldsol = this->get_solution_with_name(tempvecname) ;
	KARMARealVector& sol = this->get_solution() ;

	// if (fabs(_current_time-time) < 1e-15 )	return _dynamical_time_scale ;
	// else	_current_time = time ;

	_dynamical_time_scale = 1E-15 ;

	for (unsigned int inode=0; inode < _system.n_dofs(); inode++)
	{
		// if (fabs( sol(inode)) < 1e-14)	continue ;

		// need to uncomment next line ; i do not have context defined yet
		// call interface reference and get context
		karma_error ();

		// include a routine in IVector<T,T> to enable this directly.
		// create a new method similar to axpy: say xaydxmy
		// xaydxmy = (x \add y) divided (x \minux y)
		// compute the minimum on this vector and then find minimum
		// multiply by 0.5*delt and then return the reciprocal
// 		_dynamical_time_scale = KARMA::max<PetscReal>( _dynamical_time_scale,
// 					0.5 * _ctx->delt * fabs(( sol(inode) + oldsol(inode))/( sol(inode)-oldsol(inode))) ) ;

// 		PetscPrintf(KARMA_COMM_WORLD, "\nNode = %D, DYNTIME = %G, DELT = %G, VAL = %G", inode, _dynamical_time_scale, _ctx->time, 0.5 * _ctx->time * fabs(((*sol)(inode)+(*oldsol)(inode))/((*sol)(inode)-(*oldsol)(inode)))) ;
	}

	return 1.0/_dynamical_time_scale ;
}


unsigned int KARMAPhysicsBase::n_vars () const
{
  unsigned int tot=0;
  
  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    tot += pos->second->n_vars();

  return tot;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::predict_solution"
PetscErrorCode KARMAPhysicsBase::predict_solution ()
{
	std::string tempvecname = "_oldsolution_" ;
	KARMARealVector& oldsol = this->get_solution_with_name(tempvecname) ;
	KARMARealVector& sol = this->get_solution() ;

	PetscErrorCode ierr = VecAXPBY(sol.vec(), -1.0, 2.0, oldsol.vec()) ;CHKERRQ(ierr);

	sol.close() ;

	PetscFunctionReturn(0) ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::is_physics_converged"
bool KARMAPhysicsBase::is_physics_converged(PetscReal atol, PetscReal rtol)
{
	bool result = true ;
	KARMARealVector& resphy = this->get_residual() ;
	KARMARealVector& solphy = this->get_solution() ;

	double nrmres = resphy.l2_norm() ;
	double nrmsol = solphy.l2_norm() ;	// need to make this as the initial residual

	if (nrmres > atol || nrmres > nrmsol*rtol)
		result = false ;

	return result ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::print_info"
void KARMAPhysicsBase::print_info ()
{
	// if (KARMA::verbose)
		// PetscPrintf(PETSC_COMM_SELF, "\n[%D] : Global dofs=%D, Local dofs=%D", KARMA::processor_id, _mesh.n_nodes(), _mesh.n_local_nodes()) ;

	PetscPrintf(KARMA_COMM_WORLD, "\n\t--------------------------------------------\n") ;
	PetscPrintf(KARMA_COMM_WORLD, "\t\t%s Physics", _name.c_str()) ;
	PetscPrintf(KARMA_COMM_WORLD, "\n\t--------------------------------------------\n") ;

	_mesh.print_info();
	
	
  std::ostringstream out;
  
  out << " EquationSystems\n"
      << "  n_systems()=" << this->n_systems() << '\n';

  // Print the info for the individual systems
  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    out << pos->second->get_info();

	// Parent::print_info(std::cout);

	// std::string mesh_info = _mesh.get_info();
	// PetscPrintf(KARMA_COMM_WORLD,"\n\t%s", mesh_info.c_str()) ;
	// std::string physics_info = Parent::get_info();
	// PetscPrintf(KARMA_COMM_WORLD,"\n\t%s\n", physics_info.c_str()) ;

	// PetscPrintf(KARMA_COMM_WORLD,"\n\t  n_systems()=%d\n", this->n_systems()) ;

	// std::string primary_system_info = _system.get_info();
	// PetscPrintf(KARMA_COMM_WORLD,"\n\t 1) Primary%s\n", primary_system_info.c_str()) ;

	// // Print the info for the individual systems
	// for (unsigned int index = 0; index < _sub_systems.size(); ++index)
	// {
		// std::string aux_system_info = _sub_systems[index]->get_info();
		// PetscPrintf(KARMA_COMM_WORLD, "\n\t %d) Auxillary%s\n", index+2, aux_system_info.c_str()) ;
	// }

	PetscPrintf(KARMA_COMM_WORLD, "\n\t--------------------------------------------\n") ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::clear"
void KARMAPhysicsBase::clear ()
{
	if (_initialized)
	{
		if (_mass != NULL)
		{
			_mass = NULL ;
			
			delete _mass_solver ;
		}

		// Clear any additional parameters
		parameters.clear ();

		// clear the systems.  We must delete them
		// since we newed them!
		while (!_systems.empty())
		{
			system_iterator pos = _systems.begin();

			ISystem *sys = pos->second;
			delete sys;
			sys = NULL;

			_systems.erase (pos);
		}

#ifdef LIBMESH_PROJECTION
		delete _mesh_function ;
#endif

		delete [] _material_numbers ;

		if (!BC.empty())
		{
			for (unsigned int ivar=0; ivar < BC.size(); ivar++)
				delete BC[ivar] ;

			BC.resize(0) ;	BC.clear() ;
		}

		if (!_coupled_physics_names.empty())
		{
			this->_coupled_physics_names.clear() ;
			this->_coupled_physics.clear() ;
			for (unsigned int iphy=0; iphy < _coupled_physics_matrices.size(); iphy++)
			{
				Mat _base_mat ;
				if (_coupled_physics_matrices[iphy] != NULL)
				{
					_base_mat = _coupled_physics_matrices[iphy]->mat() ;
				}
				delete _coupled_physics_matrices[iphy] ;
				
				PetscErrorCode ierr = MatDestroy(&_base_mat) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
			}
			this->_coupled_physics_matrices.clear() ;
		}

		if (jac_operator != NULL)
			delete jac_operator ;

		if (mass_operator != NULL)
			delete mass_operator ;

		if (!prec_operator.empty())
		{
			for (unsigned int iprec=0; iprec<prec_operator.size(); iprec++)
				if (prec_operator[iprec] != NULL)
					delete prec_operator[iprec];

			prec_operator.clear() ;
		}

		_initialized = false ;
	}
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::form_initial_guess"
PetscErrorCode KARMAPhysicsBase::form_initial_guess()
{
	PetscErrorCode ierr = this->form_initial_guess(this->get_solution()) ;
	
	this->update() ;
	
	return ierr ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::clear_system_matrix"
void KARMAPhysicsBase::clear_system_matrix()
{
	_system.matrix->clear () ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::reinit"
void KARMAPhysicsBase::reinit ()
{
	karma_not_implemented () ;
	// we do not have much to do here. Just call Parent's reinit() function.
	// EquationSystems::reinit() ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::init"
void KARMAPhysicsBase::init ()
{
	if (!_initialized)
	{
		_system.add_vector("_temporary_", KARMA::AUTOMATIC) ;

		// TODO: The following is only needed for transients. Maybe delay the initialization ?
		_system.add_vector("_oldsolution_", KARMA::AUTOMATIC) ;
		_system.add_matrix("_mass_") ;

		// initialize parent data
		  const unsigned int n_sys = this->n_systems();

		  libmesh_assert (n_sys != 0);

		  // Distribute the mesh if possible
		  if (libMesh::n_processors() > 1)
			_mesh.delete_remote_elements();
		  
		  // next go over each sub-system and initialize it
		  for (unsigned int i=0; i != this->n_systems(); ++i)
			this->get_system(i).init();
			

		std::vector<unsigned int> vars ;
		vars.resize(_system.n_vars()) ;
		Utility::iota(vars.begin(), vars.end(), 0) ;
		
		// set the references for mesh-function so that we can find the solution at any given poinit
		// _mesh_function = new MeshFunction(*this, this->get_solution(), _system.get_dof_map(), vars, NULL) ;
		// _mesh_function->init() ;

		// loop over the mesh and find out the number of unique materials so that we can allocate data
		if (KARMA::processor_id == 0)
		{
			MeshBase::const_element_iterator el     = _mesh.active_local_elements_begin();
			const MeshBase::const_element_iterator el_end = _mesh.active_local_elements_end();
			PetscInt* mats = new PetscInt[_mesh.n_active_local_elem()] ;

			// default material id
			int imat = 0 ;

			for( ; el != el_end; ++el)
			{
				Elem* ele = *el ;

				try
				{
					unsigned int matnum = this->material_number (ele->id()) ;
					mats[imat++] = matnum ;
					//mats.push_back(0) ;
				}
				catch (PetscErrorCode ierr)	// bad way to handle exception. chcange this ..
				{
					mats[imat++] = 0 ;
				}
			}

			// When running in parallel, you only care about the material data in your subdomain.
			bool isunique ;
			std::vector<int> temp_mat ;
			for (int ii=0; ii < imat; ii++)
			{
				isunique = true ;
				for (unsigned int jj=0; jj < temp_mat.size(); jj++)
					if (mats[ii] == temp_mat[jj])
					{
						isunique = false ; break ;
					}
				if (isunique)
					temp_mat.push_back(mats[ii]) ;
			}

			_materials = temp_mat.size() ;
			_material_numbers = new PetscInt[_materials] ;

			std::sort(temp_mat.begin(), temp_mat.end()) ;

			for (unsigned int ii=0; ii < _materials; ii++)
				_material_numbers[ii] = temp_mat[ii] ;

			temp_mat.clear() ;

			delete [] mats ;
		}

		MPI_Bcast ( &_materials, 1, MPI_INT, 0, KARMA_COMM_WORLD) ;

		if (KARMA::processor_id != 0)
			_material_numbers = new PetscInt[_materials] ;

		PetscBarrier(PETSC_NULL) ;

		MPI_Bcast ( &_material_numbers[0], _materials, MPI_INT, 0, KARMA_COMM_WORLD) ;

		// commented out because ODE systems need not have any BC object
		// karma_assert(BC.size () > 0) ;

		_initialized = true ;

		_sub_systems.clear() ;

		unsigned int isys = 0 ;
		std::stringstream sstr ;	bool prev_found = true ;
		// populate the references for all explicit systems in our physics
		while (true)
		{
			sstr << "_sub_system_" << isys ;

			// loop till we do not find a system with the standard name pattern !
			if ( !this->has_system(sstr.str()) && !prev_found)
				break ;

			if ( !this->has_system(sstr.str()))
			{
				prev_found = false ;
				continue ;
			}

			ISystem& _sub = this->get_system(sstr.str()) ;
			_sub_systems.push_back(&_sub) ;

			sstr.str("") ;
			isys++ ;
		}

	}

}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::is_continuous"
bool KARMAPhysicsBase::is_continuous () const
{
	return true ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::find_boundary_conditions"
PetscErrorCode KARMAPhysicsBase::find_boundary_conditions()
{
	KARMAParameters param ;

	// If BCs are time dependent, just update this variable before calling parent routine.
	param.set<Real> (TIME) = _ctx->time;

	param.set<unsigned int> (DIMENSION) = _dim;

	PetscErrorCode ierr = KARMAPhysicsBase::find_boundary_conditions(param) ;

	PetscFunctionReturn(ierr) ;
}


#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::find_boundary_conditions"
PetscErrorCode KARMAPhysicsBase::find_boundary_conditions(KARMAParameters& var_param)
{
	const int dim = _mesh.mesh_dimension();

	MeshBase::const_element_iterator       el     = _mesh.local_elements_begin();
	const MeshBase::const_element_iterator end_el = _mesh.local_elements_end();

	// var_param.set<bool>("is_continuous") = this->is_continuous() ;
	var_param.set<bool>("is_time_dependent_bcs") = _is_time_dependent_bcs ;

	for (unsigned int ivar=0; ivar<BC.size(); ivar++)
		BC[ivar]->clear() ;

	for ( ; el != end_el; ++el)
	{
		const Elem* elem = *el;

		if(elem->on_boundary() || _check_all_elems_for_bcs)
		{
			for (unsigned int ivar=0; ivar<BC.size(); ivar++)
			{
				// check boundary elements unless all elements need to be investigated.
				BC[ivar]->update_boundary_conditions(elem, var_param, _check_all_elems_for_bcs) ;
			}
		}
	}

	if (KARMA::verbosity_level > 4)
	{
		//for (int ivar=0; ivar<_system.n_vars(); ivar++)
		for (unsigned int ivar=0; ivar<BC.size(); ivar++)
		{
			PetscPrintf(KARMA_COMM_WORLD, "\nBoundary conditions; Variable number = %D; Variable name = %s\n", ivar,  _system.variable_name(ivar).c_str()) ;
			BC[ivar]->print_conditions () ;
		}
	}

	PetscFunctionReturn(0) ;
}


 
void KARMAPhysicsBase::build_variable_names (std::vector<std::string>& var_names) const
{
  libmesh_assert (this->n_systems());
  
  var_names.resize (this->n_vars());

  unsigned int var_num=0;
  
  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    for (unsigned int vn=0; vn<pos->second->n_vars(); vn++)
      var_names[var_num++] = pos->second->variable_name(vn);       
}



#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::solve"
void KARMAPhysicsBase::solve ()
{
  libmesh_assert (this->n_systems());

  for (unsigned int i=0; i != this->n_systems(); ++i)
    this->get_system(i).solve();
}



#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::get_n_nz"
const std::vector<unsigned int>& KARMAPhysicsBase::get_n_nz()
{
	karma_assert(_system.sparsity().n_nz != NULL) ;
	
	return *_system.sparsity().n_nz ;
}

#undef __FUNCT__
#define __FUNCT__ "KARMAPhysicsBase::get_n_oz"
const std::vector<unsigned int>& KARMAPhysicsBase::get_n_oz()
{
	karma_assert(_system.sparsity().o_nz != NULL) ;
	
	return *_system.sparsity().o_nz ;
}

