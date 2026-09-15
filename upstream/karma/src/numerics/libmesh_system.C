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

// Local includes
#include "karma.h"
#include "karma_physics_base.h"
#include "libmesh_system.h"
#include "karma_mesh.h"
#include "karma_vector.h"
#include "karma_matrix.h"

// libMesh includes
// #include "implicit_system.h"
// #include "dof_map.h"
#include "equation_systems.h"

// ------------------------------------------------------------
// libMeshSystem implementation
libMeshSystem::libMeshSystem (KARMAPhysicsBase& es,
				const std::string& name,
				const unsigned int number,
				bool is_explicit) :
				ISystem(es, name, number, is_explicit)
{
	// TODO: not yet sure creating an equation system for every individual system
	// is going to have adverse effects for libMesh usage..
	_libmesh_eqsys = new EquationSystems(es.get_mesh()) ;
	_libmesh_system = new ImplicitSystem(*_libmesh_eqsys, name, number) ;
}


// No copy construction of System objects!
libMeshSystem::libMeshSystem (const libMeshSystem& other) :
		ISystem(other)
{
	karma_error();
}



libMeshSystem& libMeshSystem::operator= (const libMeshSystem&)
{
	karma_error();
}


libMeshSystem::~libMeshSystem ()
{
  // Clear data
  this->clear();
}



void libMeshSystem::clear ()
{
	// clear the parent's data structures first
	ISystem::clear() ;
  
	// clear all user allocated data
	_libmesh_system->clear() ;
	delete _libmesh_system ;
	
	_libmesh_eqsys->clear() ;
	delete _libmesh_eqsys ;
}



void libMeshSystem::init ()
{
	// initialize our core libMesh System object
	_libmesh_system->init() ;

	// Get the connectivity and the data distribution and
	// update the Sparsity pattern. This is needed if
	// user adds vectors/matrices to be used later.
	DofMap& dof_map = _libmesh_system->get_dof_map() ;

	// let us update the sparsity pattern so that if usre wants to
	// use the ISystem interface, everything will still work.
	_sparsity.global_dofs = dof_map.n_dofs() ;
	_sparsity.local_dofs = dof_map.n_local_dofs() ;
	_sparsity.send_list = const_cast<std::vector<unsigned int>*>(&dof_map.get_send_list()) ;
	_sparsity.n_nz = const_cast<std::vector<unsigned int>*>(&dof_map.get_n_nz()) ;
	_sparsity.o_nz = const_cast<std::vector<unsigned int>*>(&dof_map.get_n_oz()) ;

	// now init all the vectors and matrices from libMesh system
	
	// specify that the data structures are externally managed.
	// so do not allocate/deallocate in base class
	_is_externally_managed = true ;
	ISystem::init() ;
	
	// next update the local references to vectors, matrices
	// in the base ISystem object.
	// NOTE: These are only references and are managed externally
	// by the user and/or libMesh. So never clear or destroy these.
	{
		ExplicitSystem* sys = dynamic_cast<ExplicitSystem*>(_libmesh_system) ;
		KARMARealVector* libmesh_soln = dynamic_cast<KARMARealVector*>(sys->solution.get()) ;
		*solution = *libmesh_soln ;
		KARMARealVector* libmesh_locsoln = dynamic_cast<KARMARealVector*>(sys->current_local_solution.get()) ;
		*current_local_solution = *libmesh_locsoln ;
		KARMARealVector* libmesh_rhs = dynamic_cast<KARMARealVector*>(sys->rhs) ;
		*rhs = *libmesh_rhs ;
	}
	
	if (!_is_explicit_system)
	{
		ImplicitSystem* sys = dynamic_cast<ImplicitSystem*>(_libmesh_system) ;
		KARMARealMatrix* libmesh_mat = dynamic_cast<KARMARealMatrix*>(sys->matrix) ;
		*matrix = *libmesh_mat ;
	}
}



