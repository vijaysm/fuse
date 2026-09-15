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



#ifndef __libmesh_system_h__
#define __libmesh_system_h__

// C++ includes

// Local Includes
#include "isystem.h"
#include "user_sparsity.h"
#include "variable.h"

#include "dof_map.h"
#include "mesh_base.h"
// libMesh object
#include "petsc_vector.h"
#include "petsc_matrix.h"
#include "implicit_system.h"

#include "karma_vector.h"
#include "karma_matrix.h"

// Forward declarations
class KARMAPhysicsBase;

/**
 * This class provides a specific system class.  It aims
 * at providing a complete ImplicitSystem, offering nothing more 
 * than just the essentials needed to solve a equation system. Note
 * that additional vectors/matrices may be added as necessary.
 * Without the explicit system matrix, the system will act
 * purely as a utility data holder for physics (ExplicitSystem).
 */

// ------------------------------------------------------------
// libMeshSystem class definition

class libMeshSystem : public ISystem
{
public:
	
	/**
	* Constructor.  Optionally initializes required
	* data structures.
	*/
	libMeshSystem (KARMAPhysicsBase& es,
			const std::string& name,
			const unsigned int number,
				bool is_explicit=false);

	/**
	* Copy Constructor.  Explicitly disallow copy
	* by value.
	*/
	libMeshSystem(const libMeshSystem& other) ;
	
	/**
	* Operator assignment. Same as above.
	* explicitly block copy by value operations.
	*/
	libMeshSystem& operator= (const libMeshSystem&) ;
	
	/**
	* Destructor.
	*/
	virtual ~libMeshSystem ();
	
	/**
	* The type of system.
	*/
	typedef System sys_type;
	
	/**
	* @returns a clever pointer to the system.
	*/
	sys_type & system () { return *_libmesh_system; }

	/**
	* Clear all the data structures associated with
	* the system.
	*/
	virtual void clear ();
	
		/** 
	* Initializes the member data fields associated with
	* the system, based on the sparsity object that was
	* provided by the user. If \p UserSparsity is not
	* ready, this routine with throw an error.
	* NOTE: This is non-virtual.
		*/
	void init () ;
	
	/**
	* Initializes degrees of freedom on the current mesh
	* and the member data fields associated with
	* the system, so that, e.g., \p assemble() may be used.
	*/
//	virtual void init (unsigned int ndofs, unsigned int n_local_dofs,
//						std::vector<unsigned int>& send_list,
//						std::vector<unsigned int>* n_nz,
//						std::vector<unsigned int>* o_nz) ;
	
	/**
	* Update the local values to reflect the solution
	* on neighboring processors.
	*/
	virtual void update ();
	
	/**
	 * Assembles a residual in \p rhs and/or a jacobian in \p matrix,
	 * as requested.
	 *
	 * This is undefined in libMeshSystem; subclasses each have their
	 * own way of handling assembly.
	* @e Should be overloaded in derived classes.
	*/
	virtual void assemble (bool /* get_residual */ , bool /* get_jacobian */) ;
	
	/**
	* Assembles & solves the linear system Ax=b.
	*/
	virtual void solve ();

	/**
	* @returns \p "libMeshSystem".  Helps in identifying
	* the system type in an equation system file.
	* NOTE: Should we return the base \p System type from
	* libMesh ?
	*/
	virtual std::string system_type () const { return "libMeshSystem"; }
	
	/**
	* Fill the input vector \p global_soln so that it contains
	* the global solution on all processors.
	* Requires communication with all other processors.
	*/
	void update_global_solution (std::vector<Number>& global_soln) const;
	
	/**
	* Fill the input vector \p global_soln so that it contains
	* the global solution on processor \p dest_proc.
	* Requires communication with all other processors.
	*/
	void update_global_solution (std::vector<Number>& global_soln,
								const unsigned int dest_proc) const;



	/**
	 * Adds the additional vector \p vec_name to this system.  Only
	 * allowed @e prior to \p init().  All the additional vectors
	 * are similarly distributed, like the \p solution,
	 * and inititialized to zero.
	 *
	 * By default vectors added by add_vector are projected to changed grids by
	 * reinit().  To zero them instead (more efficient), pass "false" as the
	 * second argument
	 */
	virtual KARMARealVector& add_vector (const std::string& vec_name,
										 const KARMA::LinearAlgebraLayout type = KARMA::PARALLEL,
										 const bool externally_managed = false);

	/**
	 * Deletes the additional vector \p vec_name in this system.
	 */
	virtual bool delete_vector (const std::string& vec_name);

	/**
	 * Adds the additional matrix \p mat_name to this system.  Only
	 * allowed @e prior to \p assemble().  All additional matrices
	 * have the same sparsity pattern as the matrix used during
	 * solution.  When not \p System but the @e user wants to
	 * initialize the mayor matrix, then all the additional matrices,
	 * if existent, have to be initialized by the user, too.
	 */
	virtual KARMARealMatrix& add_matrix (const std::string& mat_name,
									 const bool externally_managed = false);
	
	/**
	 * Deletes the additional matrix \p mat_name to this system.
	 */
	virtual bool delete_matrix (const std::string& mat_name);


	
	
	/**
	* @returns a constant reference to this systems's \p _mesh.
	*/
	const MeshBase & get_mesh() const;
	
	/**
	* @returns a reference to this systems's \p _mesh.
	*/
	MeshBase & get_mesh();
	
	/**
	* @returns a constant reference to this system's \p _dof_map.
	*/
	const DofMap & get_dof_map() const;
	
	/**
	* @returns a writeable reference to this system's \p _dof_map.
	*/
	DofMap & get_dof_map();
	
	/**
	* @returns \p true if the system is active, \p false otherwise.
	* An active system will be solved.
	*/
	bool active () const;
	
	/**
	* Activates the system.  Only active systems are solved.
	*/
	void activate ();
	
	/**
	* Deactivates the system.  Only active systems are solved.
	*/
	void deactivate ();
	
	/**
	* Returns the number of active degrees of freedom
	* for this System.
	*/
	unsigned int n_active_dofs() const;
	
	/**
	* @returns the number of constrained degrees of freedom
	* in the system.
	*/
	unsigned int n_constrained_dofs() const;
	
	/**
	* Adds the variable \p var to the list of variables
	* for this system.  Returns the index number for the new variable.
	*/
	unsigned int add_variable (const std::string& var,
								const FEType& type,
								const std::set<subdomain_id_type> * const active_subdomains = NULL);
	
	/**
	* Adds the variable \p var to the list of variables
	* for this system.  Same as before, but assumes \p LAGRANGE
	* as default value for \p FEType.family.
	*/
	unsigned int add_variable (const std::string& var,
								const Order order = FIRST,
								const FEFamily = LAGRANGE,
								const std::set<subdomain_id_type> * const active_subdomains = NULL);

	
	/**
	* Return a constant reference to \p Variable \p var.
	*/
	const Variable& variable (unsigned int var) const;
	
	/**
	* @returns the finite element type variable number \p i.
	*/
	const FEType & variable_type (const unsigned int i) const;
	
	/**
	* @returns the finite element type for variable \p var.
	*/
	const FEType & variable_type (const std::string& var) const;
	
	/**
	* @returns \p true when the other system contains
	* identical data, up to the given threshold.  Outputs
	* some diagnostic info when \p verbose is set.
	*/
	virtual bool compare (const libMeshSystem& other_system,
						const Real threshold,
						const bool verbose) const;
	
	//--------------------------------------------------
	// The solution and solution access members

	/**
	* @returns the current solution for the specified global
	* DOF.
	*/
	Number current_solution (const unsigned int global_dof_number) const;
	
	/**
	* Zeroes all dofs in \p v that correspond to variable number \p
	* var_num.
	*/
	void zero_variable (KARMARealVector& v, unsigned int var_num) const;

  
private:

	/**
	* Maintain a true managed proxy reference to libMesh's
	* \p System object. All calls on the current system will be
 	* relayed to \p _libmesh_system .
	*/
	ImplicitSystem* _libmesh_system ;
	EquationSystems* _libmesh_eqsys ;

};



// ------------------------------------------------------------
// libMeshSystem inline methods

inline
const MeshBase & libMeshSystem::get_mesh() const
{
	return _libmesh_system->get_mesh();
}


inline
MeshBase & libMeshSystem::get_mesh()
{
	return _libmesh_system->get_mesh();
}


inline
const DofMap & libMeshSystem::get_dof_map() const
{
	return _libmesh_system->get_dof_map();
}



inline
DofMap & libMeshSystem::get_dof_map()
{
	return _libmesh_system->get_dof_map();
}



inline
bool libMeshSystem::active() const
{
	return _libmesh_system->active();
}


inline
void libMeshSystem::activate ()
{
	_libmesh_system->activate() ;
}


inline
void libMeshSystem::deactivate ()
{
	_libmesh_system->deactivate() ;
}


inline
const Variable & libMeshSystem::variable (const unsigned int i) const
{
	return _libmesh_system->variable(i);
}


inline
const FEType & libMeshSystem::variable_type (const unsigned int i) const
{
	return _libmesh_system->variable_type(i);
}



inline
const FEType & libMeshSystem::variable_type (const std::string& var) const
{
	return _libmesh_system->variable_type(var);
}


inline
unsigned int libMeshSystem::n_active_dofs() const
{
	return _libmesh_system->n_active_dofs();
}

inline
Number libMeshSystem::current_solution (const unsigned int global_dof_number) const
{
	return _libmesh_system->current_solution(global_dof_number) ;
}


inline
bool libMeshSystem::compare (const libMeshSystem& other_system,
				const Real threshold,
				const bool verbose) const
{
	return _libmesh_system->compare(*other_system._libmesh_system, threshold, verbose) ;
}

inline
void libMeshSystem::update ()
{
	// update our core libMesh System object
	_libmesh_system->update() ;
}


inline
void libMeshSystem::update_global_solution (std::vector<Number>& global_soln) const
{
	_libmesh_system->update_global_solution(global_soln) ;
}


inline
void libMeshSystem::update_global_solution (std::vector<Number>& global_soln,
											const unsigned int   dest_proc) const
{
	_libmesh_system->update_global_solution(global_soln, dest_proc) ;
}

inline
void libMeshSystem::assemble (bool get_residual, bool get_jacobian )
{
  // convey the assemble command to libMesh system
	// TODO: Check how to get one or the other in the latest version
	// _libmesh_system->assemble(get_residual, get_jacobian) ;
	_libmesh_system->assemble() ;
}


inline
void libMeshSystem::solve ()
{
	/**
	 * Basically, just make sure that \p matrix, \p rhs are \p closed()
	 * and then set the operator for the linear solver to \p matrix.
	 * Then simply call the solve method for the solver.
	 * Since the solver is provided by the user anyway, all optimizations
	 * should be specified on it a-priori.
	 * For all practical purposes, this is just a convenience proxy and
	 * the user can call \p solve() on the solver directly with \p matrix,
	 * \p solution and \p rhs references.
	 */

	// first, check if all data structures are good
// 	karma_assert (matrix != NULL);
// 	karma_assert (matrix->initialized());
// 	karma_assert (rhs    != NULL);
// 	karma_assert (rhs->initialized());
// 	karma_assert (solution    != NULL);
// 	karma_assert (solution->initialized());
	//
// 	// next check the solver.
// 	karma_assert (_solver    != NULL);
// 	karma_assert (_solver->initialized());

	// now call the solve method and update the solution and local_solution.
// 	karma_not_implemented() ;

	// convey the solve command to libMesh system
	_libmesh_system->solve() ;
}





inline
KARMARealVector& libMeshSystem::add_vector (const std::string& vec_name,
								const KARMA::LinearAlgebraLayout type,
								const bool externally_managed )
{
	if (externally_managed)
	{
		using namespace libMesh ;
		libMesh::PetscVector<PetscScalar>& pvec = dynamic_cast<libMesh::PetscVector<PetscScalar>& >(
					_libmesh_system->add_vector(vec_name, false, libMeshEnums::AUTOMATIC) ) ;
		// it looks like a memory leak but we will delete the reference
		// if delete_vector is called and so as long as user can manage this,
		// allocated memory can be retreived.
		KARMARealVector* vecref = new KARMARealVector(pvec.vec()) ;
		return ISystem::add_vector(vec_name, *vecref) ;
	}
	else
	{
		return ISystem::add_vector(vec_name, type, externally_managed) ;
	}
}



inline
bool libMeshSystem::delete_vector (const std::string& vec_name)
{
	return ISystem::delete_vector(vec_name) ;
}



inline
KARMARealMatrix& libMeshSystem::add_matrix (const std::string& mat_name,
										const bool externally_managed)
{
	karma_assert(!_is_explicit_system) ;
	
	if (externally_managed)
	{
		libMesh::PetscMatrix<PetscScalar>& pmat = dynamic_cast<libMesh::PetscMatrix<PetscScalar>& >(
				_libmesh_system->add_matrix(mat_name) ) ;
		
		// it looks like a memory leak but we will delete the reference
		// if delete_vector is called and so as long as user can manage this,
		// allocated memory can be retreived.
		return ISystem::add_matrix(mat_name, new KARMARealMatrix(pmat.mat()) ) ;
	}
	
	return ISystem::add_matrix(mat_name, externally_managed) ;
}



inline
bool libMeshSystem::delete_matrix (const std::string& mat_name)
{
	return ISystem::delete_matrix(mat_name) ;
}








#endif	// __libmesh_system_h__
