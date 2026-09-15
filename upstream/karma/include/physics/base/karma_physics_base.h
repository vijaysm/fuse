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


#ifndef __karma_physics_base_h__
#define __karma_physics_base_h__

// some common parameter identifiers used often.
#define MATERIAL_PROPERTY "material"
#define XCOORD "x"
#define YCOORD "y"
#define ZCOORD "z"
#define TIME "time"
#define DELT "delt"
#define DIMENSION "dim"

// KARMA includes
#include "io_package.h"
#include "vector_value.h"
#include "enum_order.h"

#include "ikarma.h"
#include "karma.h"
#include "karma_parameters.h"
#include "karma_nonlinear_operator.h"
#include "abstract_system_owner.h"
#include "isystem.h"
#include "karma_linear_solver.h"
// #include "karma_mesh.h"

#ifdef LIBMESH_PROJECTION
#include "mesh_function.h"
#endif

// Forward declarations
class KARMAContext ;
class KARMAPhysicsBCManager ;
class KARMAPhysicsBasePropertyManager ;
class KARMAMesh ;

namespace KARMA
{
	typedef std::vector<KARMAPhysicsBCManager* > BCs ;
}

class KARMAPhysicsBase : public IKARMA<KARMAPhysicsBase>, public KARMA::AbstractSystemOwner
{
public:

	/**
	* Constructor.  Optionally initializes required data structures.
	* In this case, a mesh representing the physics has already been created,
	* and is passed for initialization along with BC object.
	*/
	KARMAPhysicsBase (KARMAMesh& o_mesh,
					const std::string name,
					const unsigned int id,
					const KARMA::BCs& o_BCs,
					const std::string order = "FIRST",
					const std::string family = "LAGRANGE") ;

	/**
	* Constructor.  Optionally initializes required data structures.
	* In this case, a mesh representing the physics has already been created,
	* and is passed for initialization.
	*/
	KARMAPhysicsBase (KARMAMesh& o_mesh,
					const std::string name,
					const unsigned int id,
					const std::string order = "FIRST",
					const std::string family = "LAGRANGE") ;

	/**
	* Virtual Destructor. We want to call this even if the derived physics forgets !
	*/
	virtual ~KARMAPhysicsBase() ;

	unsigned int id () ;

	/**
	* @returns a constant reference to the mesh
	*/
	const KARMAMesh& get_mesh() const;

	/**
	* @returns a reference to the mesh
	*/
	KARMAMesh& get_mesh();

	KARMARealVector& get_residual() const ;

	KARMARealVector& get_residual() ;

	KARMARealMatrix& get_jacobian() const ;

	KARMARealMatrix& get_jacobian() ;

	KARMARealVector& get_solution() const ;

	KARMARealVector& get_solution() ;

	KARMARealVector& get_local_solution() const ;

	KARMARealVector& get_local_solution() ;

	KARMARealVector& get_old_solution() const ;

	KARMARealVector& get_old_solution() ;

	KARMARealVector& get_temporary_vector() const ;

	KARMARealVector& get_temporary_vector() ;

	KARMARealMatrix& get_mass_matrix() const ;

	KARMARealMatrix& get_mass_matrix() ;

	KARMALinearSolver* get_mass_solver() const ;

	KARMALinearSolver* get_mass_solver() ;

	bool is_mass_matrix_dirty() const ;

	void mass_matrix_dirty(bool) ;

	virtual void prepare_for_transient() ;

	const ISystem& get_current_system()  const ;

	const ISystem& get_sub_system(int n_system=0) const ;

	ISystem& get_current_system()  ;

	ISystem& get_sub_system(int n_system=0) ;

	// DofMap& get_dof_map()  ;

	virtual const std::vector<unsigned int>& get_send_list () = 0 ;

	virtual void step_preprocess () ;

	virtual void step_postprocess () ;

	virtual void stage_preprocess () ;

	virtual void stage_postprocess () ;

	virtual PetscErrorCode read (std::string& file_prefix) ;

	virtual PetscErrorCode write (std::string& file_prefix, libMeshEnums::IOPackage ioType) ;

	bool is_transient() const ;

	KARMANonlinearOperator* jac_operator ;

	std::vector<KARMALinearOperator*> prec_operator ;

	KARMALinearOperator* mass_operator ;

	/**
	* Clear all the data structures associated with
	* the current Physics.
	*/
	virtual void clear ();

	/**
	* Initializes all the data structures necessary for computation using
	* the current Physics.
	*/
	virtual void init ();

	/**
	* Initializes all the data structures necessary for computation of the
	* other coupled Physics objects to the current Physics.
	* This is a pure virtual function since this base class has no idea of
	* any kind of coupling with other physics. All derived classes, if
	* they have any coupling should implement this routine.
	*/
	virtual void init_coupled_physics () = 0;

	/**
	* Re-Initializes the data structures when mesh changes.
	*/
	virtual void reinit ();

	/**
	* Form initial guess and update provided vector. If exact_solution is defined,
	* then update the currentsolution with the guess at t=0; BCs are enforced
	* automatically through exact solution.
	* If exact solution is not provided, then set all values to zero. And enforce
	* dirichlet conditions as specified by user, strongly.
	*/
	virtual PetscErrorCode form_initial_guess (KARMARealVector& xs) = 0 ;

	/**
	* @returns the number of equation systems.
	*/
	unsigned int n_systems() const;

	/**
	* @returns true if the system named \p name exists within
	* this Physics object.
	*/
	bool has_system (const std::string& name) const;

	/**
	* @returns a constant reference to the system named \p name.
	* The template argument defines the return type.  For example,
	* const SteadySystem& sys = eq.get_system<SteadySystem> ("sys");
	* is an example of how the method might be used
	*/
	const ISystem & get_system (const std::string& name) const;

	/**
	* @returns a writeable referene to the system named \p name.
	* The template argument defines the return type.  For example,
	* const SteadySystem& sys = eq.get_system<SteadySystem> ("sys");
	* is an example of how the method might be used
	*/
	ISystem & get_system (const std::string& name);

	/**
	* @returns a constant reference to system number \p num.
	* The template argument defines the return type.  For example,
	* const SteadySystem& sys = eq.get_system<SteadySystem> (0);
	* is an example of how the method might be used
	*/
	const ISystem & get_system (const unsigned int num) const;

	/**
	* @returns a writeable referene to the system number \p num.
	* The template argument defines the return type.  For example,
	* const SteadySystem& sys = eq.get_system<SteadySystem> (0);
	* is an example of how the method might be used
	*/
	ISystem & get_system (const unsigned int num);

	/**
	* Add the system named \p name to the systems array.
	*/
	//template<typename systype>
	virtual ISystem& add_system (const std::string& name, bool is_explicit=false) ;

	/**
	* Remove the system named \p name from the systems array.
	* This function is now deprecated - write the
	* libmesh-devel mailing list if you need it reimplemented.
	*/
	void delete_system (const std::string& name);

	/**
	* Updates local values for all the systems
	*/
	void update ();

	bool add_coupled_physics_reference(KARMAPhysicsBase *phy) ;

	bool add_coupled_physics_reference(KARMAPhysicsBase *phy, std::string& phyname) ;

	virtual KARMARealVector& get_solution_with_name(const std::string& solname) ;

	virtual KARMARealVector& get_solution_with_name(const std::string& solname) const ;

	void compute_error(const Real dtime, const std::string& var_name, std::vector<Real>& error, const KARMAPhysicsBase* ref_system=NULL) ;

//	virtual void compute_L2_error(const Real dtime, std::vector<Real>& error) ;
//
//	virtual void compute_H1_error(const Real dtime, std::vector<Real>& error) ;
//
//	virtual void compute_L2_error(const Real dtime, const std::string sys_name, std::vector<Real>& error) ;
//
//	virtual void compute_H1_error(const Real dtime, const std::string sys_name, std::vector<Real>& error) ;
//
//	virtual void compute_L2_error(const Real dtime, const std::string sys_name, const KARMARealVector& xsoln, std::vector<Real>& error) ;
//
//	virtual void compute_H1_error(const Real dtime, const std::string sys_name, const KARMARealVector& xsoln, std::vector<Real>& error) ;
//
//	virtual void compute_L2_error(const Real dtime, const KARMAParameters params, const std::string sys_name, const KARMARealVector& xsoln, std::vector<Real>& error) ;
//
//	virtual void compute_H1_error(const Real dtime, const KARMAParameters params, const std::string sys_name, const KARMARealVector& xsoln, std::vector<Real>& error) ;

	/**
	* Update coupled physics solution values, projected on to the
	* current physics mesh. If this is performed once per Newton,
	* or as required, the other physics does not need to be invoked
	* during residual/jacobian assembly, thereby providing (added)
	* computational gain.
	*/
	virtual PetscErrorCode update_coupled_physics_solution() = 0 ;

	/**
	* A virtual method that needs to be implemented by child classes
	* so that a uniform interface to finding the boundary nodes is in
	* place. This method will update the BC for all variables and
	* hence needs to be called if the boundary mesh changes (refine/coarsen)
	* or when BCs are time-dependent functions.
	*/
	virtual PetscErrorCode find_boundary_conditions(KARMAParameters& var_param) ;

	/**
	* Computes the dynamical time scale for the current physics.
	* This is typically the minimum value of `tau` for all dofs
	* where tau = (1/(dphi/dt)) -> Check this
	*/
	PetscReal compute_dynamical_timescale (PetscReal delt, PetscReal time) ;

	/**
	* Perform an explicit prediction of the current physics solution.
	* Useful when performing OS coupling and want to obtain second-order
	* accurate solution; Also effective in accelerating the convergence
	* of Newton since this solution serves as a good initial guess for the
	* new time-step solution.
	*/
	PetscErrorCode predict_solution () ;

	/**
	* Returns a boolean representing whether the physics solution
	* has converged to a given user specified tolerance.
	* Usually the Interface knows whether solution is converged but
	* if the Physics has individual requirements for convergence checking,
	* use this routine.
	*/
	bool is_physics_converged(PetscReal atol, PetscReal rtol) ;

	/**
	* Clears the memory allocated for the system matrix;
	* Caution: Only to be used if a custom preconditioner with
	* different dof structure than the existing system is needed.
	* If not, this will be a waste of operational count =
	* [system_matrix_allocation+system_matrix_deallocation+new_matrix_allocation]
	*/
	void clear_system_matrix() ;

	/**
	* Returns the default quadrature order `recommended` to be used
	* in all the residual, jacobian evaluation routines.
	*/
	Order default_quadrature_order()  const ;

	/**
	* Returns the string identifier for the current physics.
	*/
	virtual std::string system_type () const ;

	/**
	* Returns whether a continuous FE basis is used for discretization.
	*/
	virtual bool is_continuous () const ;

	/**
	* Returns the material number associated with the Element ID.
	*/
	virtual unsigned int material_number (unsigned int elem_id) const = 0 ;

	/**
	* Returns the total number of unique materials in the current mesh.
	*/
	virtual unsigned int n_materials() const ;

	/**
	* Returns the partition (processor) that the current Element ele belongs to.
	*/
	virtual int partition_number (unsigned int elem_id/*const Elem* ele*/) const = 0 ;

	/**
	* Prints detailed and verbose information about the current physics,
	* mesh, variables and dof distribution if being run in parallel.
	*/
	virtual void print_info () ;

	/**
	* Returns the local size (number of dofs in current processor) for current physics.
	*/
	unsigned int local_size() const ;

	/**
	* Returns the global size (number of dofs in all processors) for current physics.
	*/
	unsigned int size() const ;

	/**
	* Returns the dimension of the current mesh.
	*/
	unsigned int dimension() const ;

	/**
	* Returns whether the Physics context has been initialized.
	*/
	bool is_initialized () const ;

	/**
	* Returns the last instance of time when pre_process was called.
	*/
	double cached_time () const ;

	/**
	* Sets whether all elements need to be checked for boundary constraints ?!
	*/
	bool& check_all_elems_for_bcs () ;

	/**
	* Returns whether all elements need to be checked for boundary constraints ?!
	*/
	bool check_all_elems_for_bcs () const ;

	/**
	* Returns the const reference to the boundary conditions container
	*/
	const KARMA::BCs& get_boundary_conditions() const ;

	/**
	* Returns the non-const reference to the coupled physics matrix, if present.
	* Else returns NULL.
	*/
	KARMARealMatrix* get_coupled_physics_matrix(const std::string& phyname) const ;

	/**
	* Returns the non-const reference to the boundary conditions container
	*/
	KARMA::BCs& get_boundary_conditions() ;

	/**
	* Sets whether current physics problem has time dependent BCs
	*/
	bool& time_dependent_bcs () ;
	
#ifdef LIBMESH_PROJECTION
	/**
	* Returns the const reference to the point locator to find the element containing a given point
	*/
	const PointLocatorBase& get_point_locator() const ;
#endif

	/**
	* Sets output file name to write transient data.
	*/
	void set_output_file(std::string& filename) ;

	/**
	* Sets whether the current physics is linear. By default = false;
	* If true, optimizations to utilize this information can be implemented
	* in residual and jacobian evaluations.
	*/
	bool& is_linear_physics() ;

	/**
	* Returns whether the current physics is linear.
	*/
	bool is_linear_physics() const ;

	/**
	* Returns whether current physics problem has time dependent BCs
	*/
	bool time_dependent_bcs () const ;
	
#ifdef LIBMESH_PROJECTION
	// virtual PetscErrorCode project_solution(const std::string& solution_name, KARMARealVector& output,
					// const MeshBase& ext_mesh, const KARMAParameters& params) ;

// 	virtual PetscErrorCode project_solution(KARMARealVector& output,
// 				bool use_exact_solution=false) ;

	virtual PetscErrorCode interpolate_solution(const std::vector<Point>& qpoints,
				DenseVector<Real>& solution ) const ;

	virtual PetscErrorCode interpolate_solution(const Elem* dest_elem,
				const std::vector<std::vector<Real> >& phi,
				DenseVector<Real>& solution ) const ;

	virtual PetscErrorCode compute_projection_from_quadrature_points_matrix(const Elem* dest_elem,
				DenseMatrix<Real>& solution) const ;

	virtual PetscErrorCode compute_projection_of_nodal_solution(const Elem* dest_elem,
				const DenseVector<Real>& rhs_solution, DenseVector<Real>& solution) const;
#endif

  /**
   * @returns the total number of variables in all
   * systems.
   */
  unsigned int n_vars () const;
  
	/**
	* ask the derived physics to allow Physics Base to access the base property manager
	* this allows to make common the routines that involve exact solution, gradient
	* examples: form_initial_guess, compute_L2_error, compute_H1_error etc..
	*/
	virtual const KARMAPhysicsBasePropertyManager& get_properties_base() const = 0  ;

	virtual const std::vector<unsigned int>& get_n_nz() ;

	virtual const std::vector<unsigned int>& get_n_oz() ;
	
  /**
   * Fill the input vector \p var_names with the names
   * of the variables for each system.
   */
  void build_variable_names (std::vector<std::string>& var_names) const;

	
	virtual void solve() ;

	/**
	* A typedef'd facade to EquationSystems.
	*/
	// typedef EquationSystems 	Parent ;


	/**
	* Data structure holding arbitrary parameters.
	*/
	KARMAParameters parameters;


protected:

  void adopt_abstract_system(std::unique_ptr<KARMA::AbstractSystem> system)
  {
    set_abstract_system(std::move(system));
  }

	/**
	* Form initial guess. If exact_solution is defined, then update the current
	* solution with the guess at t=0; BCs are enforced automatically through exact
	* solution.
	* If exact solution is not provided, then set all values to zero. And enforce
	* dirichlet conditions as specified by user, strongly.
	*/
	PetscErrorCode form_initial_guess () ;

	/**
	* calls the public find_boundary_conditions routine and updates all
	* the boundary nodes and constraints with default dim and time
	* params.
	*/
	PetscErrorCode find_boundary_conditions () ;

	/**
	* reference to all BCs object, for all variables
	*/
	KARMA::BCs BC ;

	/**
	* a unique ID for the current physics
	*/
	unsigned int _physics_id ;

	/**
	* using gauss-lobatto points while creating the mesh; useful only for standard cube based meshes.
	*/
	bool _use_lobatto_points ;

	/**
	* reference to the global run context
	*/
	KARMAContext* _ctx ;

	/**
	* an estimate of the dynamical time scale; could use for temporal adaptivity
	*/
	PetscReal _dynamical_time_scale ;

	/**
	* an estimate of the dynamical length scale; could use for spatial adaptivity
	*/
	PetscReal _dynamical_length_scale ;

	/**
	* a Physics owned Parameters class to invoke different calls to properties and such.
	*/
	KARMAParameters _params ;

	/**
	* file name for outputting the transient behavior of peak and/or average evolution of solution.
	*/
	std::string _transient_output_file ;

	/**
	* name of the current physics
	*/
	std::string _name ;

	/**
	* dimension of the problem being tackled by current physics
	*/
	unsigned int _dim ;

	/**
	* total number of unique materials
	*/
	unsigned int _materials ;

	/**
	* the Mass matrix that occurs due to the spatial treatment of the temporal term
	*/
	KARMARealMatrix* _mass ;

	/**
	* a linear solver for inverting the mass-matrix
	*/
	KARMALinearSolver* _mass_solver ;
	
#ifdef LIBMESH_PROJECTION
	/**
	* an object that helps to interpolate the solution at a user-specified point
	*/
	MeshFunction* _mesh_function ;
#endif

	/**
	* the list of material numbers
	*/
	int* _material_numbers ;

	/**
	* reference to the discretized mesh
	*/
	KARMAMesh& _mesh ;

	/**
	* the primary system that has all the variables of interest being solved
	*/
	ISystem& _system ;

	/**
	* other auxillary systems that can hold dependent/derived variables
	* Make sure that no matrix is allocated for these systems, since
	* all of them are explicit.
	*/
	std::vector<ISystem*> _sub_systems ;

	/**
	* reference to the DofMap connectivity between Mesh and Dof numbering
	*/
	// DofMap& _dof_map ;

	/**
	* FEM order of the polynomial basis function: CONSTANT, FIRST, SECOND ...
	*/
	const std::string _order ;

	/**
	* Type of polynomial basis function: LAGRANGE, LEGENDRE
	*/
	const std::string _family ;

	/**
	* Indicator whether init() has been called already
	*/
	bool _initialized ;

	/**
	* Indicator whether the current physics is linear. By default = false
	*/
	bool _is_linear_physics ;

	/**
	* Indicator whether KARMAPhysicsBase should create the matrices when
	* the physics is inited.
	* It is important that the derived Physics handles the matrix allocation
	* or else segfaults could occur.
	*/
	bool _parent_handle_matrices ;

	/**
	* indicator whether prepare_for_transient() has been called.
	*/
	bool _prepared_for_transient ;

	/**
	* indicator whether prepare_for_transient() has been called.
	*/
	bool _predict_solution ;

	/**
	* list of Physics names that current Physics is coupled with.
	*/
	std::vector<KARMARealMatrix*> _coupled_physics_matrices ;

	/**
	* list of Physics names that current Physics is coupled with.
	*/
	std::vector<std::string> _coupled_physics_names ;

	/**
	* list of references to Physics objects that current Physics is coupled with.
	*/
	std::vector<KARMAPhysicsBase *> _coupled_physics ;

	/**
	* the default quadrature order for residual/jacobian/error calculations.
	* any user routine can easily not use this but should take responsibility to
	* correctly integrate the given basis of Order _order.
	*/
	Order _default_quad_order ;

	/**
	* Data structure holding the systems.
	*/
	std::map<std::string, ISystem*> _systems;

	/**
	* Typedef for system iterators
	*/
	typedef std::map<std::string, ISystem*>::iterator       system_iterator;

	/**
	* Typedef for constatnt system iterators
	*/
	typedef std::map<std::string, ISystem*>::const_iterator const_system_iterator;

	/**
	* flag for looping through all elements/sides to check for BCs.
	* just a workaround if you want to assign internal boundaries artificially.
	*/
	bool _check_all_elems_for_bcs ;

	/**
	* flag for checking if mass matrix needs to be recalculated.
	*/
	bool _is_mass_mat_dirty;

	/**
	* flag for checking whether the boundary conditions are time dependent
	*/
	bool _is_time_dependent_bcs ;

	/**
	* last time pre_process was called
	*/
	double _cached_time ;

private:

	void compute_error(const Real dtime, NormType norm, const KARMAParameters params, std::vector<Real>& error) ;

	void compute_error(const Real dtime, NormType norm, const KARMAParameters params, const std::string sys_name, std::vector<Real>& error) ;

	void compute_error(const Real dtime, NormType norm, const KARMAParameters params, const std::string sys_name, const KARMARealVector& soln, std::vector<Real>& error) ;

};

// Inline methods for KARMAPhysicsBase
inline
const KARMAMesh& KARMAPhysicsBase::get_mesh () const
{
  return _mesh;
}



inline
KARMAMesh& KARMAPhysicsBase::get_mesh ()
{
  return _mesh;
}

inline
bool KARMAPhysicsBase::is_mass_matrix_dirty () const
{
	return _is_mass_mat_dirty ;
}

inline
void KARMAPhysicsBase::mass_matrix_dirty (bool is_dirty)
{
	_is_mass_mat_dirty = is_dirty ;
}


inline
bool& KARMAPhysicsBase::check_all_elems_for_bcs ()
{
	return _check_all_elems_for_bcs ;
}

inline
bool KARMAPhysicsBase::check_all_elems_for_bcs () const
{
	return _check_all_elems_for_bcs ;
}

inline
std::string KARMAPhysicsBase::system_type () const
{
	return _name ;
}

inline
unsigned int KARMAPhysicsBase::dimension() const
{
	return _dim ;
}

inline
unsigned int KARMAPhysicsBase::n_materials() const
{
	return _materials ;
}

inline
bool KARMAPhysicsBase::is_initialized () const
{
	return _initialized ;
}

inline
bool KARMAPhysicsBase::is_linear_physics () const
{
	return _is_linear_physics ;
}

inline
bool& KARMAPhysicsBase::is_linear_physics ()
{
	return _is_linear_physics ;
}

inline
Order KARMAPhysicsBase::default_quadrature_order() const
{
	return _default_quad_order ;
}


inline
unsigned int KARMAPhysicsBase::n_systems () const
{
  return _systems.size();
}



//template<typename systype>
inline
ISystem& KARMAPhysicsBase::add_system (const std::string& name, bool is_explicit)
{
	ISystem* ptr = NULL;

	if (!_systems.count(name))
	{
		ptr = new ISystem(*this, name, _systems.size(), is_explicit);

		_systems.insert (std::make_pair(name, ptr));
	}
	else
	{
		ptr = &(this->get_system(name));
	}

	// Return a dynamically casted reference to the newly added System.
	return *ptr;
}



inline
bool KARMAPhysicsBase::has_system (const std::string& name) const
{
  if (_systems.find(name) == _systems.end())
    return false;
  return true;
}




inline
const ISystem & KARMAPhysicsBase::get_system (const unsigned int num) const
{
  karma_assert (num < this->n_systems());


  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    if (pos->second->number() == num)
      break;

  // Check for errors
  if (pos == end)
    {
      KARMA_LOG(ERROR) << "ERROR: no system number " << num << " found!"
		    << std::endl;
      karma_error();
    }
  
  return *pos->second;
}




inline
ISystem & KARMAPhysicsBase::get_system (const unsigned int num)
{
  karma_assert (num < this->n_systems());
  
  const_system_iterator       pos = _systems.begin();
  const const_system_iterator end = _systems.end();

  for (; pos != end; ++pos)
    if (pos->second->number() == num)
      break;

  // Check for errors
  if (pos == end)
    {
      KARMA_LOG(ERROR) << "ERROR: no system number " << num << " found!"
		    << std::endl;
      karma_error();
    }

  return *pos->second;
}



inline
const ISystem & KARMAPhysicsBase::get_system (const std::string& name) const
{
  const_system_iterator pos = _systems.find(name);

  // Check for errors
  if (pos == _systems.end())
    {
      KARMA_LOG(ERROR) << "ERROR: no system named \"" << name << "\" found!"
		    << std::endl;
      karma_error();
    }

  return *pos->second;
}



inline
ISystem & KARMAPhysicsBase::get_system (const std::string& name)
{
  system_iterator pos = _systems.find(name);

  // Check for errors
  if (pos == _systems.end())
    {
      KARMA_LOG(ERROR) << "ERROR: no system named " << name << " found!"
		    << std::endl;
      karma_error();
    }

  return *pos->second;
}



inline
void KARMAPhysicsBase::delete_system (const std::string& name)
{
  if (!_systems.count(name))
    {
      KARMA_LOG(ERROR) << "ERROR: no system named "
                    << name  << std::endl;

      karma_error();
    }
  
  delete _systems[name];
  
  _systems.erase (name);
}



inline
KARMARealVector& KARMAPhysicsBase::get_temporary_vector()
{
	std::string tempvecname = "_temporary_" ;

	return (this->get_solution_with_name(tempvecname)) ;
}

inline
KARMARealVector& KARMAPhysicsBase::get_old_solution()
{
	std::string tempvecname = "_oldsolution_" ;

	return (this->get_solution_with_name(tempvecname)) ;
}


inline
KARMARealMatrix& KARMAPhysicsBase::get_mass_matrix()
{
	karma_assert(_mass != NULL) ;
	
	return *_mass ;
}

inline
KARMARealMatrix& KARMAPhysicsBase::get_mass_matrix() const
{
	karma_assert(_mass != NULL) ;
	
	return *_mass ;
}

// inline
// const KARMARealVector& KARMAPhysicsBase::get_solution_with_name(std::string& solname) const
// {
	// // KARMARealVector* solvec = dynamic_cast< const KARMARealVector *> (&_system.get_vector(solname)) ;
	// // return solvec ;
	// return *(dynamic_cast<const KARMARealVector *> (&_system.get_vector(solname))) ;
// }

inline
KARMARealVector& KARMAPhysicsBase::get_old_solution() const
{
	std::string tempvecname = "_oldsolution_" ;
	return (this->get_solution_with_name(tempvecname)) ;
}

inline
KARMARealVector& KARMAPhysicsBase::get_temporary_vector() const
{
	std::string tempvecname = "_temporary_" ;

	return (this->get_solution_with_name(tempvecname)) ;
}

inline
KARMALinearSolver* KARMAPhysicsBase::get_mass_solver() const
{
	return _mass_solver ;
}

inline
KARMALinearSolver* KARMAPhysicsBase::get_mass_solver()
{
	return _mass_solver ;
}

inline
unsigned int KARMAPhysicsBase::id ()
{
	return _physics_id ;
}

inline
ISystem& KARMAPhysicsBase::get_current_system()
{
	return _system ;
}

inline
ISystem& KARMAPhysicsBase::get_sub_system(int n_system)
{
	return *_sub_systems[n_system] ;
}

inline
const ISystem& KARMAPhysicsBase::get_current_system() const
{
	return _system ;
}

inline
const ISystem& KARMAPhysicsBase::get_sub_system(int n_system) const
{
	return *_sub_systems[n_system] ;
}

inline
const KARMA::BCs& KARMAPhysicsBase::get_boundary_conditions() const
{
	return BC ;
}

inline
KARMA::BCs& KARMAPhysicsBase::get_boundary_conditions()
{
	return BC ;
}

inline
bool& KARMAPhysicsBase::time_dependent_bcs ()
{
	return _is_time_dependent_bcs ;
}

inline
bool KARMAPhysicsBase::time_dependent_bcs () const
{
	return _is_time_dependent_bcs ;
}

inline
bool KARMAPhysicsBase::is_transient() const
{
	return _prepared_for_transient ;
}

inline
void KARMAPhysicsBase::set_output_file(std::string& filename)
{
	_transient_output_file = filename ;
}

inline
KARMARealMatrix* KARMAPhysicsBase::get_coupled_physics_matrix(const std::string& physics_name) const
{
	KARMARealMatrix* coupling_mat = NULL ;
	for(unsigned int iphy=0; iphy < _coupled_physics.size(); iphy++)
	{
		std::string phyname = _coupled_physics_names.at(iphy) ;

		if (!phyname.compare(physics_name))
		{
			coupling_mat = _coupled_physics_matrices[iphy] ;
			break ;
		}
	}

	return coupling_mat ;
}

#ifdef LIBMESH_PROJECTION
inline
const PointLocatorBase& KARMAPhysicsBase::get_point_locator() const
{
	if (_mesh_function != NULL)
	{
		return _mesh_function->	get_point_locator() ;
	}
	else
	{
		KARMA_LOG(ERROR) << "No Mesh function object was found and hence Locator object unavailable." ;
		karma_error() ;
	}
}
#endif

inline
unsigned int KARMAPhysicsBase::size() const
{
	return _system.solution->size() ;
}

inline
unsigned int KARMAPhysicsBase::local_size() const
{
	return _system.solution->local_size() ;
}

inline
KARMARealVector& KARMAPhysicsBase::get_residual()
{
	return *(dynamic_cast<KARMARealVector*>(_system.rhs.get())) ;
}

inline
KARMARealMatrix& KARMAPhysicsBase::get_jacobian()
{
	return *(dynamic_cast<KARMARealMatrix*>(_system.matrix.get())) ;
}

inline
KARMARealVector& KARMAPhysicsBase::get_solution()
{
	return (dynamic_cast<KARMARealVector&>(*_system.solution.get())) ;
}

inline
KARMARealVector& KARMAPhysicsBase::get_local_solution()
{
	return (dynamic_cast<KARMARealVector&>(*_system.current_local_solution.get())) ;
}

inline
KARMARealVector& KARMAPhysicsBase::get_residual() const
{
	return (dynamic_cast<KARMARealVector&>(*_system.rhs.get())) ;
}

inline
KARMARealMatrix& KARMAPhysicsBase::get_jacobian() const
{
	return (dynamic_cast<KARMARealMatrix&>(*_system.matrix.get())) ;
}

inline
KARMARealVector& KARMAPhysicsBase::get_solution() const
{
	return (dynamic_cast<KARMARealVector&>(*_system.solution.get())) ;
}

inline
KARMARealVector& KARMAPhysicsBase::get_local_solution() const
{
	return (dynamic_cast<KARMARealVector&>(*_system.current_local_solution.get())) ;
}


inline
void KARMAPhysicsBase::update ()
{
  // Localize each system's vectors
  for (unsigned int i=0; i != this->n_systems(); ++i)
    this->get_system(i).update();
}



#endif	// #define __karma_physics_base_h__
