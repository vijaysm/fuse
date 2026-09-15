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


#ifndef __karma_interface_h__
#define __karma_interface_h__

// C++ includes
#include <map>
#include <string>

// KARMA includes
#include "io_package.h"
#include "karma_context.h"
#include "karma_physics_base.h"
#include "karma_parameters.h"
#include "petsc_native_nonlinear_solver.h"


class KARMAInterface : public ReferenceCountedObject<KARMAInterface>
{
public:

	static KARMAInterface* get_interface() ;
	
	static KARMAContext* get_context () ;
	
	~KARMAInterface() ;
	
	SNES& get_SNES_context () ;
	
	KSP& get_KSP_context () ;
	
	PC& get_PC_context () ;
	
	template<typename T_sys>
	void add_physics (T_sys *p_physics) ;

	template <typename T_sys>
	const T_sys & get_physics (const std::string& name) const ;

	template <typename T_sys>
	T_sys & get_physics (const std::string& name) ;
	
	template <typename T_sys>
	const T_sys & get_physics (const unsigned int& num) const ;

	template <typename T_sys>
	T_sys & get_physics (const unsigned int& num) ;
	
	KARMAPhysicsBase* get_physics (const unsigned int& num) ;

	bool has_physics (const unsigned int& identifier) ;
	
	void step_preprocess () ;
	
	void step_postprocess () ;

	void stage_preprocess () ;
	
	void stage_postprocess () ;
	
	PetscErrorCode solve_nonlinear_system () ;

	PetscErrorCode form_initial_guess () ;

	void print_info () ;

	PetscErrorCode init() ;
	
	PetscErrorCode reinit() ;

	PetscErrorCode clear() ;

	unsigned int n_physics () const ;
	
	// sum of all dofs from all physics
	unsigned int n_dofs () const ;
	
	PetscErrorCode solve_nonlinear_system (KARMAVector& soln) ;

	PetscErrorCode compute_dynamical_timescales (KARMAVector& old_sol, KARMAVector& curr_sol, Real delt, std::vector<Real>& dyn_time) ;
	
	// PetscErrorCode predict_solution (KARMAVector& solx) ;
	
	PetscErrorCode form_initial_guess (KARMAVector& vec) ;

	/** Copy solution data from interface solution vector to all the corresponding physics.
	* 	Typically should be done after every SNES solve in the interface.
	*   The cached value of the interface solution is copied to individual physics solutions.
	*/
	PetscErrorCode synchronize_physics_solution () ;
	
	/** Copy solution data from given solution vector X to all the corresponding physics.
	* 	Typically should be done after every SNES solve in the interface.
	*   \param KARMAVector X is the global vector that is copied to individual physics solutions.
	*/
	PetscErrorCode synchronize_physics_solution (const KARMAVector& X) ;

	/** Copy solution data from all the corresponding physics to the interface solution vector.
	*   Typically should be done after initializing individual physics ;
	*   The individual physics solutions are copied to the cached value of the interface solution vector.
	*/
	PetscErrorCode synchronize_interface_solution () ;
	
	/** Copy solution data from all the corresponding physics to the given vector X.
	*   Typically should be done after initializing individual physics ;
	*   \param KARMAVector X is the global vector to which individual physics solutions are copied.
	*/
	PetscErrorCode synchronize_interface_solution (KARMAVector& X) ;
	
	/** Enforce constraints when we have hanging nodes and other constraints.
	*  Constrains the solution vector defined on the mesh for each physics's primary system.
	*/
	// PetscErrorCode enforce_constraints() ;

	/** Enforce constraints when we have hanging nodes and other constraints.
	*  Constrains the given vector 'X' defined on the mesh for each physics's primary system.
	*   \param KARMAVector X is the global solution vector if different from Interface's vector.
	*/
	// PetscErrorCode enforce_constraints(KARMAVector& X) ;

	/** Solve the mass matrix occurring from spatial discretization of temporal terms.
	*   The global Mass matrix is block diagonal.
	*   \param KARMAVector X is the global RHS vector for the mass solve.
	*   \return KARMAVector Y contains the operation of M^{-1}*X for all the physics.
	*/
	PetscErrorCode mass_solve (const KARMAVector& X, KARMAVector& Y) ;
	
	/** Return the operation of a vector on a mass matrix occurring from spatial discretization of temporal terms.
	*   This is the block Mass_Matrix*Vector for each physics.
	*   \param KARMAVector X is the global LHS vector for the mass solve.
	*   \return KARMAVector Y contains the operation of M*X for all the physics.
	*/
	PetscErrorCode mass_multiply (const KARMAVector& X, KARMAVector& Y) ;

	/** Computes the global nonlinear steady-state Residual for all physics. This residual accounts for the
	*   effects of coupling with other physics also.
	*   \param KARMAVector X represents the global solution at which residual is computed.
	*   \return KARMAVector R represents the global residual for all the physics.
	*/
	PetscErrorCode system_residual (const KARMAVector& X, KARMAVector& R) ;
	
	/** Computes the global nonlinear steady-state Jacobian matrix for all physics. 
	*   Unless specified by user, this Jacobian matrix is block-diagonal. We will ignore
	*   all the coupling with other physics. Hence, this can be seen as an OS operator matrix.
	*   \param KARMAVector X represents the global solution at which Jacobian is computed.
	*   \return KARMAMatrix J represents the global Jacobian for all the physics. (default=BLOCK_DIAGONAL)
	*/
	PetscErrorCode jacobian_matrix (const KARMAVector& X, KARMAMatrix& J) ;

	PetscErrorCode jacobian_matvec (const KARMAVector& X, KARMAVector& Y ) ;
	
	PetscErrorCode jacobian_solve (const KARMAVector& X, KARMAVector& Y ) ;
	
	PetscErrorCode preconditioner_matvec (const KARMAVector& X, KARMAVector& R) ;
	
	PetscErrorCode preconditioner_matrix (const KARMAVector& X, KARMAMatrix&  J) ;

	PetscErrorCode preconditioner_solve (const KARMAVector& X, KARMAVector& Y ) ;
	
	PetscErrorCode mass_matrix_subtract(KARMAMatrix& P, Real const_multiplier=1.0) ;
	
	PetscErrorCode get_current_residual (KARMAVector& res) ;
	
	/** A helper method that goes through each physics based on given residual vector and 
	*   checks whether the conditions for atol and rtol are met.
	*   \return true if conditions are satisfied for specified physics.
	*/
	bool is_physics_converged(KARMAVector& f, PetscReal atol, PetscReal rtol, std::string& pname) ;

	PetscErrorCode read_solution (std::string& file_prefix) ;

	PetscErrorCode write_solution (std::string& file_prefix, IOPackage ioType=VTK) ;

	PetscErrorCode get_current_residual () ;
	
	/** Sets the context while using a Shell matrix for the Jacobian or Preconditioner.
	*   \param shell_ctx represents the matrix-free context to be stored for the shell-matrices so that
	*   the computation of Jacobian, Jacobian*Vector is easily performed.
	*/
	PetscErrorCode set_shell_matrix_context(void* shell_ctx) ;
		
	/** This method needs to be called before transient simulation is performed.
	*   It is responsible for bootstrapping all the physics and getting them ready to compute
	*   the transient residuals by initializing time-dependent properties, integration vectors etc.
	*/
	void prepare_for_transient() ;
	
	/** Specifies whether everything is prepared for a transient now.
	*   All residual/jacobian calculations from now will transient based
	*   for e.g., Mass - constant * Jacobian instead of just Jacobian..
	*/
	bool is_transient() const ;
	
private:

	// singleton pattern implementation
	static KARMAInterface* __global_KARMA_interface;
	static bool interface_instantiated ;

	PetscInt _gdofs, _ldofs ;
	
	PetscInt       _maxit, _maxf ;
	PetscReal      _abstol, _rtol, _stol ;
	
	// std::vector<PetscReal> _dyn_time ;
	
	std::map< unsigned int, unsigned int > _dof_stride ;
	std::map< unsigned int, unsigned int > _ldof_stride ;
	typedef std::pair< unsigned int, unsigned int > dof_pair ;

	KARMAContext* _CTX ;
	
	KARMAParameters _params ;

	SNES _snes ;
	KSP _ksp ;
	PC _pc ;
	
	bool _initialized ;
	bool _prepared_for_transient ;
	
	// private constructor for our singleton Interface class
	KARMAInterface() ;

	std::map< unsigned int, KARMAPhysicsBase * > _physics ;
	
	PetscErrorCode create_SNES_object() ;
	
	PetscErrorCode synchronize_physics_temporary_vectors(const KARMAVector& X) ;
	PetscErrorCode synchronize_physics_residual (const KARMAVector& R) ;
	PetscErrorCode synchronize_interface_residual (KARMAVector& R) ;
	PetscErrorCode synchronize_interface_jacobian (unsigned int index, KARMAPhysicsBase* physics, KARMAMatrix& J) ;

	PetscErrorCode synchronize_physics_residual () ;
	PetscErrorCode synchronize_interface_residual () ;
	

	/**
	* Typedef for system iterators
	*/
	typedef std::map<unsigned int, KARMAPhysicsBase*>::iterator       physics_iterator;

	/**
	* Typedef for constatnt system iterators
	*/
	typedef std::map<unsigned int, KARMAPhysicsBase*>::const_iterator const_physics_iterator;

	friend class TemporalIntegrator ;
};


template<typename T_sys>
inline
void KARMAInterface::add_physics (T_sys *p_physics)
{
	unsigned int identifier = p_physics->id() ;
	if (!_physics.count(identifier))
	{
		_physics.insert (std::make_pair(identifier, p_physics));
	}
}

template <typename T_sys>
inline
const T_sys & KARMAInterface::get_physics (const std::string& identifier) const
{
  const_physics_iterator       pos = _physics.begin();
  const const_physics_iterator end = _physics.end();

  for (; pos != end; ++pos)
    if (pos->second->system_type() == identifier)
      break;

  // Check for errors
  if (pos == end)
    {
      std::cerr << "ERROR: no system named " << identifier << " found!" << std::endl;
      karma_error();
    }

  // Attempt dynamic cast
  T_sys* ptr = dynamic_cast<T_sys*>(pos->second);

  // Check for failure of dynamic cast
  if (ptr == NULL)
    {
      std::cerr << "ERROR: cannot convert system " << identifier << " to requested type!" << std::endl;
      karma_error();
    }
  
  return *ptr;
}

template <typename T_sys>
inline
T_sys & KARMAInterface::get_physics (const std::string& identifier)
{
  const_physics_iterator       pos = _physics.begin();
  const const_physics_iterator end = _physics.end();

  for (; pos != end; ++pos)
    if (pos->second->system_type() == identifier)
      break;

  // Check for errors
  if (pos == end)
    {
      std::cerr << "ERROR: no system named " << identifier << " found!" << std::endl;
      karma_error();
    }

  // Attempt dynamic cast
  T_sys* ptr = dynamic_cast<T_sys*>(pos->second);

  // Check for failure of dynamic cast
  if (ptr == NULL)
    {
      std::cerr << "ERROR: cannot convert system " << identifier << " to requested type!" << std::endl;
      karma_error();
    }
  
  return *ptr;
}


inline
bool KARMAInterface::has_physics (const unsigned int& identifier)
{
  if (_physics.find(identifier) == _physics.end())
    return false;
  return true;
}


template <typename T_sys>
inline
const T_sys & KARMAInterface::get_physics (const unsigned int& identifier) const
{
  karma_assert (identifier < this->n_physics());
  
  const_physics_iterator       pos = _physics.find(identifier) ;
  
  // Check for errors
  if (pos == _physics.end())
  {
      std::cerr << "ERROR: no system number " << identifier << " found!" << std::endl;
      karma_error();
  }

  // Attempt dynamic cast
  T_sys* ptr = dynamic_cast<T_sys*>(pos->second);

  // Check for failure of dynamic cast
  if (ptr == NULL)
    {
      std::cerr << "ERROR: cannot convert system " << identifier << " to requested type!" << std::endl;
      karma_error();
    }
  
  return *ptr;
}


template <typename T_sys>
inline
T_sys & KARMAInterface::get_physics (const unsigned int& identifier)
{
  karma_assert (identifier < this->n_physics());
  
  const_physics_iterator       pos = _physics.find(identifier) ;
  
  // Check for errors
  if (pos == _physics.end())
  {
      std::cerr << "ERROR: no system number " << identifier << " found!" << std::endl;
      karma_error();
  }

  // Attempt dynamic cast
  T_sys* ptr = dynamic_cast<T_sys*>(pos->second);

  // Check for failure of dynamic cast
  if (ptr == NULL)
    {
      std::cerr << "ERROR: cannot convert system "
		<< identifier << " to requested type!"
		<< std::endl;
      karma_error();
    }

  return *ptr; 
}


inline
KARMAPhysicsBase* KARMAInterface::get_physics (const unsigned int& identifier)
{
	karma_assert (identifier < this->n_physics());

	const_physics_iterator       pos = _physics.begin();
	const const_physics_iterator end = _physics.end();

	for (; pos != end; ++pos)
	if (pos->second->id() == identifier)
	  break;

	// Check for errors
	if (pos == end)
	{
		std::cerr << "ERROR: no system number " << identifier << " found!" << std::endl;
		karma_error();
	}

	return pos->second ;
}

inline
SNES& KARMAInterface::get_SNES_context ()
{
	return _snes ;
}


inline
KSP& KARMAInterface::get_KSP_context ()
{
	return _ksp ;
}

inline
PC& KARMAInterface::get_PC_context ()
{
	return _pc ;
}


inline
unsigned int KARMAInterface::n_physics () const
{
  return _physics.size();
}


inline
unsigned int KARMAInterface::n_dofs () const
{
	return _gdofs ;
}

inline
bool KARMAInterface::is_transient() const
{
	return _prepared_for_transient ;
}


#endif	// #define __karma_interface_h__
