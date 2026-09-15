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



#ifndef __isystem_h__
#define __isystem_h__

// C++ includes

// Local Includes
#include "karma_error.h"
#include "user_sparsity.h"
#include "karma_vector.h"
#include "karma_matrix.h"

// Forward declarations
class KARMAPhysicsBase;


/**
 * This class provides an abstract system class.  It aims
 * at providing the necessary definition needed to solve an 
 * equation system. 
 */

// ------------------------------------------------------------
// ISystem class definition

class ISystem : public IKARMA<ISystem>
{
public:

	/**
	* Constructor.  Optionally initializes required
	* data structures.
	*/
	ISystem (KARMAPhysicsBase& es,
		  const std::string& name,
		  const unsigned int number,
			  bool is_explicit=false);

	/**
	* Copy Constructor.  Explicitly disallow copy
	* by value.
	*/
	ISystem(const ISystem& other) ;

	/**
	* Operator assignment. Same as above.
	* explicitly block copy by value operations.
	*/
	ISystem& operator= (const ISystem&) ;

	/**
	* Destructor.
	*/
	virtual ~ISystem ();

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
	virtual void init (unsigned int ndofs, unsigned int n_local_dofs,
					   std::vector<unsigned int>& send_list,
					   std::vector<unsigned int>* n_nz,
					   std::vector<unsigned int>* o_nz) ;

	/**
	* Update the local values to reflect the solution
	* on neighboring processors.
	*/
	virtual void update ();

	/**
	* Assembles & solves the linear system Ax=b.
	*/
	virtual void solve ();

	/**
	* @returns \p "Implicit".  Helps in identifying
	* the system type in an equation system file.
	*/
	virtual std::string system_type () const { return "ISystem"; }

	/**
	* Accepts a pointer to a linear solver appropriate for use in
	* the current system. This is usually called by the Physics
	* which understands the details of solver, preconditioner and
	* other relevant parameters necessary to obtain a stable
	* convergence.
	* If a solver is not attached, \p assemble() will throw an error.
	*/
// 	virtual void attach_linear_solver(ISolver* ) ;

	/**
	* @returns the system number.
	*/
	unsigned int number () const;

	/**
	* Fill the input vector \p global_soln so that it contains
	* the global solution on all processors.
	* Requires communication with all other processors.
	*/
	virtual void update_global_solution (std::vector<Number>& global_soln) const;

	/**
	* Vector iterator typedefs.
	*/
	typedef std::map<std::string, KARMARealVector* >::iterator       vectors_iterator;
	typedef std::map<std::string, KARMARealVector* >::const_iterator const_vectors_iterator;


	/**
	* Beginning of vectors container
	*/
	vectors_iterator vectors_begin ();

	/**
	* Beginning of vectors container
	*/
	const_vectors_iterator vectors_begin () const;

	/**
	* End of vectors container
	*/
	vectors_iterator vectors_end ();

	/**
	* End of vectors container
	*/
	const_vectors_iterator vectors_end () const;

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
	 * Adds the user-provided vector with \p vec_name to this system.
	 * By default, these type of vectors are user-managed and so ISystem
	 * will take not responsibility to init(), clear() the vector.
	 */
	KARMARealVector& add_vector (  const std::string& vec_name,
									KARMARealVector& vec) ;

	/**
	* @returns \p true if this \p System has a vector associated with the
	* given name, \p false otherwise.
	*/
	bool have_vector (const std::string& vec_name) const;

	/**
	* @returns the number of vectors (in addition to the solution)
	* handled by this system
	* This is the size of the \p _vectors map
	*/
	unsigned int n_vectors () const;

	/**
	 * Deletes the additional vector \p vec_name in this system.
	 */
	virtual bool delete_vector (const std::string& vec_name);

	/**
	* @returns a const pointer to the vector if this \p System has a
	* vector associated with the given name, \p NULL otherwise.
	*/
	const KARMARealVector* request_vector (const std::string& vec_name) const;

	/**
	* @returns a pointer to the vector if this \p System has a
	* vector associated with the given name, \p NULL otherwise.
	*/
	KARMARealVector* request_vector (const std::string& vec_name);

	/**
	* @returns a const pointer to this system's @e additional vector
	* number \p vec_num (where the vectors are counted starting with
	* 0), or returns \p NULL if the system has no such vector.
	*/
	const KARMARealVector* request_vector (const unsigned int vec_num) const;

	/**
	* @returns a writeable pointer to this system's @e additional
	* vector number \p vec_num (where the vectors are counted starting
	* with 0), or returns \p NULL if the system has no such vector.
	*/
	KARMARealVector* request_vector (const unsigned int vec_num);

	/**
	* @returns a const reference to this system's @e additional vector
	* named \p vec_name.  Access is only granted when the vector is already
	* properly initialized.
	*/
	const KARMARealVector& get_vector (const std::string& vec_name) const;

	/**
	* @returns a writeable reference to this system's @e additional vector
	* named \p vec_name.  Access is only granted when the vector is already
	* properly initialized.
	*/
	KARMARealVector& get_vector (const std::string& vec_name);

	/**
	* @returns a const reference to this system's @e additional vector
	* number \p vec_num (where the vectors are counted starting with
	* 0).
	*/
	const KARMARealVector& get_vector (const unsigned int vec_num) const;

	/**
	* @returns a writeable reference to this system's @e additional
	* vector number \p vec_num (where the vectors are counted starting
	* with 0).
	*/
	KARMARealVector& get_vector (const unsigned int vec_num);

	/**
	* @returns the name of this system's @e additional vector number \p
	* vec_num (where the vectors are counted starting with 0).
	*/
	const std::string & vector_name (const unsigned int vec_num);

	/**
	* Matrix iterator typedefs.
	*/
	typedef std::map<std::string, KARMARealMatrix* >::iterator        matrices_iterator;
	typedef std::map<std::string, KARMARealMatrix* >::const_iterator  const_matrices_iterator;

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
	 * Adds the user-provided vector with \p vec_name to this system.
	 * By default, these type of vectors are user-managed and so ISystem
	 * will take not responsibility to init(), clear() the vector.
	 */
	virtual KARMARealMatrix& add_matrix(  const std::string& mat_name,
									  KARMARealMatrix& mat) ;
	
	/**
	 * This routine allows the user to add a matrix to the collection,
	 * even after the system has been initialized. In fact, that is one
	 * of the necessary conditions for this to proceed correctly.
	 * It will return the existing matrix, if it exists, initialize it
	 * if it is not already and return; else it will create a new matrix,
	 * add to the collection, initialize and return the reference for
	 * the same.
	 */
	KARMARealMatrix& init_matrix (const std::string& mat_name) ;
	
	/**
	* Deletes the additional matrix \p mat_name to this system.
	*/
	virtual bool delete_matrix (const std::string& mat_name);

	/**
	* @returns \p true if this \p System has a matrix associated with the
	* given name, \p false otherwise.
	*/
	bool have_matrix (const std::string& mat_name) const;

	/**
	* @returns a const pointer to this system's @e additional matrix
	* named \p mat_name, or returns \p NULL if no matrix by that name
	* exists.
	*/
	const KARMARealMatrix* request_matrix (const std::string& mat_name) const;

	/**
	* @returns a writable pointer to this system's @e additional matrix
	* named \p mat_name, or returns \p NULL if no matrix by that name
	* exists.
	*/
	KARMARealMatrix* request_matrix (const std::string& mat_name);

	/**
	* @returns a const reference to this system's @e additional matrix
	* named \p mat_name.  @e None of these matrices is involved in the 
	* solution process.  Access is only granted when the matrix is already
	* properly initialized.
	*/
	const KARMARealMatrix& get_matrix (const std::string& mat_name) const;

	/**
	* @returns a writeable reference to this system's @e additional matrix
	* named \p mat_name.  @e None of these matrices is involved in the 
	* solution process.  Access is only granted when the matrix is already
	* properly initialized.
	*/
	KARMARealMatrix& get_matrix (const std::string& mat_name);

	/**
	* @returns the number of matrices handled by this system
	*/
	unsigned int n_matrices () const;

	/**
	* @returns the const reference to UserSparsity object handle for the system
	*/
	const UserSparsity& sparsity () const { return _sparsity ; }

	/**
	* @returns the reference to UserSparsity object handle for the system
	*/
	UserSparsity& sparsity () { return _sparsity ; }

	/**
	* The system matrix.  Implicit systems are characterized by
	* the need to solve the linear system Ax=b.  This is the
	* right-hand-side vector b.
	*/
	AutoPtr<KARMARealVector> rhs;

	/**
	* The system matrix.  Implicit systems are characterized by
	* the need to solve the linear system Ax=b.  This is the
	* system matrix A.
	*/
	AutoPtr<KARMARealMatrix> matrix;

	/**
	* @returns the number of variables in the system
	*/
	unsigned int n_vars() const;

	/**
	 * @returns the reference to number of degrees of freedom in the system
	 * allow user to set this size information.
	 */
	unsigned int& n_dofs() { return _sparsity.global_dofs ;}

	/**
	 * @returns the reference to number of degrees of freedom local
	 * to this processor; allow user to set this size information.
	 */
	unsigned int& n_local_dofs() { return _sparsity.local_dofs ;}

	/**
	* @returns the number of degrees of freedom in the system
	*/
	unsigned int n_dofs() const { return _sparsity.global_dofs ;}

	/**
	* @returns the number of degrees of freedom local
	* to this processor
	*/
	unsigned int n_local_dofs() const  { return _sparsity.local_dofs ;}

	/**
	* Adds the variable \p var to the list of variables
	* for this system.  Returns the index number for the new variable.
	*/
	virtual unsigned int add_variable (const std::string& var);

	/** 
	* Return a constant reference to \p Variable \p var.
	*/
// 	const Variable& variable (unsigned int var) const;

	/**
	* @returns true if a variable named \p var exists in this System
	*/
	bool have_variable(const std::string& var) const;

	/**
	* @returns the name of variable \p i.
	*/
	const std::string & variable_name(const unsigned int i) const;

	/**
	* @returns the variable number assoicated with
	* the user-specified variable named \p var.
	*/
	unsigned short int variable_number (const std::string& var) const;

	/**
	* @returns \p true when the other system contains
	* identical data, up to the given threshold.  Outputs
	* some diagnostic info when \p verbose is set.
	*/
	virtual bool compare (const ISystem& other_system,
						const Real threshold,
						const bool verbose) const;

	/**
	* @returns the system name.
	*/
	const std::string & name () const;

	//--------------------------------------------------
	// The solution and solution access members

	/**
	* Data structure to hold solution values.
	*/
	AutoPtr<KARMARealVector> solution;

	/**
	* All the values I need to compute my contribution
	* to the simulation at hand.  Think of this as the
	* current solution with any ghost values needed from
	* other processors.  This vector is necessarily larger
	* than the \p solution vector in the case of a parallel
	* simulation.  The \p update() member is used to synchronize
	* the contents of the \p solution and \p current_local_solution
	* vectors.
	*/
	AutoPtr<KARMARealVector> current_local_solution;

	/**
	* @returns a writeable reference to the flag that indicates whether the system matrix
	* and other matrices hence, should be initialized when init_matrices() is called. 
	* If this is set to "false" and init() is called, user is responsible for allocating memory
	* for all the matrices or else this function can be called again to set to "true" with a 
	* call to reinit() on the system.
	*/
	bool& delay_matrix_initialization() ;

	/**
	* This routine primarily prepares the current system for usage by initializing 
	* the matrices associated with this system. It is imperative that \p n_dofs and \p n_local_dofs
	* have been set a-priori before calling this routine. If delay_matrix_initialization was set,
	* this routine can be used to re-allocate system and other matrices. 
	* Internally, each physics can call this routine during prepare_for_transient() when  
	* delayed allocation is preferred or before if a SS problem is being solved.
	*/
	bool prepare_for_usage() ;

	/**
	 * Sets the preference whether ghosted vectors are needed
	 * instead of full serial vectors for all except \p solution
	 * and \p rhs .
	 */
	void enable_ghosted_vectors();

  
protected:

	/**
	* Initializes the matrices associated with this system.
	*/
	virtual void init_matrices (const std::vector<unsigned int>& n_nz, const std::vector<unsigned int>& o_nz) ;

	/**
	* Add the system matrix to the \p _matrices data structure.
	* Useful in initialization.
	*/
	void add_system_matrix ();

	/**
	* Add the system right-hand-side vector to the \p _vectors data structure.
	* Useful in initialization.
	*/
	void add_system_rhs ();

	/**
	* Constant reference to the \p EquationSystems object
	* used for the simulation.
	*/
	KARMAPhysicsBase& _equation_systems;

	/**
	 * A reference to the system solver.
	 */
// 	ISolver* _solver ;

	/**
	* A name associated with this system.
	*/
	const std::string _sys_name;

	/**
	* The number associated with this system
	*/
	const unsigned int _sys_number;

	/**
	* The \p Variables in this \p System.
	*/
	std::vector<std::string> _variable_names;

	/**
	* The variable numbers corresponding to user-specified
	* names, useful for name-based lookups.
	*/
	std::map<std::string, unsigned short int> _variable_numbers;

	/**
	* Some systems need an arbitrary number of vectors.
	* This map allows names to be associated with arbitrary
	* vectors.  All the vectors in this map will be distributed
	* in the same way as the solution vector.
	*/
	std::map<std::string, KARMARealVector* > _vectors;

	/**
	 * Some systems need an arbitrary number of matrices.
	 */
	std::map<std::string, KARMARealMatrix* > _matrices;

	/**
	* Holds the type of a vector
	*/
	std::map<std::string, KARMA::LinearAlgebraLayout> _vector_types;

	/**
	 * \p true when the vector in the collection is externally managed
	 * by either the derived system or by user himself. Hence, do not 
	 * perform any non-const operations on the vector from ISystem
	 * directly.
	 */
	std::map<std::string, bool> _is_externally_managed_vector;

	/**
	 * \p true when the matrix in the collection is externally managed
	 * by either the derived system or by user himself. Hence, do not
	 * perform any non-const operations on the matrix from ISystem
	 * directly.
	 */
	std::map<std::string, bool> _is_externally_managed_matrix;

	/**
	* \p true when no matrices are needed; Only \p solution and \p rhs vectors are allocated.
	*/
	bool _is_explicit_system;

	/**
	* Some systems need an arbitrary matrix sparsity that is not dictated by dof_map.
	*/
	UserSparsity _sparsity;

	/**
	* \p true when matrix initialization will be taken care of by user. \p false otherwise.
	*/
	bool _delay_matrix_initialization ;
	
// private:

	/**
 	* \p true when the system has been initialized
	*/
	bool _is_initialized;

	/**
	 * \p true when the system has been prepared and ready for usage
	 * i.e., all vectors/matrices have been allocated according to
	 * provided SparsityPattern and can be used to assemble/solve/IO etc.
	 */
	bool _prepared_for_usage;

	/**
	 * \p true when user wants Ghosted vectors instead of Serial vectors.
	 */
	bool _ghosted_vectors_enabled;

	/**
	 * \p true when either one of vector or matrix in collection is user managed.
	 * We will automatically assume then that the \p solution and \p local_solution
	 * are also user managed.
	 */
	bool _is_externally_managed;

};



// ------------------------------------------------------------
// ISystem inline methods
inline
bool ISystem::have_matrix (const std::string& mat_name) const
{
  return (_matrices.count(mat_name));
}

inline
bool ISystem::have_vector (const std::string& vec_name) const
{
	return (_vectors.count(vec_name));
}


inline
unsigned int ISystem::n_matrices () const
{
 return _matrices.size(); 
}

inline
bool& ISystem::delay_matrix_initialization()
{
	return _delay_matrix_initialization ;
}


inline
const std::string & ISystem::name() const
{
	return _sys_name;
}



inline
unsigned int ISystem::number() const
{
	return _sys_number;
}


inline
unsigned int ISystem::n_vars() const
{
	return _variable_names.size();
}


inline
const std::string & ISystem::variable_name (const unsigned int i) const
{
	karma_assert (i < _variable_names.size());

	return _variable_names[i] ;
}


inline
unsigned int ISystem::add_variable (const std::string& var)
{
	// check if variable name already exists
	karma_assert() ;

	// this is a new variable. add to the list
	_variable_names.push_back(var) ;

	// variable numbers are zero index based
	_variable_numbers[var] = _variable_names.size()-1 ;

	return _variable_numbers[var] ;
}


inline
bool ISystem::have_variable(const std::string& var) const
{
	return _variable_numbers.count(var);
}


inline
unsigned int ISystem::n_vectors () const
{
	return _vectors.size();
}

inline
ISystem::vectors_iterator ISystem::vectors_begin ()
{
	return _vectors.begin();
}

inline
ISystem::const_vectors_iterator ISystem::vectors_begin () const
{
	return _vectors.begin();
}

inline
ISystem::vectors_iterator ISystem::vectors_end ()
{
	return _vectors.end();
}

inline
ISystem::const_vectors_iterator ISystem::vectors_end () const
{
	return _vectors.end();
}

inline
void ISystem::enable_ghosted_vectors()
{
	_ghosted_vectors_enabled = true ;
}

inline
void ISystem::add_system_rhs ()
{
	// Possible that we cleared the _vectors but
	// forgot to NULL-out the rhs?
	if (_vectors.empty()) rhs.reset() ;

	// Only need to add the rhs if it isn't there
	// already!
	if (rhs.get() == NULL)
		*rhs = (this->add_vector ("RHS Vector", KARMA::PARALLEL, false));

	karma_assert (rhs.get() != NULL);
}


inline
void ISystem::add_system_matrix ()
{
	// if this is an explicit system, then do not create the matrix
	if (_is_explicit_system)
		return ;
	
	// Possible that we cleared the _matrices but
	// forgot to NULL-out the matrix?
	if (_matrices.empty()) matrix.reset() ;
	
	// Only need to add the matrix if it isn't there
	// already!
	if (matrix.get() == NULL)
		*matrix = this->add_matrix ("System Matrix", false) ;
	
	karma_assert (matrix.get() != NULL);
}

inline
void ISystem::init_matrices (const std::vector<unsigned int>& n_nz, const std::vector<unsigned int>& o_nz)
{
	karma_not_implemented() ;
}

#endif	// __isystem_h__
