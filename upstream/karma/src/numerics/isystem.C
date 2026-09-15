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
// #include "karma_physics_base.h"

// #include "dof_map.h"
#include "isystem.h"

// ------------------------------------------------------------
// ISystem implementation
ISystem::ISystem (KARMAPhysicsBase& es,
				const std::string& name,
				const unsigned int number,
				bool is_explicit) :
				matrix            			(NULL),
				rhs							(NULL),
				_delay_matrix_initialization(false),
				solution					(NULL),
				current_local_solution		(NULL),
 				_equation_systems			(es),
// 				_solver						(NULL),
				_sys_name					(name),
				_sys_number					(number),
				_is_explicit_system			(is_explicit),
				_is_initialized				(false),
				_prepared_for_usage			(false),
				_is_externally_managed		(false)
{
}


// No copy construction of System objects!
ISystem::ISystem (const ISystem& other) :
				IKARMA<ISystem>(),
 				_equation_systems			(other._equation_systems),
				_sys_name					(other._sys_name),
				_sys_number					(other._sys_number)
{
	karma_exception("Copy constructor disallowed for all System objects.");
}



ISystem& ISystem::operator= (const ISystem&)
{
	karma_exception("Operator= override is disallowed for all System objects.");
}


ISystem::~ISystem ()
{
  // Clear data
  this->clear();
}



void ISystem::clear ()
{
	_variable_names.clear();
  
	_variable_numbers.clear();

	if (!_is_externally_managed)
	{
		solution->clear ();
	
		rhs->clear() ;
	
		current_local_solution->clear ();
	
		// clear any user-added vectors
		if (_vectors.size())
		{
			for (vectors_iterator pos = _vectors.begin(); pos != _vectors.end(); ++pos)
			{
				pos->second->clear ();
				if (!_is_externally_managed_vector[pos->first])
					delete pos->second;
				pos->second = NULL;
			}
		
			_vectors.clear();
			_vector_types.clear();
		}
		
		// clear any user-added matrices
		if (_matrices.size())
		{
			for (matrices_iterator pos = _matrices.begin(); pos != _matrices.end(); ++pos)
			{
				pos->second->clear ();
				
				if (!_is_externally_managed_matrix[pos->first])
					delete pos->second;
				pos->second = NULL;
				
// 				if (_delay_matrix_initialization)	// really ???
// 				{
// 					PetscErrorCode ierr = MatDestroy(pos->second->mat()) ;	CHKERRABORT(KARMA_COMM_WORLD,ierr);
// 				}
			}
			
			_matrices.clear();
		}
		
		// NULL the matrix.
		matrix->clear();
	}
	else
	{
		_vectors.clear();
		_vector_types.clear();
		_matrices.clear();
	}

	_is_initialized = false ;
	
}


void ISystem::init ()
{
	// check if we have initialized already
	if (!_is_initialized)
	{
		// update the status of external data-structure management flag
		// first check the vectors
		for (vectors_iterator pos = _vectors.begin(); pos != _vectors.end(); ++pos)
			_is_externally_managed = _is_externally_managed || _is_externally_managed_vector[pos->first] ;

		// next check the matrices
		for (matrices_iterator pos = _matrices.begin(); pos != _matrices.end(); ++pos)
			_is_externally_managed = _is_externally_managed || _is_externally_managed_matrix[pos->first] ;

		/**
		* NOTE: Even if this System is externally_managed, we can add more vectors and matrices
		* via calls to add_vector and add_matrix.
		* But the overriding system should set the sparsity pattern based on discretization/connectivity
		* before this will work.
		 */

		// check if we have everything we need to proceed
		karma_assert (_sparsity.is_ready()) ;

		_is_initialized = true ;
		
		// Not entirely sure if we should we make this assumption !
		// Shouldn't the user be able to manage only auxillary vectors/matrices and
		// not control the main solution vectors ?
		if (!_is_externally_managed)
		{
			// Resize the solution conformal to the user specified distribution
			solution->init (_sparsity.global_dofs, _sparsity.local_dofs, false, KARMA::PARALLEL);
		
			// Resize the current_local_solution for the current mesh
			if (_ghosted_vectors_enabled)
			{
				current_local_solution->init (_sparsity.global_dofs, _sparsity.local_dofs,
									*_sparsity.send_list, false);
			}
			else
			{
				current_local_solution->init (_sparsity.global_dofs, false, KARMA::SERIAL);
			}
	
			// check if we have a flag for every vector
			karma_assert(_is_externally_managed_vector.size() == _vectors.size()) ;
			
			// initialize & zero other vectors, if necessary
			for (vectors_iterator pos = _vectors.begin(); pos != _vectors.end(); ++pos)
			{
				// if user manages it, do not initialize it here.
				if (_is_externally_managed_vector[pos->first])
					continue ;
				
				KARMA::LinearAlgebraLayout type = _vector_types[pos->first];
				
				if(type == KARMA::GHOSTED && _ghosted_vectors_enabled)
				{
					pos->second->init (_sparsity.global_dofs, _sparsity.local_dofs,
							*_sparsity.send_list, false);
				}
				else
				{
					// NOTE: should we change the type if user specifically did not ask for GHOSTED ?
					pos->second->init (_sparsity.global_dofs, _sparsity.local_dofs, false, type);
				}
			}

			// Add the system matrix.
			this->add_system_matrix ();
	
			// if user wants to allocate the matrices separately, that is fine
			// but if not, use the given sparsity index to allocate the operators
			if (!_delay_matrix_initialization)
			{
					// check whether our sparsity patterns are NULL
				karma_assert(n_nz != NULL && o_nz != NULL) ;
					
					// Initialize the matrices for the system
				karma_assert (matrix != NULL);
			
				// Check for quick return in case the system matrix
				// (and by extension all the matrices) has already
				// been initialized.
				if (matrix->initialized())
					return;
			
				this->prepare_for_usage() ;
			}
		}
		else
		{
			// almost nothing is done with this call
			// but it maintains consistency in the code path
			this->prepare_for_usage() ;
		}
		
	}
}


void ISystem::init (unsigned int n_dofs, unsigned int n_local_dofs,
					std::vector<unsigned int>& send_list,
					std::vector<unsigned int>* n_nz, std::vector<unsigned int>* o_nz)
{
	// if user spent some time setting up the sparsity correctly, then
	// let us not over-write anything yet.
	if (!_sparsity.is_ready())
	{
		_sparsity.global_dofs = n_dofs ;
		_sparsity.local_dofs = n_local_dofs ;
		_sparsity.send_list = &send_list ;
	
		// if the given n_nz and o_nz are NULL, provide crude, default values
		_sparsity.nnz_per_row = (n_nz == NULL ? 30 : 0) ;
		_sparsity.onz_per_row = (o_nz == NULL ? 10 : 0) ;
		_sparsity.n_nz = n_nz ;
		_sparsity.o_nz = o_nz ;
	}

	// now that sparsity pattern is updated, call the real init()
	this->init() ;
}



bool ISystem::prepare_for_usage ()
{
	// we shouldn't be preparing for usage without calling init first
	if (!_is_initialized)
		return false ;

	/**
	 * check whether this is an implicit system
	 * then no need to add the matrices.
	 * if user needs it, add/initialize it explicitly with
	 * \p init_matrix routine.
	 */
	if (_is_explicit_system) 
		return true ;

	if (!_prepared_for_usage)
	{
		// check if our sparsity pattern is ready for usage
		if (!_sparsity.is_ready())
		{
			karma_exception("The global and/or local degrees-of-freedom cannot be zero.") ;
		}

		if (!_is_externally_managed)
		{
			// check if we have a flag for every vector
			karma_assert(_is_externally_managed_matrix.size() == _matrices.size()) ;
			
			// Initialize matrices
			for (matrices_iterator pos = _matrices.begin(); pos != _matrices.end(); ++pos)
			{
				// if user manages it, do not initialize it here.
				if (_is_externally_managed_matrix[pos->first])
					continue ;
				
				karma_assert (!pos->second->initialized());
				if (_sparsity.n_nz == NULL)
				{
					pos->second->init(_sparsity.global_dofs, _sparsity.global_dofs, _sparsity.local_dofs, _sparsity.local_dofs,
									_sparsity.nnz_per_row, _sparsity.onz_per_row) ;
				}
				else
				{
					pos->second->init(_sparsity.global_dofs, _sparsity.global_dofs, _sparsity.local_dofs, _sparsity.local_dofs,
									*_sparsity.n_nz, *_sparsity.o_nz) ;
				}
		
				// let us also initialize the data to zero
				pos->second->zero ();
		
			}
		
			// the original system matrix reference was destroyed. So update the new reference.
			matrix = libMesh::AutoPtr<KARMARealMatrix>(_matrices["System Matrix"]) ;
		}
		
		_prepared_for_usage = true ;
	}
	
	return _prepared_for_usage ;
}


void ISystem::solve ()
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
	karma_assert (matrix != NULL);
	karma_assert (matrix->initialized());
	karma_assert (rhs    != NULL);
	karma_assert (rhs->initialized());
	karma_assert (solution    != NULL);
	karma_assert (solution->initialized());

	// next check the solver.
// 	karma_assert (_solver    != NULL);
// 	karma_assert (_solver->initialized());

	// now call the solve method and update the solution and local_solution.
	karma_not_implemented() ;
}


KARMARealMatrix& ISystem::add_matrix (  const std::string& mat_name,
									KARMARealMatrix& mat)
{
	// Return the matrix if it is already there.
	if (this->have_matrix(mat_name))
		return *(_matrices[mat_name]);

	// insert the user-provided vector in the collection
	// Otherwise build the matrix and return it.
	_matrices.insert (std::make_pair (mat_name, &mat));
	
	// since the matrix was user-provided, ISystem will not manage it
	_is_externally_managed_matrix.insert(std::make_pair (mat_name, true)) ;

	return mat;
}



KARMARealMatrix& ISystem::add_matrix (const std::string& mat_name,
								  const bool externally_managed)
{
	// Return the matrix if it is already there.
	if (this->have_matrix(mat_name))
		return *(_matrices[mat_name]);
	
	// Otherwise build the matrix and return it.
	_matrices.insert (std::make_pair (mat_name, new KARMARealMatrix()));
	// by default, ISystem will manage the matrix
	_is_externally_managed_matrix.insert(std::make_pair (mat_name, externally_managed)) ;
	
	return *_matrices[mat_name];
}


KARMARealMatrix& ISystem::init_matrix (const std::string& mat_name)
{
	// first check that we are initialized and prepared_for_usage
	if (_is_initialized && _prepared_for_usage)
	{
		// Return the matrix if it is already there.
		KARMARealMatrix* mat = NULL ;
		if (this->have_matrix(mat_name))	// check if it exists already
			mat = (_matrices[mat_name]);
		else	// Otherwise build the matrix and return it.
			*mat = this->add_matrix(mat_name) ;	

		if (_is_externally_managed_matrix[mat_name])
			return *mat ;
		
		/**
		* check that it has not been initialized before.
		* if it has been initialized, possibly the old Matrix will be cleared
		* and re-inited with the new data or we might silently not use
		* the current sparsity at all .. and this can be dangerous.
		*/
		karma_assert (!mat->initialized());
		
		// Tell the matrices about the nnz assignment and initialize
		// NOTE: Should we differentiate between the user provided sparsity
		mat->init(_sparsity.global_dofs, _sparsity.global_dofs, _sparsity.local_dofs, _sparsity.local_dofs, *_sparsity.n_nz, *_sparsity.o_nz) ;
		// let us also initialize the data to zero
		mat->zero ();
	}
	
	return *_matrices[mat_name];
}


bool ISystem::delete_matrix (const std::string& mat_name)
{
	// Make sure the matrix exists
	matrices_iterator pos = _matrices.find (mat_name);
	
	if (pos == _matrices.end())
		return false;
	
	// free the memmory allocated for the matrix before erasing the entry
	// only if ISystem is not current managing this reference.
	if (pos->second != NULL && !_is_externally_managed_matrix[mat_name])
		delete pos->second ;
	
	_matrices.erase (pos) ;
	
	return true;
}


const KARMARealMatrix * ISystem::request_matrix (const std::string& mat_name) const
{
	// Make sure the matrix exists
	const_matrices_iterator pos = _matrices.find (mat_name);
	
	if (pos == _matrices.end())
		return NULL;
	
	return pos->second;
}



KARMARealMatrix * ISystem::request_matrix (const std::string& mat_name)
{
	// Make sure the matrix exists
	matrices_iterator pos = _matrices.find (mat_name);
	
	if (pos == _matrices.end())
		return NULL;
	
	return pos->second;
}



const KARMARealMatrix & ISystem::get_matrix (const std::string& mat_name) const
{
	// Make sure the matrix exists
	const_matrices_iterator pos = _matrices.find (mat_name);
	
	if (pos == _matrices.end())
	{
		std::cerr << "ERROR: matrix "
		<< mat_name
		<< " does not exist in this system!"
		<< std::endl;
		karma_error();
	}
	
	return *(pos->second);
}



KARMARealMatrix & ISystem::get_matrix (const std::string& mat_name)
{
	// Make sure the matrix exists
	matrices_iterator pos = _matrices.find (mat_name);
	
	if (pos == _matrices.end())
	{
		std::cerr << "ERROR: matrix "
		<< mat_name
		<< " does not exist in this system!"
		<< std::endl;
		karma_error();
	}
	
	return *(pos->second);
}



// NOTE: Bad idea. Maintain a reference ourselves and let the
// user manipulate only the parameters at run-time.
// Allows for specializations pertaining to each specific physics.
// void ISystem::attach_linear_solver(ISolver* solver)
// {
// 	_solver = solver ;
// }


void ISystem::update ()
{
	// Check sizes
	karma_assert (current_local_solution->size() == solution->size());
	// More processors than elements => empty send_list
	//  karma_assert (!send_list.empty());
	karma_assert (send_list.size() <= solution->size());

	// Create current_local_solution from solution.  This will
	// put a local copy of solution into current_local_solution.
	// Only the necessary values (specified by the send_list)
	// are copied to minimize communication
	if (_sparsity.send_list != NULL)
	{
		solution->localize (*current_local_solution, *_sparsity.send_list);
	}
	else
	{
		// since send_list is NULL, we essentially perform an all-gather !
		solution->localize (*current_local_solution);
	}
}


bool ISystem::compare (const ISystem& other_system,
					  const Real threshold,
					  const bool verbose) const
{
	// we do not care for matrices, but only for vectors
	if (verbose)
	{
		KARMA_LOG(NONE) << "  Systems \"" << _sys_name << "\"" << std::endl;
		KARMA_LOG(NONE) << "   comparing matrices not supported." << std::endl;
		KARMA_LOG(NONE) << "   comparing names...";
	}

	// compare the name: 0 means identical
	const int name_result = _sys_name.compare(other_system.name());
	if (verbose)
	{
		if (name_result == 0)
			KARMA_LOG(NONE) << " identical." << std::endl;
		else
			KARMA_LOG(NONE) << "  names not identical." << std::endl;
		KARMA_LOG(NONE) << "   comparing solution vector...";
	}


	// compare the solution: -1 means identical
	const int solu_result = solution->compare (*other_system.solution.get(),
			threshold);

	if (verbose)
	{
		if (solu_result == -1)
			KARMA_LOG(NONE) << " identical up to threshold." << std::endl;
		else
			KARMA_LOG(NONE) << "  first difference occured at index = "
					<< solu_result << "." << std::endl;
	}

	
	// safety check, whether we handle at least the same number
	// of vectors
	std::vector<int> ov_result;

	if (this->n_vectors() != other_system.n_vectors())
	{
		if (verbose)
		{
			KARMA_LOG(NONE) << "   Fatal difference. This system handles "
					<< this->n_vectors() << " add'l vectors," << std::endl
					<< "   while the other system handles "
					<< other_system.n_vectors()
					<< " add'l vectors." << std::endl
					<< "   Aborting comparison." << std::endl;
		}
		return false;
	}
	else if (this->n_vectors() == 0)
	{
      // there are no additional vectors...
		ov_result.clear ();
	}
	else
	{
      // compare other vectors
		for (const_vectors_iterator pos = _vectors.begin();
				   pos != _vectors.end(); ++pos)
		{
			if (verbose)
				KARMA_LOG(NONE) << "   comparing vector \""
						<< pos->first << "\" ...";

	  // assume they have the same name
			const KARMARealVector& other_system_vector =
					other_system.get_vector(pos->first);

			ov_result.push_back(pos->second->compare (other_system_vector,
								threshold));

			if (verbose)
			{
				if (ov_result[ov_result.size()-1] == -1)
					KARMA_LOG(NONE) << " identical up to threshold." << std::endl;
				else
					KARMA_LOG(NONE) << " first difference occured at" << std::endl
							<< "   index = " << ov_result[ov_result.size()-1] << "." << std::endl;
			}

		}

	} // finished comparing additional vectors


	bool overall_result;
       
  // sum up the results
	if ((name_result==0) && (solu_result==-1))
	{
		if (ov_result.size()==0)
			overall_result = true;
		else
		{
			bool ov_identical;
			unsigned int n    = 0;
			do
			{
				ov_identical = (ov_result[n]==-1);
				n++;
			}
			while (ov_identical && n<ov_result.size());
			overall_result = ov_identical;
		}
	}
	else
		overall_result = false;

	if (verbose)
	{
		KARMA_LOG(NONE) << "   finished comparisons, ";
		if (overall_result)
			KARMA_LOG(NONE) << "found no differences." << std::endl << std::endl;
		else
			KARMA_LOG(NONE) << "found differences." << std::endl << std::endl;
	}
	  
	return overall_result;
}


void ISystem::update_global_solution (std::vector<Number>& global_soln) const
{
	global_soln.resize (solution->size());

	solution->localize (global_soln);
}


KARMARealVector& ISystem::add_vector (  const std::string& vec_name,
										KARMARealVector& vec)
{
	// Return the vector if it is already there.
	if (this->have_vector(vec_name))
		return *(_vectors[vec_name]);

	// insert the user-provided vector in the collection
	_vectors.insert (std::make_pair (vec_name, &vec));
	_vector_types.insert (std::make_pair (vec_name, vec.type()));
	
	// since the vector was user-provided, ISystem will not manage it
	_is_externally_managed_vector.insert(std::make_pair (vec_name, true)) ;

	return vec;
}


KARMARealVector& ISystem::add_vector (  const std::string& vec_name,
										const KARMA::LinearAlgebraLayout type,
										const bool externally_managed)
{
	// Return the vector if it is already there.
	if (this->have_vector(vec_name))
		return *(_vectors[vec_name]);

  // Otherwise build the vector
	// PetscVector<double>* buf = dynamic_cast<PetscVector<double>* >(KARMARealVectorBase<double>::build().release());
	KARMARealVector* buf = new KARMARealVector(type) ;
	_vectors.insert (std::make_pair (vec_name, buf));
	_vector_types.insert (std::make_pair (vec_name, type));
	// by default, ISystem will manage the vector
	_is_externally_managed_vector.insert(std::make_pair (vec_name, externally_managed)) ;

	// check whether sparsity data is updated
	karma_assert(_sparsity.is_ready()) ;
	
	// Initialize it if necessary
	if(type == KARMA::GHOSTED && _ghosted_vectors_enabled)
	{
		buf->init (_sparsity.global_dofs, _sparsity.local_dofs,
				*_sparsity.send_list, false);
	}
	else
	{
		// NOTE: should we change the type if user specifically did not ask for GHOSTED ?
		buf->init (_sparsity.global_dofs, _sparsity.local_dofs, false, type);
	}

	return *buf;
}



const KARMARealVector* ISystem::request_vector (const std::string& vec_name) const
{
	const_vectors_iterator pos = _vectors.find(vec_name);
  
	if (pos == _vectors.end())
		return NULL;
  
	return pos->second;
}



KARMARealVector* ISystem::request_vector (const std::string& vec_name)
{
	vectors_iterator pos = _vectors.find(vec_name);
  
	if (pos == _vectors.end())
		return NULL;
  
	return pos->second;
}



const KARMARealVector* ISystem::request_vector (const unsigned int vec_num) const
{
	const_vectors_iterator v = vectors_begin();
	const_vectors_iterator v_end = vectors_end();
	unsigned int num = 0;
	while((num<vec_num) && (v!=v_end))
	{
		num++;
		++v;
	}
	if (v==v_end)
		return NULL;
	return v->second;
}



KARMARealVector* ISystem::request_vector (const unsigned int vec_num)
{
	vectors_iterator v = vectors_begin();
	vectors_iterator v_end = vectors_end();
	unsigned int num = 0;
	while((num<vec_num) && (v!=v_end))
	{
		num++;
		++v;
	}
	if (v==v_end)
		return NULL;
	return v->second;
}


bool ISystem::delete_vector (const std::string& vec_name)
{
	// Make sure the matrix exists
	vectors_iterator pos = _vectors.find (vec_name);
	
	if (pos == _vectors.end())
		return false;
	
	// free the memmory allocated for the matrix before erasing the entry
	// only if ISystem is not current managing this reference.
	if (pos->second != NULL && !_is_externally_managed_vector[vec_name])
		delete pos->second ;
	
	_vectors.erase (pos) ;
	
	return true;
}


const KARMARealVector& ISystem::get_vector (const std::string& vec_name) const
{
  // Make sure the vector exists
	const_vectors_iterator pos = _vectors.find(vec_name);
  
	if (pos == _vectors.end())
	{
		libMesh::err << "ERROR: vector "
				<< vec_name
				<< " does not exist in this system!"
				<< std::endl;
		karma_error();
	}
  
	return *(pos->second);
}



KARMARealVector& ISystem::get_vector (const std::string& vec_name)
{
  // Make sure the vector exists
	vectors_iterator pos = _vectors.find(vec_name);
  
	if (pos == _vectors.end())
	{
		libMesh::err << "ERROR: vector "
				<< vec_name
				<< " does not exist in this system!"
				<< std::endl;
		karma_error();
	}
  
	return *(pos->second);
}



const KARMARealVector& ISystem::get_vector (const unsigned int vec_num) const
{
	const_vectors_iterator v = vectors_begin();
	const_vectors_iterator v_end = vectors_end();
	unsigned int num = 0;
	while((num<vec_num) && (v!=v_end))
	{
		num++;
		++v;
	}
	karma_assert(v!=v_end);
	return *(v->second);
}



KARMARealVector& ISystem::get_vector (const unsigned int vec_num)
{
	vectors_iterator v = vectors_begin();
	vectors_iterator v_end = vectors_end();
	unsigned int num = 0;
	while((num<vec_num) && (v!=v_end))
	{
		num++;
		++v;
	}
	karma_assert(v!=v_end);
	return *(v->second);
}



const std::string& ISystem::vector_name (const unsigned int vec_num)
{
	const_vectors_iterator v = vectors_begin();
	const_vectors_iterator v_end = vectors_end();
	unsigned int num = 0;
	while((num<vec_num) && (v!=v_end))
	{
		num++;
		++v;
	}
	karma_assert(v!=v_end);
	return v->first;
}

