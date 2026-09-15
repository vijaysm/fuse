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


#ifndef __petsc_native_vector_h__
#define __petsc_native_vector_h__


#include "karma_config.h"

#ifdef KARMA_HAVE_PETSC

// Local includes
#include "ivector.h"

// Petsc includes
#include "petscvec.h"

// Forward Declarations
namespace libMesh
{
	template <typename T> class DenseVector;
	template <typename T> class DenseSubVector;
}

// Forward declaration
template <typename T>
class PetscNativeVector ;

// specialize for PetscNativeVector<T>
template<typename T>
struct VectorTraits<PetscNativeVector<T> >
{
	/**
	 *  Create some traits to define Scalar and Composite types.
	 */
	typedef T ScalarType ;
	typedef T CompositeType ;
	typedef PetscNativeVector<T> VectorType ;
};


/**
 * Petsc vector. Provides a nice interface to the
 * Petsc C-based data structures for parallel vectors.
 */
template <typename T>
class PetscNativeVector : public IVector<PetscNativeVector<T> >
{
public:
	
	typedef IVector<PetscNativeVector<T> > Parent ;
	typedef typename VectorTraits<PetscNativeVector<T> >::ScalarType ScalarType ;
	typedef typename VectorTraits<PetscNativeVector<T> >::CompositeType CompositeType ;
	typedef typename VectorTraits<PetscNativeVector<T> >::VectorType VectorType ;
	
	/**
	*  Dummy-Constructor. Dimension=0
	*/
	explicit
	PetscNativeVector (const KARMA::LinearAlgebraLayout type = KARMA::AUTOMATIC);
	
	/**
	* Constructor. Set dimension to \p n and initialize all elements with zero.
	*/
	explicit
	PetscNativeVector (const unsigned int n,
				const KARMA::LinearAlgebraLayout type = KARMA::AUTOMATIC);
	
	/**
	* Constructor. Set local dimension to \p n_local, the global dimension
	* to \p n, and initialize all elements with zero.
	*/
	PetscNativeVector (const unsigned int n,
			const unsigned int n_local,
				const KARMA::LinearAlgebraLayout type = KARMA::AUTOMATIC);
	
	/**
	* Constructor. Set local dimension to \p n_local, the global
	* dimension to \p n, but additionally reserve memory for the
	* indices specified by the \p ghost argument.
	*/
	PetscNativeVector (const unsigned int N,
			const unsigned int n_local,
			const std::vector<unsigned int>& ghost,
				const KARMA::LinearAlgebraLayout type = KARMA::AUTOMATIC);
	
	/**
	* Constructor.  Creates a PetscNativeVector assuming you already have a
	* valid PETSc Vec object.  In this case, \p v is NOT destroyed by the
	* PetscNativeVector constructor when this object goes out of scope.
	* This allows ownership of \p v to remain with the original creator,
	* and to simply provide additional functionality with the PetscNativeVector.
	*/
	PetscNativeVector(Vec v);

	/**
	* Copy Constructor.  Creates a PetscNativeVector assuming you already have a
	* valid PetscNativeVector pre-allocated.  In this case, the underlying data structures
	* in PetscNativeVector \p v is copied. This is similar to clone.
	*/
	PetscNativeVector(const PetscNativeVector<T>& v);
	
	/**
	* Builds a \p PetscNativeVector with a default constructor.
	* This allows us the option to pre-set some user-specific data
	* but highly improbable since we do not know the size of the
	* vector yet ! Anyway, as of now, this will serve as a helper.
	*/
	static AutoPtr<PetscNativeVector<T> > build();
	
	/**
	* Destructor, deallocates memory. Made virtual to allow
	* for derived classes to behave properly.
	*/
	virtual ~PetscNativeVector ();
	
	/**
	* Call the assemble functions
	*/
	virtual void close ();
	
	/**
	* @returns the \p PetscNativeVector<T> to a pristine state.
	*/
	virtual void clear ();
	
	/**
	* Set all entries to zero. Equivalent to \p v = 0, but more obvious and
	* faster.
	*/
	virtual void zero ();
	
	/**
	 * Creates a vector which has the same type, size and partitioning
	 * as this vector, but whose data is either copied or zeroed out
	 * depending on \p copy_values. Note that the new vector returned as
	 * an \p AutoPtr has separate memory allocated and needs to be cleared
	 * by the user.
	 */
	virtual AutoPtr<Parent> deep_clone (bool copy_values=false) const ;
	
	/**
	 * Creates a copy of this vector and returns it in an \p AutoPtr.
	 */
	virtual AutoPtr<Parent> clone () const ;
	
	/**
	* Change the dimension of the vector to \p N. The reserved memory for
	* this vector remains unchanged if possible, to make things faster, but
	* this may waste some memory, so take this in the back of your head.
	* However, if \p N==0 all memory is freed, i.e. if you want to resize
	* the vector and release the memory not needed, you have to first call
	* \p init(0) and then \p init(N). This cited behaviour is analogous
	* to that of the STL containers.
	*
	* On \p fast==false, the vector is filled by
	* zeros.
	*/
	virtual void init (const unsigned int N,
			const unsigned int n_local,
			const bool  = false,
			const KARMA::LinearAlgebraLayout = KARMA::AUTOMATIC);
	
	/**
	* Create a vector that holds tha local indices plus those specified
	* in the \p ghost argument; type = GHOSTED
	*/
	virtual void init (const unsigned int ,
				const unsigned int ,
				const std::vector<unsigned int>& ,
				const bool = false);
	
	/**
	* Creates a vector that has the same dimension and storage type as
	* \p other. \p fast determines whether we need to zero out the
	* entries in the new vector.
	*/
	virtual void init (const Parent& other,
						const bool = false) ;
	
	/**
	* \f$U(0-N) = s\f$: fill all components.
	*/
	virtual Parent& operator= (ScalarType const s);
	
	/**
	*  \f$U = V\f$: copy all components.
	*/
	virtual Parent& operator= (const Parent& V);
	
	/**
	*  \f$U = V\f$: copy all components.
	*/
	virtual Parent& operator= (const std::vector<T> &v);
	
	/**
	*  Assignment operator. \f$U = V\f$: copy all components.
	*/
	PetscNativeVector<T>& operator= (const PetscNativeVector<T>& v) ;

	/**
	* Access components, returns \p U(i); A const version.
	*/
	virtual ScalarType const operator() (const unsigned int i) const;
	
	/**
	* Access components, returns \p U(i).
	*/
	virtual ScalarType& operator() (const unsigned int i) ;
	
	/**
	* Access components, returns \p U[i]; Same as U(i).
	*/
	virtual CompositeType const operator[] (const unsigned int i) const;
	
	/**
	* Access components, returns \p U(i). Same as U(i).
	*/
	virtual CompositeType& operator[] (const unsigned int i) ;
	
	/**
	* Access multiple components at once.  Overloaded method that
	* should be faster (probably much faster) than calling \p
	* operator() individually for each index.
	*/
	virtual void operator() (const std::vector<unsigned int>& index, std::vector<CompositeType>& values) const ;
	
	/**
	* @returns the minimum element in the vector.
	* In case of complex numbers, this returns the minimum
	* Real part.
	*/
	virtual ScalarType min () const;
	
	/**
	* @returns the maximum element in the vector.
	* In case of complex numbers, this returns the maximum
	* Real part.
	*/
	virtual ScalarType max () const;
	
	/**
	* @returns the sum of values in a vector
	*/
	virtual ScalarType sum () const;
	
	/**
	* @returns the \f$l_1\f$-norm of the vector, i.e.
	* the sum of the absolute values.
	*/
	virtual ScalarType l1_norm () const;
	
	/**
	* @returns the \f$l_2\f$-norm of the vector, i.e.
	* the square root of the sum of the
	* squares of the elements.
	*/
	virtual ScalarType l2_norm () const;
	
	/**
	* @returns the maximum absolute value of the
	* elements of this vector, which is the
	* \f$l_\infty\f$-norm of a vector.
	*/
	virtual ScalarType linfty_norm () const;

	/**
	* @returns the \f$l_1\f$-norm of the vector, i.e.
	* the sum of the absolute values for the specified
	* entries in the vector.
	*
	* Note that the indices must necessary live on this
	* processor.
	*/
	ScalarType subset_l1_norm (const std::set<unsigned int> & indices) const;

	/**
	* @returns the \f$l_2\f$-norm of the vector, i.e.
	* the square root of the sum of the
	* squares of the elements for the specified entries
	* in the vector.
	*
	* Note that the indices must necessary live on this
	* processor.
	*/
	ScalarType subset_l2_norm (const std::set<unsigned int> & indices) const;

	/**
	* @returns the maximum absolute value of the
	* specified entries of this vector, which is the
	* \f$l_\infty\f$-norm of a vector.
	*
	* Note that the indices must necessary live on this
	* processor.
	*/
	ScalarType subset_linfty_norm (const std::set<unsigned int> & indices) const;

	/**
	* @returns dimension of the vector. This
	* function was formerly called \p n(), but
	* was renamed to get the \p PetscNativeVector<T> class
	* closer to the C++ standard library's
	* \p std::vector container.
	*/
	virtual unsigned int size () const;
	
	/**
	* @returns the local size of the vector
	* (index_stop-index_start)
	*/
	virtual unsigned int local_size() const;
	
	/**
	* @returns Determine whether global vector index
	* is owned by this process.
	*/
	bool owns_index(unsigned int i) const ;
	
	/**
	* @returns the index of the first vector element
	* actually stored on this processor
	*/
	unsigned int first_local_index() const;
	
	/**
	* @returns the index of the last vector element
	* actually stored on this processor
	*/
	unsigned int last_local_index() const;
	
	/**
	* Addition operator.
	* Shift each component of this vector by a.
	*/
	virtual Parent& operator += (const ScalarType a) ;
	
	/**
	* Addition operator.
	* Fast equivalent to \p U.add(1, V).
	*/
	virtual Parent& operator += (const Parent& V) ;
	
	/**
	* Subtraction operator.
	* Shift each component of this vector by a.
	*/
	virtual Parent& operator -= (const ScalarType a) ;
	
	/**
	* Subtraction operator.
	* Fast equivalent to \p U.add(-1, V).
	*/
	virtual Parent& operator -= (const Parent& V) ;
	
	/**
	* v(i) = value
	*/
	virtual void insert (CompositeType const value, const unsigned int index);
	
	/**
	* v(i) += value
	*/
	virtual void add (CompositeType const value, const unsigned int index);

	/**
	* \f$ U=v \f$ where v is a std::vector<T>
	* and you want to specify WHERE to insert it
	*/
	virtual void insert (const std::vector<CompositeType>& values,
				const std::vector<unsigned int>& index);
	
	/**
	* \f$ U+=v \f$ where v is a std::vector<T>
	* and you want to specify WHERE to insert it
	*/
	virtual void add (const std::vector<CompositeType>& values,
				const std::vector<unsigned int>& index);
		
	/**
	* \f$ U+=V \f$ .
	* Simple vector addition, equal to the
	* \p operator +=.
	*/
	virtual void add (const Parent& V);
	
	/**
	* \f$U+=V \f$ where U and V are type
	* DenseVector<T> and you
	* want to specify WHERE to add
	* the DenseVector<T> V
	*/
	void add (const libMesh::DenseVector<T>& V,
			const std::vector<unsigned int>& indices);
	
	/**
	* \f$ U=V \f$ where V is type
	* DenseVector<T> and you
	* want to specify WHERE to insert it
	*/
	void insert (const libMesh::DenseVector<T>& V,
				const std::vector<unsigned int>& indices);
	
	/**
	* \f$ U=V \f$ where V is type
	* DenseSubVector<T> and you
	* want to specify WHERE to insert it
	*/
	void insert (const libMesh::DenseSubVector<T>& V,
			     const std::vector<unsigned int>& dof_indices);
	
	/**
	 * Shift each element of the
	 * vector by the given \p factor[0].
	 * Optionally, can pass an array to scale each
	 * element independently. This would then function
	 * like a pointwise_multiply for a single vector case
	 * but provides regular scale interface for block vector.
	 */
	virtual void shift (const ScalarType factor) ;
	
	/**
	 * Scale each element of the
	 * vector by the given \p factor[0].
	 * Optionally, can pass an array to scale each
	 * element independently. This would then function
	 * like a pointwise_multiply for a single vector case
	 * but provides regular scale interface for block vector.
	 */
	virtual void scale (const ScalarType factor) ;
	
	/**
	* Shift each element of the
	* vector by the given \p factor[0].
	* Optionally, can pass an array to scale each
	* element independently. This would then function
	 
	 /**
	 * Add multiple of given vector (AXPY operation)
	 * this += a*X
	 */
	virtual void axpy(const ScalarType a, const Parent& x) ;
	
	/**
	 * Add multiple of given vector (AYPX operation)
	 * this = a*this + x
	 */
	virtual void aypx(const ScalarType a, const Parent& x) ;
	
	/**
	 * Add multiple of given vector (AXPY operation)
	 * this = a*x + b*this
	 */
	virtual void axpby(const ScalarType a, const Parent& x, const ScalarType b) ;
	
	/**
	 * Add multiple of given vector (AXPY operation)
	 * this = a*x + b*y + c*this
	 */
	virtual void axpbypcz(const ScalarType a, const Parent& x, const ScalarType b, const Parent& y, const ScalarType c) ;
	
	/**
	* Add multiple of given vector (AXPY operation)
	* this += \sum_{i=0}^{n}(a_i*x_i)
	*/
	virtual void maxpy(const std::vector<ScalarType>& a, const std::vector<Parent* >& x) ;
	
	/**
	* v = abs(v)... that is, each entry in v is replaced
	* by its absolute value.
	*/
	virtual void abs();
	
	/**
	* v = 1./v... that is, each entry in v is replaced
	* by its reciprocal value.
	*/
	virtual void reciprocal() ;
	
	/**
	* Computes the dot product, p = U.V
	*/
	virtual ScalarType dot(const Parent& ) const;
	
	/**
	* Creates a local vector \p v_local containing
	* only information relevant to this processor, as
	* defined by the \p send_list.
	*/
	virtual void localize (Parent& v_local,
			const std::vector<unsigned int>* send_list = NULL) const;
	
	/**
	* Updates a local vector with selected values from neighboring
	* processors, as defined by \p send_list.
	*/
	void localize (const unsigned int first_local_idx,
			const unsigned int last_local_idx,
			const std::vector<unsigned int>& send_list);
	
	/**
	* Creates a local copy of the global vector in
	* \p v_local only on processor \p proc_id.  By
	* default the data is sent to processor 0.  This method
	* is useful for outputting data from one processor.
	*/
	void localize_to_one (std::vector<T>& v_local,
			const unsigned int proc_id=0) const;
	
	/**
	* @returns \p -1 when \p this is equivalent to \p other_vector,
	* up to the given \p threshold.  When differences occur,
	* the return value contains the first index \p i where
	* the difference \p (a[i]-b[i]) exceeded the threshold.  When
	* no threshold is given, the \p KARMA \p TOLERANCE
	* is used.
	*/
	virtual int compare (const Parent& other_vector,
				const Real threshold = TOLERANCE) const ;
	
	/**
	* @returns \p -1 when \p this is equivalent to \p other_vector,
	* up to the given local relative \p threshold.  When differences
	* occur, the return value contains the first index where
	* the difference \p (a[i]-b[i])/max(a[i],b[i]) exceeded the
	* threshold.  When no threshold is given, the \p KARMA
	* \p TOLERANCE is used.
	*/
	int local_relative_compare (const Parent& other_vector,
										const Real threshold = TOLERANCE) const ;
	
	/**
	* @returns \p -1 when \p this is equivalent to \p other_vector,
	* up to the given local relative \p threshold.  When differences
	* occur, the return value contains the first index where
	* the difference \p (a[i]-b[i])/max_j(a[j],b[j]) exceeded the
	* threshold.  When no threshold is given, the \p KARMA
	* \p TOLERANCE is used.
	*/
	int global_relative_compare (const Parent& other_vector,
										const Real threshold = TOLERANCE) const ;
	
	/**
	* Computes the pointwise (i.e. component-wise) product of \p vec1
	* and \p vec2 and stores the result in \p *this.
	*/
	virtual void pointwise_mult (const Parent& vec1,
					const Parent& vec2);
	
	/**
	* Computes the pointwise (i.e. component-wise) division of \p vec1
	* and \p vec2 and stores the result in \p *this.
	*/
	virtual void pointwise_divide (const Parent& vec1,
					const Parent& vec2) ;
	
	/**
	* Prints the local contents of the vector, by default to
	* std::cout
	*/
	virtual void print(bool is_global=true, std::ostream& os=std::cout) const ;
	
	/**
	* Print the contents of the matrix in Matlab's
	* sparse matrix format. Optionally prints the
	* matrix to the file named \p name.  If \p name
	* is not specified it is dumped to the screen.
	*/
	virtual void print_matlab(const std::string name="NULL") const ;

	
	/**
	* Creates a "subvector" from this vector using the rows indices
	* of the "rows" array.
	*/
	virtual void create_subvector(Parent& subvector,
				const std::vector<unsigned int>& rows) const;
	
	/**
	* Swaps the raw PETSc vector context pointers.
	*/
	virtual void swap (Parent& v);
	
	/**
	* Returns the raw PETSc vector context pointer.  Note this is generally
	* not required in user-level code. Just don't do anything crazy like
	* calling VecDestroy()!
	*/
	virtual Vec vec () { karma_assert (_vec != NULL); return _vec; }

	/**
	* Returns the raw PETSc vector context pointer.  Note this is generally
	* not required in user-level code. Just don't do anything crazy like
	* calling VecDestroy()!
	*/
	virtual const Vec& vec () const { karma_assert (_vec != NULL); return _vec; }

private:

	/**
	 * Actual Petsc vector datatype
	 * to hold vector entries
	 */
	Vec _vec;
	
	/**
	* Maps the global index \p i to the corresponding global index. If
	* the index is not a ghost cell, this is done by subtraction the
	* number of the first local index.  If it is a ghost cell, it has
	* to be looked up in the map.
	*/
	unsigned int map_global_to_local_index(const unsigned int i) const;
	
	/**
	* If \p true, the actual Petsc array of the values of the vector is
	* currently accessible.  That means that the members \p _local_form
	* and \p _values are valid.
	*/
	mutable bool _array_is_present;
	
#ifndef NDEBUG
	/**
	* Size of the local form, for being used in assertations.  The
	* contents of this field are only valid if the vector is ghosted
	* and \p _array_is_present is \p true.
	*/
	mutable unsigned int _local_size;
#endif
	
	/**
	* Petsc vector datatype to hold the local form of a ghosted vector.
	* The contents of this field are only valid if the vector is
	* ghosted and \p _array_is_present is \p true.
	*/
	mutable Vec _local_form;
	
	/**
	* Pointer to the actual Petsc array of the values of the vector.
	* This pointer is only valid if \p _array_is_present is \p true.
	*/
	mutable PetscScalar* _values;
	
	/**
	* Queries the array (and the local form if the vector is ghosted)
	* from Petsc.
	*/
	void _get_array(void) const;
	
	/**
	* Restores the array (and the local form if the vector is ghosted)
	* to Petsc.
	*/
	void _restore_array(void) const;
	
	/**
	* Type for map that maps global to local ghost cells.
	*/
	typedef std::map<unsigned int,unsigned int> GlobalToLocalMap;
	
	/**
	* Map that maps global to local ghost cells (will be empty if not
	* in ghost cell mode).
	*/
	GlobalToLocalMap _global_to_local_map;
	
	/**
	* This boolean value should only be set to false
	* for the constructor which takes a PETSc Vec object.
	*/
	bool _destroy_vec_on_exit;

};


/*----------------------- Inline functions ----------------------------------*/



template <typename T>
inline
PetscNativeVector<T>::PetscNativeVector (const KARMA::LinearAlgebraLayout type)
  : IVector<PetscNativeVector<T> >(type),
    _array_is_present(false),
    _local_form(NULL),
    _values(NULL),
    _global_to_local_map(),
    _destroy_vec_on_exit(true)
{
  this->_type = type;
}



template <typename T>
inline
PetscNativeVector<T>::PetscNativeVector (const unsigned int n,
                             const KARMA::LinearAlgebraLayout type)
  : IVector<PetscNativeVector<T> >(type),
    _array_is_present(false),
    _local_form(NULL),
    _values(NULL),
    _global_to_local_map(),
    _destroy_vec_on_exit(true)
{
  this->init(n, n, false, type);
}



template <typename T>
inline
PetscNativeVector<T>::PetscNativeVector (const unsigned int n,
			     const unsigned int n_local,
                             const KARMA::LinearAlgebraLayout type)
  : IVector<PetscNativeVector<T> >(type),
    _array_is_present(false),
    _local_form(NULL),
    _values(NULL),
    _global_to_local_map(),
    _destroy_vec_on_exit(true)
{
  this->init(n, n_local, false, type);
}



template <typename T>
inline
PetscNativeVector<T>::PetscNativeVector (const unsigned int n,
			     const unsigned int n_local,
			     const std::vector<unsigned int>& ghost,
                             const KARMA::LinearAlgebraLayout type)
  : IVector<PetscNativeVector<T> >(KARMA::GHOSTED),
    _array_is_present(false),
    _local_form(NULL),
    _values(NULL),
    _global_to_local_map(),
    _destroy_vec_on_exit(true)
{
  this->init(n, n_local, ghost, false);
}





template <typename T>
inline
PetscNativeVector<T>::PetscNativeVector (Vec v)
  : IVector<PetscNativeVector<T> >(KARMA::AUTOMATIC),
    _array_is_present(false),
    _local_form(NULL),
    _values(NULL),
    _global_to_local_map(),
    _destroy_vec_on_exit(false)
{
  this->_vec = v;
  this->_is_closed = true;
  this->_is_initialized = true;

  /* We need to ask PETSc about the (local to global) ghost value
     mapping and create the inverse mapping out of it.  */
  PetscErrorCode ierr=0, petsc_size=0, petsc_local_size=0;
  ierr = VecGetSize(_vec, &petsc_size);
  CHKERRABORT(KARMA_COMM_WORLD,ierr);
  ierr = VecGetLocalSize(_vec, &petsc_local_size);
  CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // NOTE: The following check seems to assume that all parallel vectors
  // are ghosted by default. KARMA::PARALLEL will never be set in this case.
  // We need a better way to look at Vec and then decide the type.
  
  /* \p petsc_local_size is the number of non-ghost values.  If it
     equals the global size, then we are a serial vector, and there
     are no ghost values.  */
  if(petsc_size!=petsc_local_size)
    {
		ISLocalToGlobalMapping mapping;
		ierr = VecGetLocalToGlobalMapping(_vec, &mapping);
		CHKERRABORT(libMesh::COMM_WORLD,ierr);

      // If is a sparsely stored vector, set up our new mapping
      if (mapping)
        {
          const unsigned int local_size = static_cast<unsigned int>(petsc_local_size);
          const unsigned int ghost_begin = static_cast<unsigned int>(petsc_local_size);
          const unsigned int ghost_end = static_cast<unsigned int>(mapping->n);
          for(unsigned int i=ghost_begin; i<ghost_end; i++)
			_global_to_local_map[mapping->indices[i]] = i-local_size;
		  this->_type = KARMA::GHOSTED;
		}
      else
		this->_type = KARMA::PARALLEL;
    }
  else
    this->_type = KARMA::SERIAL;

}



template <typename T>
inline
PetscNativeVector<T>::PetscNativeVector (const PetscNativeVector<T>& v)
	: IVector<PetscNativeVector<T> >(v),
	_array_is_present(false),
	_local_form(NULL),
	_values(NULL),
    _global_to_local_map(),
    _destroy_vec_on_exit(false)
{
  this->_vec = v._vec;
  this->_is_closed = true;
  this->_is_initialized = true;

  /* We need to ask PETSc about the (local to global) ghost value
     mapping and create the inverse mapping out of it.  */
  PetscErrorCode ierr=0, petsc_size=0, petsc_local_size=0;
  ierr = VecGetSize(_vec, &petsc_size);
  CHKERRABORT(KARMA_COMM_WORLD,ierr);
  ierr = VecGetLocalSize(_vec, &petsc_local_size);
  CHKERRABORT(KARMA_COMM_WORLD,ierr);

	/* \p petsc_local_size is the number of non-ghost values.  If it
     equals the global size, then we are a serial vector, and there
     are no ghost values.  */
	if(petsc_size!=petsc_local_size)
    {
		ISLocalToGlobalMapping mapping;
		ierr = VecGetLocalToGlobalMapping(_vec, &mapping);
		CHKERRABORT(libMesh::COMM_WORLD,ierr);

      // If is a sparsely stored vector, set up our new mapping
      if (mapping)
        {
          const unsigned int local_size = static_cast<unsigned int>(petsc_local_size);
          const unsigned int ghost_begin = static_cast<unsigned int>(petsc_local_size);
          const unsigned int ghost_end = static_cast<unsigned int>(mapping->n);
          for(unsigned int i=ghost_begin; i<ghost_end; i++)
	    	_global_to_local_map[mapping->indices[i]] = i-local_size;
	  this->_type = KARMA::GHOSTED;
	}
    else
		this->_type = KARMA::PARALLEL;
    }
	else
    	this->_type = KARMA::SERIAL;

}


template <typename T>
inline
PetscNativeVector<T>::~PetscNativeVector ()
{
  this->clear ();
}



template <typename T>
inline
void PetscNativeVector<T>::init (const unsigned int n,
						   const unsigned int n_local,
						   const bool fast,
                           const KARMA::LinearAlgebraLayout type)
{
  PetscErrorCode ierr=0;
  int petsc_n=static_cast<int>(n);
  int petsc_n_local=static_cast<int>(n_local);


  // Clear initialized vectors 
  if (this->initialized())
    this->clear();

  if (type == KARMA::AUTOMATIC)
    {
      if (n == n_local)
        this->_type = KARMA::SERIAL;
      else
        this->_type = KARMA::PARALLEL;
    }
  else
    this->_type = type;

  karma_assert ((this->_type==KARMA::SERIAL && n==n_local) ||
                  this->_type==KARMA::PARALLEL);
  
  // create a sequential vector if on only 1 processor 
  if (this->_type == KARMA::SERIAL)
    {
      ierr = VecCreateSeq (PETSC_COMM_SELF, petsc_n, &_vec);
             CHKERRABORT(PETSC_COMM_SELF,ierr);
      
      ierr = VecSetFromOptions (_vec);
             CHKERRABORT(PETSC_COMM_SELF,ierr);
    }
  // otherwise create an MPI-enabled vector
  else if (this->_type == KARMA::PARALLEL)
    {
      karma_assert (n_local <= n);
      ierr = VecCreateMPI (KARMA_COMM_WORLD, petsc_n_local, petsc_n,
			   &_vec);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
      
      ierr = VecSetFromOptions (_vec);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }  
  else
    karma_error();
  
  this->_is_initialized = true;
  this->_is_closed = true;
  
  
  if (fast == false)
    this->zero ();
}




template <typename T>
inline
void PetscNativeVector<T>::init (const unsigned int n,
			   const unsigned int n_local,
			   const std::vector<unsigned int>& ghost,
			   const bool fast)
{
  PetscErrorCode ierr=0;
  int petsc_n=static_cast<int>(n);
  int petsc_n_local=static_cast<int>(n_local);
  int petsc_n_ghost=static_cast<int>(ghost.size());

  // If the mesh is not disjoint, every processor will either have
  // all the dofs, none of the dofs, or some non-zero dofs at the
  // boundary between processors.
  //
  // However we can't assert this, because someone might want to
  // construct a GHOSTED vector which doesn't include neighbor element
  // dofs.  Boyce tried to do so in user code, and we're going to want
  // to do so in System::project_vector().
  //
  // karma_assert(n_local == 0 || n_local == n || !ghost.empty());

  int* petsc_ghost = ghost.empty() ? PETSC_NULL :
    const_cast<int*>(reinterpret_cast<const int*>(&ghost[0]));

  // Clear initialized vectors 
  if (this->initialized())
    this->clear();

  this->_type = KARMA::GHOSTED;

  /* Make the global-to-local ghost cell map.  */
  for(unsigned int i=0; i<ghost.size(); i++)
    {
      _global_to_local_map[ghost[i]] = i;
    }

  /* Create vector.  */
  ierr = VecCreateGhost (KARMA_COMM_WORLD, petsc_n_local, petsc_n,
			 petsc_n_ghost, petsc_ghost,
			 &_vec);
  CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
  ierr = VecSetFromOptions (_vec);
  CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
  this->_is_initialized = true;
  this->_is_closed = true;
  
  if (fast == false)
    this->zero ();
}



template <typename T>
inline
void PetscNativeVector<T>::init (const Parent& other,
                           const bool fast)
{
  // Clear initialized vectors 
  if (this->initialized())
    this->clear();

  const PetscNativeVector<T>& v = dynamic_cast<const PetscNativeVector<T>&>(other);

  // Other vector should restore array.
  if(v.initialized())
    {
      v._restore_array();
    }

  this->_global_to_local_map = v._global_to_local_map;
  this->_is_closed      = v._is_closed;
  this->_is_initialized = v._is_initialized;
  this->_type = v._type;

  if (v.size() != 0)
    {
      PetscErrorCode ierr = 0;

      ierr = VecDuplicate (v._vec, &this->_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  
  if (fast == false)
    this->zero ();
}




template <typename T>
inline
void PetscNativeVector<T>::close ()
{
  this->_restore_array();
  
  PetscErrorCode ierr=0;
  
  ierr = VecAssemblyBegin(_vec);
  CHKERRABORT(KARMA_COMM_WORLD,ierr);
  ierr = VecAssemblyEnd(_vec);
  CHKERRABORT(KARMA_COMM_WORLD,ierr);

  if(this->type() == KARMA::GHOSTED)
    {
      ierr = VecGhostUpdateBegin(_vec,INSERT_VALUES,SCATTER_FORWARD);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
      ierr = VecGhostUpdateEnd(_vec,INSERT_VALUES,SCATTER_FORWARD);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

  this->_is_closed = true;
}



template <typename T>
inline
void PetscNativeVector<T>::clear ()
{
  if (this->initialized())
    this->_restore_array();

  if ((this->initialized()) && (this->_destroy_vec_on_exit))
    {
      PetscErrorCode ierr=0;

      ierr = VecDestroy(&_vec);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

  this->_is_closed = this->_is_initialized = false;

  _global_to_local_map.clear();
}



template <typename T>
inline
void PetscNativeVector<T>::zero ()
{
  this->_restore_array();
  
  PetscErrorCode ierr=0;

  PetscScalar z=0.;

  if(this->type() != KARMA::GHOSTED)
    {
      // 2.3.x & newer
      ierr = VecSet (_vec, z);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  else
    {
      /* Vectors that include ghost values require a special
	 handling.  */
      Vec loc_vec;
      ierr = VecGhostGetLocalForm (_vec,&loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);

	  // 2.3.x & newer
      ierr = VecSet (loc_vec, z);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
      ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
}



template <typename T>
inline
AutoPtr<typename PetscNativeVector<T>::Parent> PetscNativeVector<T>::deep_clone (bool copy_values) const
{
	PetscNativeVector<T>* cloned_vector = new PetscNativeVector<T>();
	
	cloned_vector->init(*this, !copy_values);
	
	if(copy_values)
	{
		PetscErrorCode ierr = VecCopy( _vec, cloned_vector->_vec);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	
	return AutoPtr<Parent> ( cloned_vector) ;
}



template <typename T>
inline
AutoPtr<typename PetscNativeVector<T>::Parent> PetscNativeVector<T>::clone () const
{
	PetscNativeVector<T>* cloned_vector = new PetscNativeVector<T>();
	
	cloned_vector->init(*this, true);
	
	*cloned_vector = *this;
	
	return AutoPtr<Parent> ( cloned_vector) ;
}



template <typename T>
inline
unsigned int PetscNativeVector<T>::size () const
{
	karma_assert (this->initialized());
	
	PetscErrorCode ierr=0, petsc_size=0;
	
	if (!this->initialized())
	return 0;
	
	ierr = VecGetSize(_vec, &petsc_size);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<unsigned int>(petsc_size);
}



template <typename T>
inline
unsigned int PetscNativeVector<T>::local_size () const
{
	karma_assert (this->initialized());
	
	PetscErrorCode ierr=0, petsc_size=0;
	
	ierr = VecGetLocalSize(_vec, &petsc_size);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<unsigned int>(petsc_size);
}



template <typename T>
inline
bool PetscNativeVector<T>::owns_index (unsigned int i) const
{
	karma_assert (this->initialized());
	
	PetscErrorCode ierr=0, petsc_first=0, petsc_last=0;
	
	ierr = VecGetOwnershipRange (_vec, &petsc_first, &petsc_last);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<unsigned int>((i >= petsc_first && i < petsc_last));
}


template <typename T>
inline
unsigned int PetscNativeVector<T>::first_local_index () const
{
	karma_assert (this->initialized());
	
	PetscErrorCode ierr=0, petsc_first=0, petsc_last=0;
	
	ierr = VecGetOwnershipRange (_vec, &petsc_first, &petsc_last);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<unsigned int>(petsc_first);
}



template <typename T>
inline
unsigned int PetscNativeVector<T>::last_local_index () const
{
	karma_assert (this->initialized());
	
	PetscErrorCode ierr=0, petsc_first=0, petsc_last=0;
	
	ierr = VecGetOwnershipRange (_vec, &petsc_first, &petsc_last);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<unsigned int>(petsc_last);
}



template <typename T>
inline
unsigned int PetscNativeVector<T>::map_global_to_local_index (const unsigned int i) const
{
	karma_assert (this->initialized());
	
	PetscErrorCode ierr=0, petsc_first=0, petsc_last=0;
	ierr = VecGetOwnershipRange (_vec, &petsc_first, &petsc_last);CHKERRABORT(KARMA_COMM_WORLD,ierr);
	const unsigned int first = static_cast<unsigned int>(petsc_first);
	const unsigned int last = static_cast<unsigned int>(petsc_last);
	
	if((i>=first) && (i<last))
	{
		return i-first;
	}
	
	GlobalToLocalMap::const_iterator it = _global_to_local_map.find(i);
	karma_assert (it!=_global_to_local_map.end());
	return it->second+last-first;
}


template <typename T>
inline
typename PetscNativeVector<T>::CompositeType const PetscNativeVector<T>::operator[] (const unsigned int i) const
{
	return (*this)(i) ;
}

template <typename T>
inline
typename PetscNativeVector<T>::CompositeType& PetscNativeVector<T>::operator[] (const unsigned int i)
{
	return (*this)(i) ;
}

template <typename T>
inline
typename PetscNativeVector<T>::ScalarType const PetscNativeVector<T>::operator() (const unsigned int i) const
{
	this->_get_array();
	
	const unsigned int local_index = this->map_global_to_local_index(i);
	
#ifndef NDEBUG
	if(this->type() == KARMA::GHOSTED)
		{
		karma_assert(local_index<_local_size);
		}
#endif
	
	return static_cast<T>(_values[local_index]);
}



template <typename T>
inline
typename PetscNativeVector<T>::ScalarType& PetscNativeVector<T>::operator() (const unsigned int i)
{
	// if the pointer to the base array for the PETSc vector is NULL,
	// call get_array to obtain the reference in _values.
	// if not, allow consequetive calls to maintain the same data until
	// close() or _restore_array() is called explicitly
	// NOTE: This should speed up the value set routine due to lack of
	// multiple calls to VecSetValue.
	if (!_array_is_present && _values == 0)	// if we have dont the array
		this->_get_array();
	
	const unsigned int local_index = this->map_global_to_local_index(i);
	
#ifndef NDEBUG
	if(this->type() == KARMA::GHOSTED)
		{
		karma_assert(local_index<_local_size);
		}
#endif

	return static_cast<T&>(_values[local_index]);
}


template <typename T>
inline
void PetscNativeVector<T>::operator() (const std::vector<unsigned int>& index, std::vector<CompositeType>& values) const
{
	this->_get_array();
	
	const unsigned int num = index.size();
	values.resize(num);
	
	for(unsigned int i=0; i<num; i++)
	{
		const unsigned int local_index = this->map_global_to_local_index(index[i]);
#ifndef NDEBUG
		if(this->type() == KARMA::GHOSTED)
		{
			karma_assert(local_index<_local_size);
		}
#endif
		values[i] = static_cast<T>(_values[local_index]);
    }
}


template <typename T>
inline
void PetscNativeVector<T>::reciprocal()
{
	this->_restore_array();
	
	PetscErrorCode ierr = 0;
	ierr = VecReciprocal(_vec) ;	CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


template <typename T>
inline
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::min () const
{
	this->_restore_array();
	
	int index=0, ierr=0;
	PetscScalar min=0.;
	
	ierr = VecMin (_vec, &index, &min);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	// this return value is correct: VecMin returns a PetscReal
	return static_cast<T>(min);
}



template <typename T>
inline
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::max() const
{
  this->_restore_array();

  int index=0, ierr=0;
  PetscScalar max=0.;

  ierr = VecMax (_vec, &index, &max);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // this return value is correct: VecMax returns a PetscReal
  return static_cast<T>(max);
}



template <typename T>
inline
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::sum () const
{
  this->_restore_array();
  
  PetscErrorCode ierr=0;
  PetscScalar value=0.;
  
  ierr = VecSum (_vec, &value);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
  return static_cast<T>(value);
}


template <typename T>
inline
void PetscNativeVector<T>::swap (Parent& other)
{
  IVector<PetscNativeVector<T> >::swap(other);

  PetscNativeVector<T>& v = dynamic_cast<PetscNativeVector<T>&>(other);

  std::swap(_vec, v._vec);
  std::swap(_destroy_vec_on_exit, v._destroy_vec_on_exit);
  std::swap(_global_to_local_map, v._global_to_local_map);
  std::swap(_array_is_present, v._array_is_present);
  std::swap(_local_form, v._local_form);
  std::swap(_values, v._values);
}



template <typename T>
inline
void PetscNativeVector<T>::_get_array(void) const
{
  karma_assert (this->initialized());
  if(!_array_is_present)
    {
      PetscErrorCode ierr=0;
      if(this->type() != KARMA::GHOSTED)
	{
	  ierr = VecGetArray(_vec, &_values);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
      else
	{
	  ierr = VecGhostGetLocalForm (_vec,&_local_form);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGetArray(_local_form, &_values);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
#ifndef NDEBUG
	  int local_size = 0;
	  ierr = VecGetLocalSize(_local_form, &local_size);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  _local_size = static_cast<unsigned int>(local_size);
#endif
	}
      _array_is_present = true;
    }
}



template <typename T>
inline
void PetscNativeVector<T>::_restore_array(void) const
{
  karma_assert (this->initialized());
  if(_array_is_present)
    {
      PetscErrorCode ierr=0;
      if(this->type() != KARMA::GHOSTED)
	{
	  ierr = VecRestoreArray (_vec, &_values);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  _values = NULL;
	}
      else
	{
	  ierr = VecRestoreArray (_local_form, &_values);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  _values = NULL;
	  ierr = VecGhostRestoreLocalForm (_vec,&_local_form);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  _local_form = NULL;
#ifndef NDEBUG
	  _local_size = 0;
#endif
	}
      _array_is_present = false;
    }
}


template <typename T>
inline
void PetscNativeVector<T>::print(bool is_global, std::ostream& os) const
{
	karma_assert(this->closed()) ;

	PetscErrorCode ierr ;
	PetscViewer viewer ;
	ierr = PetscViewerCreate(KARMA_COMM_WORLD, 
							 &viewer) ;
			CHKERRABORT(KARMA_COMM_WORLD, ierr);
	
	/**
	* Set the format to ASCII and then print the 
	* matrix using PETSc native View.
	*/
	ierr = PetscViewerSetFormat (PETSC_VIEWER_STDOUT_WORLD,
			   PETSC_VIEWER_ASCII_INFO_DETAIL);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	ierr = VecView (_vec, PETSC_VIEWER_STDOUT_WORLD);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	ierr = PetscViewerDestroy(&viewer) ;
			CHKERRABORT(KARMA_COMM_WORLD, ierr);
	
}

template <typename T>
inline
void PetscNativeVector<T>::print_matlab(const std::string name) const
{
	karma_assert(this->closed()) ;

	PetscErrorCode ierr ;
	PetscViewer viewer ;
	
	ierr = PetscViewerCreate(KARMA_COMM_WORLD, 
							 &viewer) ;
			CHKERRABORT(KARMA_COMM_WORLD, ierr);
	
	/**
	* Create an ASCII file containing the matrix
	* if a filename was provided.  
	*/
	if (name != "NULL")
	{
	  ierr = PetscViewerASCIIOpen( KARMA_COMM_WORLD,
				   name.c_str(),
				   &viewer);
			 CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  
	  ierr = PetscViewerSetFormat (viewer,
				   PETSC_VIEWER_ASCII_MATLAB);
			 CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	  ierr = VecView (_vec, viewer);
			 CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	
	/**
	* Otherwise the matrix will be dumped to the screen.
	*/
	else
	{
	  ierr = PetscViewerSetFormat (PETSC_VIEWER_STDOUT_WORLD,
				   PETSC_VIEWER_ASCII_MATLAB);
			 CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	  ierr = VecView (_vec, PETSC_VIEWER_STDOUT_WORLD);
			 CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	
	ierr = PetscViewerDestroy(&viewer) ;
			CHKERRABORT(KARMA_COMM_WORLD, ierr);
}



#endif // #ifdef KARMA_HAVE_PETSC


#endif // #define __petsc_native_vector_h__

