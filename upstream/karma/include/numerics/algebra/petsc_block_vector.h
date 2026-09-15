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


#ifndef __petsc_block_vector_h__
#define __petsc_block_vector_h__

#include "karma_configuration.h"

#if KARMA_HAVE_PETSC && PETSC_VERSION_GREATER_THAN(3,1,8)

// Local includes
#include "ivector.h"

// Petsc includes
#include "petscvec.h"

// Forward declaration
template <typename T>
class PetscNativeVector ;
template <typename T>
class PetscBlockVector ;

// specialize for PetscBlockVector<T>
template<typename T>
struct VectorTraits<PetscBlockVector<T> >
{
	/**
	 *  Create some traits to define Scalar and Composite types.
	 */
	typedef T ScalarType ;
	typedef PetscNativeVector<T>* CompositeType ;
	typedef PetscBlockVector<T> VectorType ;
};


/**
 * Petsc Block vector. Provides a nice interface to the
 * Petsc C-based NEST type data structures for parallel vectors.
 */
template <typename T>
class PetscBlockVector : public IVector<PetscBlockVector<T> >
{
public:

	typedef IVector<PetscBlockVector<T> > Parent ;
	typedef typename VectorTraits<PetscBlockVector<T> >::ScalarType ScalarType ;
	typedef typename VectorTraits<PetscBlockVector<T> >::CompositeType CompositeType ;
	typedef typename VectorTraits<PetscBlockVector<T> >::VectorType Type ;

	/**
	* Constructor. No data is set yet but user needs to call init
	* to allocate blocks and set the vector references.
	*/
	explicit
	PetscBlockVector() ;

	/**
	* Constructor. Takes the number of blocks as argument and initializes
	* the vector with empty blocks, ready to be assembled.
	*/
	PetscBlockVector (const unsigned int n) ;
	/**
	 * Constructor.Creates a PetscBlockVector assuming you already have a
	 * valid PETSc Block_Vec object.  In this case, \p v is NOT destroyed by the
	 * PetscBlockVector constructor when this object goes out of scope.
	 * This allows ownership of \p v to remain with the original creator,
	 * and to simply provide additional functionality with the PetscBlockVector.
	*/
	PetscBlockVector(Vec v) ;

	/**
	* Copy Constructor.  Creates a PetscNativeVector assuming you already have a
	* valid PetscNativeVector pre-allocated.  In this case, the underlying data structures
	* in PetscNativeVector \p v is copied. This is similar to clone.
	*/
	PetscBlockVector(const PetscBlockVector<T>& v) ;
	
	/**
	* Builds a \p PetscNativeVector with a default constructor.
	* This allows us the option to pre-set some user-specific data
	* but highly improbable since we do not know the size of the
	* vector yet ! Anyway, as of now, this will serve as a helper.
	*/
	static AutoPtr<PetscBlockVector<T> > build();

	/**
	* Destructor, deallocates memory. Made virtual to allow
	* for derived classes to behave properly.
	*/
	virtual ~PetscBlockVector() ;
	
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
	* Creates a vector that has the same dimension and storage type as
	* \p other. \p fast determines whether we need to zero out the
	* entries in the new vector.
	*/
	virtual void init (const Parent& other,
						const bool = false) ;
						
	/**
	* call init with n_local = N; type = KARMA::BLOCK
	*/
	virtual void init (const unsigned int N,
			const bool  = false);
	
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
	virtual Parent& operator= (const std::vector<CompositeType > &v);
	
	/**
	 * Access block components, returns \p U(i); A const version.
	 */
	virtual CompositeType const operator[] (unsigned int i) const;
	
	/**
	 * Access block components, returns \p U(i).
	 */
	virtual CompositeType& operator[] (unsigned int i) ;
	
	/**
	 * Access scalar components, returns \p U[block](i); A const version.
	 */
	virtual ScalarType const operator() (unsigned int i) const;
	
	/**
	 * Access scalar components, returns \p U[block](i).
	 */
	virtual ScalarType& operator() (unsigned int i) ;
	
	/**
	 * Access multiple components at once.  Overloaded method that
	 * should be faster (probably much faster) than calling \p
	 * operator() individually for each index.
	 */
	virtual void operator() (const std::vector<unsigned int>& index, std::vector<CompositeType >& values) const ;
	
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
	 * @returns dimension of the vector. This
	 * function was formerly called \p n(), but
	 * was renamed to get the \p PetscNativeVector<T> class
	 * closer to the C++ standard library's
	 * \p std::vector container.
	 */
	virtual unsigned int size () const	{	return _size ;	}
	
	/**
	 * @returns the local size of the vector
	 * (index_stop-index_start)
	 */
	virtual unsigned int local_size() const	{	return _size ;	}
	
	/**
	* Addition operator.
	* Shift each component of this vector by a.
	*/
	virtual Parent& operator += (const ScalarType a) ;
	
	/**
	* Addition operator.
	* Fast equivalent to \p U.add(1, V).
	*/
	virtual Parent& operator += (const Parent&V) ;
	
	/**
	* Subtraction operator.
	* Shift each component of this vector by a.
	*/
	virtual Parent& operator -= (const ScalarType a) ;
	
	/**
	* Subtraction operator.
	* Fast equivalent to \p U.add(-1, V).
	*/
	virtual Parent& operator -= (const Parent&V) ;
	
	/**
	 * \f$ U_i=V_i \f$ where V is type
	 * DenseVector<T> and you
	 * want to specify WHERE to insert it
	 */
	virtual void insert (CompositeType const &V,
						const unsigned int index);
	
	/**
	 * \f$U(0-LIBMESH_DIM)+=s\f$.
	 * Addition of \p s to all components. Note
	 * that \p s is a scalar and not a vector.
	 */
	virtual void add (CompositeType const &V,
					const unsigned int index);

	/**
	 * \f$ U=v \f$ where v is a std::vector<T>
	 * and you want to specify WHERE to insert it
	 */
	virtual void insert (const std::vector<CompositeType>& values,
						 const std::vector<unsigned int>& index);
	
	/**
	 * Access multiple components at once.  Overloaded method that
	 * should be faster (probably much faster) than calling \p
	 * operator() individually for each index.
	 */
	virtual void add(const std::vector<CompositeType>& values,
					 const std::vector<unsigned int>& index) ;
	
	/**
	 * \f$U=V\f$, where U and V are type
	 * IVector<VectorTraits<PetscBlockVector<T> > > and you
	 * want to specify WHERE to insert
	 * the Parent vector \p V
	 */
	virtual void insert (const Parent& V,
						 const std::vector<unsigned int>& indices);
		
	/**
	 * \f$U+=V\f$, where U and V are type
	 * IVector<VectorTraits<PetscBlockVector<T> > > and you
	 * want to specify WHERE to insert
	 * the Parent vector \p V
	 */
	virtual void add (const Parent& V,
						 const std::vector<unsigned int>& indices);
		
	/**
	 * \f$U+=V\f$, where U and V are type
	 * IVector<VectorTraits<PetscBlockVector<T> > >.
	 */
	virtual void add (const Parent& V);
	
	/**
	* Shift each element of the
	* vector by the given \p factor[0].
	* Optionally, can pass an array to shift each
	* element independently. This would then function
	* like a pointwise_multiply for a single vector case
	* but provides regular shift interface for block vector.
	*/
	virtual void shift (const std::vector<ScalarType>& factor) ;
	
	/**
	* Scale each element of the
	* vector by the given \p factor[0].
	* Optionally, can pass an array to scale each
	* element independently. This would then function
	* like a pointwise_multiply for a single vector case
	* but provides regular scale interface for block vector.
	*/
	virtual void scale (const std::vector<ScalarType>& factor) ;
	
	/**
	 * Shift all elements of the individual
	 * vectors by the given factor.
	 */
	virtual void shift (const ScalarType factor) ;
	
	/**
	 * Scale all elements of the individual
	 * vectors by the given factor.
	 */
	virtual void scale (const ScalarType factor);
	
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
	virtual ScalarType dot(const Parent& V) const;

	/**
	* Creates a copy of the global vector \p IVector<VectorTraits<PetscBlockVector<T> > > in the
	* local vector \p v_local.
	*/
	virtual void localize (Parent& v_local,
						   const std::vector<unsigned int>* send_list = NULL) const;

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
	* Creates a "subvector" with a block-representation from this
	* vector using the block indices of the \p rows array.
	*/
	//virtual void create_subvector(Parent& subvector,
	//							  const std::vector<unsigned int>& rows) const;
	
	/**
	 * Swaps the raw PETSc vector context pointers.
	 */
	virtual void swap (Parent& v);
	
	// block-vector specific routines

	/**
	 * Creates a "subvector" with a non-block-representation from this
	 * vector. The 
	 */
	virtual void merge(PetscNativeVector<T>&, const bool aggregate=false) const ;
	
	/**
	* Returns the raw PETSc block vector context pointer.  Note this is generally
	* not required in user-level code. Just don't do anything crazy like
	* calling VecDestroy()!
	*/
	virtual Vec vec () { karma_assert (_vec != PETSC_NULL); return _vec; }

	/**
	 * Returns the raw PETSc vector context pointer.  Note this is generally
	 * not required in user-level code. Just don't do anything crazy like
	 * calling VecDestroy()!
	 */
	virtual const Vec& vec () const { karma_assert (_vec != PETSC_NULL); return _vec; }

private:
	
	bool _destroy_vec_on_exit, _proxy ;
	int _size ;
	Vec _vec ;
	std::vector<CompositeType> _block_vector_obj_refs ;

};

// dummy function to please the compiler for now
int VecBlockSetValue(Vec, unsigned int, Vec, InsertMode)
{
}

// Inline methods
template <typename T>
inline
PetscBlockVector<T>::PetscBlockVector ()
	: IVector<PetscBlockVector<T> >(KARMA::BLOCK),
	_destroy_vec_on_exit(true),
	_proxy(false)
{
	this->init(0) ;
}

template <typename T>
inline
PetscBlockVector<T>::PetscBlockVector (const unsigned int N)
	: IVector<PetscBlockVector<T> >(KARMA::BLOCK),
	_destroy_vec_on_exit(true),
	_proxy(false)
{
	this->init(N) ;
}

template <typename T>
inline
PetscBlockVector<T>::PetscBlockVector(const PetscBlockVector<T>& v)
		: Parent(v),
		_vec( v._vec ),
		_destroy_vec_on_exit (v._destroy_vec_on_exit),
		_proxy (true)/*v._proxy*/,
		_size(v._size)
{
	_block_vector_obj_refs.resize(_size) ;
	for (unsigned int i=0; i < _size; i++)
		_block_vector_obj_refs[i] = v._block_vector_obj_refs[i] ;
}


template <typename T>
inline
void PetscBlockVector<T>::zero()
{
	for (int index=0; index < _size; index++)
		_block_vector_obj_refs[index]->zero() ;
}


template <typename T>
inline
void PetscBlockVector<T>::insert (const std::vector<PetscBlockVector<T>::CompositeType>& v,
					const std::vector<unsigned int>& indices)
{
	karma_assert(indices.size() == v.size()) ;
	//	karma_assert(_size == v.size()) ;	// Not necessarily true..

	PetscErrorCode ierr = 0 ;
	for (unsigned int i=0; i < v.size(); i++)
	{
		if (_proxy)	// if we allocated for this block, free the memory
			delete _block_vector_obj_refs[indices[i]] ;
		
		_block_vector_obj_refs[indices[i]] = v[i] ;

		ierr = VecNestSetValue( _vec, indices[i], v[i]->vec(), INSERT_VALUES) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}

template <typename T>
inline
void PetscBlockVector<T>::add (const std::vector<PetscBlockVector<T>::CompositeType>& v,
					 const std::vector<unsigned int>& indices)
{
	karma_assert(indices.size() == v.size()) ;
	//	karma_assert(_size == v.size()) ;	// Not necessarily true..

	PetscErrorCode ierr = 0 ;
	for (unsigned int i=0; i < v.size(); i++)
	{
		// _block_vector_obj_refs[indices[i]] = v[i] ;

		ierr = VecNestSetValue( _vec, indices[i], v[i]->vec(), ADD_VALUES) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}


template <typename T>
inline
void PetscBlockVector<T>::insert (const Parent& V_in,
					 const std::vector<unsigned int>& indices)
{
	karma_assert(indices.size() == V_in.size()) ;
	//	karma_assert(_size == V.size()) ;	// Not necessarily true..

	const PetscBlockVector<T>& V = dynamic_cast<const PetscBlockVector<T> &>(V_in) ;

	PetscErrorCode ierr = 0 ;
	for (unsigned int i=0; i < V.size(); i++)
	{
		if (_proxy)	// if we allocated for this block, free the memory
			delete _block_vector_obj_refs[indices[i]] ;
		
		_block_vector_obj_refs[indices[i]] = V[i] ;

		ierr = VecNestSetValue( _vec, indices[i], V[i]->vec(), INSERT_VALUES) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}


template <typename T>
inline
void PetscBlockVector<T>::add (const Parent& V_in,
				  const std::vector<unsigned int>& indices)
{
	karma_assert(indices.size() == V_in.size()) ;
	//	karma_assert(_size == V.size()) ;	// Not necessarily true..

	const PetscBlockVector<T>& V = dynamic_cast<const PetscBlockVector<T> &>(V_in) ;

	PetscErrorCode ierr = 0 ;
	for (unsigned int i=0; i < V.size(); i++)
	{
		// _block_vector_obj_refs[indices[i]] = V[i] ;

		ierr = VecNestSetValue( _vec, indices[i], V[i]->vec(), ADD_VALUES) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}


template <typename T>
inline
void PetscBlockVector<T>::insert (CompositeType const &V,
					 const unsigned int index)
{
	karma_assert(index < _size) ;

	if (_proxy)	// if we allocated for this block, free the memory
		delete _block_vector_obj_refs[index] ;
		
	_block_vector_obj_refs[index] = V ;

	PetscErrorCode ierr = VecNestSetValue( _vec, index, V->vec(), INSERT_VALUES) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


template <typename T>
inline
void PetscBlockVector<T>::add (CompositeType const &V,
				  const unsigned int index)
{
	karma_assert(index < _size) ;
	karma_assert(_block_vector_obj_refs[index] != NULL) ;

	//_block_vector_obj_refs[index] = V ;

	PetscErrorCode ierr = VecNestSetValue( _vec, index, V.vec(), ADD_VALUES) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


template <typename T>
inline
void PetscBlockVector<T>::add (const Parent& V_in)
{
	karma_assert(V_in.size() == _size) ;

	const PetscBlockVector<T>& V = dynamic_cast<const PetscBlockVector<T> &>(V_in) ;

	for (unsigned int i=0; i < _size; i++)
	{
		const CompositeType* vecref = V[i] ;
		
		karma_assert(vecref != NULL) ;
	
		PetscErrorCode ierr = VecNestSetValue( _vec, i, vecref->vec(), ADD_VALUES) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	this->close() ;
}


template <typename T>
inline
AutoPtr<typename PetscBlockVector<T>::Parent> PetscBlockVector<T>::deep_clone (bool copy_values) const
{
	PetscBlockVector<T>* cloned_vector (new PetscBlockVector<T>);
	
	cloned_vector->init(*this, !copy_values);
	
	if(copy_values)
	{
		PetscErrorCode ierr = VecCopy( _vec, cloned_vector->_vec);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	
	return AutoPtr<Parent >( cloned_vector );
}



template <typename T>
inline
AutoPtr<typename PetscBlockVector<T>::Parent> PetscBlockVector<T>::clone () const
{
	AutoPtr<Parent > cloned_vector (new PetscBlockVector<T>);
	
	cloned_vector->init(*this, true);
	
	*cloned_vector = *this;
	
	return cloned_vector;
}

template <typename T>
inline
typename PetscBlockVector<T>::CompositeType& PetscBlockVector<T>::operator[] (unsigned int index)
{
	karma_assert(index < _size) ;
	karma_assert(_block_vector_obj_refs[index] != NULL) ;

	// HACK: should we directly work on just the block vector ?
	// Then we need to get and restore block vectors consistently.
	return _block_vector_obj_refs[index] ;
}

template <typename T>
inline
typename PetscBlockVector<T>::CompositeType const PetscBlockVector<T>::operator[] (unsigned int index) const
{
	karma_assert(index < _size) ;
	karma_assert(_block_vector_obj_refs[index] != NULL) ;
	
	// HACK: should we directly work on just the block vector ?
	// Then we need to get and restore block vectors consistently.
	return _block_vector_obj_refs[index] ;
}


template <typename T>
inline
typename PetscBlockVector<T>::ScalarType& PetscBlockVector<T>::operator() (unsigned int index)
{
	karma_assert(index < _size) ;
	karma_assert(_block_vector_obj_refs[0] != NULL) ;

	// HACK: should we directly work on just the block vector ?
	// Then we need to get and restore block vectors consistently.
	return _block_vector_obj_refs[0](index) ;
}

template <typename T>
inline
typename PetscBlockVector<T>::ScalarType const PetscBlockVector<T>::operator() (unsigned int index) const
{
	karma_assert(index < _size) ;
	karma_assert(_block_vector_obj_refs[0] != NULL) ;
	
	// HACK: should we directly work on just the block vector ?
	// Then we need to get and restore block vectors consistently.
	return _block_vector_obj_refs[0](index) ;
}


template <typename T>
inline
void PetscBlockVector<T>::swap (Parent& other)
{
	Parent::swap(other);
	
	PetscBlockVector<T>& v = dynamic_cast<PetscBlockVector<T>&>(other);
	
	std::swap(_vec, v._vec);
	std::swap(_destroy_vec_on_exit, v._destroy_vec_on_exit);
	std::swap(_proxy, v._proxy);
	std::swap(_block_vector_obj_refs, v._block_vector_obj_refs);
	std::swap(_size, v._size);
}


template <typename T>
inline
void PetscBlockVector<T>::print(bool is_global, std::ostream& os) const
{
	karma_assert(this->closed()) ;

	PetscErrorCode ierr ;
	PetscViewer viewer ;
	ierr = PetscViewerCreate(KARMA_COMM_WORLD, &viewer) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRABORT(KARMA_COMM_WORLD, ierr);
	ierr = VecView(_vec, viewer) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
	ierr = PetscViewerDestroy(&viewer) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
	
}

template <typename T>
inline
void PetscBlockVector<T>::print_matlab(const std::string name) const
{
	karma_assert(this->closed()) ;

	PetscErrorCode ierr ;
	PetscViewer viewer ;
	ierr = PetscViewerCreate(KARMA_COMM_WORLD, &viewer) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
	ierr = PetscViewerSetType(viewer, PETSCVIEWERMATLAB);CHKERRABORT(KARMA_COMM_WORLD, ierr);
	ierr = VecView(_vec, viewer);CHKERRABORT(KARMA_COMM_WORLD, ierr);
	ierr = PetscViewerDestroy(&viewer) ;CHKERRABORT(KARMA_COMM_WORLD, ierr);
}



#endif

#endif	// #define __petsc_block_vector_h__
