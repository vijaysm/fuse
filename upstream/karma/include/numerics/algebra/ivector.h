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

#ifndef __ivector_h__
#define __ivector_h__


// C++ includes
#include <set>
#include <memory>

// Local includes
#include "karma.h"
#include "karma_error.h"
#include "karma_utils.h"
#include "linear_algebra_layout.h"

// create a Traits class for Vectors
template<typename derived_vector_type>	// based on LinearAlgebraLayout enum
struct VectorTraits;

/**
 * IVector. Provides a uniform interface
 * to vector storage. The class is inspired by
 * libMesh's NumericVector and Dolfin's GenericVector.
 *
 * @author Vijay S. Mahadevan
 */
template <typename derived_vector_type>
class IVector : public IKARMA<IVector<derived_vector_type> >
{
public:

	typedef IVector<derived_vector_type> Self ;
	typedef typename VectorTraits<derived_vector_type>::ScalarType ScalarType ;
	typedef typename VectorTraits<derived_vector_type>::CompositeType CompositeType ;
	typedef typename VectorTraits<derived_vector_type>::VectorType VectorType ;

	/**
	 *  Constructor. Explicitly just stores the type of vector representing 
	 *  the layout. It is either SEQUENTIAL/PARALLEL/GHOSTED/NESTED.
	 */
	explicit
	IVector (const KARMA::LinearAlgebraLayout type = KARMA::AUTOMATIC);
	
	/**
	* Destructor, deallocates memory. Made virtual to allow
	* for derived classes to behave properly.
	*/
	virtual ~IVector ();
	
	/**
	 * Copy Constructor.  Creates a IVector assuming you already have a
	 * valid concrete instance pre-allocated and initialized.
	 */
	IVector(const Self& v);
	
	/**
	* @returns true if the vector has been initialized,
	* false otherwise.
	*/
	bool initialized() const { return _is_initialized; }
	
	/**
	* @returns the type (SERIAL, PARALLEL, GHOSTED, BLOCK) of the vector.
	*/
	KARMA::LinearAlgebraLayout type() const { return _type; }
	
	/**
	* @returns the type (SERIAL, PARALLEL, GHOSTED, BLOCK) of the vector.
	*/
	KARMA::LinearAlgebraLayout & type() { return _type; }
	
	/**
	* @returns true if the vector is closed and ready for
	* computation, false otherwise.
	*/
	bool closed() const { return _is_closed; }
	
	/**
	* Call the assemble functions
	*/
	virtual 
	void close () = 0;
	
	/**
	 * Initialize the vector with the type representing the layout. It is  
	 * either SEQUENTIAL/PARALLEL/GHOSTED/NESTED.
	 */
	virtual 
	void init () ;
	
	/**
	 * Creates a vector that has the same dimension and storage type as
	 * \p other. \p fast determines whether we need to zero out the
	 * entries in the new vector.
	 */
	virtual void init (const Self& other,
					   const bool = false) = 0;
	
	/**
	* @returns the \p IVector<derived_vector_type> to a pristine state.
	*/
	virtual 
	void clear () ;
	
	/**
	* Set all entries to zero. Equivalent to \p v = 0, but more obvious and
	* faster.
	*/
	virtual 
	void zero () = 0;

	/**
	* Creates a vector which has the same type, size and partitioning
	* as this vector, but whose data is either copied or zeroed out
	* depending on \p copy_values. Note that the new vector returned as
	* an \p AutoPtr has separate memory allocated and needs to be cleared
	* by the user.
	*/
	virtual AutoPtr<Self > deep_clone (bool copy_values=false) const = 0;
	
	/**
	* Creates a copy of this vector and returns it in an \p AutoPtr.
	*/
	virtual AutoPtr<Self > clone () const = 0;
	
	/**
	* \f$U(0-N) = s\f$: fill all components.
	*/
	virtual Self & operator= (ScalarType const s) = 0;
	
	/**
	*  \f$U = V\f$: copy all components.
	*/
	virtual Self & operator= (const Self &V) = 0;
	
	/**
	 * Access block components, returns \p U[i]; A const version.
	 */
	virtual CompositeType const operator[] (const unsigned int i) const = 0;
	
	/**
	 * Access block components, returns \p U[i].
	 */
	virtual CompositeType& operator[] (const unsigned int i) = 0 ;
	
	/**
	* Access leaf components, returns \p U(i); A const version.
	*/
	virtual ScalarType const operator() (const unsigned int i) const = 0;
	
	/**
	* Access leaf components, returns \p U(i).
	*/
	virtual ScalarType& operator() (const unsigned int i) = 0 ;
	
	/**
	 * Access multiple components at once.  \p values will be resized,
	 * if necessary, and filled.  The default implementation calls \p
	 * operator() for each index, but some implementations may supply
	 * faster methods here.
	 */
	virtual void operator() (const std::vector<unsigned int>& index, std::vector<CompositeType>& values) const = 0 ;
	
	/**
	* @returns the minimum element in the vector.
	* In case of complex numbers, this returns the minimum.
	*/
	virtual ScalarType min () const = 0;
	
	/**
	* @returns the maximum element in the vector.
	* In case of complex numbers, this returns the maximum.
	*/
	virtual ScalarType max () const = 0;
	
	/**
	* returns the sum of the elements in a vector
	*/
	virtual ScalarType sum() const = 0;
	
	/**
	* @returns the \f$l_1\f$-norm of the vector, i.e.
	* the sum of the absolute values.
	*/
	virtual ScalarType l1_norm () const = 0;
	
	/**
	* @returns the \f$l_2\f$-norm of the vector, i.e.
	* the square root of the sum of the
	* squares of the elements.
	*/
	virtual ScalarType l2_norm () const = 0;
	
	/**
	* @returns the maximum absolute value of the
	* elements of this vector, which is the
	* \f$l_\infty\f$-norm of a vector.
	*/
	virtual ScalarType linfty_norm () const = 0;
	
	// /**
	// * @returns the \f$l_1\f$-norm of the vector, i.e.
	// * the sum of the absolute values for the specified
	// * entries in the vector.
	// *
	// * Note that the indices must necessary live on this
	// * processor.
	// */
	// virtual ScalarType subset_l1_norm (const std::set<unsigned int> & indices) const;
	
	// /**
	// * @returns the \f$l_2\f$-norm of the vector, i.e.
	// * the square root of the sum of the
	// * squares of the elements for the specified entries
	// * in the vector.
	// *
	// * Note that the indices must necessary live on this
	// * processor.
	// */
	// virtual ScalarType subset_l2_norm (const std::set<unsigned int> & indices) const;
	
	// /**
	// * @returns the maximum absolute value of the
	// * specified entries of this vector, which is the
	// * \f$l_\infty\f$-norm of a vector.
	// *
	// * Note that the indices must necessary live on this
	// * processor.
	// */
	// virtual ScalarType subset_linfty_norm (const std::set<unsigned int> & indices) const;
	
	/**
	* @returns dimension of the vector. This
	* function was formerly called \p n(), but
	* was renamed to get the \p IVector<derived_vector_type> class
	* closer to the C++ standard library's
	* \p std::vector container.
	*/
	virtual unsigned int size () const = 0;
	
	/**
	* @returns the local size of the vector
	* (index_stop-index_start).
	* In ghost cell mode, this does *not* include the ghost cells.
	*/
	virtual unsigned int local_size () const = 0;
	
	/**
	* Addition operator.
	* Shift each component of this vector by a.
	*/
	virtual Self& operator += (const ScalarType a) = 0 ;
	
	/**
	* Addition operator.
	* Fast equivalent to \p U.add(1, V).
	*/
	virtual Self& operator += (const Self &V) = 0 ;
	
	/**
	* Subtraction operator.
	* Shift each component of this vector by a.
	*/
	virtual Self & operator -= (const ScalarType a) = 0 ;
	
	/**
	* Subtraction operator.
	* Fast equivalent to \p U.add(-1, V).
	*/
	virtual Self & operator -= (const Self &V) = 0 ;
	
	/**
	* Multiplication operator.
	* Equivalent to \p U.scale(a)
	*/
	Self & operator *= (const std::vector<ScalarType>& a)
	{ this->scale(a); return *this; }
	
	/**
	* Multiplication operator.
	* Equivalent to \p U.*V
	*/
	Self & operator *= (const Self &V)
	{ this->pointwise_mult(V); return *this; }
	
	/**
	* Division operator.
	* Equivalent to \p U.scale(1./a)
	*/
//	Self & operator /= (const std::vector<ScalarType>& a_in)
//	{
//		std::vector<ScalarType> a(a_in.begin(), a_in.end()) ;
//		for (unsigned int i=0; i < a.size(); i++)
//			a[i] = 1.0/a[i] ;
//		this->scale(a);
//		return *this;
//	}
	
	/**
	* Division operator.
	* Equivalent to \p U./V
	*/
	Self & operator /= (const Self &V)
	{ this->pointwise_divide(V); return *this; }
	
	/**
	* v(i) = value
	*/
	virtual void insert (CompositeType const value, const unsigned int index) = 0;
	
	/**
	* v(i) += value
	*/
	virtual void add (CompositeType const value, const unsigned int index) = 0;

	/**
	* \f$ U=v \f$ where v is a std::vector<ScalarType>
	* and you want to specify WHERE to insert it
	*/
	virtual void insert (const std::vector<CompositeType>& values,
				const std::vector<unsigned int>& index) = 0;
	
	/**
	* \f$ U+=v \f$ where v is a std::vector<ScalarType>
	* and you want to specify WHERE to insert it
	*/
	virtual void add (const std::vector<CompositeType>& values,
				const std::vector<unsigned int>& index) = 0;
	
	/**
	* \f$U+=V\f$:
	* Simple vector addition, equal to the
	* \p operator +=.
	*/
	virtual void add (const Self& V) = 0;
	
	/**
	 * Shift each element of the
	 * vector by the given \p factor[0].
	 * Optionally, can pass an array to scale each
	 * element independently. This would then function
	 * like a pointwise_multiply for a single vector case
	 * but provides regular scale interface for block vector.
	 */
	virtual void shift (const ScalarType factor) = 0 ;
	
	/**
	* Shift each element of the
	* vector by the given \p factor[0].
	* Optionally, can pass an array to scale each
	* element independently. This would then function
	* like a pointwise_multiply for a single vector case
	* but provides regular scale interface for block vector.
	*/
	//virtual void shift (const std::vector<ScalarType>& factor) = 0;
	
	/**
	 * Scale each element of the
	 * vector by the given \p factor[0].
	 * Optionally, can pass an array to scale each
	 * element independently. This would then function
	 * like a pointwise_multiply for a single vector case
	 * but provides regular scale interface for block vector.
	 */
	virtual void scale (const ScalarType factor) = 0 ;
	
	/**
	* Scale each element of the
	* vector by the given \p factor[0].
	* Optionally, can pass an array to scale each
	* element independently. This would then function
	* like a pointwise_multiply for a single vector case
	* but provides regular scale interface for block vector.
	*/
	//virtual void scale (const std::vector<ScalarType>& factor) = 0;
	
	/**
	* Add multiple of given vector (AXPY operation)
	* this += a*X
	*/
	virtual void axpy(const ScalarType a, const Self& x) = 0;
	
	/**
	* Add multiple of given vector (AYPX operation)
	* this = a*this + x
	*/
	virtual void aypx(const ScalarType a, const Self& x) = 0;
	
	/**
	* Add multiple of given vector (AXPY operation)
	* this = a*x + b*this
	*/
	virtual void axpby(const ScalarType a, const Self& x, const ScalarType b) = 0;
	
	/**
	* Add multiple of given vector (AXPY operation)
	* this = a*x + b*y + c*this
	*/
	virtual void axpbypcz(const ScalarType a, const Self& x, const ScalarType b, const Self& y, const ScalarType c) = 0;
	
	/**
	* Add multiple of given vector (AXPY operation)
	* this += \sum_{i=0}^{n}(a_i*x_i)
	*/
	virtual void maxpy(const std::vector<ScalarType>& a, const std::vector<Self* >& x) = 0;
	
	/**
	* v = abs(v)... that is, each entry in v is replaced
	* by its absolute value.
	*/
	virtual void abs() = 0;
	
	/**
	* v = 1./v... that is, each entry in v is replaced
	* by its reciprocal value.
	*/
	virtual void reciprocal() = 0;
	
	/**
	* Computes the dot product, p = U.V
	*/
	virtual ScalarType dot(const Self&) const = 0;
	
	
	/**
	 * Creates a local vector \p v_local containing
	 * only information relevant to this processor, as
	 * defined by the \p send_list.
	 */
	virtual void localize (Self& v_local,
						   const std::vector<unsigned int>* send_list = NULL) const = 0;
	
	/**
	* @returns \p -1 when \p this is equivalent to \p other_vector,
	* up to the given \p threshold.  When differences occur,
	* the return value contains the first index \p i where
	* the difference \p (a[i]-b[i]) exceeded the threshold.  When
	* no threshold is given, the \p KARMA \p TOLERANCE
	* is used.
	*/
	virtual int compare (const Self &other_vector,
				const Real threshold = TOLERANCE) const = 0;
	
	/**
	* Computes the pointwise (i.e. component-wise) product of \p vec1
	* and \p vec2 and stores the result in \p *this.
	*/
	virtual void pointwise_mult (const Self& vec1,
					const Self& vec2) = 0;
	
	/**
	* Computes the pointwise (i.e. component-wise) division of \p vec1
	* and \p vec2 and stores the result in \p *this.
	*/
	virtual void pointwise_divide (const Self& vec1,
					const Self& vec2) = 0;
	
	/**
	* Prints the local contents of the vector, by default to
	* std::cout
	*/
	virtual void print(bool is_global=true, std::ostream& os=std::cout) const = 0;
	
	/**
	* Print the contents of the matrix in Matlab's
	* sparse matrix format. Optionally prints the
	* matrix to the file named \p name.  If \p name
	* is not specified it is dumped to the screen.
	*/
	virtual void print_matlab(const std::string name="NULL") const = 0 ;
	
	/**
	* Creates the subvector "subvector" from the indices in the
	* "rows" array.  Similar to the create_submatrix routine for
	* the SparseMatrix class, it is currently only implemented for
	* PetscVectors.
	*/
	virtual void create_subvector(Self& ,
				const std::vector<unsigned int>& ) const = 0 ;
	
	/**
	* Exchanges the values/sizes of two vectors.  There should be
	* enough indirection in subclasses to make this an O(1) header-swap
	* operation.
	*/
	virtual void swap (Self &v);

	/**
	 * Helper routines to set/return the prefix name set for a vector.
	 */
	const std::string name () const
	{	return _name ;	}
	
	std::string& name ()
	{	return _name ;	}
	

protected:

	
	/**
	* Store the name or the prefix for the current vector to identify uniquely.
	* Useful when setting options from command-line or file
	*/
	std::string _name ;
	
	/**
	* Flag to see if the Numeric
	* assemble routines have been called yet
	*/
	mutable bool _is_closed;

	/**
	* Flag to tell if init
	* has been called yet
	*/
	bool _is_initialized;
	
	/**
	* Type of vector
	*/
	KARMA::LinearAlgebraLayout _type;
};


/*----------------------- Inline functions ----------------------------------*/



template <typename derived_vector_type>
inline
IVector<derived_vector_type>::IVector (const KARMA::LinearAlgebraLayout type) :
		_name(""),
		_is_closed(false),
		_is_initialized(false),
		_type(type)
{
}


template <typename derived_vector_type>
inline
IVector<derived_vector_type>::IVector(const IVector<derived_vector_type>& v) :
		_name(v._name),
		_is_closed(v._is_closed),
		_is_initialized(v._is_initialized),
		_type(v._type)
{
}



template <typename derived_vector_type>
inline
IVector<derived_vector_type>::~IVector ()
{
	clear ();
}


template <typename derived_vector_type>
inline
void IVector<derived_vector_type>::init ()
{
	this->close() ;
	_is_closed      = true;
	_is_initialized = true;
}



template <typename derived_vector_type>
inline
void IVector<derived_vector_type>::clear ()
{
	_is_closed      = false;
	_is_initialized = false;
}



template <typename derived_vector_type>
inline
void  IVector<derived_vector_type>::swap (IVector<derived_vector_type> &v)
{
	std::swap(_name, v._name);
	std::swap(_is_closed, v._is_closed);
	std::swap(_is_initialized, v._is_initialized);
	std::swap(_type, v._type);
}


#endif  // #ifdef __ivector_h__
