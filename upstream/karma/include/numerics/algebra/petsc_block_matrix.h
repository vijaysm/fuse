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


#ifndef __petsc_block_matrix_h__
#define __petsc_block_matrix_h__

#include "karma_config.h"

#ifdef USE_PETSC_BLOCK_EXTENSIONS

// petsc related includes
#include "petsc_native_matrix.h"
#include "petsc_block_vector.h"
#include "petscext_mat.h"

// Forward declarations
class KARMALinearOperator ;
class KARMAParameters ;

// Forward declaration
template <typename T>
class PetscBlockMatrix ;

// specialize for PetscBlockMatrix<T,LayoutType>
template<typename T>
struct MatrixTraits<T, KARMA::BLOCK>			// BLOCKED
{
	/**
	 *  Create some traits to define BASE and AGGREGATE types.
	 */
	static const unsigned int TypeID = 4 ;
	typedef T ScalarType ;
	typedef PetscNativeMatrix<ScalarType>* CompositeType ;
	typedef PetscBlockMatrix<ScalarType> MatrixType ;
	typedef PetscBlockVector<ScalarType> VectorType ;
};


template <typename T>
class PetscBlockMatrix : public IMatrix<MatrixTraits<T, KARMA::BLOCK> >
{
public:

	typedef typename MatrixTraits<T, KARMA::BLOCK>::ScalarType ScalarType ;
	typedef typename MatrixTraits<T, KARMA::BLOCK>::CompositeType CompositeType ;
	typedef typename MatrixTraits<T, KARMA::BLOCK>::MatrixType MatrixType ;
	typedef typename MatrixTraits<T, KARMA::BLOCK>::VectorType VectorType ;
	
	/**
	* Constructor; initializes the matrix to
	* be empty, without any structure, i.e.
	* the matrix is not usable at all. This
	* constructor is therefore only useful
	* for matrices which are members of a
	* class. All other matrices should be
	* created at a point in the data flow
	* where all necessary information is
	* available.
	*
	* You have to initialize
	* the matrix before usage with
	* \p init(...).
	*/
	explicit
	PetscBlockMatrix (const KARMA::LinearAlgebraLayout type = KARMA::AUTOMATIC);

	/**
	* Constructor. No data is set yet but user needs to call init
	* to allocate blocks and set the vector references.
	*/
	PetscBlockMatrix(int total_blocks) ;

	/**
	 * Constructor. Creates a PetscBlockMatrix assuming you already have a
	 * valid PETSc Block_Vec object.  In this case, \p v is NOT destroyed by the
	 * PetscBlockMatrix constructor when this object goes out of scope.
	 * This allows ownership of \p v to remain with the original creator,
	 * and to simply provide additional functionality with the PetscBlockMatrix.
	*/
	PetscBlockMatrix(Mat usermat) ;

	/**
	* Copy Constructor.  Creates a PetscNativeVector assuming you already have a
	* valid PetscNativeVector pre-allocated.  In this case, the underlying data structures
	* in PetscNativeVector \p v is copied. This is similar to clone.
	*/
	PetscBlockMatrix(const PetscBlockMatrix<T>& v);
	
	/**
	* Builds a \p PetscNativeVector with a default constructor.
	* This allows us the option to pre-set some user-specific data
	* but highly improbable since we do not know the size of the
	* vector yet ! Anyway, as of now, this will serve as a helper.
	*/
	static AutoPtr<PetscBlockMatrix<T> > build();

	/**
	* Destructor, deallocates memory. Made virtual to allow
	* for derived classes to behave properly.
	*/
	virtual ~PetscBlockMatrix() ;
	
	/**
	 * Creates a vector which has the same type, size and partitioning
	 * as this vector, but whose data is either copied or zeroed out
	 * depending on \p copy_values. Note that the new vector returned as
	 * an \p AutoPtr has separate memory allocated and needs to be cleared
	 * by the user.
	 */
	virtual AutoPtr<IMatrix<MatrixTraits<T, KARMA::BLOCK> > > deep_clone (bool copy_values=false) const ;
	
	/**
	 * Creates a copy of this vector and returns it in an \p AutoPtr.
	 */
	virtual AutoPtr<IMatrix<MatrixTraits<T, KARMA::BLOCK> > > clone () const ;
	
	/**
	* Initialize a Sparse matrix that is of global
	* dimension \f$ m \times  n \f$ with local dimensions
	* \f$ m_l \times n_l \f$.  \p nnz is the number of on-processor
	* nonzeros per row (defaults to 30).
	* \p noz is the number of on-processor
	* nonzeros per row (defaults to 10).
	* Note: This simply creates a single matrix block [1x1].
	*/
	virtual void init (const unsigned int m,
				const unsigned int n,
				const unsigned int m_l,
				const unsigned int n_l,
				const unsigned int nnz=30,
				const unsigned int noz=10) { karma_not_implemented() ; }
  
	/**
	* Initialize a Sparse matrix that is of global
	* dimension \f$ m \times  n \f$ with local dimensions
	* \f$ m_l \times n_l \f$.  \p nnz is the number of on-processor
	* nonzeros per row (defaults to 30).
	* \p noz is the number of on-processor
	* nonzeros per row (defaults to 10).
	* Note: This simply creates a single matrix block [1x1].
	*/
	virtual void init (const unsigned int m,
						const unsigned int n,
						const unsigned int m_l,
						const unsigned int n_l,
						const std::vector<unsigned int>& n_nz,
						const std::vector<unsigned int>& n_oz) { karma_not_implemented() ; }
	
 	/**
	* Initialize a Sparse matrix that is of global
	* dimension \f$ m \times  n \f$ with local dimensions
	* \f$ m_l \times n_l \f$.  \p oper is the Linear operator
	* representing the discrete matrix form and can be used
	* to perform all the operations as if we had the actual
	* matrix allocated in memory.
	* Note: This simply creates a single matrix block [1x1].
	*/
	virtual void init (const unsigned int m,
					   const unsigned int n,
					   const unsigned int m_l,
					   const unsigned int n_l,
					   const KARMALinearOperator* oper)  { karma_not_implemented() ; }

	/**
	* Initialize the matrix with a pre-allocated STL vector. With the
	* appropriate global \p N and \p n_local unknowns, either a 
	* SEQUENTIAL or PARALLEL layout can be created. After this init()
	* call, the user needs to clear the \p matrices array if \p copy_array
	* is set to false. Also the \p xindices and \p yindices are given in
	* row-major-ordering for the corresponding entries of \p matrices.
	*/
	virtual void init (const unsigned int m,
					   const unsigned int n,
					   const unsigned int m_l,
					   const unsigned int n_l,
					   const std::vector<unsigned int>& xindices,
					   const std::vector<unsigned int>& yindices,
					   const std::vector<CompositeType>& matrices,
					   const bool copy_arrays=true,
			const KARMA::LinearAlgebraLayout = KARMA::AUTOMATIC) ;
	
	/**
	* Creates a matrix that has the same dimension and storage type as
	* \p other. \p fast determines whether we need to copy all the
	* entries in the new matrix or just the layout.
	*/
	virtual void init (const IMatrix<MatrixTraits<T, KARMA::BLOCK> >& other,
						const bool = true) { karma_not_implemented() ; }
	
	/**
	* Call the Sparse assemble routines.
	* sends necessary messages to other
	* processors
	*/
	virtual void close () ;
	
	/**
	* Release all memory and return
	* to a state just like after
	* having called the default
	* constructor.
	*/
	virtual void clear () ;
	
	/**
	* Set all entries to 0.
	*/
	virtual void zero () ;

	/**
	* @returns \p m, the row-dimension of
	* the matrix where the marix is \f$ M \times N \f$.
	*/
	virtual unsigned int m () const { return _size; } ;
	
	/**
	* @returns \p n, the column-dimension of
	* the matrix where the marix is \f$ M \times N \f$.
	*/
	virtual unsigned int n () const { return _size; } ;
	
	/**
	* return row_start, the index of the first
	* matrix row stored on this processor
	*/
	virtual unsigned int row_start () const { return 0;  } ;
	
	/**
	* return row_stop, the index of the last
	* matrix row (+1) stored on this processor
	*/
	virtual unsigned int row_stop () const { return _size; } ;
	
	/**
	* return <\p m, \p n> as a pair indicating the
	* global row and column dimensions of the matrix
	*/
	virtual std::pair<unsigned int, unsigned int> size () const ;
	
	/**
	* return <\p m_l, \p n_l> as a pair indicating the
	* local row and column dimensions of the matrix
	*/
	virtual std::pair<unsigned int, unsigned int> local_size () const ;
	
	/**
	* Set the block matrix \p (i,j) to \p value.
	* Throws an error if the both a matrix and
	* operator exists. 
	*/
	virtual void insert (CompositeType const value,
				const unsigned int i,
				const unsigned int j);

	/**
	* Set the block operator \p (i,j) to \p value.
	* Throws an error if the both a matrix and
	* operator exists. 
	*/
	virtual void insert (const KARMALinearOperator* value,
				const unsigned int i,
				const unsigned int j);

	/**
	* Add the block matrix \p value to location \p (i,j).
	* Throws an error if the a matrix does not exist
	* already in this location. 
	* NOTE: It does not make sense to do the same with
	* the LinearOperator.
	*/
	virtual void add (CompositeType const value,
				const unsigned int i,
				const unsigned int j);

	/**
	* Add a Sparse matrix \p _X, scaled with \p _a, to \p this,
	* stores the result in \p this:
	* \f$\texttt{this} = \_a*\_X + \texttt{this} \f$.
	*/
	virtual void axpy (const T, IMatrix<MatrixTraits<T, KARMA::BLOCK> >&,
					KARMA::MatrixStructure mstr=KARMA::Different_NNZ_Pattern ) {karma_not_implemented();}
		
	/**
	* Add \p this Sparse matrix, scaled with \p _a, to \p _X,
	* and store the result in \p this:
	* \f$\texttt{this} = \_X + \_a*\texttt{this} \f$.
	*/
	virtual void aypx (const T, IMatrix<MatrixTraits<T, KARMA::BLOCK> >&,
					KARMA::MatrixStructure mstr=KARMA::Different_NNZ_Pattern ) {karma_not_implemented();}
	
	/**
	* Return the value of the entry
	* \p (i,j).  This may be an
	* expensive operation and you
	* should always take care where
	* to call this function.  In
	* order to avoid abuse, this
	* function throws an exception
	* if the required element does
	* not exist in the matrix.
	*
	* In case you want a function
	* that returns zero instead (for
	* entries that are not in the
	* sparsity pattern of the
	* matrix), use the \p el
	* function.
	*/
	virtual const ScalarType operator() (unsigned int i,
				unsigned int j) const ;
	
	/**
	* Return the block value of the entry
	* \p (i,j).  This is a simple look-up
	* for block matrices in the stored 
	* reference array.
	*
	* Note that this reference
	* could be null and so user needs to take 
	* care to check if the value is safe to use.
	*/
	virtual const CompositeType operator[] (const std::pair<unsigned int, unsigned int>& index) const ;
	
	virtual void operator() (std::vector<CompositeType>&) ;
	
	virtual void operator() (std::vector<KARMALinearOperator*>&) ;
	
	/**
	* Return the l1-norm of the matrix, that is
	* \f$|M|_1=max_{j=1 \ldots n}\sum_{i=1 \ldots m}
	* |M_ij|\f$,
	* (max. sum of columns).
	* This is the
	* natural matrix norm that is compatible
	* to the l1-norm for vectors, i.e.
	* \f$|Mv|_1\leq |M|_1 |v|_1\f$.
	*/
	virtual Real l1_norm () const { karma_not_implemented() ; }
	
	/**
	* Return the l2-norm of the matrix or the frobenius
	* norm, that is
	* \f$|M|_2=\sqrt(\sum_{i=1}^{m} \sum_{j=1}^{n} \|a_{ij}\|^2.
	* This is the
	* natural matrix norm that is compatible
	* to the l1-norm for vectors, i.e.
	* \f$|Mv|_1\leq |M|_1 |v|_1\f$.
	*/
	virtual Real l2_norm () const { karma_not_implemented() ; }
	
	/**
	* Return the linfty-norm of the
	* matrix, that is
	* \f$|M|_\infty=max_{i=1 \ldots m}
	* \sum_{j=1 \ldots n} |M_ij|\f$,
	* (max. sum of rows).
	* This is the
	* natural matrix norm that is compatible
	* to the linfty-norm of vectors, i.e.
	* \f$|Mv|_\infty \leq |M|_\infty |v|_\infty\f$.
	*/
	virtual Real linfty_norm () const { karma_not_implemented() ; }
	
	/**
	* Shift each element of the diagonal of
	* matrix by the given \p factor[0].
	* Optionally, can pass an array to scale each
	* element independently. This would then function
	* like a pointwise_multiply for a single matrix case
	* but provides regular scale interface for block matrix.
	*/
	virtual void shift (const T factor)  { karma_not_implemented() ; }
	
	/**
	* Scale each element of the
	* matrix by the given \p factor.
	* For the block matrix, this just represents scaling each
	* block row independently. This would then function
	* like a pointwise_multiply for a single matrix case
	* but provides regular scale interface for block matrix.
	*/
	virtual void scale (const std::vector<T>& factor)  { karma_not_implemented() ; }
	
	/**
	* Diagonal Scale each element of the
	* matrix by the given \p factor_left or \p factor_right.
	* These vectors correspondingly scale the matrix from the 
	* left and right respectively.
	* For the block matrix, this just represents scaling each
	* block row independently. This would then function
	* like a pointwise_multiply for a single matrix case
	* but provides regular scale interface for block matrix.
	* Either \p factor_left or \p factor_right can be NULL.
	*/
	virtual void scale (const std::vector<T>* factor_left, const std::vector<T>* factor_right)  { karma_not_implemented() ; }
	
	virtual void scale (const VectorType* factor_left, const VectorType* factor_right)  { karma_not_implemented() ; }
	
	
	/**
	* see if Sparse matrix has a Symmetric structure
	* i.e., A = A^T
	*/
	virtual bool is_symmetric() const { karma_not_implemented() ; }
	
	/**
	* Print the relevant information about the matrix-sparsity
	* such as nnz_allocated, nnz_used, nnz_unneeded
	* approximate memory allocated, etc..
	*/
	virtual void print_info() const { karma_not_implemented() ; }
	
	/**
	* Print the contents of the matrix to the screen
	* in a package-personalized style, if available.
	*/
	virtual void print(std::ostream& os=libMesh::out) const { karma_not_implemented() ; }
	
	/**
	* Print the contents of the matrix in Matlab's
	* sparse matrix format. Optionally prints the
	* matrix to the file named \p name.  If \p name
	* is not specified it is dumped to the screen.
	*/
	virtual void print_matlab(const std::string name="NULL") const { karma_not_implemented() ; }
	
	
	/**
	* Multiplies the matrix with \p arg and stores the result in \p
	* dest.
	*/
	virtual void multiply (const VectorType& v_in,
			   VectorType& v_out) const  { karma_not_implemented() ; }
	
	virtual void multiply(VectorType& , VectorType&, std::vector<KARMAParameters*>& ) ;
	
	/**
	* Multiplies the matrix with \p arg and stores the result in \p
	* dest.
	*/
	virtual void transpose_multiply (const VectorType& v_in,
			   VectorType& v_out) const  { karma_not_implemented() ; }
	
	/**
	* Multiplies the matrix with \p arg and adds the result to \p dest.
	*/
	virtual void multiply_add (const VectorType& v_in,
			   VectorType& v_out) const  { karma_not_implemented() ; }
	
	virtual void multiply_add(VectorType& , VectorType&, std::vector<KARMAParameters*>& ) ;

	/**
	* Multiplies the matrix with \p arg and adds the result to \p dest.
	*/
	virtual void transpose_multiply_add (const VectorType& v_in,
			   VectorType& v_out) const  { karma_not_implemented() ; }
	
	/**
	* Copies the diagonal part of the matrix into \p dest.
	*/
	virtual void get_diagonal (VectorType& dest) const  { karma_not_implemented() ; }
	
	/**
	* Copies the transpose of the matrix into \p dest, which may be
	* *this.
	*/
	virtual void get_transpose (IMatrix<MatrixTraits<T, KARMA::BLOCK> >& dest) const  { karma_not_implemented() ; }
	
	/**
	* Transposes the current matrix and stores in the same allocated space.
	* NOTE: For unsymmetric matrices, this might need re-allocation if
	* the nnz/row increases.
	*/
	virtual void transpose ()  { karma_not_implemented() ; }

	/**
	 * Returns the raw PETSc matrix context pointer.  Note this is generally
	 * not required in user-level code. Just don't do anything crazy like
	 * calling MatDestroy()!
	 */
	Mat mat () { karma_assert (_mat != NULL); return _mat; }

	const Mat& mat () const { karma_assert (_mat != NULL); return _mat; }
	
	/**
	* Print the contents of the matrix in Matlab's
	* sparse matrix format. Optionally prints the
	* matrix to the file named \p name.  If \p name
	* is not specified it is dumped to the screen.
	*/
	virtual bool& fill_complete()
	{
		return _fill_complete ;
	}
	
// 	virtual AutoPtr<PetscBlockMatrix<T> > create_view()
// 	{return AutoPtr<PetscBlockMatrix<T> >(NULL) ;}	// Take a parameter for BD(block-diag), BLD, BUD
	
private:
	
	/**
	* Protected implementation of the create_submatrix and reinit_submatrix
	* routines.  Note that this function must be redefined in derived classes
	* for it to work properly!
	*/
	virtual void _get_submatrix(IMatrix<MatrixTraits<T, KARMA::BLOCK> >& ,
					const std::vector<unsigned int>& ,
					const std::vector<unsigned int>& ,
					const bool) const ;

	bool _external, _fill_complete ;
	int _size ;
	Mat _mat ;
	std::vector<CompositeType > _block_matrix_refs ;
	std::vector<KARMALinearOperator* > _block_matrix_ops ;

};



template <typename T>
inline
PetscBlockMatrix<T>::PetscBlockMatrix (const KARMA::LinearAlgebraLayout type)
		: IMatrix<MatrixTraits<T, KARMA::BLOCK> >(type)
{
	_is_initialized = false ;
	_size = 0 ;
	_external = false ;
	_fill_complete = false ;
}

template <typename T>
inline
PetscBlockMatrix<T>::PetscBlockMatrix (int size)
		: IMatrix<MatrixTraits<T, KARMA::BLOCK> >(KARMA::BLOCK)
{
	_is_initialized = false ;
	_size = size ;
	_external = false ;
	_fill_complete = false ;
}

template <typename T>
inline
PetscBlockMatrix<T>::PetscBlockMatrix(Mat usermat)
		: IMatrix<MatrixTraits<T, KARMA::BLOCK> >(KARMA::BLOCK)
{
	karma_not_implemented() ;
	
		// // check if this is a block-matrix and get appropriate references for matrix parts
		// unsigned int xi = 0, yi = 0 ;
		// for (unsigned int j=0; j < yindices.size(); j++)
		// {
			// // get the index of block row
			// unsigned int xi = std::find(xindices.begin(), xindices.end(), j) - xindices.begin();
			// this->insert(matrices[j], xindices[xi], yindices[j]) ;
		// }
		// this->close() ;
		
		// // if this is a single matrix, then set the size to 1 and create a 1x1 block
		
	_is_initialized = true ;
	MatGetSize(usermat, &_size, PETSC_NULL);
	_external = true ;
	_fill_complete = false ;
	_mat = usermat ;
	_block_matrix_ops.resize(0) ;
	_block_matrix_refs.resize(_size*_size) ;
	
}


template <typename T>
inline
AutoPtr<IMatrix<MatrixTraits<T, KARMA::BLOCK> > > PetscBlockMatrix<T>::deep_clone (bool copy_values) const
{
	PetscBlockMatrix<T>* cloned_matrix = new PetscBlockMatrix<T>();
	
	cloned_matrix->init(*this, !copy_values);
	
	return AutoPtr<IMatrix<MatrixTraits<T, KARMA::BLOCK> > > ( dynamic_cast<IMatrix<MatrixTraits<T, KARMA::BLOCK> >*>(cloned_matrix) ) ;
}



template <typename T>
inline
AutoPtr<IMatrix<MatrixTraits<T, KARMA::BLOCK> > > PetscBlockMatrix<T>::clone () const
{
	AutoPtr<IMatrix<MatrixTraits<T, KARMA::BLOCK> > > cloned_matrix (new PetscBlockMatrix<T>);
	
	// no need to copy because we maintina the same _mat reference
	cloned_matrix->init(*this, true);
	
	*cloned_matrix = *this;
	
	return cloned_matrix;
}


template <typename T>
inline
void PetscBlockMatrix<T>::operator() (std::vector<PetscBlockMatrix<T>::CompositeType>& other)
{
	other.resize(_block_matrix_refs.size()) ;
	for (unsigned int i=0; i<other.size(); i++)
		other[i] = _block_matrix_refs[i] ;
}

template <typename T>
inline
void PetscBlockMatrix<T>::operator() (std::vector<KARMALinearOperator*>& other)
{
	other.resize(_block_matrix_ops.size()) ;
	for (unsigned int i=0; i<other.size(); i++)
		other[i] = _block_matrix_ops[i] ;
}

template <typename T>
inline
std::pair<unsigned int, unsigned int> PetscBlockMatrix<T>::size () const
{
	return std::make_pair<unsigned int, unsigned int>(_size, _size) ;
}
	
template <typename T>
inline
std::pair<unsigned int, unsigned int> PetscBlockMatrix<T>::local_size () const
{
	return std::make_pair<unsigned int, unsigned int>(_size, _size) ;
}
	


#endif

#endif	// #define __petsc_block_matrix_h__
