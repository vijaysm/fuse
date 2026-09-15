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


#ifndef __petsc_native_matrix_h__
#define __petsc_native_matrix_h__

#include "karma_config.h"

#ifdef KARMA_HAVE_PETSC

// Local includes
#include "imatrix.h"
#include "parallel.h"

// Macro to identify and debug functions which should be called in
// parallel on parallel matrices but which may be called in serial on
// serial matrices.  This macro will only be valid inside non-static
// PetscNativeMatrix methods
#undef semiparallel_only
#ifndef NDEBUG
#include <cstring>

#define semiparallel_only() do { if (this->initialized()) { const char *mytype; \
	MatGetType(_mat,&mytype); \
	if (!strcmp(mytype, MATSEQAIJ)) \
	parallel_only(); } } while (0)
#else
#define semiparallel_only()
#endif

/**
* Petsc include files.
*/
#include "petscmat.h"


// Forward Declarations
namespace libMesh
{
	template <typename T> class DenseMatrix;
}


// Forward declaration
template <typename T>
class PetscNativeMatrix ;
template <typename T>
class PetscNativeVector ;

// specialize for PetscNativeMatrix<T>
template<typename T>
struct MatrixTraits<PetscNativeMatrix<T> >
{
	/**
	 *  Create some traits to define Scalar and Composite types.
	 */
	typedef T ScalarType ;
	typedef T CompositeType ;
	typedef PetscNativeMatrix<ScalarType> MatrixType ;
	typedef PetscNativeVector<ScalarType> VectorType ;
};


/**
	* Petsc matrix. Provides a nice interface to the
	* Petsc C-based data structures for parallel,
	* sparse matrices.
*/

template <typename T>
class PetscNativeMatrix : public IMatrix<PetscNativeMatrix<T> > 
{
public:
	
	
	typedef IMatrix<PetscNativeMatrix<T> > BaseMatrix ;
	typedef typename MatrixTraits<PetscNativeMatrix<T> >::ScalarType ScalarType ;
	typedef typename MatrixTraits<PetscNativeMatrix<T> >::CompositeType CompositeType ;
	typedef typename MatrixTraits<PetscNativeMatrix<T> >::MatrixType MatrixType ;
	typedef typename MatrixTraits<PetscNativeMatrix<T> >::VectorType VectorType ;
	
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
	PetscNativeMatrix (const KARMA::LinearAlgebraLayout type = KARMA::AUTOMATIC);

	/**
	* Constructor.  Creates a PetscNativeMatrix assuming you already
	* have a valid Mat object.  In this case, m is NOT destroyed
	* by the PetscNativeMatrix destructor when this object goes out of scope.
	* This allows ownership of m to remain with the original creator,
	* and to simply provide additional functionality with the PetscNativeMatrix.
	*/
	PetscNativeMatrix (Mat m);

	/**
	* Copy Constructor.  Creates a PetscNativeMatrix assuming you already
	* have a valid Mat object.  In this case, m is NOT destroyed
	* by the PetscNativeMatrix destructor when this object goes out of scope.
	* This allows ownership of m to remain with the original creator,
	* and to simply provide additional functionality with the PetscNativeMatrix.
	*/
	PetscNativeMatrix (const PetscNativeMatrix<T>& m);

	/**
	* Destructor. Free all memory, but do not
	* release the memory of the sparsity
	* structure.
	*/
	virtual ~PetscNativeMatrix ();

	/**
	 * Creates a matrix which has the same type, size and partitioning
	 * as this matrix, but whose data is either copied or zeroed out
	 * depending on \p copy_values. Note that the new matrix returned as
	 * an \p AutoPtr has separate memory allocated and needs to be cleared
	 * by the user.
	 */
	virtual AutoPtr<BaseMatrix> deep_clone (bool copy_values=false) const ;
	
	/**
	 * Creates a copy of this matrix and returns it in an \p AutoPtr.
	 */
	virtual AutoPtr<BaseMatrix> clone () const  ;
	
	/**
	* Initialize a Petsc matrix that is of global
	* dimension \f$ m \times  n \f$ with local dimensions
	* \f$ m_l \times n_l \f$.  \p nnz is the number of on-processor
	* nonzeros per row (defaults to 30).
	* \p noz is the number of on-processor
	* nonzeros per row (defaults to 30).
	*/
	void init (const unsigned int m,
			const unsigned int n,
			const unsigned int m_l,
			const unsigned int n_l,
			const unsigned int nnz=30,
			const unsigned int noz=10);

 	/**
	* Initialize a Sparse matrix that is of global
	* dimension \f$ m \times  n \f$ with local dimensions
	* \f$ m_l \times n_l \f$.  \p nnz is the number of on-processor
	* nonzeros per row (defaults to 30).
	* \p noz is the number of on-processor
	* nonzeros per row (defaults to 10).
	*/
	virtual void init (const unsigned int m,
					   const unsigned int n,
					   const unsigned int m_l,
					   const unsigned int n_l,
					   const std::vector<unsigned int>& n_nz,
					   const std::vector<unsigned int>& n_oz) ;
					   
	
 	/**
	* Initialize a Sparse matrix that is of global
	* dimension \f$ m \times  n \f$ with local dimensions
	* \f$ m_l \times n_l \f$.  \p oper is the Linear operator
	* representing the discrete matrix form and can be used
	* to perform all the operations as if we had the actual
	* matrix allocated in memory.
	*/
	// virtual void init (const unsigned int m,
	//				   const unsigned int n,
	//				   const unsigned int m_l,
	//				   const unsigned int n_l,
	//				   const KARMALinearOperator* oper) ;

	/**
	* Creates a matrix that has the same dimension and storage type as
	* \p other. \p fast determines whether we need to copy all the
	* entries in the new matrix or just the layout.
	*/
	virtual void init (const BaseMatrix& other,
						const bool = true) ;
	
	/**
	* Call the Petsc assemble routines.
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
	* Set all entries to 0. This method retains
	* sparsity structure.
	*/
	virtual void zero ();

	/**
	* Set all row entries to 0 then puts diag_value in the diagonal entry
	*/
	virtual void zero_rows (std::vector<int> & rows, T diag_value = 0.0);

	/**
	* @returns true if the matrix is assembled and ready for
	* computation, false otherwise.
	*/
	virtual bool closed() const ;

	/**
	* @returns \p m, the row-dimension of
	* the matrix where the marix is \f$ M \times N \f$.
	*/
	virtual unsigned int m () const;

	/**
	* @returns \p n, the column-dimension of
	* the matrix where the marix is \f$ M \times N \f$.
	*/
	virtual unsigned int n () const;

	/**
	* return row_start, the index of the first
	* matrix row stored on this processor
	*/
	virtual unsigned int row_start () const;

	/**
	* return row_stop, the index of the last
	* matrix row (+1) stored on this processor
	*/
	virtual unsigned int row_stop () const;

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
	* Set the element \p (i,j) to \p value.
	* Throws an error if the entry does
	* not exist. Still, it is allowed to store
	* zero values in non-existent fields.
	*/
	void insert (T const value,
				const unsigned int i,
				const unsigned int j);
	
	/**
	* Add \p value to the element
	* \p (i,j).  Throws an error if
	* the entry does not
	* exist. Still, it is allowed to
	* store zero values in
	* non-existent fields.
	*/
	void add (T const value,
			const unsigned int i,
			const unsigned int j);

	/**
	* Add the full matrix to the
	* Petsc matrix.  This is useful
	* for adding an element matrix
	* at assembly time
	*/
	void add (const libMesh::DenseMatrix<T> &dm,
					const std::vector<unsigned int> &rows,
					const std::vector<unsigned int> &cols);

	/**
	* Same, but assumes the row and column maps are the same.
	* Thus the matrix \p dm must be square.
	*/
	void add (const libMesh::DenseMatrix<T> &dm,
					const std::vector<unsigned int> &dof_indices);
	
	/**
	 * Add a Sparse matrix \p X, scaled with \p a, to \p this,
	 * and stores the result in \p this:
	 * \f$\texttt{this} = a*X + \texttt{this} \f$.
	 * Use this with caution, the sparse matrices need to have the
	 * same nonzero pattern, otherwise \p PETSc will crash!
	 * It is advisable to not only allocate appropriate memory with
	 * \p init() , but also explicitly zero the terms of \p this
	 * whenever you add a non-zero value to \p X.  Note: \p X will
	 * be closed, if not already done, before performing any work.
	 */
	virtual void axpy (const T, BaseMatrix&, KARMA::MatrixStructure mstr=KARMA::Different_NNZ_Pattern ) ;

	/**
	* Add \p this Sparse matrix, scaled with \p _a, to \p _X,
	* and store the result in \p this:
	* \f$\texttt{this} = X + a*\texttt{this} \f$.
	* Use this with caution, the sparse matrices need to have the
	* same nonzero pattern, otherwise \p PETSc will crash!
	* It is advisable to not only allocate appropriate memory with
	* \p init() , but also explicitly zero the terms of \p this
	* whenever you add a non-zero value to \p X.  Note: \p X will
	* be closed, if not already done, before performing any work.
	*/
	virtual void aypx (const T, BaseMatrix&, KARMA::MatrixStructure mstr=KARMA::Different_NNZ_Pattern ) ;

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
	* \p (i,j).  This has the same cost 
	* as operator() for native matrices.
	*/
	virtual CompositeType const operator[] (const std::pair<unsigned int, unsigned int>& index) const ;
	
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
	virtual Real l1_norm () const ;

	/**
	* Return the l2-norm of the matrix or the frobenius
	* norm, that is
	* \f$|M|_2=\sqrt(\sum_{i=1}^{m} \sum_{j=1}^{n} \|a_{ij}\|^2.
	* This is the
	* natural matrix norm that is compatible
	* to the l1-norm for vectors, i.e.
	* \f$|Mv|_1\leq |M|_1 |v|_1\f$.
	*/
	virtual Real l2_norm () const ;

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
	virtual Real linfty_norm () const ;

	/**
	* Add the elements on each row and
	* store it in \p rowsums.  i.e.,
	* \f$v_i=\sum_{j=1 \ldots n} M_ij\f$,
	* for all rows. 
	*/
	virtual void sum_row (VectorType& rowsums) const ;
	
	/**
	* Add the elements on each row and
	* store it in \p rowsums.  i.e.,
	* \f$v_i=\max_{j=1 \ldots n} (M_ij)\f$,
	* for all rows. 
	*/
	virtual void max_row (VectorType& max_rows) const ;
	
	/**
	* Add the elements on each row and
	* store it in \p rowsums.  i.e.,
	* \f$v_i=\min_{j=1 \ldots n} (M_ij)\f$,
	* for all rows. 
	*/
	virtual void min_row (VectorType& min_rows) const ;
	
	/**
	* Shift each element of the
	* matrix by the given \p factor[0].
	* Optionally, can pass an array to scale each
	* element independently. This would then function
	* like a pointwise_multiply for a single matrix case
	* but provides regular scale interface for block matrix.
	*/
	virtual void shift (const T factor) ;
	
	/**
	* Scale each element of the
	* matrix by the given \p factor.
	* For the block matrix, this just represents scaling each
	* block row independently. This would then function
	* like a pointwise_multiply for a single matrix case
	* but provides regular scale interface for block matrix.
	*/
	virtual void scale (const std::vector<T>& factor) ;
	
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
	virtual void scale (const std::vector<T>* factor_left, const std::vector<T>* factor_right) ;
	
	virtual void scale (const VectorType* factor_left, const VectorType* factor_right) ;
	
	/**
	* see if Sparse matrix has a Symmetric structure
	* i.e., A = A^T
	*/
	virtual bool is_symmetric() const ;

	/**
	* Print the relevant information about the matrix-sparsity
	* such as nnz_allocated, nnz_used, nnz_unneeded
	* approximate memory allocated, etc..
	*/
	virtual void print_info() const ;
  
	/**
	* Print the contents of the matrix to the screen
	* with the PETSc viewer.  This function only allows
	* printing to standard out, this is because we have
	* limited ourselves to one PETSc implementation for
	* writing.
	*/
	virtual void print(std::ostream& os=std::cout) const;

	/**
	* Print the contents of the matrix in Matlab's
	* sparse matrix format. Optionally prints the
	* matrix to the file named \p name.  If \p name
	* is not specified it is dumped to the screen.
	*/
	virtual void print_matlab(const std::string name="NULL") const;

	/**
	* Multiplies the matrix with \p arg and stores the result in \p
	* dest.
	*/
	virtual void multiply (const VectorType& v_in,
			   VectorType& v_out) const ;
	
	/**
	* Multiplies the matrix with \p arg and stores the result in \p
	* dest.
	*/
	virtual void transpose_multiply (const VectorType& v_in,
			   VectorType& v_out) const ;
	
	/**
	* Multiplies the matrix with \p arg and adds the result to \p dest.
	*/
	virtual void multiply_add (const VectorType& v_in,
			   VectorType& v_out) const ;
	
	/**
	* Multiplies the matrix with \p arg and adds the result to \p dest.
	*/
	virtual void transpose_multiply_add (const VectorType& v_in,
			   VectorType& v_out) const ;
	
	/**
	* Copies the diagonal part of the matrix into \p dest.
	*/
	virtual void get_diagonal (VectorType& dest) const;

	/**
	* Copies the transpose of the matrix into \p dest, which may be
	* *this.
	*/
	virtual void get_transpose (IMatrix<PetscNativeMatrix<T> >  & dest) const;

  /**
	 * Transposes the current matrix and stores in the same allocated space.
	 * NOTE: For unsymmetric matrices, this might need re-allocation if
	 * the nnz/row increases.
   */
	virtual void transpose () ;

	/**
	* Swaps the raw PETSc matrix context pointers.
	*/
	virtual void swap (BaseMatrix&);

	/**
	* Returns the raw PETSc matrix context pointer.  Note this is generally
	* not required in user-level code. Just don't do anything crazy like
	* calling MatDestroy()!
	*/
	Mat mat () { karma_assert (_mat != NULL); return _mat; }

	const Mat& mat () const { karma_assert (_mat != NULL); return _mat; }
	
protected:

	/**
	* This function either creates or re-initializes
	* a matrix called "submatrix" which is defined
	* by the row and column indices given in the "rows" and "cols" entries.
	* This function is implemented in terms of the MatGetSubMatrix()
	* routine of PETSc.  The boolean reuse_submatrix parameter determines
	* whether or not PETSc will treat "submatrix" as one which has already
	* been used (had memory allocated) or as a new matrix.
	*/
	virtual void _get_submatrix(IMatrix<PetscNativeMatrix<T> >  & submatrix,
								const std::vector<unsigned int>& rows,
								const std::vector<unsigned int>& cols,
								const bool reuse_submatrix) const;

private:

	/**
	* Petsc matrix datatype to store values
	*/
	Mat _mat;

	/**
	* This boolean value should only be set to false
	* for the constructor which takes a PETSc Mat object.
	*/
	bool _destroy_mat_on_exit;
};




//-----------------------------------------------------------------------
// PetscNativeMatrix inline members
template <typename T>
inline
PetscNativeMatrix<T>::PetscNativeMatrix(const KARMA::LinearAlgebraLayout type)
		: IMatrix<PetscNativeMatrix<T> >  (type),
		_destroy_mat_on_exit(true)
{}




template <typename T>
inline
PetscNativeMatrix<T>::PetscNativeMatrix(Mat m)
		: IMatrix<PetscNativeMatrix<T> >  (KARMA::AUTOMATIC),
		_destroy_mat_on_exit(false)
{
	this->_mat = m;
	this->_is_initialized = true;
}



template <typename T>
inline
PetscNativeMatrix<T>::PetscNativeMatrix(const PetscNativeMatrix<T>& m)
		: IMatrix<PetscNativeMatrix<T> >  (m._type),
		_destroy_mat_on_exit(m._destroy_mat_on_exit)
{
	this->_mat = m._mat;
	this->_is_initialized = true;
	BaseMatrix::_is_closed = true;
}


template <typename T>
inline
PetscNativeMatrix<T>::~PetscNativeMatrix()
{
	this->clear();
}



template <typename T>
inline
AutoPtr<IMatrix<PetscNativeMatrix<T> > > PetscNativeMatrix<T>::deep_clone (bool copy_values) const
{
	PetscNativeMatrix<T>* cloned_matrix = new PetscNativeMatrix<T>();
	
	cloned_matrix->init(*this, !copy_values);
	
	return AutoPtr<IMatrix<PetscNativeMatrix<T> > > (cloned_matrix) ;
}



template <typename T>
inline
AutoPtr<IMatrix<PetscNativeMatrix<T> > > PetscNativeMatrix<T>::clone () const
{
	PetscNativeMatrix<T> *cloned_matrix = new PetscNativeMatrix<T>();
	
	// no need to copy because we maintain the same _mat reference
	cloned_matrix->init(*this, true);
	
	*cloned_matrix = *this;
	
	return AutoPtr<IMatrix<PetscNativeMatrix<T> > > (cloned_matrix);
}



template <typename T>
inline
void PetscNativeMatrix<T>::close ()
{
	semiparallel_only();

// BSK - 1/19/2004
// strictly this check should be OK, but it seems to
// fail on matrix-free matrices.  Do they falsely
// state they are assembled?  Check with the developers...
//   if (this->closed())
//     return;

	PetscErrorCode ierr=0;

	ierr = MatAssemblyBegin (_mat, MAT_FINAL_ASSEMBLY);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = MatAssemblyEnd   (_mat, MAT_FINAL_ASSEMBLY);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	BaseMatrix::_is_closed = true ;
}



template <typename T>
inline
unsigned int PetscNativeMatrix<T>::m () const
{
	karma_assert (this->initialized());

	int petsc_m=0, petsc_n=0, ierr=0;

	ierr = MatGetSize (_mat, &petsc_m, &petsc_n);

	return static_cast<unsigned int>(petsc_m);
}



template <typename T>
inline
unsigned int PetscNativeMatrix<T>::n () const
{
	karma_assert (this->initialized());

	int petsc_m=0, petsc_n=0, ierr=0;

	ierr = MatGetSize (_mat, &petsc_m, &petsc_n);

	return static_cast<unsigned int>(petsc_n);
}



template <typename T>
inline
unsigned int PetscNativeMatrix<T>::row_start () const
{
	karma_assert (this->initialized());

	int start=0, stop=0, ierr=0;

	ierr = MatGetOwnershipRange(_mat, &start, &stop);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	return static_cast<unsigned int>(start);
}



template <typename T>
inline
unsigned int PetscNativeMatrix<T>::row_stop () const
{
	karma_assert (this->initialized());

	int start=0, stop=0, ierr=0;

	ierr = MatGetOwnershipRange(_mat, &start, &stop);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	return static_cast<unsigned int>(stop);
}

template <typename T>
inline
std::pair<unsigned int, unsigned int> PetscNativeMatrix<T>::size () const
{
	karma_assert (this->initialized());

	int petsc_m=0, petsc_n=0, ierr=0;

	ierr = MatGetSize (_mat, &petsc_m, &petsc_n);

	return std::make_pair(petsc_m, petsc_n);
}


template <typename T>
inline
std::pair<unsigned int, unsigned int> PetscNativeMatrix<T>::local_size () const
{
	karma_assert (this->initialized());

	int petsc_m=0, petsc_n=0, ierr=0;

	ierr = MatGetLocalSize (_mat, &petsc_m, &petsc_n);

	return std::make_pair(petsc_m, petsc_n);
}


template <typename T>
inline
void PetscNativeMatrix<T>::insert (T const value,
						const unsigned int i,
						const unsigned int j)
{
	karma_assert (this->initialized());

	PetscErrorCode ierr=0, i_val=i, j_val=j;

	PetscScalar petsc_value = static_cast<PetscScalar>(value);
	ierr = MatSetValues(_mat, 1, &i_val, 1, &j_val,
			&petsc_value, INSERT_VALUES);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);
}



template <typename T>
inline
void PetscNativeMatrix<T>::add (T const value,
						const unsigned int i, 
						const unsigned int j)
{
	karma_assert (this->initialized());

	PetscErrorCode ierr=0, i_val=i, j_val=j;

	PetscScalar petsc_value = static_cast<PetscScalar>(value);
	ierr = MatSetValues(_mat, 1, &i_val, 1, &j_val,
			&petsc_value, ADD_VALUES);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);
}



template <typename T>
inline
void PetscNativeMatrix<T>::add(const libMesh::DenseMatrix<T>& dm,
						const std::vector<unsigned int>& dof_indices)
{
	this->add (dm, dof_indices, dof_indices);
}


template <typename T>
inline
void PetscNativeMatrix<T>::axpy (const T a_in, IMatrix<PetscNativeMatrix<T> >   &X_in, KARMA::MatrixStructure mstr_in)
{
	karma_assert (this->initialized());

	// sanity check. but this will not avoid
	// crash due to incompatible sparsity structure...
	// we hope the user chose the right option or
	// our default Different_NNZ_Pattern is good enough
	// to keep us in the clear with only a run-time
	// performance hit.
	karma_assert (this->m() == X_in.m());
	karma_assert (this->n() == X_in.n());

	PetscScalar     a = static_cast<PetscScalar>      (a_in);
	PetscNativeMatrix<T>* X = dynamic_cast<PetscNativeMatrix<T>*> (&X_in);
	MatStructure mstr = static_cast<MatStructure>      (mstr_in);

	karma_assert (X != NULL);

	PetscErrorCode ierr=0;

	// the matrix from which we copy the values has to be assembled/closed
	karma_assert(X->closed());

	semiparallel_only();

// 3.0 & newer style
	ierr = MatAXPY(_mat, a, X->_mat, mstr);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);

}



template <typename T>
inline
void PetscNativeMatrix<T>::aypx (const T a_in, IMatrix<PetscNativeMatrix<T> >   &X_in, KARMA::MatrixStructure mstr_in)
{
	karma_assert (this->initialized());

	// sanity check. but this will not avoid
	// crash due to incompatible sparsity structure...
	// we hope the user chose the right option or
	// our default Different_NNZ_Pattern is good enough
	// to keep us in the clear with only a run-time
	// performance hit.
	karma_assert (this->m() == X_in.m());
	karma_assert (this->n() == X_in.n());

	PetscScalar     a = static_cast<PetscScalar>      (a_in);
	PetscNativeMatrix<T>* X = dynamic_cast<PetscNativeMatrix<T>*> (&X_in);
	MatStructure mstr = static_cast<MatStructure>      (mstr_in);

	karma_assert (X != NULL);

	PetscErrorCode ierr=0;

	// the matrix from which we copy the values has to be assembled/closed
	karma_assert(X->closed());

	semiparallel_only();

	// 2.3.x & newer
	ierr = MatAYPX(_mat, a, X->_mat, mstr);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);

}


// TODO:: This routine seems very inefficient.. Optimize using MatGetValues
// template <typename T>
// inline
// T PetscNativeMatrix<T>::get (const unsigned int i, const unsigned int j)
// {
// 	karma_assert (this->initialized());
// 
// #if PETSC_VERSION_LESS_THAN(2,2,1)
// 	
// 	// PETSc 2.2.0 & older
// 	PetscScalar *petsc_row;
// 	int* petsc_cols;
// 
// #else
// 	
// 	// PETSc 2.2.1 & newer
// 	const PetscScalar *petsc_row;
// 	const PetscInt    *petsc_cols;
// 
// #endif
// 
// 	
// 	// If the entry is not in the sparse matrix, it is 0.
// 	T* value=new T[1];
// 	
// 	int ierr=0 ;
// 	int ncols=0,
// 		i_val=static_cast<int>(i),
// 		j_val=static_cast<int>(j);
// 
// 	
// 	// the matrix needs to be closed for this to work
// 	// this->close();
// 	// but closing it is a semiparallel operation; we want operator()
// 	// to run on one processor.
// 	karma_assert(this->closed());
// 	
// 	ierr = MatGetRow(_mat, i_val, &ncols, &petsc_cols, &petsc_row);
// 	CHKERRABORT(KARMA_COMM_WORLD,ierr);
// 	
// 	// Perform a binary search to find the contiguous index in
// 	// petsc_cols (resp. petsc_row) corresponding to global index j_val
// 	std::pair<const int*, const int*> p =
// 		std::equal_range (&petsc_cols[0], &petsc_cols[0] + ncols, j_val);
// 
// // Found an entry for j_val
// 	if (p.first != p.second)
// 	{
// 	// The entry in the contiguous row corresponding
// 	// to the j_val column of interest
// 		const int j = std::distance (const_cast<int*>(&petsc_cols[0]),
// 									const_cast<int*>(p.first));
// 
// 		karma_assert (j < ncols);
// 		karma_assert (petsc_cols[j] == j_val);
// 
// 		*value = static_cast<T> (petsc_row[j]);
// 	}
// 
// 	ierr  = MatRestoreRow(_mat, i_val,
// 							&ncols, &petsc_cols, &petsc_row);
// 	CHKERRABORT(KARMA_COMM_WORLD,ierr);
// 
// 	return AutoPtr<T>(value);
// }


template <typename T>
inline
const typename PetscNativeMatrix<T>::ScalarType PetscNativeMatrix<T>::operator() (unsigned int i, unsigned int j) const
{
	karma_assert (this->initialized());
	
	/**
	 * This implementation is straight-forward and should hopefully be lot faster
	 * than the non-const version above. 
	 * TODO: profile both methods and choose an optimal implementation for both.
	 */
	PetscScalar petsc_value;
	
	PetscErrorCode ierr=0;
	int ncols=0,
		i_val=static_cast<int>(i),
		j_val=static_cast<int>(j);
	
	// It is only valid for local entries and so check if we are within the range.
	const unsigned int start = this->row_start() ;
	const unsigned int end = this->row_stop() ;
	
	int root_proc = -1 ;
	if (i >= start && i < end)
		root_proc = KARMA::processor_id ;
	
	// bingo ! the value we want resides in the current processor's space
	if (root_proc == KARMA::processor_id)
	{
		ierr = MatGetValues(_mat, 1, &i_val, 1, &j_val, &petsc_value) ;
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
		
		// at some point, stop using libMesh's parallel routines.. we might
		// want to work on a different mpi_communicator that is specific 
		// to the current matrix.
		// broadcast the local value from root_proc
		libMesh::Parallel::broadcast<PetscScalar>(petsc_value, root_proc) ;
	}
	else
	{
		// at some point, stop using libMesh's parallel routines.. we might
		// want to work on a different mpi_communicator that is specific 
		// to the current matrix.
		// receive the broadcast of the value from root_proc
		libMesh::Parallel::broadcast<PetscScalar>(petsc_value, root_proc) ;
	}
	
	// wait for all processors to finish ?
	// libMesh::Parallel::barrier() ;
	
	return static_cast<T> (petsc_value);
}


template <typename T>
inline
typename PetscNativeMatrix<T>::CompositeType const PetscNativeMatrix<T>::operator[] (const std::pair<unsigned int, unsigned int>& index) const
{
	return operator()(index.first, index.second) ;
}

template <typename T>
inline
void PetscNativeMatrix<T>::multiply (const PetscNativeMatrix<T>::VectorType& v_in,
			   PetscNativeMatrix<T>::VectorType& v_out) const
{
	// Make sure the data passed in are really of Petsc types
  const PetscNativeVector<T>* V = dynamic_cast<const PetscNativeVector<T>*>(&v_in);
  PetscNativeVector<T>* Z = dynamic_cast<PetscNativeVector<T>*>(&v_out);

	// close the current matrix
  // this->close();
  karma_assert (this->closed());
  
  // Perform the actual PETSc call to compute matvec product
  PetscErrorCode ierr = MatMult(_mat, V->vec(), Z->vec());
         CHKERRABORT(libMesh::COMM_WORLD,ierr); 
}

template <typename T>
inline
void PetscNativeMatrix<T>::transpose_multiply (const PetscNativeMatrix<T>::VectorType& v_in,
			   PetscNativeMatrix<T>::VectorType& v_out) const
{
// Make sure the data passed in are really of Petsc types
  const PetscNativeVector<T>* V = dynamic_cast<const PetscNativeVector<T>*>(&v_in);
  PetscNativeVector<T>* Z = dynamic_cast<PetscNativeVector<T>*>(&v_out);

	// close the current matrix
  // this->close();
  karma_assert (this->closed());
  
  // Perform the actual PETSc call to compute tranpose matvec product
  PetscErrorCode ierr = MatMultTranspose(_mat, V->vec(), Z->vec());
         CHKERRABORT(libMesh::COMM_WORLD,ierr); 
}


template <typename T>
inline
void PetscNativeMatrix<T>::multiply_add (const PetscNativeMatrix<T>::VectorType& v_in,
			   PetscNativeMatrix<T>::VectorType& v_out) const
{
	// Make sure the data passed in are really of Petsc types
  const PetscNativeVector<T>* V = dynamic_cast<const PetscNativeVector<T>*>(&v_in);
  PetscNativeVector<T>* Z = dynamic_cast<PetscNativeVector<T>*>(&v_out);

	// close the current matrix
  // this->close();
  karma_assert (this->closed());
  
  // Perform the actual PETSc call to compute matvec product
  PetscErrorCode ierr = MatMultAdd(_mat, V->vec(), Z->vec(), Z->vec());
         CHKERRABORT(libMesh::COMM_WORLD,ierr); 
}


template <typename T>
inline
void PetscNativeMatrix<T>::transpose_multiply_add (const PetscNativeMatrix<T>::VectorType& v_in,
			   PetscNativeMatrix<T>::VectorType& v_out) const
{
	// Make sure the data passed in are really of Petsc types
  const PetscNativeVector<T>* V = dynamic_cast<const PetscNativeVector<T>*>(&v_in);
  PetscNativeVector<T>* Z = dynamic_cast<PetscNativeVector<T>*>(&v_out);

	// close the current matrix
  // this->close();
  karma_assert (this->closed());
  
  // Perform the actual PETSc call to compute matvec product
  PetscErrorCode ierr = MatMultTransposeAdd(_mat, V->vec(), Z->vec(), Z->vec());
         CHKERRABORT(libMesh::COMM_WORLD,ierr); 
}

template <typename T>
inline
bool PetscNativeMatrix<T>::closed() const
{
	karma_assert (this->initialized());

	PetscErrorCode ierr=0;
	PetscBool is_assembled;

	ierr = MatAssembled(_mat, &is_assembled);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	BaseMatrix::_is_closed = (is_assembled == PETSC_TRUE) ;
	return BaseMatrix::_is_closed;
}


template <typename T>
inline
void PetscNativeMatrix<T>::swap(IMatrix<PetscNativeMatrix<T> >   &m)
{
	IMatrix<PetscNativeMatrix<T> >::swap(m) ;
	
	PetscNativeMatrix<T>& other = dynamic_cast<PetscNativeMatrix<T>&>(m) ;
	
	std::swap(_mat, other._mat);
	std::swap(_destroy_mat_on_exit, other._destroy_mat_on_exit);
}


template <typename T>
inline
bool PetscNativeMatrix<T>::is_symmetric() const
{
	// check if it is assembled already
	karma_assert (this->closed());
	
	PetscErrorCode ierr=0;
	PetscBool symmetric;

	// we use a default tolerance of 1e-6 for checking. can make this
	// as a user option in the future
	ierr = MatIsSymmetric(_mat, 1e-6, &symmetric);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// also could provide the user an option to just check the sparsity 
	// of the structure to be symmetric instead of values = FAST
	return (symmetric == PETSC_TRUE);
}



template <typename T>
inline
void PetscNativeMatrix<T>::print_info() const
{
	// check if it is assembled already
	karma_assert (this->closed());
	
	PetscErrorCode ierr=0;
	MatInfo minfo ;
	
	// we use a default tolerance of 1e-6 for checking. can make this
	// as a user option in the future
	ierr = MatGetInfo(_mat, MAT_GLOBAL_SUM, &minfo);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	// NOTE: For now, printing only the nnz's and the memory allocated
	KARMA_LOG(INFO) << "-----------Matrix Information------------" ;
	KARMA_LOG(INFO) << "Total number of non-zeros [allocated, used, unneeded]: " <<
			minfo.nz_allocated << ", " << minfo.nz_used << ", " << minfo.nz_unneeded ;
	KARMA_LOG(INFO) << "Total memory allocated: " << minfo.memory ;
	KARMA_LOG(INFO) << "-----------------------------------------" ;
	

}

template <typename T>
inline
void PetscNativeMatrix<T>::print(std::ostream& os) const
{
	karma_assert (this->initialized());

#ifndef NDEBUG
	if (os != std::cout)
		KARMA_LOG(WARNING) << "PETSc can only print to std::cout!" << std::endl;
#endif

	PetscErrorCode ierr=0;
	
	ierr = MatView(_mat, PETSC_VIEWER_STDOUT_SELF);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


#endif	// KARMA_HAVE_PETSC

#endif // #define __petsc_native_matrix_h__

