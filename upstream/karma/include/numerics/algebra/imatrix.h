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

#ifndef __imatrix_h__
#define __imatrix_h__


// C++ includes
#include <iomanip>
#include <vector>

// Local includes
#include "ikarma.h"
#include "karma.h"
#include "karma_error.h"
#include "linear_algebra_layout.h"
#include "matrix_structure.h"

// forward declarations
class UserSparsity;
class KARMALinearOperator ;


// create a Traits class for Matrix
template<typename derived_matrix_type>	// based on LinearAlgebraLayout enum
struct MatrixTraits; 

/**
 * Generic matrix interface. This class contains
 * pure virtual members that must be overloaded
 * in derived classes.  Using a derived class
 * allows for uniform access to dense/sparse matrices
 * from various different solver packages in
 * different formats.
 *
 * @author Vijay S. Mahadevan
 */
template<typename derived_matrix_type>
class IMatrix : public IKARMA<IMatrix<derived_matrix_type> >
{
public:

	typedef typename MatrixTraits<derived_matrix_type>::ScalarType ScalarType ;
	typedef typename MatrixTraits<derived_matrix_type>::CompositeType CompositeType ;
	typedef typename MatrixTraits<derived_matrix_type>::MatrixType MatrixType ;
	typedef typename MatrixTraits<derived_matrix_type>::VectorType VectorType ;

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
	IMatrix (const KARMA::LinearAlgebraLayout type = KARMA::AUTOMATIC);

	/**
	* Destructor. Free all memory, including
	* the memory of the sparsity
	* structure.
	*/
	virtual ~IMatrix ();

	/**
	* @returns true if the matrix has been initialized,
	* false otherwise.
	*/
	virtual 
	bool initialized() const { return _is_initialized; }
	
	/**
	* @returns the const type (SERIAL, PARALLEL, <GHOSTED>, BLOCK) of the vector.
	* i.e., it specifies whether the matrix is Sequential, MPI-based, or Blocked.
	* Currently the Ghosted matrix implementation is not available but in principle,
	* you could imagine ASM preconditioners operating on a similar structure.
	*/
	KARMA::LinearAlgebraLayout type() const { return _type; }
	
	/**
	* @returns the type (SERIAL, PARALLEL, <GHOSTED>, BLOCK) of the vector.
	* i.e., it specifies whether the matrix is Sequential, MPI-based, or Blocked.
	* Currently the Ghosted matrix implementation is not available but in principle,
	* you could imagine ASM preconditioners operating on a similar structure.
	*/
	KARMA::LinearAlgebraLayout & type() { return _type; }
	
	/**
	 * Return the const immutable name/prefix for a matrix as set by user.
	 */
	const std::string name () const
	{	return _name ;	}
	
	/**
	 * Return the writable name/prefix for a matrix.
	 */
	std::string& name ()
	{	return _name ;	}
	
	/**
	 * @returns true if the matrix is assembled and ready for
	 * computation, false otherwise.
	 */
	virtual bool closed() const { return _is_closed; }
	
	/**
	 * Call the Sparse assemble routines.
	 * sends necessary messages to other
	 * processors
	 */
	virtual void close () = 0;
	
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
					   const unsigned int nnz=30,
					   const unsigned int noz=10) = 0;
	
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
	virtual void zero () = 0;
	
	
	/**
	* \p returns true if this sparse matrix format needs to be fed the
	* graph of the sparse matrix.  This is true in the case of the \p LaspackMatrix,
	* but not for the \p PetscMatrix.  In the case where the full graph is not
	* required we can efficiently approximate it to provide a good estimate of the
	* required size of the sparse matrix.
	*/
	virtual bool need_full_sparsity_pattern() const
	{ return false; }

	/**
	 * Creates a matrix which has the same type, size and partitioning
	 * as this matrix, but whose data is either copied or zeroed out
	 * depending on \p copy_values. Note that the new matrix returned as
	 * an \p AutoPtr has separate memory allocated and needs to be cleared
	 * by the user.
	 */
	virtual AutoPtr<IMatrix<derived_matrix_type> > deep_clone (bool copy_values=false) const = 0 ;
	
	/**
	 * Creates a copy of this matrix and returns it in an \p AutoPtr.
	 */
	virtual AutoPtr<IMatrix<derived_matrix_type> > clone () const = 0 ;
	
	/**
	* Updates the matrix sparsity pattern. When your \p IMatrix<derived_matrix_type> 
	* implementation does not need this data simply do
	* not overload this method.
	*/
	virtual void update_sparsity_pattern (const UserSparsity&)
	{
		if (this->need_full_sparsity_pattern())
		{
			// concrete matrix classes need to overload the 
			// sparsity update if they need it.
			karma_error() ;
		}
	}
	
	/**
	* Initialize a Sparse matrix that is of global
	* dimension \f$ m \times  n \f$ with local dimensions
	* \f$ m_l \times n_l \f$.  \p nnz is the number of on-processor
	* nonzeros per row (defaults to 30).
	* \p noz is the number of on-processor
	* nonzeros per row (defaults to 10).
	*/
	// virtual void init (const unsigned int m,
	//					const unsigned int n,
	//					const unsigned int m_l,
	//					const unsigned int n_l,
	//					const std::vector<unsigned int>& n_nz,
	//					const std::vector<unsigned int>& n_oz) = 0;
	
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
	//				   const KARMALinearOperator* oper) = 0;

	// /**
	// * Initialize the matrix with a pre-allocated STL vector. With the
	// * appropriate global \p N and \p n_local unknowns, either a 
	// * SEQUENTIAL or PARALLEL layout can be created. After this init()
	// * call, the user needs to clear the \p matrices array if \p copy_array
	// * is set to false. Also the \p xindices and \p yindices are given in
	// * row-major-ordering for the corresponding entries of \p matrices.
	// */
	// virtual void init (const unsigned int m,
					   // const unsigned int n,
					   // const unsigned int m_l,
					   // const unsigned int n_l,
					   // const std::vector<unsigned int>& xindices,
					   // const std::vector<unsigned int>& yindices,
					   // const std::vector<CompositeType>& matrices,
					   // const bool copy_arrays=true,
			// const KARMA::LinearAlgebraLayout = KARMA::AUTOMATIC) = 0;
	
	/**
	* Creates a matrix that has the same dimension and storage type as
	* \p other. \p fast determines whether we need to copy all the
	* entries in the new matrix or just the layout.
	*/
	// virtual void init (const IMatrix<derived_matrix_type> & other,
	//					const bool = true) = 0 ;

	/**
	* return row_start, the index of the first
	* matrix row stored on this processor
	*/
	virtual unsigned int row_start () const = 0;
	
	/**
	* return row_stop, the index of the last
	* matrix row (+1) stored on this processor
	*/
	virtual unsigned int row_stop () const = 0;
	
	/**
	* return <\p m, \p n> as a pair indicating the
	* global row and column dimensions of the matrix
	*/
	virtual std::pair<unsigned int, unsigned int> size () const = 0;
	
	/**
	* return <\p m_l, \p n_l> as a pair indicating the
	* local row and column dimensions of the matrix
	*/
	virtual std::pair<unsigned int, unsigned int> local_size () const = 0;
	
	/**
	* Set the element \p (i,j) to \p value.
	* Throws an error if the entry does
	* not exist. Still, it is allowed to store
	* zero values in non-existent fields.
	*/
	virtual void insert (CompositeType const value,
				const unsigned int i,
				const unsigned int j) = 0 ;
	
	/**
	* Add \p value to the element
	* \p (i,j).  Throws an error if
	* the entry does not
	* exist. Still, it is allowed to
	* store zero values in
	* non-existent fields.
	*/
	virtual void add (CompositeType const value,
			const unsigned int i,
			const unsigned int j) = 0 ;
		
	/**
	* Add a Sparse matrix \p _X, scaled with \p _a, to \p this,
	* stores the result in \p this:
	* \f$\texttt{this} = \_a*\_X + \texttt{this} \f$.
	*/
	virtual void axpy (const ScalarType, IMatrix<derived_matrix_type>  &, 
					KARMA::MatrixStructure mstr=KARMA::Different_NNZ_Pattern ) = 0;
		
	/**
	* Add \p this Sparse matrix, scaled with \p _a, to \p _X,
	* and store the result in \p this:
	* \f$\texttt{this} = \_X + \_a*\texttt{this} \f$.
	*/
	virtual void aypx (const ScalarType, IMatrix<derived_matrix_type>  &, 
					KARMA::MatrixStructure mstr=KARMA::Different_NNZ_Pattern ) = 0;
	
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
				unsigned int j) const = 0;
	
	/**
	* Return the block value of the entry
	* \p (i,j).  This is a simple look-up
	* for block matrices but the same cost 
	* as operator() for non-blocked matrices.
	*
	* Also, in the blocked case, this reference
	* could be null and so user need to take 
	* care to check if the value is safe to use.
	*/
	virtual CompositeType const operator[] (const std::pair<unsigned int, unsigned int>& index) const = 0;
	
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
	virtual Real l1_norm () const = 0;
	
	/**
	* Return the l2-norm of the matrix or the frobenius
	* norm, that is
	* \f$|M|_2=\sqrt(\sum_{i=1}^{m} \sum_{j=1}^{n} \|a_{ij}\|^2.
	* This is the
	* natural matrix norm that is compatible
	* to the l1-norm for vectors, i.e.
	* \f$|Mv|_1\leq |M|_1 |v|_1\f$.
	*/
	virtual Real l2_norm () const = 0;
	
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
	virtual Real linfty_norm () const = 0;
	
	/**
	* Shift each element of the diagonal of
	* matrix by the given \p factor[0].
	* Optionally, can pass an array to scale each
	* element independently. This would then function
	* like a pointwise_multiply for a single matrix case
	* but provides regular scale interface for block matrix.
	*/
	virtual void shift (const ScalarType factor) = 0;
	
	/**
	* Scale each element of the
	* matrix by the given \p factor.
	* For the block matrix, this just represents scaling each
	* block row independently. This would then function
	* like a pointwise_multiply for a single matrix case
	* but provides regular scale interface for block matrix.
	*/
	virtual void scale (const std::vector<ScalarType>& factor) = 0;
	
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
	virtual void scale (const std::vector<ScalarType>* factor_left, const std::vector<ScalarType>* factor_right) = 0;

	
	virtual void scale (const VectorType* factor_left, const VectorType* factor_right) = 0;
	
	/**
	* see if Sparse matrix has a Symmetric structure
	* i.e., A = A^T
	*/
	virtual bool is_symmetric() const = 0;
	
	/**
	* Print the relevant information about the matrix-sparsity
	* such as nnz_allocated, nnz_used, nnz_unneeded
	* approximate memory allocated, etc..
	*/
	virtual void print_info() const = 0;
	
	/**
	* Print the contents of the matrix to the screen
	* in a package-personalized style, if available.
	*/
	virtual void print(std::ostream& os=std::cout) const = 0;
	
	/**
	* Print the contents of the matrix in Matlab's
	* sparse matrix format. Optionally prints the
	* matrix to the file named \p name.  If \p name
	* is not specified it is dumped to the screen.
	*/
	virtual void print_matlab(const std::string name="NULL") const = 0;
	
	/**
	* This function creates a matrix called "submatrix" which is defined
	* by the row and column indices given in the "rows" and "cols" entries.
	* Currently this operation is only defined for the PetscMatrix type.
	*/
	virtual void create_submatrix(IMatrix<derived_matrix_type> & submatrix,
				const std::vector<unsigned int>& rows,
				const std::vector<unsigned int>& cols) const
	{
		this->_get_submatrix(submatrix,
				rows,
				cols,
				false); // false means DO NOT REUSE submatrix
	}

	/**
	* This function is similar to the one above, but it allows you to reuse
	* the existing sparsity pattern of "submatrix" instead of reallocating
	* it again.  This should hopefully be more efficient if you are frequently
	* extracting submatrices of the same size.
	*/
	virtual void reinit_submatrix(IMatrix<derived_matrix_type> & submatrix,
				const std::vector<unsigned int>& rows,
				const std::vector<unsigned int>& cols) const
	{
		this->_get_submatrix(submatrix,
				rows,
				cols,
				true); // true means REUSE submatrix
	}
	
	/**
	* Multiplies the matrix with \p arg and stores the result in \p
	* dest.
	*/
	virtual void multiply (const VectorType& v_in,
							VectorType& v_out) const = 0;
	
	/**
	* Multiplies the matrix with \p arg and stores the result in \p
	* dest.
	*/
	virtual void transpose_multiply (const VectorType& v_in,
			   VectorType& v_out) const = 0;
	
	/**
	* Multiplies the matrix with \p arg and adds the result to \p dest.
	*/
	virtual void multiply_add (const VectorType& v_in,
			   VectorType& v_out) const = 0;
	
	/**
	* Multiplies the matrix with \p arg and adds the result to \p dest.
	*/
	virtual void transpose_multiply_add (const VectorType& v_in,
			   VectorType& v_out) const = 0;
	
	/**
	* Copies the diagonal part of the matrix into \p diag.
	*/
	virtual void get_diagonal (VectorType& diag) const = 0;
	
	/**
	* Copies the transpose of the matrix into \p dest, which may be
	* *this.
	*/
	virtual void get_transpose (IMatrix<derived_matrix_type> & dest) const = 0;
	
	/**
	* Transposes the current matrix and stores in the same allocated space.
	* NOTE: For unsymmetric matrices, this might need re-allocation if
	* the nnz/row increases.
	*/
	virtual void transpose () = 0;

	/**
	* Swaps the raw matrix context pointers.
	*/
	virtual void swap (IMatrix<derived_matrix_type>  &) ;
	
protected:
	
	/**
	* Protected implementation of the create_submatrix and reinit_submatrix
	* routines.  Note that this function must be redefined in derived classes
	* for it to work properly!
	*/
	virtual void _get_submatrix(IMatrix<derived_matrix_type> & ,
					const std::vector<unsigned int>& ,
					const std::vector<unsigned int>& ,
					const bool) const = 0 ;
	
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



//-----------------------------------------------------------------------
// IMatrix inline members
template<typename derived_matrix_type>
inline
IMatrix<derived_matrix_type> ::IMatrix (const KARMA::LinearAlgebraLayout type) :
  _is_closed(false),
  _is_initialized(false),
  _type(type)
{}



template<typename derived_matrix_type>
inline
IMatrix<derived_matrix_type> ::~IMatrix ()
{
	clear() ;
}


template<typename derived_matrix_type>
inline
void IMatrix<derived_matrix_type> ::clear ()
{
  _is_closed      = false;
  _is_initialized = false;
}



template<typename derived_matrix_type>
inline
void  IMatrix<derived_matrix_type>::swap (IMatrix<derived_matrix_type> &m)
{
  std::swap(_is_closed, m._is_closed);
  std::swap(_is_initialized, m._is_initialized);
  std::swap(_type, m._type);
}



#endif // #ifndef __imatrix_h__
