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


#include "karma_config.h"


#ifdef KARMA_HAVE_PETSC

// Local includes
#include "dense_matrix.h"
#include "petsc_native_vector.h"
#include "petsc_native_matrix.h"


//-----------------------------------------------------------------------
// PetscNativeMatrix members
template <typename T>
void PetscNativeMatrix<T>::init (const unsigned int m,
			   const unsigned int n,
			   const unsigned int m_l,
			   const unsigned int n_l,
			   const unsigned int nnz,
			   const unsigned int noz)
{
  // We allow 0x0 matrices now
  //if ((m==0) || (n==0))
  //  return;

  // Clear initialized matrices
  if (this->initialized())
    this->clear();

  this->_is_initialized = true;

  
  PetscErrorCode ierr     = 0;
  int m_global = static_cast<int>(m);
  int n_global = static_cast<int>(n);
  int m_local  = static_cast<int>(m_l);
  int n_local  = static_cast<int>(n_l);
  int n_nz     = static_cast<int>(nnz);
  int n_oz     = static_cast<int>(noz);
  
  // create a sequential matrix on one processor
  if (KARMA::num_processors == 1)
    {
      karma_assert ((m_l == m) && (n_l == n));

      // Create matrix.  Revisit later to do preallocation and make more efficient
      ierr = MatCreateSeqAIJ (KARMA_COMM_WORLD, m_global, n_global,
			      n_nz, PETSC_NULL, &_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
      ierr = MatSetFromOptions (_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  else
    {
      parallel_only();

      ierr = MatCreateMPIAIJ (KARMA_COMM_WORLD, m_local, n_local, m_global, n_global,
			      n_nz, PETSC_NULL, n_oz, PETSC_NULL, &_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
      ierr = MatSetFromOptions (_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

  this->zero ();
}




template <typename T>
void PetscNativeMatrix<T>::init (const unsigned int m,
					   const unsigned int n,
					   const unsigned int m_l,
					   const unsigned int n_l,
					   const std::vector<unsigned int>& n_nz,
					   const std::vector<unsigned int>& n_oz)
{ 
  // We allow 0x0 matrices now
  //if (m==0)
  //  return;
  
  // if we have an empty nnz array, call the default initializer
  if (n_nz.empty())
    this->init(m, n, m_l, n_l) ;
  
  // Clear initialized matrices
  if (this->initialized())
    this->clear();

  this->_is_initialized = true;

  PetscErrorCode ierr     = 0;
  int m_global = static_cast<int>(m);
  int n_global = static_cast<int>(n);
  int m_local  = static_cast<int>(m_l);
  int n_local  = static_cast<int>(n_l);

  // Make sure the sparsity pattern isn't empty unless the matrix is 0x0
  karma_assert (n_nz.size() == n_l);
  karma_assert (n_oz.size() == n_l);
  
  // create a sequential matrix on one processor
  if (KARMA::num_processors == 1)
    {
      karma_assert ((m_l == m) && (n_l == n));
	  
      ierr = MatCreateSeqAIJ (KARMA_COMM_WORLD, m_global, n_global,
			        PETSC_NULL, (int*) &n_nz[0], &_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
      ierr = MatSetFromOptions (_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  else
    {
      parallel_only();

      ierr = MatCreateMPIAIJ (KARMA_COMM_WORLD,
			        m_local, n_local,
			        m_global, n_global,
			        PETSC_NULL, (int*) &n_nz[0],
			        PETSC_NULL, (int*) &n_oz[0], &_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
      ierr = MatSetFromOptions (_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

  this->zero();
}


//template <typename T>
//void PetscNativeMatrix<T>::init (const unsigned int m,
//					   const unsigned int n,
//					   const unsigned int m_l,
//					   const unsigned int n_l,
//					   const KARMALinearOperator* oper)
//{
//	karma_not_implemented() ;
//}


template <typename T>
void PetscNativeMatrix<T>::init (const IMatrix<PetscNativeMatrix<T> > & other,
                           const bool fast)
{
  // Clear initialized matrix 
  if (this->initialized())
    this->clear();

  const PetscNativeMatrix<T>& m = dynamic_cast<const PetscNativeMatrix<T>&>(other);

  this->_is_closed      = m._is_closed;
  this->_is_initialized = m._is_initialized;
  this->_type = m._type;

  if (m.m() != 0 && m.n() != 0)
    {
      PetscErrorCode ierr = 0;

	  if (fast)
		ierr = MatDuplicate (m._mat, MAT_DO_NOT_COPY_VALUES, &this->_mat);
	  else
		ierr = MatDuplicate (m._mat, MAT_COPY_VALUES, &this->_mat);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  
}


template <typename T>
void PetscNativeMatrix<T>::clear ()
{
  PetscErrorCode ierr=0;
  
  if ((this->initialized()) && (this->_destroy_mat_on_exit))
    {
      semiparallel_only();
  
      ierr = MatDestroy (&_mat);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
      
      this->_is_initialized = false;
    }
}



template <typename T>
void PetscNativeMatrix<T>::zero ()
{
  karma_assert (this->initialized());

  semiparallel_only();
  
  PetscErrorCode ierr=0;

  ierr = MatZeroEntries(_mat);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);
}

template <typename T>
void PetscNativeMatrix<T>::zero_rows (std::vector<int> & rows, T diag_value)
{
  karma_assert (this->initialized());

  semiparallel_only();

  PetscErrorCode ierr=0;

  if(!rows.empty())
    ierr = MatZeroRows(_mat, rows.size(), &rows[0], diag_value, PETSC_NULL, PETSC_NULL);
  else
    ierr = MatZeroRows(_mat, 0, PETSC_NULL, diag_value, PETSC_NULL, PETSC_NULL);

  CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


template <typename T>
Real PetscNativeMatrix<T>::l1_norm () const
{
  karma_assert (this->initialized());
  
  semiparallel_only();

  PetscErrorCode ierr=0;
  PetscReal petsc_value;
  
  karma_assert (this->closed());

  ierr = MatNorm(_mat, NORM_1, &petsc_value);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  return static_cast<Real>(petsc_value);
}



template <typename T>
Real PetscNativeMatrix<T>::l2_norm () const
{
  karma_assert (this->initialized());
  
  semiparallel_only();

  PetscErrorCode ierr=0;
  PetscReal petsc_value;
  
  karma_assert (this->closed());

  ierr = MatNorm(_mat, NORM_FROBENIUS, &petsc_value);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  return static_cast<Real>(petsc_value);
}



template <typename T>
Real PetscNativeMatrix<T>::linfty_norm () const
{
  karma_assert (this->initialized());
  
  semiparallel_only();

  PetscErrorCode ierr=0;
  PetscReal petsc_value;
  
  karma_assert (this->closed());

  ierr = MatNorm(_mat, NORM_INFINITY, &petsc_value);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  return static_cast<Real>(petsc_value);
}



template <typename T>
void PetscNativeMatrix<T>::print_matlab (const std::string name) const
{
  karma_assert (this->initialized());

  semiparallel_only();

  karma_assert (this->closed());
  // this->close();
  
  PetscErrorCode ierr=0;
  PetscViewer petsc_viewer;


  ierr = PetscViewerCreate (KARMA_COMM_WORLD,
			    &petsc_viewer);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  /**
   * Create an ASCII file containing the matrix
   * if a filename was provided.  
   */
  if (name != "NULL")
    {
      ierr = PetscViewerASCIIOpen( KARMA_COMM_WORLD,
				   name.c_str(),
				   &petsc_viewer);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
      
      ierr = PetscViewerSetFormat (petsc_viewer,
				   PETSC_VIEWER_ASCII_MATLAB);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
      ierr = MatView (_mat, petsc_viewer);
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
  
      ierr = MatView (_mat, PETSC_VIEWER_STDOUT_WORLD);
             CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }


  /**
   * Destroy the viewer.
   */
  ierr = PetscViewerDestroy (&petsc_viewer);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);
}




template <typename T>
void PetscNativeMatrix<T>::add(const DenseMatrix<T>& dm,
				const std::vector<unsigned int>& rows,
				const std::vector<unsigned int>& cols)
{
  karma_assert (this->initialized());
  
  const unsigned int m = dm.m();
  const unsigned int n = dm.n();

  karma_assert (rows.size() == m);
  karma_assert (cols.size() == n);
  
  PetscErrorCode ierr=0;

  // These casts are required for PETSc <= 2.1.5
  ierr = MatSetValues(_mat,
		      m, (int*) &rows[0],
		      n, (int*) &cols[0],
		      (PetscScalar*) &dm.get_values()[0],
		      ADD_VALUES);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);
}





template <typename T>
void PetscNativeMatrix<T>::_get_submatrix(IMatrix<PetscNativeMatrix<T> > & submatrix,
				    const std::vector<unsigned int> &rows,
				    const std::vector<unsigned int> &cols,
				    const bool reuse_submatrix) const
{
  // Can only extract submatrices from closed matrices
	karma_assert(this->closed()) ;
  // this->close();

  // Make sure the SparseMatrix passed in is really a PetscNativeMatrix
  PetscNativeMatrix<T>* petsc_submatrix = dynamic_cast<PetscNativeMatrix<T>*>(&submatrix);

  // If we're not reusing submatrix and submatrix is already initialized
  // then we need to clear it, otherwise we get a memory leak.
  if( !reuse_submatrix && submatrix.initialized() )
    submatrix.clear();
  
  // Construct row and column index sets.
  PetscErrorCode ierr=0;
  IS isrow, iscol;

  ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 rows.size(),
			 (int*) &rows[0],
			 PETSC_USE_POINTER,
			 &isrow); CHKERRABORT(KARMA_COMM_WORLD,ierr);

  ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 cols.size(),
		     (int*) &cols[0],
			 PETSC_USE_POINTER,
			 &iscol); CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Extract submatrix
  ierr = MatGetSubMatrix(_mat,
			 isrow,
			 iscol,
			 (reuse_submatrix ? MAT_REUSE_MATRIX : MAT_INITIAL_MATRIX),
			 &(petsc_submatrix->_mat));  CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Specify that the new submatrix is initialized and close it.
  petsc_submatrix->_is_initialized = true;
  petsc_submatrix->close();

  // Clean up PETSc data structures
  ierr = ISDestroy(&isrow); CHKERRABORT(KARMA_COMM_WORLD,ierr);
  ierr = ISDestroy(&iscol); CHKERRABORT(KARMA_COMM_WORLD,ierr);
}



template <typename T>
void PetscNativeMatrix<T>::get_diagonal (PetscNativeMatrix<T>::VectorType& dest) const
{
  // Make sure the IVector passed in is really a PetscNativeVector
  PetscNativeMatrix<T>::VectorType& petsc_dest = dynamic_cast<PetscNativeMatrix<T>::VectorType&>(dest);

  // Call PETSc function.
  // Needs a const_cast since PETSc does not work with const.
  PetscErrorCode ierr =
    MatGetDiagonal(const_cast<PetscNativeMatrix<T>*>(this)->mat(),petsc_dest.vec()); CHKERRABORT(KARMA_COMM_WORLD,ierr);

}



template <typename T>
void PetscNativeMatrix<T>::sum_row (PetscNativeMatrix<T>::VectorType& rowsums) const
{
  // Call PETSc function.
  PetscErrorCode ierr =
    MatGetRowSum(this->mat(),rowsums.vec()); CHKERRABORT(KARMA_COMM_WORLD,ierr);

}



template <typename T>
void PetscNativeMatrix<T>::max_row (PetscNativeMatrix<T>::VectorType& max_rows) const
{
  // Call PETSc function.
  PetscErrorCode ierr = MatGetRowMax(this->mat(), max_rows.vec(), PETSC_NULL); CHKERRABORT(KARMA_COMM_WORLD,ierr);

}



template <typename T>
void PetscNativeMatrix<T>::min_row (PetscNativeMatrix<T>::VectorType& min_rows) const
{
  // Call PETSc function.
  PetscErrorCode ierr = MatGetRowMin(this->mat(), min_rows.vec(), PETSC_NULL); CHKERRABORT(KARMA_COMM_WORLD,ierr);

}



template <typename T>
void PetscNativeMatrix<T>::shift (const T factor)
{
	const PetscScalar& shift = static_cast<const PetscScalar&>(factor) ;
	
  // Call PETSc function.
  PetscErrorCode ierr = MatShift(_mat, shift); CHKERRABORT(KARMA_COMM_WORLD,ierr);

}



template <typename T>
void PetscNativeMatrix<T>::scale (const std::vector<T>& factor)
{
	/**
	* if only one value is provided, we scale all elements of the matrix;
	* alternately if size() == number of rows, scale each row with the factor;
	*/
	karma_assert(factor.size() == 1 || factor.size() == this->m()) ;

	PetscScalar scale = static_cast<PetscScalar>(factor[0]) ;

	// Call PETSc function.
	// Needs a const_cast since PETSc does not work with const.
	if (factor.size() == 1)
	{
		PetscErrorCode ierr =
		MatScale(_mat, scale); CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	else
	{
		karma_not_implemented() ;

		// This is slightly tricky. we need to re-use the array provided by
		// the user but need to get the layout from the Matrix.
		// How to sync both these operations optimally ??
		// PetscErrorCode ierr ;
		// Vec v;
		// ierr = MatGetVecs(_mat, &v, PETSC_NULL) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		// MatDiagonalScale(_mat, scale); CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}



template <typename T>
void PetscNativeMatrix<T>::scale (const std::vector<T>* factor_left, const std::vector<T>* factor_right)
{
	/**
	* if only one value is provided, we scale all elements of the matrix;
	* alternately if size() == number of rows, scale each row with the factor;
	*/
	karma_assert(factor_left != NULL && factor_left->size() == this->n()) ;
	karma_assert(factor_right != NULL && factor_right->size() == this->m()) ;

	karma_experimental() ;

	// This is slightly tricky. we need to re-use the array provided by
	// the user but need to get the layout from the Matrix.
	// How to sync both these operations optimally ??
	PetscErrorCode ierr ;
	// we usually do not use BAIJ formats in KARMA but if from external source,
	// just follow the semantics correctly.
	PetscInt bs;
	ierr = MatGetBlockSize(_mat, &bs); CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	Vec vleft=PETSC_NULL, vright=PETSC_NULL;
	
	if (factor_right != NULL)
	{
		ierr = VecSetSizes(vright, this->n(), PETSC_DETERMINE); CHKERRABORT(KARMA_COMM_WORLD,ierr);
		ierr = VecSetBlockSize(vright, bs); CHKERRABORT(KARMA_COMM_WORLD,ierr);
		if (KARMA::num_processors > 1)
			VecSetType(vright,VECMPI);
		else
			VecSetType(vright,VECSEQ);
		const PetscScalar* rightarray = &factor_right->front() ;
		ierr = VecReplaceArray(vright, rightarray) ; CHKERRABORT(KARMA_COMM_WORLD,ierr);
		// if (KARMA::num_processors > 1)
			// ierr = VecCreateMPIWithArray(v, PETSC_DETERMINE, rightsize, factor_right[0]);
		// else
			// ierr = VecCreateMPIWithArray(v, PETSC_DETERMINE, rightsize, factor_right[0]);
		// CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	
	if (factor_left != NULL)
	{
		ierr = VecSetSizes(vleft, this->m(), PETSC_DETERMINE); CHKERRABORT(KARMA_COMM_WORLD,ierr);
		ierr = VecSetBlockSize(vleft, bs); CHKERRABORT(KARMA_COMM_WORLD,ierr);
		if (KARMA::num_processors > 1)
			VecSetType(vleft,VECMPI);
		else
			VecSetType(vleft,VECSEQ);
		const PetscScalar* leftarray = &factor_left->front() ;
		ierr = VecReplaceArray(vleft, leftarray) ; CHKERRABORT(KARMA_COMM_WORLD,ierr);
		// if (KARMA::num_processors > 1)
			// ierr = VecCreateMPIWithArray(v, PETSC_DETERMINE, rightsize, factor_left[0]);
		// else
			// ierr = VecCreateMPIWithArray(v, PETSC_DETERMINE, rightsize, factor_left[0]);
		// CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	
	/**
	* Call PETSc function.
	* MatDiagonalScale() computes A = LAR, where 
	* L = a diagonal matrix (stored as a vector), 
	* R = a diagonal matrix (stored as a vector) 
	*/
	ierr = MatDiagonalScale(_mat, vleft, vright); CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	if (factor_right != NULL)
	{
		ierr = VecDestroy(&vright); CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	if (factor_left != NULL)
	{
		ierr = VecDestroy(&vleft); CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}



template <typename T>
void PetscNativeMatrix<T>::scale (const PetscNativeMatrix<T>::VectorType* factor_left, const PetscNativeMatrix<T>::VectorType* factor_right)
{
	Vec vleft=PETSC_NULL, vright=PETSC_NULL;
	if (factor_left != NULL)
	{
		const PetscNativeMatrix<T>::VectorType& vector = dynamic_cast<const PetscNativeMatrix<T>::VectorType&>(*factor_left) ;
		vleft = vector.vec() ;
	}
	if (factor_right != NULL)
	{
		const PetscNativeMatrix<T>::VectorType& vector = dynamic_cast<const PetscNativeMatrix<T>::VectorType&>(*factor_right) ;
		vright = vector.vec() ;
	}
	
  // Call PETSc function.
  // Needs a const_cast since PETSc does not work with const.
  PetscErrorCode ierr = MatDiagonalScale(_mat, vleft, vright); 
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

}


template <typename T>
void PetscNativeMatrix<T>::get_transpose (IMatrix<PetscNativeMatrix<T> > & dest) const
{
	// Make sure the IMatrix passed in is really a PetscNativeMatrix
	PetscNativeMatrix<T>& petsc_dest = dynamic_cast<PetscNativeMatrix<T>&>(dest);

	// If we aren't reusing the matrix then need to clear dest,
	// otherwise we get a memory leak
	if(&petsc_dest != this)
		dest.clear();

	PetscErrorCode ierr;
	// FIXME - we can probably use MAT_REUSE_MATRIX in more situations
	if (&petsc_dest == this)
		ierr = MatTranspose(_mat,MAT_REUSE_MATRIX,&petsc_dest._mat);
	else
		ierr = MatTranspose(_mat,MAT_INITIAL_MATRIX,&petsc_dest._mat);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Specify that the transposed matrix is initialized and close it.
	petsc_dest._is_initialized = true;
	petsc_dest.close();
}


template <typename T>
void PetscNativeMatrix<T>::transpose ()
{
	karma_assert(this->initialized()) ;

	PetscErrorCode ierr;
	ierr = MatTranspose(_mat,MAT_REUSE_MATRIX,&_mat);
	CHKERRABORT(KARMA_COMM_WORLD,ierr);

	// Let us now lose the matrix.
	this->close();
}


//------------------------------------------------------------------
// Explicit instantiations
template class PetscNativeMatrix<PetscScalar>;


#endif // #ifdef KARMA_HAVE_PETSC
