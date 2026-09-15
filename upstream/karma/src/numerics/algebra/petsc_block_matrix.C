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

#ifdef USE_PETSC_BLOCK_EXTENSIONS

#include "petsc_block_matrix.h"
#include "petsc_block_vector.h"

#include "karma_linear_operator.h"
#include "karma_parameters.h"

template <typename T>
PetscBlockMatrix<T>::~PetscBlockMatrix ()
{
	if (_external)
	{
		_mat = NULL ;	// user given matrix. allow user to destroy
		_size = 0 ;
		_is_initialized = false ;
		_block_matrix_refs.clear() ;
	}
}

// provide block matrix in compressed CSR format
template <typename T>
void PetscBlockMatrix<T>::init (const unsigned int m,
					   const unsigned int n,
					   const unsigned int m_l,
					   const unsigned int n_l,
					   const std::vector<unsigned int>& xindices,
					   const std::vector<unsigned int>& yindices,
					   const std::vector<PetscBlockMatrix<T>::CompositeType>& matrices,
					   const bool copy_arrays,
					   const KARMA::LinearAlgebraLayout)
{
	if (!_is_initialized)
	{
		_is_initialized = true ;
		
		PetscErrorCode ierr = 0 ;
		
		ierr = MatCreate( KARMA_COMM_WORLD, &_mat );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		ierr = MatSetSizes( _mat, _size, _size, _size, _size );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		ierr = MatSetType( _mat, "block" );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		_block_matrix_ops.resize(_size*_size) ;
		_block_matrix_refs.resize(_size*_size) ;
		
		unsigned int xi = 0, yi = 0 ;
		for (unsigned int j=0; j < yindices.size(); j++)
		{
			// get the index of block row
			unsigned int xi = std::find(xindices.begin(), xindices.end(), j) - xindices.begin();
			this->insert(matrices[j], xindices[xi], yindices[j]) ;
		}
		this->close() ;
	}
}

template <typename T>
void PetscBlockMatrix<T>::clear ()
{
	if (_is_initialized && !_external)
	{
		IMatrix<MatrixTraits<T, KARMA::BLOCK> >::clear() ;
		_is_initialized = false ;

		PetscErrorCode ierr = 0 ;
		ierr = MatDestroy(_mat) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		
		_block_matrix_ops.clear() ;
		_block_matrix_refs.clear() ;
		_size = 0 ;
	}
}

template <typename T>
void PetscBlockMatrix<T>::close()
{
	PetscErrorCode ierr = 0 ;
	
	ierr = MatAssemblyBegin(_mat, MAT_FINAL_ASSEMBLY);CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = MatAssemblyEnd(_mat, MAT_FINAL_ASSEMBLY);CHKERRABORT(KARMA_COMM_WORLD,ierr);
}

template <typename T>
void PetscBlockMatrix<T>::zero()
{
	PetscErrorCode ierr = 0 ;
	
	for (unsigned int ix=0; ix < _size; ix++)
		for (unsigned int iy=0; iy < _size; iy++)
		{
			if (_block_matrix_refs[ix*_size+iy] != NULL)
				_block_matrix_refs[ix*_size+iy]->zero() ;
		}
}


template <typename T>
void PetscBlockMatrix<T>::insert(const KARMALinearOperator* b_op, const unsigned int m, const unsigned int n)
{
	karma_assert(m < _size) ;
	karma_assert(n < _size) ;
	
	_block_matrix_ops[m*_size+n] = b_op ;
}

template <typename T>
void PetscBlockMatrix<T>::insert(PetscBlockMatrix<T>::CompositeType const bmat, const unsigned int m, const unsigned int n)
{
	karma_assert(m < _size) ;
	karma_assert(n < _size) ;
	
	_block_matrix_refs[m*_size+n] = bmat ;
	
	PetscErrorCode ierr = 0 ;
	ierr = MatBlockSetValue( _mat, m, n, bmat->mat(), DIFFERENT_NONZERO_PATTERN, INSERT_VALUES );CHKERRABORT(KARMA_COMM_WORLD,ierr);
}

template <typename T>
void PetscBlockMatrix<T>::add(PetscBlockMatrix<T>::CompositeType const bmat, const unsigned int m, const unsigned int n)
{
	karma_assert(m < _size) ;
	karma_assert(n < _size) ;
	
	_block_matrix_refs[m*_size+n] = bmat ;
	
	PetscErrorCode ierr = 0 ;
	ierr = MatBlockSetValue( _mat, m, n, bmat->mat(), DIFFERENT_NONZERO_PATTERN, ADD_VALUES );CHKERRABORT(KARMA_COMM_WORLD,ierr);
}

// template <typename T>
// PetscNativeMatrix<T>* PetscBlockMatrix<T>::get (unsigned int m, unsigned int n)
// {
// 	karma_assert(m < _size) ;
// 	karma_assert(n < _size) ;
// 	karma_assert(_block_matrix_refs[m*_size+n] != NULL) ;
// 	
// 	// PetscErrorCode MatBlockGetSubMatrix( Mat A, PetscInt ridx, PetscInt cidx, Mat *sa ) ;
// 	return (_block_matrix_refs[m*_size+n]) ;
// }

template <typename T>
const PetscBlockMatrix<T>::ScalarType PetscBlockMatrix<T>::operator() (unsigned int m, unsigned int n) const
{
	karma_assert(m < _size) ;
	karma_assert(n < _size) ;
	karma_assert(_block_matrix_refs[0] != NULL) ;
	
	return (*_block_matrix_refs[0])(m,n) ;
}

template <typename T>
const PetscBlockMatrix<T>::CompositeType PetscBlockMatrix<T>::operator[] (const std::pair<unsigned int, unsigned int>& index) const
{
	karma_assert(index.first < _size) ;
	karma_assert(index.second < _size) ;
	karma_assert(_block_matrix_refs[index.first*_size+index.second] != NULL) ;
	
	return (_block_matrix_refs[index.first*_size+index.second]) ;
}


template <typename T>
void PetscBlockMatrix<T>::_get_submatrix(IMatrix<MatrixTraits<T, KARMA::BLOCK> > & ,
					const std::vector<unsigned int>& ,
					const std::vector<unsigned int>& ,
					const bool) const
{
	karma_not_implemented() ;
}


// PetscErrorCode MatBlockGetSubMatrices( Mat A, Mat ***sa )
// PetscErrorCode MatBlockRestoreSubMatrices( Mat A )
template <typename T>
void PetscBlockMatrix<T>::multiply(PetscBlockMatrix<T>::VectorType& x, PetscBlockMatrix<T>::VectorType& y, std::vector<KARMAParameters*>& params)
{
	// check if the sizes are conforming
	karma_assert(x.local_size() == y.local_size()) ;
	karma_assert(x.size() == y.size()) ;

	// zero out the output vector first
	y.zero() ;
	
	multiply_add(x, y, params) ;
}

template <typename T>
void PetscBlockMatrix<T>::multiply_add(PetscBlockMatrix<T>::VectorType& x, PetscBlockMatrix<T>::VectorType& y, std::vector<KARMAParameters*>& params)
{
	// check if the sizes are conforming
	karma_assert(x.local_size() == y.local_size()) ;
	karma_assert(x.size() == y.size()) ;
	
	// Not sure if we need the check below because it is only needed when
	// the linear operator is used to obtain the operator application.
	// karma_assert(params.size() < _size) ;
	
	PetscErrorCode ierr = 0 ;
	
	for (int ix=0; ix < _size; ix++)
	{
		PetscNativeVector<T>* yblock = y[ix] ;
		
		karma_assert(yblock != NULL) ;
		
		for (int iy=0; iy < _size; iy++)
		{
			PetscNativeVector<T>* xblock = x[iy] ;
			
			karma_assert(xblock != NULL) ;
			// karma_assert(xblock->local_size() == yblock->local_size()) ;
			// karma_assert(xblock->size() == yblock->size()) ;
			
			PetscNativeMatrix<T>* ablock = _block_matrix_refs[ix*_size+iy] ;
			
			if (ablock != NULL)
			{
				ierr = MatMultAdd(ablock->mat(), xblock->vec(), yblock->vec(), yblock->vec()) ;CHKERRABORT(KARMA_COMM_WORLD,ierr) ;
			}
			else
			{
				KARMALinearOperator* ablockopt = _block_matrix_ops[ix*_size+iy] ;
				
				// NOTE: Technically, even if is_matrix() is available, we can proceed further
				// but it is non-optimal to create/allocate the matrix for this.
				// So ensure that the matvec operation is defined and available for the given operator.
				if (!ablockopt->is_matvec())
				{
					KARMA_LOG(ERROR) << "No matrix-vector operator found for BlockMatrix with index (" << ix << ", " << iy << ")." ;
					karma_error() ;
				}
				
				KARMAParameters* phy_params = params[ix] ;
				
				// if we call the operator, need a non-null parameter specification
				karma_assert(phy_params != NULL) ;
				
				// TODO: Maybe create a temporary vector ? Or else the matvec operation
				// will over-write the existing yblock..
				karma_error() ;
				// ierr = ablockopt->matvec(*xblock, *yblock, *phy_params) ;CHKERRABORT(KARMA_COMM_WORLD,ierr) ;
			}
		}
	}
	
	y.close() ;
}

//------------------------------------------------------------------
// Explicit instantiations
template class PetscBlockMatrix<PetscScalar>;

#endif
