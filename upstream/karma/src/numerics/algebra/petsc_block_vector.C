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


#include "karma_configuration.h"

#if KARMA_HAVE_PETSC && PETSC_VERSION_GREATER_THAN(3,1,8)

#include "karma.h"
//#include "karma_parameters.h"
#include "petsc_native_vector.h"

// C++ includes
#include <numeric>
#include "petsc_block_vector.h"
#include "utility.h"

template <typename T>
AutoPtr<PetscBlockVector<T> >
PetscBlockVector<T>::build()
{
	// Build the PETSc NEST vector
	AutoPtr<PetscBlockVector<T> > ap(new PetscBlockVector<T>);
	return ap;
}

template <typename T>
int PetscBlockVector<T>::compare (const Parent &other_vector,
			       const Real threshold) const
{
	// TODO: Loop over each vector and call compare recursively.
	karma_not_implemented() ;
}

template <typename T>
PetscBlockVector<T>::PetscBlockVector (Vec uservec)
{
	// first, let us check the type of the given vector
	// IF type = block, set internal reference directly.
	// ELSE, create a block vector with size=1 and make the v_0=uservec
	const VecType type ;
	VecGetType(uservec, &type) ;
	_proxy = true ;
	
	const std::string block_vector_name (VECNEST) ;
	
	if (!block_vector_name.compare(type))
	{
		_destroy_vec_on_exit = false ;

		// set the true vector reference
		_vec = uservec ;
		
		PetscErrorCode ierr ;
		Vec* bvecs ;

		// get the references to the block vectors
		ierr = VecNestGetSubVecs(_vec, &_size, &bvecs) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		
		// resize our list based on the new count
		_block_vector_obj_refs.resize(_size) ;
		
		// loop over the vector and create temporary PetscNativeVector wrappers
		for(unsigned int index=0; index < _size; index++)
		{
			if (bvecs[index])
			{
				// check if this is a valid Vec object
				// PetscBool flg ;
				// ierr = VecValid(bvecs[index], &flg) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
				// if (flg)
					_block_vector_obj_refs[index] = new PetscNativeVector<T>(bvecs[index]) ;
			}
			else
				_block_vector_obj_refs[index] = NULL ;
		}
		//ierr = VecBlockRestoreSubVectors(_vec) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		// bvecs = NULL ;
		// delete [] bvecs ;
	}
	else
	{
		_size = 1 ;
		_destroy_vec_on_exit = true ;

		// create a new block vector
		PetscErrorCode ierr = 0 ;
		ierr = VecCreate( KARMA_COMM_WORLD, &_vec );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		ierr = VecSetSizes( _vec, _size, _size );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		ierr = VecSetType( _vec, VECNEST );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		
		_block_vector_obj_refs.resize(1) ;
		
		// set the 1st block of the vector to the user-provided reference
		ierr = VecBlockSetValue( _vec, 0, uservec, INSERT_VALUES) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		// also create a wrapper over the vector
		_block_vector_obj_refs[0] = new PetscNativeVector<T>(uservec) ;
	}
		
	this->close() ;
}


template <typename T>
PetscBlockVector<T>::~PetscBlockVector ()
{
	clear() ;
}

template <typename T>
void PetscBlockVector<T>::init (const unsigned int N,
								const bool )
{
	if (!Parent::_is_initialized)
	{
		_size = N ;
		Parent::_type = KARMA::BLOCK ;
		
		PetscErrorCode ierr = 0 ;
		
		ierr = VecCreate( KARMA_COMM_WORLD, &_vec );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		ierr = VecSetSizes( _vec, _size, _size );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		ierr = VecSetType( _vec, VECNEST );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		_block_vector_obj_refs.resize(_size) ;
		
		/**
		 * Should we set it to true ? 
		 * maybe check whether all blocks are set
		 * before doing this.
		 */
		Parent::init() ;
	}
}


template <typename T>
void PetscBlockVector<T>::init (const Parent& other,
								const bool)
{
	if (!Parent::_is_initialized)
	{
		const PetscBlockVector<T>& othervec = dynamic_cast<const PetscBlockVector<T>&>(other) ;
		_size = othervec.size() ;
		Parent::_type = KARMA::BLOCK ;
		
		PetscErrorCode ierr = 0 ;
		
		ierr = VecDuplicate( othervec.vec(), &_vec );CHKERRABORT(KARMA_COMM_WORLD,ierr);
		_block_vector_obj_refs.resize(_size) ;
		for (unsigned int i=0; i < _size; i++)
		{
			_block_vector_obj_refs[i] = new PetscNativeVector<T>(*othervec[i]) ;
		}
		
		/**
		 * Should we set it to true ? 
		 * maybe check whether all blocks are set
		 * before doing this.
		 */
		Parent::init() ;
	}
}


template <typename T>
void PetscBlockVector<T>::pointwise_mult (const Parent& vec1,
										  const Parent& vec2)
{
	karma_assert(this->closed()) ;
	
	karma_not_implemented() ;
	
	PetscErrorCode ierr = 0;
	
	// Convert arguments to PetscNativeVector*.
	const PetscBlockVector<T>& vec1_petsc = dynamic_cast<const PetscBlockVector<T>&>(vec1);
	const PetscBlockVector<T>& vec2_petsc = dynamic_cast<const PetscBlockVector<T>&>(vec2);
	
	// Call PETSc function.
	for (unsigned int i=0; i < _size; i++)
	{
		ierr = VecPointwiseMult((*this)[i].vec(),
								  vec1_petsc[i].vec(),
								  vec2_petsc[i].vec());
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}



template <typename T>
void PetscBlockVector<T>::pointwise_divide (const Parent& vec1,
											const Parent& vec2)
{
	karma_assert(this->closed()) ;
	
	PetscErrorCode ierr = 0;
	
	// Convert arguments to PetscNativeVector*.
	const PetscBlockVector<T>& vec1_petsc = dynamic_cast<const PetscBlockVector<T>&>(vec1);
	const PetscBlockVector<T>& vec2_petsc = dynamic_cast<const PetscBlockVector<T>&>(vec2);
	
	// Call PETSc function.
	for (unsigned int i=0; i < _size; i++)
	{
		ierr = VecPointwiseDivide((*this)[i].vec(),
									vec1_petsc[i].vec(),
									vec2_petsc[i].vec());
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}

}


template <typename T>
typename PetscBlockVector<T>::ScalarType PetscBlockVector<T>::sum () const
{
	karma_assert(this->closed()) ;
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscScalar value=0.;
	
	ierr = VecSum (_vec, &value);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<T>(value);
}


template <typename T>
typename PetscBlockVector<T>::ScalarType PetscBlockVector<T>::min () const
{
	karma_assert(this->closed()) ;
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscScalar value=0.;
	
	ierr = VecMin (_vec, &value);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<T>(value);
}


template <typename T>
typename PetscBlockVector<T>::ScalarType PetscBlockVector<T>::max () const
{
	karma_assert(this->closed()) ;
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscScalar value=0.;
	
	ierr = VecMax (_vec, &value);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<T>(value);
}


template <typename T>
typename PetscBlockVector<T>::ScalarType PetscBlockVector<T>::l1_norm () const
{
	karma_assert(this->closed()) ;
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscReal value=0.;
	
	ierr = VecNorm (_vec, NORM_1, &value);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<Real>(value);
}


template <typename T>
typename PetscBlockVector<T>::ScalarType PetscBlockVector<T>::l2_norm () const
{
	karma_assert(this->closed()) ;
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscReal value=0.;
	
	ierr = VecNorm (_vec, NORM_2, &value);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<Real>(value);
}


template <typename T>
typename PetscBlockVector<T>::ScalarType PetscBlockVector<T>::linfty_norm () const
{
	karma_assert(this->closed()) ;
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscReal value=0.;
	
	ierr = VecNorm (_vec, NORM_INFINITY, &value);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<Real>(value);
}


template <typename T>
void PetscBlockVector<T>::clear ()
{
	if (Parent::_is_initialized)
	{
		// KARMA_LOG(INFO) << "Destroying block vector: " << _name ;
		
		if (_destroy_vec_on_exit)
		{
			PetscErrorCode ierr = VecDestroy(&_vec) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
		
		if (_proxy)
		{
			for (unsigned int i=0; i < _size; i++)
				delete _block_vector_obj_refs[i] ;
		}
		
		_vec = NULL ;	// user given vector. allow user to destroy
		_block_vector_obj_refs.clear() ;
		_size = 0 ;
		
		Parent::clear() ;
	}
}

template <typename T>
void PetscBlockVector<T>::close()
{
	// propagate the call to all internal vectors
	for (int i=0; i < _size; i++)
		_block_vector_obj_refs[i]->close() ;

	PetscErrorCode ierr = 0 ;
		
	ierr = VecAssemblyBegin(_vec);CHKERRABORT(KARMA_COMM_WORLD,ierr);
	ierr = VecAssemblyEnd(_vec);CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	this->_is_closed = true ;
}


template <typename T>
void PetscBlockVector<T>::operator() (const std::vector<unsigned int>& index, std::vector<CompositeType>& values) const
{
	values.clear() ;
	values.resize(_size) ;

	for (unsigned int i=0; i < index.size(); i++)
		values[i] = _block_vector_obj_refs [index[i]] ;
}


template <typename T>
void PetscBlockVector<T>::shift (const ScalarType factor)
{
	for (unsigned int i=0; i < _size; i++)
	{
		PetscErrorCode ierr = VecShift( _block_vector_obj_refs[i]->vec(), factor) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}

template <typename T>
void PetscBlockVector<T>::shift (const std::vector<ScalarType>& factor)
{
	karma_assert(factor.size() == _size) ;
	
	for (unsigned int i=0; i < _size; i++)
	{
		PetscErrorCode ierr = VecShift( _block_vector_obj_refs[i]->vec(), factor[i]) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}

template <typename T>
void PetscBlockVector<T>::scale (const std::vector<ScalarType>& factor)
{
	karma_assert(factor.size() == _size) ;
	
	for (unsigned int i=0; i < _size; i++)
	{
		PetscErrorCode ierr = VecScale( _block_vector_obj_refs[i]->vec(), factor[i]) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}

template <typename T>
void PetscBlockVector<T>::scale (const ScalarType factor)
{
	for (unsigned int i=0; i < _size; i++)
	{
		PetscErrorCode ierr = VecScale( _block_vector_obj_refs[i]->vec(), factor) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}


template <typename T>
void PetscBlockVector<T>::axpy(const ScalarType a, const Parent& x)
{
	const PetscBlockVector<T>& v = dynamic_cast<const PetscBlockVector<T>&>(x) ;
	PetscErrorCode ierr = VecAXPY(_vec, a, v.vec()) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


template <typename T>
void PetscBlockVector<T>::aypx(const ScalarType a, const Parent& x)
{
	const PetscBlockVector<T>& v = dynamic_cast<const PetscBlockVector<T>&>(x) ;
	PetscErrorCode ierr = VecAYPX( _vec, a, v.vec()) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


template <typename T>
void PetscBlockVector<T>::axpby(const ScalarType a, const Parent& x, const ScalarType b)
{
	const PetscBlockVector<T>& v = dynamic_cast<const PetscBlockVector<T>&>(x) ;
	PetscErrorCode ierr = VecAXPBY( _vec, a, b, v.vec()) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


template <typename T>
void PetscBlockVector<T>::axpbypcz(const ScalarType a, const Parent& x, const ScalarType b, const Parent& y, const ScalarType c)
{
	const PetscBlockVector<T>& v = dynamic_cast<const PetscBlockVector<T>&>(x) ;
	const PetscBlockVector<T>& w = dynamic_cast<const PetscBlockVector<T>&>(y) ;
	PetscErrorCode ierr = VecAXPBYPCZ(_vec, a, b, c, v._vec, w._vec);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
}


template <typename T>
void PetscBlockVector<T>::maxpy(const std::vector<ScalarType>& a, const std::vector<Parent* >& x)
{
	Vec mvecs[x.size()] ;
	PetscScalar alpha[x.size()] ;
	
	for (unsigned int i=0; i < x.size(); i++)
	{
		// get the PetscNativeVector and its native Vec reference
		mvecs[i] = x[i]->vec() ;
		alpha[i] = static_cast<PetscScalar>(a[i]) ;
	}

	PetscErrorCode ierr = VecMAXPY( _vec, x.size(), alpha, mvecs) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
}

template <typename T>
void PetscBlockVector<T>::abs()
{
	for (unsigned int i=0; i < _size; i++)
	{
		PetscErrorCode ierr = VecAbs( _block_vector_obj_refs[i]->vec()) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}

template <typename T>
void PetscBlockVector<T>::reciprocal()
{
	for (unsigned int i=0; i < _size; i++)
	{
		PetscErrorCode ierr = VecReciprocal( _block_vector_obj_refs[i]->vec()) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}


template <typename T>
typename PetscBlockVector<T>::ScalarType PetscBlockVector<T>::dot(const Parent& V_in) const
{
	karma_assert(_size == V_in.size()) ;
	
	const PetscBlockVector<T>& V = dynamic_cast<const PetscBlockVector<T> &>(V_in) ;
	
	Real value = 0.0 ;
	for (unsigned int i=0; i < _size; i++)
	{
		value += _block_vector_obj_refs[i]->dot(*V._block_vector_obj_refs[i]) ;
	}

	return value ;
}


template <typename T>
void PetscBlockVector<T>::merge(PetscNativeVector<T>& fullvec, const bool aggregate) const
{
	// make sure we have atleast one vector in our block representation
	karma_assert(_size > 0) ;
	
	/**
	 * NOTE: Make this configurable. Let the user decide if we should aggregate
	 * a block vector in to a single vector by placing the entires accordingly
	 * or whether just return the first block. For now, by default, only the
	 * first block will be returned.
	 */
	if (aggregate)
	{
		/**
		 * NOTE: Not sure how to distribute the new vector. The options could be
		 * 			1) Find total dofs in the current processor and use that
		 * 			2) Make it completely serial and then let user decide afterwards ?
		 * 			3) Actually the block_vector implementation in PetscExt has a utility
		 * 			   function to do exactly this. Call VecBlockMergeSubVectors function as
		 * 		*******************************************************************************
		 * 		VecBlockMergeSubVectors( Vec x, InsertMode addv, const VecType type, Vec *mX );
		 * 		*******************************************************************************
		 * 
		 * For now, going with Option (3). If any anomalies exist, switch this out with our
		 * custom implementation.
		 * 
		 */
		fullvec.clear() ;
		Vec rawvec = fullvec.vec() ;
		PetscErrorCode ierr = VecNestMergeSubVecs (_vec, INSERT_VALUES, VECMPI, &rawvec) ;CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
		fullvec.close() ;
	}
	else
	{
		// return the reference to the first block vector in the collection
		fullvec = *_block_vector_obj_refs[0] ;
	}
}


// void PetscBlockVector::multiply(PetscNativeVector<T>& x, PetscNativeVector<T>& y, KARMAParameters* params)
// {
	// // check if the sizes are conforming
	// karma_assert(x.local_size() == y.local_size()) ;
	// karma_assert(x.size() == y.size()) ;
	// // Should we check if x and y are block vectors too ?? If not this is almost impossible
	// // without additional data like DofMap or something. For now, just assume a stride
	
	// // zero out the output vector first
	// y.zero() ;
	
	// for (int ix=0; ix < _size; ix++)
		// for (int iy=0; iy < _size; iy++)
		// {
		// }
// }

template <typename T>
typename PetscBlockVector<T>::Parent&
PetscBlockVector<T>::operator= (PetscBlockVector<T>::ScalarType const v)
{
	// loop over the nested vectors and call VecSet on them
	for (unsigned int i=0; i < _size; i++)
	{
		PetscErrorCode ierr = VecSet( _block_vector_obj_refs[i]->vec(), v) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	
	return *this ;
}

template <typename T>
typename PetscBlockVector<T>::Parent&
PetscBlockVector<T>::operator= (const Parent& V_in)
{
	karma_assert(_size == V_in.size()) ;
	
	const PetscBlockVector<T>& V = dynamic_cast<const PetscBlockVector<T> &>(V_in) ;
	PetscErrorCode ierr = VecCopy(V.vec(), _vec) ;	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	this->close() ;
	return *this ;
}

template <typename T>
typename PetscBlockVector<T>::Parent&
PetscBlockVector<T>::operator= (const std::vector<PetscBlockVector<T>::CompositeType > &v)
{
	karma_assert(v.size() == _size) ;
	
	// insert all the data in the corresponding blocks
	std::vector<unsigned int> indices ;
	indices.resize(v.size()) ;
	std::iota(indices.begin(), indices.end(), 0u) ;
	
	this->insert(v, indices) ;

	this->close() ;
	
	return *this ;
}

template <typename T>
typename PetscBlockVector<T>::Parent&
PetscBlockVector<T>::operator += (const Parent& v)
{
	karma_assert(this->closed()) ;
	karma_assert(_size == v.size()) ;
	
	this->add(v);
	
	return *this;
}


template <typename T>
typename PetscBlockVector<T>::Parent&
PetscBlockVector<T>::operator += (const ScalarType a)
{
	karma_assert(this->closed()) ;
	
	PetscErrorCode ierr = VecShift(_vec, a);
	
	return *this;
}


template <typename T>
typename PetscBlockVector<T>::Parent&
PetscBlockVector<T>::operator -= (const Parent& v)
{
	karma_assert(this->closed()) ;
	karma_assert(v.closed());
	
	this->axpy(-1.0, v) ;
	
	return *this;
}


template <typename T>
typename PetscBlockVector<T>::Parent&
PetscBlockVector<T>::operator -= (const ScalarType a)
{
	karma_assert(this->closed()) ;
	
	// call the Petsc add method
	PetscErrorCode ierr = VecShift(_vec, -a);
	
	return *this;
}


template <typename T>
void PetscBlockVector<T>::localize (Parent& v_local,
						   const std::vector<unsigned int>* send_list) const
{
	// loop over all objects in _block_vector_obj_refs and if they match 
	// an entry in the send_list, call localize on the CompositeType.
	karma_not_implemented() ;
}



//------------------------------------------------------------------
// Explicit instantiations
// template class IKARMA<IVector<VectorTraits<PetscScalar, 4> > > ;

// First instantiate the Traits class for Real type
// template class VectorTraits<PetscScalar, 4>;
// Next, instantiate the actual Native vector
template class PetscBlockVector<PetscScalar>;

// NOTE: We have no complex vectors here for now. Not even sure
// if a block vector can have both double and complex types as
// different blocks

#endif
