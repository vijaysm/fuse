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

// Local Includes
#include "petsc_native_vector.h"
#include "dense_vector.h"
#include "dense_subvector.h"
#include "parallel.h"
#include "utility.h"


//-----------------------------------------------------------------------
// PetscNativeVector members

template <typename T>
AutoPtr<PetscNativeVector<T> >
PetscNativeVector<T>::build()
{
	// Build the PETSc vector
	AutoPtr<PetscNativeVector<T> > ap(new PetscNativeVector<T>);
	return ap;
}


template <typename T>
int PetscNativeVector<T>::compare (const Parent& other_vector,
			       const Real threshold) const
{
	karma_assert (this->initialized());
	karma_assert (other_vector.initialized());

	const PetscNativeVector<T>& V = dynamic_cast<const PetscNativeVector<T>&>(other_vector) ;
	
	karma_assert (this->first_local_index() == V.first_local_index());
	karma_assert (this->last_local_index()  == V.last_local_index());

	if (!_array_is_present)
		this->_get_array() ;
	
	if (!V._array_is_present)
		V._get_array() ;

	int first_different_i = std::numeric_limits<int>::max();
	unsigned int i = first_local_index();

	do
	{
		if ( std::abs( _values[i] - V._values[i] ) > threshold )
		{
			first_different_i = i;
			break ;	// no need to continue if value is unequal !
		}
		else
			i++;
	}
	while (first_different_i==std::numeric_limits<int>::max() && i < last_local_index());
	
	// Find the correct first differing index in parallel
	Parallel::min(first_different_i);
	
	if (first_different_i == std::numeric_limits<int>::max())
		return -1;
	
	return first_different_i;
}


template <typename T>
int PetscNativeVector<T>::local_relative_compare (const Parent &other_vector,
			                      const Real threshold) const
{
	karma_assert (this->initialized());
	karma_assert (other_vector.initialized());
	
	const PetscNativeVector<T>& V = dynamic_cast<const PetscNativeVector<T>&>(other_vector) ;
	
	karma_assert (this->first_local_index() == V.first_local_index());
	karma_assert (this->last_local_index()  == V.last_local_index());

	if (!_array_is_present)
		this->_get_array() ;
	
	if (!V._array_is_present)
		V._get_array() ;

	int first_different_i = std::numeric_limits<int>::max();
	unsigned int i = first_local_index();
	
	do
	{
		if ( std::abs( _values[i] - V._values[i] ) > threshold *
				   std::max(std::abs(_values[i]), std::abs(V._values[i]) ) )
			first_different_i = i;
		else
			i++;
	}
	while (first_different_i==std::numeric_limits<int>::max() && i<last_local_index());
	
	// Find the correct first differing index in parallel
	Parallel::min(first_different_i);
	
	if (first_different_i == std::numeric_limits<int>::max())
		return -1;
	
	return first_different_i;
}


template <typename T>
int PetscNativeVector<T>::global_relative_compare (const Parent &other_vector,
			                       const Real threshold) const
{
	karma_assert (this->initialized());
	karma_assert (other_vector.initialized());
	
	const PetscNativeVector<T>& V = dynamic_cast<const PetscNativeVector<T>&>(other_vector) ;
	
	karma_assert (this->first_local_index() == V.first_local_index());
	karma_assert (this->last_local_index()  == V.last_local_index());

	if (!_array_is_present)
		this->_get_array() ;
	
	if (!V._array_is_present)
		V._get_array() ;

	int first_different_i = std::numeric_limits<int>::max();
	unsigned int i = first_local_index();
	
	const Real my_norm = this->linfty_norm();
	const Real other_norm = other_vector.linfty_norm();
	const Real abs_threshold = std::max(my_norm, other_norm) * threshold;
	
	do
	{
		if ( std::abs( _values[i] - V._values[i] ) > abs_threshold )
			first_different_i = i;
		else
			i++;
	}
	while (first_different_i==std::numeric_limits<int>::max() && i<last_local_index());
	
	// Find the correct first differing index in parallel
	Parallel::min(first_different_i);
	
	if (first_different_i == std::numeric_limits<int>::max())
		return -1;
	
	return first_different_i;
}


template <typename T>
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::subset_l1_norm (const std::set<unsigned int> & indices) const
{
	const PetscNativeVector<T> & v = *this;
	
	std::set<unsigned int>::const_iterator it = indices.begin();
	const std::set<unsigned int>::const_iterator it_end = indices.end();
	
	T norm = 0;
	
	for(; it!=it_end; ++it)
		norm += std::abs(v(*it));
	
	libMesh::Parallel::sum(norm);
	
	return norm;
}

template <typename T>
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::subset_l2_norm (const std::set<unsigned int> & indices) const
{
	const PetscNativeVector<T> & v = *this;
	
	std::set<unsigned int>::const_iterator it = indices.begin();
	const std::set<unsigned int>::const_iterator it_end = indices.end();
	
	Real norm = 0;
	
	for(; it!=it_end; ++it)
		norm += libmesh_norm(v(*it));
	
	libMesh::Parallel::sum(norm);
	
	return std::sqrt(norm);
}

template <typename T>
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::subset_linfty_norm (const std::set<unsigned int> & indices) const
{
	const PetscNativeVector<T> & v = *this;
	
	std::set<unsigned int>::const_iterator it = indices.begin();
	const std::set<unsigned int>::const_iterator it_end = indices.end();
	
	T norm = 0;
	
	for(; it!=it_end; ++it)
	{
		T value = std::abs(v(*it));
		if(value > norm)
			norm = value;
	}
	
	libMesh::Parallel::max(norm);
	
	return norm;
}


template <typename T>
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::l1_norm () const
{
	this->_restore_array();
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscReal value=0.;
	
	ierr = VecNorm (_vec, NORM_1, &value);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<Real>(value);
}



template <typename T>
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::l2_norm () const
{
	this->_restore_array();
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscReal value=0.;
	
	ierr = VecNorm (_vec, NORM_2, &value);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<Real>(value);
}




template <typename T>
typename PetscNativeVector<T>::ScalarType PetscNativeVector<T>::linfty_norm () const
{
	this->_restore_array();
	karma_assert(this->closed());
	
	PetscErrorCode ierr=0;
	PetscReal value=0.;
	
	ierr = VecNorm (_vec, NORM_INFINITY, &value);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return static_cast<Real>(value);
}




template <typename T>
typename PetscNativeVector<T>::Parent&
PetscNativeVector<T>::operator += (const Parent& v)
{
	this->_restore_array();
	karma_assert(this->closed());
	
	this->add(v);
	
	return *this;
}



template <typename T>
typename PetscNativeVector<T>::Parent&
PetscNativeVector<T>::operator += (const ScalarType a)
{
	this->_restore_array();
	karma_assert(this->closed());
	
	PetscErrorCode ierr = VecShift(_vec, a);
	
	return *this;
}



template <typename T>
typename PetscNativeVector<T>::Parent&
PetscNativeVector<T>::operator -= (const Parent& v)
{
	this->_restore_array();
	karma_assert(this->closed());
	
	this->axpy(-1.0, v) ;
	
	return *this;
}


template <typename T>
typename PetscNativeVector<T>::Parent&
PetscNativeVector<T>::operator -= (const ScalarType a)
{
	this->_restore_array();
	karma_assert(this->closed());
	
	PetscErrorCode ierr = VecShift(_vec, -a);
	
	return *this;
}




template <typename T>
void PetscNativeVector<T>::insert (CompositeType const value, const unsigned int i)
{
  this->_restore_array();
  karma_assert(i<size());
  
  PetscErrorCode ierr=0;
  int i_val = static_cast<int>(i);
  PetscScalar petsc_value = static_cast<PetscScalar>(value);

  ierr = VecSetValues (_vec, 1, &i_val, &petsc_value, INSERT_VALUES);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  this->_is_closed = false;
}



template <typename T>
void PetscNativeVector<T>::add (CompositeType const value, const unsigned int i)
{
  this->_restore_array();
  karma_assert(i<size());
  
  PetscErrorCode ierr=0;
  int i_val = static_cast<int>(i);
  PetscScalar petsc_value = static_cast<PetscScalar>(value);

  ierr = VecSetValues (_vec, 1, &i_val, &petsc_value, ADD_VALUES);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  this->_is_closed = false;
}



template <typename T>
void PetscNativeVector<T>::insert (const std::vector<CompositeType>& v,
				 const std::vector<unsigned int>& dof_indices)
{
  this->_restore_array();
  karma_assert (v.size() == dof_indices.size());
  karma_assert (this->closed()) ;
  
  // should we call close or warn user if not closed ?
  PetscErrorCode ierr = VecSetValues (_vec, v.size(), (int*)&dof_indices[0], &v[0], INSERT_VALUES);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  this->_is_closed = false;
}


template <typename T>
void PetscNativeVector<T>::add (const std::vector<CompositeType>& v,
				 const std::vector<unsigned int>& dof_indices)
{
  this->_restore_array();
  karma_assert (v.size() == dof_indices.size());
  karma_assert (this->closed()) ;
  
  // should we call close or warn user if not closed ?
  PetscErrorCode ierr = VecSetValues (_vec, v.size(), (int*)&dof_indices[0], &v[0], ADD_VALUES);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  this->_is_closed = false;
}



template <typename T>
void PetscNativeVector<T>::add (const Parent& V)
{
	karma_assert (V.size() == this->size());

	const PetscNativeVector<T>& vector = dynamic_cast<const PetscNativeVector<T>& > (V) ;
	
	PetscErrorCode ierr = VecAXPY(_vec, 1.0, vector.vec()) ;CHKERRABORT(KARMA_COMM_WORLD,ierr);
}




template <typename T>
void PetscNativeVector<T>::add (const libMesh::DenseVector<T>& V,
				 const std::vector<unsigned int>& dof_indices)
{
	karma_assert (V.size() == dof_indices.size());
	
	for (unsigned int i=0; i<V.size(); i++)
		this->add (V(i), dof_indices[i]);
}



template <typename T>
void PetscNativeVector<T>::insert (const libMesh::DenseVector<T>& V,
			     const std::vector<unsigned int>& dof_indices)
{
	karma_assert (V.size() == dof_indices.size());
	
	for (unsigned int i=0; i<V.size(); i++)
		this->insert (V(i), dof_indices[i]);
}



template <typename T>
void PetscNativeVector<T>::insert (const libMesh::DenseSubVector<T>& V,
			     const std::vector<unsigned int>& dof_indices)
{
	karma_assert (V.size() == dof_indices.size());
	
	for (unsigned int i=0; i<V.size(); i++)
		this->insert (V(i), dof_indices[i]);
}



template <typename T>
void PetscNativeVector<T>::shift (const ScalarType v_in)
{
  this->_restore_array();

  PetscErrorCode ierr=0;
  PetscScalar* values;
  const PetscScalar v = static_cast<PetscScalar>(v_in);  

  if(this->type() != KARMA::GHOSTED)
    {
      const int n   = static_cast<int>(this->local_size());
      const int fli = static_cast<int>(this->first_local_index());
      
      for (int i=0; i<n; i++)
	{
	  ierr = VecGetArray (_vec, &values);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  
	  int ig = fli + i;      
	  
	  PetscScalar value = (values[i] + v);
	  
	  ierr = VecRestoreArray (_vec, &values);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  
	  ierr = VecSetValues (_vec, 1, &ig, &value, INSERT_VALUES);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr); 
	}
    }
  else
    {
      /* Vectors that include ghost values require a special
	 handling.  */
      Vec loc_vec;
      ierr = VecGhostGetLocalForm (_vec,&loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);

      int n=0;
      ierr = VecGetSize(loc_vec, &n);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
      
      for (int i=0; i<n; i++)
	{
	  ierr = VecGetArray (loc_vec, &values);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  
	  PetscScalar value = (values[i] + v);
	  
	  ierr = VecRestoreArray (loc_vec, &values);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  
	  ierr = VecSetValues (loc_vec, 1, &i, &value, INSERT_VALUES);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr); 
	}

      ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

  this->_is_closed = false;
}



template <typename T>
void PetscNativeVector<T>::scale (const ScalarType factor_in)
{
  this->_restore_array();

  PetscErrorCode ierr = 0;
  PetscScalar factor = static_cast<PetscScalar>(factor_in);
  
  if(this->type() != KARMA::GHOSTED)
    {
      // 2.3.x & later style	 
      ierr = VecScale(_vec, factor);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  else
    {
      Vec loc_vec;
      ierr = VecGhostGetLocalForm (_vec,&loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);

      // 2.3.x & later style	 
      ierr = VecScale(loc_vec, factor);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);

      ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
}


template <typename T>
void PetscNativeVector<T>::axpy (const ScalarType a_in, const Parent& v_in)
{
  this->_restore_array();

  PetscErrorCode ierr = 0;
  PetscScalar a = static_cast<PetscScalar>(a_in);

  // Make sure the NumericVector passed in is really a PetscNativeVector
  const PetscNativeVector<T>* v = libmesh_cast_ptr<const PetscNativeVector<T>*>(&v_in);
  v->_restore_array();

  karma_assert(this->size() == v->size());
  
  if(this->type() != KARMA::GHOSTED)
    {
      // 2.3.x & later style
      ierr = VecAXPY(_vec, a, v->_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  else
    {
      Vec loc_vec;
      Vec v_loc_vec;
      ierr = VecGhostGetLocalForm (_vec,&loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
      ierr = VecGhostGetLocalForm (v->_vec,&v_loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);

      // 2.3.x & later style
      ierr = VecAXPY(loc_vec, a, v_loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);

      ierr = VecGhostRestoreLocalForm (v->_vec,&v_loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
      ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
}


template <typename T>
void PetscNativeVector<T>::aypx(const ScalarType a, const Parent& x)
{
	// TODO: make this synonymous to axpy function above
	// with specialization for GHOSTED
  this->_restore_array();

  const PetscNativeVector<T>& v = dynamic_cast<const PetscNativeVector<T>&>(x) ;
  PetscErrorCode ierr = 0;

  ierr = VecAYPX(_vec, a, v._vec);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
}

template <typename T>
void PetscNativeVector<T>::axpby(const ScalarType a, const Parent& x, const ScalarType b)
{
	// TODO: make this synonymous to axpy function above
	// with specialization for GHOSTED
  this->_restore_array();

  const PetscNativeVector<T>& v = dynamic_cast<const PetscNativeVector<T>&>(x) ;
  PetscErrorCode ierr = 0;

  ierr = VecAXPBY(_vec, a, b, v._vec);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
}

template <typename T>
void PetscNativeVector<T>::axpbypcz(const ScalarType a, const Parent& x, const ScalarType b, const Parent& y, const ScalarType c)
{
	// TODO: make this synonymous to axpy function above
	// with specialization for GHOSTED
  this->_restore_array();

  const PetscNativeVector<T>& v = dynamic_cast<const PetscNativeVector<T>&>(x) ;
  const PetscNativeVector<T>& w = dynamic_cast<const PetscNativeVector<T>&>(y) ;
  PetscErrorCode ierr = 0;

  ierr = VecAXPBYPCZ(_vec, a, b, c, v._vec, w._vec);	CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
}

template <typename T>
void PetscNativeVector<T>::maxpy(const std::vector<ScalarType>& a, const std::vector<Parent* >& vecs)
{
	// TODO: make this synonymous to axpy function above
	// with specialization for GHOSTED
	this->_restore_array();
	
	PetscInt nv = vecs.size() ;
	Vec rawvecs[nv] ;
	for (unsigned int i=0; i < nv; i++)
	{
		const PetscNativeVector<T>* v = dynamic_cast<const PetscNativeVector<T>* >(vecs[i]) ;
		rawvecs[i] = v->_vec ;
	}
	
	PetscErrorCode ierr = 0;
	ierr = VecMAXPY(_vec, nv, &a[0], rawvecs);	CHKERRABORT(KARMA_COMM_WORLD,ierr);

}

template <typename T>
void PetscNativeVector<T>::abs()
{
	this->_restore_array();
	
	PetscErrorCode ierr = 0;
	
	if(this->type() != KARMA::GHOSTED)
	{
		ierr = VecAbs(_vec);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	else
	{
		Vec loc_vec;
		ierr = VecGhostGetLocalForm (_vec,&loc_vec);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
		ierr = VecAbs(loc_vec);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
		ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
}

template <typename T>
typename PetscNativeVector<T>::ScalarType 
PetscNativeVector<T>::dot (const Parent& V) const
{
	this->_restore_array();
	
	// Error flag
	PetscErrorCode ierr = 0;
	
	// Return value
	Real value=0.;
	
	// Make sure the NumericVector passed in is really a PetscNativeVector
	const PetscNativeVector<T>* v = libmesh_cast_ptr<const PetscNativeVector<T>*>(&V);
	
	// 2.3.x (at least) style.  Untested for previous versions.
	ierr = VecDot(this->_vec, v->_vec, &value);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	return value;
}




template <typename T>
typename PetscNativeVector<T>::Parent&
PetscNativeVector<T>::operator = (ScalarType const s_in)
{
  this->_restore_array();
  karma_assert(this->closed());

  PetscErrorCode ierr = 0;
  PetscScalar s = static_cast<PetscScalar>(s_in);

	if (this->size() != 0)
	{
		if(this->type() != KARMA::GHOSTED)
		{
			// 2.3.x & later style
			ierr = VecSet(_vec, s);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
		else
		{
			Vec loc_vec;
			ierr = VecGhostGetLocalForm (_vec,&loc_vec);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
		
			// 2.3.x & later style
			ierr = VecSet(loc_vec, s);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
		
			ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
		}
	}
	
	return *this;
}



template <typename T>
typename PetscNativeVector<T>::Parent&
PetscNativeVector<T>::operator = (const Parent& v_in)
{
	// Make sure the NumericVector passed in is really a PetscNativeVector
	const PetscNativeVector<T>* v = libmesh_cast_ptr<const PetscNativeVector<T>*>(&v_in);
	
	*this = *v;
	
	return *this;
}



template <typename T>
PetscNativeVector<T>&
PetscNativeVector<T>::operator = (const PetscNativeVector<T>& v)
{
	this->_restore_array();
	v._restore_array();
	
	karma_assert (this->size() == v.size());
	karma_assert (this->local_size() == v.local_size());
	karma_assert (v.closed());
	this->_is_closed = true;
	
	PetscErrorCode ierr = 0;

	if((this->type()==KARMA::PARALLEL) && (v.type()==KARMA::GHOSTED))
	{
		/* Allow assignment of a ghosted to a parallel vector since this
		causes no difficulty.  See discussion in libmesh-devel of
		June 24, 2010.  */
		ierr = VecCopy (v._vec, this->_vec);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	}
	else
	{
		/* In all other cases, we assert that both vectors are of equal
		type.  */
		karma_assert (this->_type == v._type);
		karma_assert (this->_global_to_local_map == v._global_to_local_map);
		
		if (v.size() != 0)
		{
			if(this->type() != KARMA::GHOSTED)
			{
				ierr = VecCopy (v._vec, this->_vec);
				CHKERRABORT(KARMA_COMM_WORLD,ierr);
			}
			else
			{
				Vec loc_vec;
				Vec v_loc_vec;
				ierr = VecGhostGetLocalForm (_vec,&loc_vec);
				CHKERRABORT(KARMA_COMM_WORLD,ierr);
				ierr = VecGhostGetLocalForm (v._vec,&v_loc_vec);
				CHKERRABORT(KARMA_COMM_WORLD,ierr);
			
				ierr = VecCopy (v_loc_vec, loc_vec);
				CHKERRABORT(KARMA_COMM_WORLD,ierr);
			
				ierr = VecGhostRestoreLocalForm (v._vec,&v_loc_vec);
				CHKERRABORT(KARMA_COMM_WORLD,ierr);
				ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
				CHKERRABORT(KARMA_COMM_WORLD,ierr);
			}
		}
	}

	return *this;
}



template <typename T>
typename PetscNativeVector<T>::Parent&
PetscNativeVector<T>::operator = (const std::vector<T>& v)
{
  this->_restore_array();

  const unsigned int nl   = this->local_size();
  const unsigned int ioff = this->first_local_index();
  PetscErrorCode ierr=0;
  PetscScalar* values;
      
  /**
   * Case 1:  The vector is the same size of
   * The global vector.  Only add the local components.
   */
  if (this->size() == v.size())
    {
      ierr = VecGetArray (_vec, &values);
 	     CHKERRABORT(KARMA_COMM_WORLD,ierr);

      for (unsigned int i=0; i<nl; i++)
		values[i] =  static_cast<PetscScalar>(v[i+ioff]);
	
      ierr = VecRestoreArray (_vec, &values);
	     CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

  /**
   * Case 2: The vector is the same size as our local
   * piece.  Insert directly to the local piece.
   */
  else
    {
      karma_assert (this->local_size() == v.size());

      ierr = VecGetArray (_vec, &values);
	     CHKERRABORT(KARMA_COMM_WORLD,ierr);

      for (unsigned int i=0; i<nl; i++)
		values[i] = static_cast<PetscScalar>(v[i]);
	
      ierr = VecRestoreArray (_vec, &values);
	     CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

	// Make sure ghost dofs are up to date
	if (this->type() == KARMA::GHOSTED)
    	this->close();

  return *this;
}


/*
template <typename T>
void PetscNativeVector<T>::localize (Parent& v_local_in) const
{
  this->_restore_array();

  // Make sure the NumericVector passed in is really a PetscNativeVector
  PetscNativeVector<T>* v_local = libmesh_cast_ptr<PetscNativeVector<T>*>(&v_local_in);

  karma_assert (v_local != NULL);
  karma_assert (v_local->size() == this->size());

  PetscErrorCode ierr = 0;
  const int n = this->size();

  IS is;
  VecScatter scatter;

  // Create idx, idx[i] = i;
  std::vector<int> idx(n); Utility::iota (idx.begin(), idx.end(), 0);

  // Create the index set & scatter object
  ierr = ISCreateGeneral(KARMA_COMM_WORLD, n, &idx[0], PETSC_USE_POINTER, &is);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  ierr = VecScatterCreate(_vec,          is,
			  v_local->_vec, is,
			  &scatter);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Perform the scatter
  ierr = VecScatterBegin(scatter, _vec, v_local->_vec,
			 INSERT_VALUES, SCATTER_FORWARD);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
  ierr = VecScatterEnd  (scatter, _vec, v_local->_vec,
                         INSERT_VALUES, SCATTER_FORWARD);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Clean up
  ierr = ISDestroy (&is);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
  ierr = VecScatterDestroy(&scatter);
         CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Make sure ghost dofs are up to date
		 if (v_local->type() == KARMA::GHOSTED)
    v_local->close();
}
*/


template <typename T>
void PetscNativeVector<T>::localize (Parent& v_local_in,
			       const std::vector<unsigned int>* send_list) const
{
	this->_restore_array();
	
	// Make sure the NumericVector passed in is really a PetscNativeVector
	PetscNativeVector<T>* v_local = dynamic_cast<PetscNativeVector<T>*>(&v_local_in);
	
	karma_assert (v_local != NULL);
	karma_assert (v_local->size() == this->size());
	unsigned int n_sl = 0; 
	if (send_list != NULL)
	{
		karma_assert (send_list.size()     <= v_local->size());
		n_sl = send_list->size() ;
	}
	
	PetscErrorCode ierr=0;
	
	IS is;
	VecScatter scatter;
	
	std::vector<int> idx(n_sl + this->local_size());

	if (send_list != NULL)	
	{
		for (unsigned int i=0; i<n_sl; i++)
			idx[i] = static_cast<int>((*send_list)[i]);
		for (int i = 0; i < this->local_size(); i++)
			idx[n_sl+i] = i + this->first_local_index();
	}
	else
	{
		for (int i=0; i < this->local_size(); i++)
			idx[i] = i + this->first_local_index();
	}
	
	// Create the index set & scatter object
	if (idx.empty())
		ierr = ISCreateGeneral(KARMA_COMM_WORLD,
							n_sl+this->local_size(), PETSC_NULL, PETSC_USE_POINTER, &is);
	else
		ierr = ISCreateGeneral(KARMA_COMM_WORLD,
				n_sl+this->local_size(), &idx[0], PETSC_USE_POINTER, &is);
		CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	ierr = VecScatterCreate(_vec,          is,
				v_local->_vec, is,
				&scatter);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	
	// Perform the scatter
	ierr = VecScatterBegin(scatter, _vec, v_local->_vec,
							INSERT_VALUES, SCATTER_FORWARD);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	ierr = VecScatterEnd  (scatter, _vec, v_local->_vec,
							INSERT_VALUES, SCATTER_FORWARD);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	
	// Clean up
	ierr = ISDestroy (&is);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	ierr = VecScatterDestroy(&scatter);
			CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	// Make sure ghost dofs are up to date
	if (v_local->type() == KARMA::GHOSTED)
		v_local->close();
}


template <typename T>
void PetscNativeVector<T>::localize (const unsigned int first_local_idx,
			       const unsigned int last_local_idx,
			       const std::vector<unsigned int>& send_list)
{
  this->_restore_array();

  // Only good for serial vectors.
  // karma_assert (this->size() == this->local_size());
  karma_assert (last_local_idx >= first_local_idx);
  karma_assert (send_list.size() <= this->size());
  karma_assert (last_local_idx < this->size());
  
  const unsigned int size       = this->size();
  const unsigned int local_size = (last_local_idx - first_local_idx + 1);
  PetscErrorCode ierr=0;
  
  // Don't bother for serial cases
  if ((first_local_idx == 0) &&
      (local_size == size))
    return;
  
  
  // Build a parallel vector, initialize it with the local
  // parts of (*this)
  PetscNativeVector<T> parallel_vec;
  // we could use the constructor directly but do not want to set values = 0: 3rd argument
  parallel_vec.init (size, local_size, true, KARMA::PARALLEL);

  // Copy part of *this into the parallel_vec
  {
    IS is;
    VecScatter scatter;

    // Create idx, idx[i] = i+first_local_idx;
    std::vector<int> idx(local_size);
    Utility::iota (idx.begin(), idx.end(), first_local_idx);

    // Create the index set & scatter object
	ierr = ISCreateGeneral(KARMA_COMM_WORLD, local_size, &idx[0], PETSC_USE_POINTER, &is);
           CHKERRABORT(KARMA_COMM_WORLD,ierr);

    ierr = VecScatterCreate(_vec,              is,
			    parallel_vec._vec, is,
			    &scatter);
           CHKERRABORT(KARMA_COMM_WORLD,ierr);

    // Perform the scatter
    ierr = VecScatterBegin(scatter, _vec, parallel_vec._vec,
			   INSERT_VALUES, SCATTER_FORWARD);
           CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
    ierr = VecScatterEnd  (scatter, _vec, parallel_vec._vec,
			   INSERT_VALUES, SCATTER_FORWARD);
           CHKERRABORT(KARMA_COMM_WORLD,ierr);

    // Clean up
    ierr = ISDestroy (&is);
           CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
    ierr = VecScatterDestroy(&scatter);
           CHKERRABORT(KARMA_COMM_WORLD,ierr);
  }

  // localize like normal
  parallel_vec.close();
  parallel_vec.localize (*this, &send_list);
  this->close();
}



// Full specialization for Real datatypes
#ifdef LIBMESH_USE_REAL_NUMBERS

// NOTE: Partial specialization is only for AUTOMATIC layout type
template <>
void PetscNativeVector<Real>::localize_to_one (std::vector<Real>& v_local,
					 const unsigned int pid) const
{
  this->_restore_array();

  PetscErrorCode ierr=0;
  const int n  = size();
  const int nl = local_size();
  PetscScalar *values;

  v_local.resize(n);

  // only one processor
  if (n == nl)
    {      
      ierr = VecGetArray (_vec, &values);
	     CHKERRABORT(KARMA_COMM_WORLD,ierr);

      for (int i=0; i<n; i++)
	v_local[i] = static_cast<Real>(values[i]);

      ierr = VecRestoreArray (_vec, &values);
	     CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

  // otherwise multiple processors
  else
    {
      unsigned int ioff = this->first_local_index();
      std::vector<Real> local_values (n, 0.);
      
      {
	ierr = VecGetArray (_vec, &values);
	       CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	for (int i=0; i<nl; i++)
	  local_values[i+ioff] = static_cast<Real>(values[i]);
	
	ierr = VecRestoreArray (_vec, &values);
	       CHKERRABORT(KARMA_COMM_WORLD,ierr);
      }
      

      MPI_Reduce (&local_values[0], &v_local[0], n, MPI_REAL, MPI_SUM,
		  pid, KARMA_COMM_WORLD);
    }
}

#endif


// Full specialization for Complex datatypes
#ifdef LIBMESH_USE_COMPLEX_NUMBERS
// NOTE: Partial specialization is only for AUTOMATIC layout type
template <>
void PetscNativeVector<Complex>::localize_to_one (std::vector<Complex>& v_local,
					    const unsigned int pid) const
{
  this->_restore_array();

  PetscErrorCode ierr=0;
  const int n  = size();
  const int nl = local_size();
  PetscScalar *values;

  
  v_local.resize(n);

  
  for (int i=0; i<n; i++)
    v_local[i] = 0.;
  
  // only one processor
  if (n == nl)
    {      
      ierr = VecGetArray (_vec, &values);
	     CHKERRABORT(KARMA_COMM_WORLD,ierr);

      for (int i=0; i<n; i++)
	v_local[i] = static_cast<Complex>(values[i]);

      ierr = VecRestoreArray (_vec, &values);
	     CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }

  // otherwise multiple processors
  else
    {
      unsigned int ioff = this->first_local_index();

      /* in here the local values are stored, acting as send buffer for MPI
       * initialize to zero, since we collect using MPI_SUM
       */
      std::vector<Real> real_local_values(n, 0.);
      std::vector<Real> imag_local_values(n, 0.);

      {
	ierr = VecGetArray (_vec, &values);
	       CHKERRABORT(KARMA_COMM_WORLD,ierr);
	
	// provide my local share to the real and imag buffers
	for (int i=0; i<nl; i++)
	  {
	    real_local_values[i+ioff] = static_cast<Complex>(values[i]).real();
	    imag_local_values[i+ioff] = static_cast<Complex>(values[i]).imag();
	  }

	ierr = VecRestoreArray (_vec, &values);
	       CHKERRABORT(KARMA_COMM_WORLD,ierr);
      }
   
      /* have buffers of the real and imaginary part of v_local.
       * Once MPI_Reduce() collected all the real and imaginary
       * parts in these std::vector<Real>, the values can be 
       * copied to v_local
       */
      std::vector<Real> real_v_local(n);
      std::vector<Real> imag_v_local(n);

      // collect entries from other proc's in real_v_local, imag_v_local
      MPI_Reduce (&real_local_values[0], &real_v_local[0], n, 
		  MPI_REAL, MPI_SUM,
		  pid, KARMA_COMM_WORLD);	

      MPI_Reduce (&imag_local_values[0], &imag_v_local[0], n, 
		  MPI_REAL, MPI_SUM,
		  pid, KARMA_COMM_WORLD);	

      // copy real_v_local and imag_v_local to v_local
      for (int i=0; i<n; i++)
	v_local[i] = Complex(real_v_local[i], imag_v_local[i]);
    }  
}

#endif




template <typename T>
void PetscNativeVector<T>::pointwise_mult (const Parent& vec1,
				     const Parent& vec2)
{
  this->_restore_array();

  PetscErrorCode ierr = 0;

  // Convert arguments to PetscNativeVector*.
  const PetscNativeVector<T>* vec1_petsc = libmesh_cast_ptr<const PetscNativeVector<T>*>(&vec1);
  const PetscNativeVector<T>* vec2_petsc = libmesh_cast_ptr<const PetscNativeVector<T>*>(&vec2);

  // Call PETSc function.

  if(this->type() != KARMA::GHOSTED)
    {
      ierr = VecPointwiseMult(this->vec(),
			      const_cast<PetscNativeVector<T>*>(vec1_petsc)->vec(),
			      const_cast<PetscNativeVector<T>*>(vec2_petsc)->vec());
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  else
    {
	  Vec loc_vec;
	  Vec v1_loc_vec;
	  Vec v2_loc_vec;
	  ierr = VecGhostGetLocalForm (_vec,&loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGhostGetLocalForm (const_cast<PetscNativeVector<T>*>(vec1_petsc)->vec(),&v1_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGhostGetLocalForm (const_cast<PetscNativeVector<T>*>(vec2_petsc)->vec(),&v2_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);

	  ierr = VecPointwiseMult(loc_vec,v1_loc_vec,v2_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);

	  ierr = VecGhostRestoreLocalForm (const_cast<PetscNativeVector<T>*>(vec1_petsc)->vec(),&v1_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGhostRestoreLocalForm (const_cast<PetscNativeVector<T>*>(vec2_petsc)->vec(),&v2_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
}


template <typename T>
void PetscNativeVector<T>::pointwise_divide (const Parent& vec1,
				     const Parent& vec2)
{
  this->_restore_array();

  PetscErrorCode ierr = 0;

  // Convert arguments to PetscNativeVector*.
  const PetscNativeVector<T>* vec1_petsc = libmesh_cast_ptr<const PetscNativeVector<T>*>(&vec1);
  const PetscNativeVector<T>* vec2_petsc = libmesh_cast_ptr<const PetscNativeVector<T>*>(&vec2);

  // Call PETSc function.
  if(this->type() != KARMA::GHOSTED)
    {
      ierr = VecPointwiseDivide(this->vec(),
			      const_cast<PetscNativeVector<T>*>(vec1_petsc)->vec(),
			      const_cast<PetscNativeVector<T>*>(vec2_petsc)->vec());
      CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
  else
    {
	  Vec loc_vec;
	  Vec v1_loc_vec;
	  Vec v2_loc_vec;
	  ierr = VecGhostGetLocalForm (_vec,&loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGhostGetLocalForm (const_cast<PetscNativeVector<T>*>(vec1_petsc)->vec(),&v1_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGhostGetLocalForm (const_cast<PetscNativeVector<T>*>(vec2_petsc)->vec(),&v2_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);

	  ierr = VecPointwiseDivide(loc_vec,v1_loc_vec,v2_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);

	  ierr = VecGhostRestoreLocalForm (const_cast<PetscNativeVector<T>*>(vec1_petsc)->vec(),&v1_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGhostRestoreLocalForm (const_cast<PetscNativeVector<T>*>(vec2_petsc)->vec(),&v2_loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
	  ierr = VecGhostRestoreLocalForm (_vec,&loc_vec);
	  CHKERRABORT(KARMA_COMM_WORLD,ierr);
    }
}





template <typename T>
void PetscNativeVector<T>::create_subvector(Parent& subvector,
				      const std::vector<unsigned int>& rows) const
{
  this->_restore_array();

  // PETSc data structures
  IS parent_is, subvector_is;
  VecScatter scatter;
  PetscErrorCode ierr = 0;
  
  // Make sure the passed in subvector is really a PetscNativeVector
  PetscNativeVector<T>* petsc_subvector = libmesh_cast_ptr<PetscNativeVector<T>*>(&subvector);
  
  // If the petsc_subvector is already initialized, we assume that the
  // user has already allocated the *correct* amount of space for it.
  // If not, we use the appropriate PETSc routines to initialize it.
  if (!petsc_subvector->initialized())
    {
      // Initialize the petsc_subvector to have enough space to hold
      // the entries which will be scattered into it.  Note: such an
      // init() function (where we let PETSc decide the number of local
      // entries) is not currently offered by the PetscNativeVector
      // class.  Should we differentiate here between sequential and
      // parallel vector creation based on KARMA::n_processors() ?
      ierr = VecCreateMPI(KARMA_COMM_WORLD,
			  PETSC_DECIDE,          // n_local
			  rows.size(),           // n_global
			  &(petsc_subvector->_vec)); CHKERRABORT(KARMA_COMM_WORLD,ierr);

      ierr = VecSetFromOptions (petsc_subvector->_vec); CHKERRABORT(KARMA_COMM_WORLD,ierr);

      // Mark the subvector as initialized
      petsc_subvector->_is_initialized = true;
    }
  else
    {
      petsc_subvector->_restore_array();
    }
  
  // Use iota to fill an array with entries [0,1,2,3,4,...rows.size()]
  std::vector<int> idx(rows.size());
  Utility::iota (idx.begin(), idx.end(), 0);

  // Construct index sets
  ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 rows.size(),
			 (int*) &rows[0],
			 PETSC_USE_POINTER,
			 &parent_is); CHKERRABORT(KARMA_COMM_WORLD,ierr);

  ierr = ISCreateGeneral(KARMA_COMM_WORLD,
			 rows.size(),
			 (int*) &idx[0],
			 PETSC_USE_POINTER,
			 &subvector_is); CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Construct the scatter object
  ierr = VecScatterCreate(this->_vec,
			  parent_is,
			  petsc_subvector->_vec,
			  subvector_is,
			  &scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr);

  // Actually perform the scatter
  ierr = VecScatterBegin(scatter,
			 this->_vec,
			 petsc_subvector->_vec,
			 INSERT_VALUES,
			 SCATTER_FORWARD); CHKERRABORT(KARMA_COMM_WORLD,ierr);

  ierr = VecScatterEnd(scatter,
		       this->_vec,
		       petsc_subvector->_vec,
		       INSERT_VALUES,
		       SCATTER_FORWARD); CHKERRABORT(KARMA_COMM_WORLD,ierr);
  
  // Clean up 
  ierr = ISDestroy(&parent_is);       CHKERRABORT(KARMA_COMM_WORLD,ierr);
  ierr = ISDestroy(&subvector_is);    CHKERRABORT(KARMA_COMM_WORLD,ierr);
  ierr = VecScatterDestroy(&scatter); CHKERRABORT(KARMA_COMM_WORLD,ierr); 

}



//------------------------------------------------------------------
// Explicit instantiations
template class PetscNativeVector<PetscScalar>;

#endif // #ifdef KARMA_HAVE_PETSC
