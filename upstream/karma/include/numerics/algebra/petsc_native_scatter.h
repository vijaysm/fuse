#ifndef __petsc_native_scatter_h__
#define __petsc_native_scatter_h__

#include "karma_dense.h"

#include <petscmat.h>
#include <petscvec.h>

#include <span>

namespace KARMA
{
/** Values are staged only; Mat/Vec assembly remains the collective caller's responsibility. */
void scatter( Mat matrix, const RealDenseMatrix& values, std::span< const PetscInt > rows,
              std::span< const PetscInt > columns, InsertMode mode );
void scatter_local( Mat matrix, const RealDenseMatrix& values, std::span< const PetscInt > rows,
                    std::span< const PetscInt > columns, InsertMode mode );
void scatter( Vec vector, const RealDenseVector& values, std::span< const PetscInt > indices, InsertMode mode );
void scatter_local( Vec vector, const RealDenseVector& values, std::span< const PetscInt > indices, InsertMode mode );

inline void add( Mat matrix, const RealDenseMatrix& values, std::span< const PetscInt > rows,
                 std::span< const PetscInt > columns )
{ scatter( matrix, values, rows, columns, ADD_VALUES ); }
inline void add_local( Mat matrix, const RealDenseMatrix& values, std::span< const PetscInt > rows,
                       std::span< const PetscInt > columns )
{ scatter_local( matrix, values, rows, columns, ADD_VALUES ); }
inline void add( Vec vector, const RealDenseVector& values, std::span< const PetscInt > indices )
{ scatter( vector, values, indices, ADD_VALUES ); }
inline void add_local( Vec vector, const RealDenseVector& values, std::span< const PetscInt > indices )
{ scatter_local( vector, values, indices, ADD_VALUES ); }
}  // namespace KARMA

#endif
