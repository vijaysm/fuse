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

#ifndef __karma_boundary_condition_h__
#define __karma_boundary_condition_h__

// C++ includes
#include <vector>

// KARMA includes
#include "karma.h"
#include "bc_type.h"
#include "static_vector.h"

/**
 * This implementation is for the KARMABoundaryCondition class container.
 * It is a base class for all the different types of boundary conditions that are
 * commonly encountered in physics applications.
 * The derived members can be Dirichlet, Neumann and Robin.
 * The virtual methods need to be overridden for these derived types.
 *
 * @author Vijay M, 2009
 */

// Create a fixed vector of dimension=3 to store vector valued functions
typedef KARMA::StaticVector< double, 3 > FaceValues;

// ------------------------------------------------------------
// KARMABoundaryCondition class definition
class KARMABoundaryCondition
{
  public:
    /**
     *  Default constructor.
     *  By default, only one Real variable is needed to depict this B.C.
     *  This is true for Dirichlet (value) and Neumann (derivative).
     *  Modern typed Robin data use 3 values (a,b,c) for a*(k grad(u).n)+b*u=c.
     *  Legacy callbacks retain their historical interpretation.
     */
    KARMABoundaryCondition( KARMA::PHYSICS::BCType type, bool time_dependent_bc = false );

    /**
     *  Copy constructor.
     *  Copies the type and data from supplied copy.
     */
    KARMABoundaryCondition( const KARMABoundaryCondition& dummy );

    /**
     *  Copy constructor. Initializes and calls Parent KARMABoundaryCondition
     *  and copies the data from the supplied object.
     */
    KARMABoundaryCondition& operator=( const KARMABoundaryCondition& dummy );

    /**
     * Destoys all the function pointers allocated.
     */
    virtual ~KARMABoundaryCondition();

    std::vector< double >& values();

    const std::vector< double >& values() const;

    KARMA::PHYSICS::BCType type() const;

    FaceValues& normal();

    FaceValues& normal() const;

    void set( double val );

    void set( unsigned i, double val );

    double& operator[]( unsigned i );

    double operator[]( unsigned i ) const;

    bool is_time_dependent_bcs() const;

  private:
    // boundary condition type
    KARMA::PHYSICS::BCType _type;

    // boundary condition data for scalar-valued types
    std::vector< double > _values;

    // boundary condition data for vector-valued types
    FaceValues* _normal;

    bool _is_time_dependent_bcs;
};

// Inline methods for KARMABoundaryCondition
inline std::vector< double >& KARMABoundaryCondition::values()
{ return _values; }

inline const std::vector< double >& KARMABoundaryCondition::values() const
{ return _values; }

inline KARMA::PHYSICS::BCType KARMABoundaryCondition::type() const
{ return _type; }

inline FaceValues& KARMABoundaryCondition::normal()
{ return *_normal; }

inline FaceValues& KARMABoundaryCondition::normal() const
{ return *_normal; }

inline void KARMABoundaryCondition::set( double value )
{
    // check if there is atleast one value allocated
    karma_assert( _values.size() > 0 );

    _values[0] = value;
}

inline void KARMABoundaryCondition::set( unsigned i, double value )
{
    // check if enough size is allocated
    karma_assert( i < _values.size() );

    _values[i] = value;
}

inline double& KARMABoundaryCondition::operator[]( unsigned i )
{
    karma_assert( i < _values.size() );

    return _values[i];
}

inline double KARMABoundaryCondition::operator[]( unsigned i ) const
{
    karma_assert( i < _values.size() );

    return _values[i];
}

inline bool KARMABoundaryCondition::is_time_dependent_bcs() const
{ return _is_time_dependent_bcs; }

#endif  // define __karma_boundary_condition_h__
