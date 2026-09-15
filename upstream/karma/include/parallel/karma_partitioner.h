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


#ifndef __karma_partitioner_h__
#define __karma_partitioner_h__

// Local Includes -----------------------------------

// Forward Declarations
class KARMAMesh;

#include "partitioner.h"
#include "partitioner_type.h"

#include "karma_base.h"

/**
 * The \p KARMAPartitioner uses the Metis or Parmetis graph partitioner
 * to partition the elements depending on the context
 */

// ------------------------------------------------------------
// KARMAPartitioner class definition
class KARMAPartitioner
{
 public:

  /**
   * Constructor.
   */
  KARMAPartitioner (KARMA::PartitionerType part_type=KARMA::Parmetis) ;
  
  ~KARMAPartitioner () ;
  
 /**
   * This is the actual partitioning method which is overloaded.  
   * It is called via the public partition() method by the user.
   */
  void Partition (KARMAMesh& mesh,
				const unsigned int n);
				
 /**
   * Parmetis can handle dynamically repartitioning a mesh such
   * that the redistribution costs are minimized.  This method
   * takes a previously partitioned domain (which may have
   * then been adaptively refined) and repartitions it.
   */
  void RePartition (KARMAMesh& mesh,
				const unsigned int n);
  
 private:

/**
   * Use ParMetis if available and if not, use METIS
   *TODO: If both are not available, need to throw an error here
   */
   Partitioner* m_partitioner ;

};

#endif // #define __karma_partitioner_h__
