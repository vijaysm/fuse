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


// The KARMA Finite Element Library.
// Copyright (C) 2002-2008 Benjamin S. Kirk, John W. Peterson, Roy H. Stogner
  
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
  
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
  
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA


#ifndef __linear_algebra_factory_h
#define __linear_algebra_factory_h

#include <string>

namespace KARMA
{

	// Helper factory class with all static methods
	// Is omnipotent in terms of knowing how to create the
	// different objects necessary for linear algebra operations.
	class KARMALinearSolver;
	template <typename T> class KARMAMatrixBase;
	template <typename T> class KARMAVectorBase;
	
	class LinearAlgebraFactory
	{
	public:
		
		/// Constructor
		LinearAlgebraFactory() {}
		
		/// Destructor
		virtual ~LinearAlgebraFactory() {}
		
		/// Create empty matrix
		virtual AutoPtr<KARMAMatrixBase<T> > create_matrix() const = 0;
		
		/// Create empty vector (global)
		virtual AutoPtr<KARMAVectorBase<T> > create_vector() const = 0;
		
		/// Create empty vector (local)
		virtual AutoPtr<KARMAVectorBase<T> > create_local_vector() const = 0;
		
		/// Create empty sparsity pattern (returning zero if not used/needed)
		virtual AutoPtr<UserSparsity> create_pattern() const = 0;
		
		/// Create LU solver
		virtual AutoPtr<KARMALinearSolver<T> > create_lu_solver() const = 0;
		
		/// Create Krylov solver
		virtual AutoPtr<KARMAVectorBase<T> > create_krylov_solver(std::string& method,
															std::string& pc) const = 0;
	
	};

}

#endif	// __linear_algebra_factory_h
