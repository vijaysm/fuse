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


#include <vector>
#include <cassert>

#include "mesh.h"
#include "elem.h"
#include "node.h"

#include "karma_error.h"
#include "karma_mesh_utils.h"

namespace KARMA
{
	// Extract a 2-D surface from a 3-D volume mesh
	Mesh& karma_mesh_extract_face(const Mesh& parent, unsigned int direction, double , double )
	{
		assert(parent.mesh_dimension() == 3) ;
		assert(direction > 0 && direction < 4) ;

		parent.print_info() ;
	
		Mesh* face_mesh = new Mesh(2) ;

		face_mesh->clear() ;

		return *face_mesh ;
	}
}
