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


#ifndef __io_package_h__
#define __io_package_h__

namespace KARMA
{

	/**
	 * The library or file format used to write mesh and solution data.
	 *
	 * This enumeration was previously taken from libMesh's
	 * \p enum_io_package.h.  KARMA now defines it so that the I/O layer
	 * does not depend on libMesh; the enumerators and their order are kept
	 * compatible with the original for the values KARMA uses.
	 */
	enum IOPackage
	{
		TECPLOT=0,
		GMV=1,
		GMSH=2,
		VTK=3,
		EXODUS_II=4,
		NEMESIS=5,
		INVALID_IO_PACKAGE=6
	};

}

/**
 * KARMA code refers to these enumerators unqualified, and in a few places
 * still spells the scope as \p libMeshEnums.  Both spellings resolve here.
 */
using KARMA::IOPackage;
using KARMA::TECPLOT;
using KARMA::GMV;
using KARMA::GMSH;
using KARMA::VTK;
using KARMA::EXODUS_II;
using KARMA::NEMESIS;
using KARMA::INVALID_IO_PACKAGE;

namespace libMeshEnums
{
	using KARMA::IOPackage;
}

#endif // #define __io_package_h__
