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
// Link-and-run check for the KARMA core library.  This deliberately touches
// one symbol from each part of the library that is currently built, so that
// a successful run means the archive is not merely well-formed but usable.

#include <petsc.h>

#include <iostream>
#include <sstream>

#include "bc_type.h"
#include "enum_utils.h"
#include "io_package.h"
#include "karma_types.h"
#include "static_vector.h"

namespace
{
	int failures = 0;

	void check(bool condition, const std::string& what)
	{
		if (!condition)
		{
			std::cerr << "FAIL: " << what << std::endl;
			++failures;
		}
	}
}

int main(int argc, char** argv)
{
	PetscErrorCode ierr = PetscInitialize(&argc, &argv, NULL, NULL);
	if (ierr) return ierr;

	PetscMPIInt rank = 0, size = 1;
	MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
	MPI_Comm_size(PETSC_COMM_WORLD, &size);

	// -- scalar types ------------------------------------------------------
	// These come from karma_types.h and used to be supplied by libMesh.
	check(sizeof(Real) == sizeof(double), "Real is double");
	check(TOLERANCE > 0.0, "TOLERANCE is positive");

	// -- AutoPtr, now std::unique_ptr --------------------------------------
	{
		AutoPtr<Real> owned(new Real(2.5));
		check(*owned == 2.5, "AutoPtr dereference");

		AutoPtr<Real> moved(std::move(owned));
		check(owned == nullptr, "AutoPtr transfers ownership on move");
		check(*moved == 2.5, "AutoPtr retains value across move");
	}

	// -- enum reflection ---------------------------------------------------
	{
		using KARMA::PHYSICS::BCType;
		using KARMA::PHYSICS::Neumann;

		const std::string name = KARMA::Utility::enum_to_string(Neumann);
		check(name == "Neumann", "enum_to_string(Neumann)");

		const BCType back = KARMA::Utility::string_to_enum<BCType>("Robin");
		check(back == KARMA::PHYSICS::Robin, "string_to_enum(Robin)");

		check(KARMA::Utility::enum_to_int(Neumann) == 1, "enum_to_int(Neumann)");
	}

	// -- the I/O package enum that replaced libMesh's ----------------------
	{
		IOPackage fmt = VTK;
		check(fmt == KARMA::VTK, "IOPackage resolves unqualified and qualified");
	}

	// -- a template instantiated in the library, not the header ------------
	{
		KARMA::StaticVector<double, 3> v;
		v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
		check(v(2) == 3.0, "StaticVector element access");

		std::ostringstream os;
		v.print(os);
		check(!os.str().empty(), "StaticVector::print writes output");
	}

	// -- PETSc is live, not merely linked ----------------------------------
	{
		Vec x;
		ierr = VecCreate(PETSC_COMM_WORLD, &x);              CHKERRQ(ierr);
		ierr = VecSetSizes(x, PETSC_DECIDE, 10);             CHKERRQ(ierr);
		ierr = VecSetFromOptions(x);                         CHKERRQ(ierr);
		ierr = VecSet(x, 3.0);                               CHKERRQ(ierr);

		PetscReal norm = 0.0;
		ierr = VecNorm(x, NORM_1, &norm);                    CHKERRQ(ierr);
		check(std::abs(norm - 30.0) < 1.e-10, "PETSc Vec 1-norm");

		ierr = VecDestroy(&x);                               CHKERRQ(ierr);
	}

	if (rank == 0)
	{
		if (failures == 0)
			std::cout << "KARMA smoke test passed on " << size
			          << " rank(s)." << std::endl;
		else
			std::cout << "KARMA smoke test: " << failures
			          << " failure(s)." << std::endl;
	}

	ierr = PetscFinalize();
	if (ierr) return ierr;

	return failures == 0 ? 0 : 1;
}
