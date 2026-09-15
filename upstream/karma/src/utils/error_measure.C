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


#include "karma.h"
#include "equation_systems.h"
#include "mesh.h"
#include "point_locator_base.h"

// Define the Finite Element object.
#include "fe.h"
// Define Gauss quadrature rules.
#include "quadrature_gauss.h"
#include "quadrature_trap.h"
#include "quadrature_jacobi.h"

#include "implicit_system.h"

// Define the DofMap, which handles degree of freedom indexing.
#include "dof_map.h"

#include "petsc_vector.h"

#undef __FUNCT__
#define __FUNCT__ "error_measure"
std::vector<Real> error_measure(int dim, std::string solFile, std::string refFile, int varnum=-1, bool verbose=false)
{
	Mesh refMesh(dim) ;
	//refMesh.read(fineFile) ;
	EquationSystems reference (refMesh) ;
	reference.init() ;
	PetscPrintf(libMesh::COMM_WORLD, "\nReading reference data from %s", refFile.c_str()) ;
	reference.read(refFile, READ, EquationSystems::READ_HEADER | EquationSystems::READ_DATA | EquationSystems::READ_ADDITIONAL_DATA) ;
	PetscPrintf(libMesh::COMM_WORLD, "\tDone..");
	PetscPrintf(libMesh::COMM_WORLD, "\nIniting solution eqnsys") ;
	
	PetscPrintf(libMesh::COMM_WORLD, "\tDone..");
	
	const PointLocatorBase& el_locator = refMesh.point_locator() ;
	std::vector<Real> refsol ;
	reference.build_solution_vector(refsol) ;
	PetscPrintf(libMesh::COMM_WORLD, "\tDone..");
	
	Mesh solMesh(dim) ;
	//solMesh.read(solFile) ;
	EquationSystems solution (solMesh) ;
	PetscPrintf(libMesh::COMM_WORLD, "\nIniting solution eqnsys") ;
	solution.init() ;
	PetscPrintf(libMesh::COMM_WORLD, "\tDone..\nReading solution data from %s", solFile.c_str()) ;
	solution.read(solFile, READ, EquationSystems::READ_HEADER | EquationSystems::READ_DATA | EquationSystems::READ_ADDITIONAL_DATA) ;
	PetscPrintf(libMesh::COMM_WORLD, "\tDone..");
	std::vector<Real> sol ;
	solution.build_solution_vector(sol) ;
	PetscPrintf(libMesh::COMM_WORLD, "\tDone..");
	
	std::vector<Real> error ;
	error.resize((varnum < 0) ? solution.n_vars() : 1) ;

	const DofMap& ref_dof_map = reference.get_system(0).get_dof_map();
	const DofMap& sol_dof_map = solution.get_system(0).get_dof_map();

	FEType fe_type = ref_dof_map.variable_type(0);
	AutoPtr<FEBase> ref_fe (FEBase::build(dim, fe_type));
	AutoPtr<FEBase> sol_fe (FEBase::build(dim, fe_type));

	// Use higher order quadrature than that used to solve to gain better estimate on the error
	QGauss qrule (dim, TENTH);
	
	// Tell the finite element object to use our quadrature rule.
	sol_fe->attach_quadrature_rule (&qrule);
	ref_fe->attach_quadrature_rule (&qrule);
	
	const std::vector<Real>& JxW = sol_fe->get_JxW();
	const std::vector<Point>& q_point = sol_fe->get_xyz();
	const std::vector<std::vector<Real> >& sol_phi = sol_fe->get_phi();
	const std::vector<std::vector<Real> >& ref_phi = ref_fe->get_phi();
	
	std::vector<unsigned int> sol_dof_indices;
	std::vector<unsigned int> ref_dof_indices;
	
	MeshBase::const_element_iterator       el     = solMesh.active_local_elements_begin();
	const MeshBase::const_element_iterator end_el = solMesh.active_local_elements_end();
	
	if (verbose)
		PetscPrintf(libMesh::COMM_WORLD, "\n[X\tY\tZ]\tExact\tNumerical\tError\n") ;

	for ( ; el != end_el; ++el)
	{
		const Elem* elem = *el;

		sol_fe->reinit (elem);
		
		for (unsigned int qp=0; qp<qrule.n_points(); qp++)
		{
			const Real x = q_point[qp](0);
			const Real y = q_point[qp](1);
			const Real z = q_point[qp](2);
			
			// find the exact solution at the point
			const Elem* ref_elem = el_locator(q_point[qp]) ;
			ref_fe->reinit(ref_elem) ;
			
			if (verbose)
				PetscPrintf(libMesh::COMM_WORLD, "\nPos=[%G, %G, %G]\t", x, y, z) ;
					
			for (unsigned int ivar=0; ivar<reference.n_vars(); ivar++)
			{
				ref_dof_map.dof_indices (ref_elem, ref_dof_indices, ivar);
				sol_dof_map.dof_indices (elem, sol_dof_indices, ivar);
				
				// evaluate the reference solution at the point
				Real exact = 0.0 ;
				for (unsigned int i=0; i<ref_phi.size(); i++)
					exact += ref_phi[i][qp]*refsol[ref_dof_indices[i]];
				
				// evaluate the numerical solution at the point
				Real val = 0.0 ;
				for (unsigned int i=0; i<sol_phi.size(); i++)
					val += sol_phi[i][qp]*sol[sol_dof_indices[i]];
				
				error[ivar] += JxW[qp] * (exact-val) * (exact-val) ;
				
				if (verbose)
					PetscPrintf(libMesh::COMM_WORLD, "\nVar=[%D]\t%G\t%G\t%G", ivar, exact, val, exact-val) ;
			}
		}
	}

	std::vector<Real> errnrm ;
	errnrm.resize(error.size()) ;

	// since this is done in parallel, reduce the contribution to compute the L2 norm value  
	MPI_Reduce(&error[0], &errnrm[0], error.size(), MPI_DOUBLE, MPI_SUM, 0, KARMA_COMM_WORLD) ;
	
	PetscPrintf(KARMA_COMM_WORLD, "\nNorm of error w.r.t exact solution.") ;
	for (unsigned int ivar=0; ivar<errnrm.size(); ivar++)
	{
		errnrm[ivar] = sqrt(errnrm[ivar]) ;
		PetscPrintf(KARMA_COMM_WORLD, "\nVariable [%D] = %G", ivar, errnrm[ivar]) ;
	}

	error.clear() ;
	
	return errnrm ;

}
