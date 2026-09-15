// KARMA: K(C)ode for Accident and Reactor Modeling Analysis
// Copyright (C) 2008-2009  Vijay S Mahadevan
//
// This program is free software; you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation; either version 2 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along
// with this program; if not, write to the Free Software Foundation, Inc.,
// 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
//

#include "karma.h"
#include "karma_log.h"
#include "karma_config.h"
#include "karma_test_suite.h"

#include <fstream>

class KARMAPackageTest : public KARMATestSuite
{
public:
	KARMAPackageTest (int argc, char** argv)
	: KARMATestSuite(argc, argv)
	{
		TEST_ADD(KARMAPackageTest::mpi_init)
#ifdef KARMA_HAVE_PETSC
		TEST_ADD(KARMAPackageTest::petsc_init)
#endif
#ifdef KARMA_HAVE_LIBMESH
		TEST_ADD(KARMAPackageTest::libmesh_init)
#endif
	}
	
	virtual ~KARMAPackageTest()
	{
	}
	
private:

	void mpi_init() ; // simple test for MPI initialization

#ifdef KARMA_HAVE_PETSC
	void petsc_init() ; // simple test for PETSc initialization
#endif
#ifdef KARMA_HAVE_LIBMESH
	void libmesh_init() ; // simple test for libMesh initialization
#endif
};


void KARMAPackageTest::mpi_init()
{
	// Check if the MPI_COMM instance is non null
    //
    TEST_ASSERT(KARMA_COMM_WORLD != MPI_COMM_NULL)
    
	// Check if the total number of procs > 0
    // Asserts whether MPI initialized correctly
    TEST_ASSERT(KARMA::num_processors > 0)

	KARMA_LOG(INFO) << "Checking for number of MPI_COMM processes: [" << KARMA::num_processors << "]" << std::endl ;

	// KARMAPackage will go out of scope now...
	return ;
}

#ifdef KARMA_HAVE_PETSC
void KARMAPackageTest::petsc_init()
{
	KARMA_LOG(INFO) << "Checking for basic operations in PETSc." << std::endl ;

	// In this test, create a vector and compute its norm
	Vec petsc_vec ;
	PetscErrorCode ierr ;

	KARMA_LOG(INFO) << "Creating a PETSc Vector of size 10: VecCreate(vector) + VecSize(vector, 10)" << std::endl ;
	ierr = VecCreate(KARMA_COMM_WORLD, &petsc_vec) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
	// create a 10 element vector
	ierr = VecSetSizes(petsc_vec,PETSC_DECIDE,10);	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
	ierr = VecSetFromOptions(petsc_vec);	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
	
	KARMA_LOG(INFO) << "Call VecSet(vector, 1.0)" << std::endl ;
	ierr = VecSet(petsc_vec, 1.0) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;

	PetscScalar norm ;
	KARMA_LOG(INFO) << "Call VecNorm(vector)" << std::endl ;
	ierr = VecNorm(petsc_vec, NORM_2, &norm) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;

	KARMA_LOG(INFO) << "Computed L_2 norm: " << norm << " = " << std::sqrt(10.0) << std::endl ;
	
	// Check if the L2_NORM of 1x10 is sqrt(10)
    TEST_ASSERT_DELTA(norm, std::sqrt(10.0), 1e-8)
	
	KARMA_LOG(INFO) << "Call VecDestroy(vector)" << std::endl ;
	ierr = VecDestroy(&petsc_vec) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
}	
#endif

#ifdef KARMA_HAVE_LIBMESH
#include <fstream>
#include "mesh.h"
#include "mesh_generation.h"
void KARMAPackageTest::libmesh_init()
{
	// In this test, create a libMesh structed 1-d mesh
	Mesh mesh;
	
	KARMA_LOG(INFO) << "Create a 1-d mesh with 5 elements." << std::endl ;
	
    MeshTools::Generation::build_line(mesh,5,0.,1.,EDGE2);	// with 5 elements of type EDGE2
	
	mesh.print_info() ;

	KARMA_LOG(INFO) << "Write out the mesh to GMSH file." << std::endl ;

	const std::string output_mesh_file = "output/mesh-1d.msh" ;
	mesh.write(output_mesh_file) ;	// write the mesh in gmsh format
	
	KARMA_LOG(INFO) << "Check if the mesh file exists." << std::endl ;

	// open a stream and try to read from it..
	std::ifstream mesh_file(output_mesh_file.c_str()) ;
	// Check if the L2_NORM of 1x10 is sqrt(10)
    TEST_ASSERT(mesh_file.good())
    mesh_file.close() ;
	
}	
#endif

int main (int argc, char** argv)
{
	try
	{
		KARMAPackageTest test(argc, argv) ;

		const KARMATestSuite::OutputType otype = KARMATestSuite::Html ;
	
		std::auto_ptr<Test::Output> output_format = KARMATestSuite::get_output_format(otype) ;
		test.run (*output_format) ;

		if (otype == KARMATestSuite::Html)
		{
			std::ofstream unit_test_output ;
			unit_test_output.open("output/unit_test_results.html") ;
			
			Test::HtmlOutput* html = dynamic_cast<Test::HtmlOutput*>(output_format.get());

			if (html && unit_test_output.is_open())
			{
				html->generate(unit_test_output, true, "KARMAPackageTest");
				unit_test_output.close() ;
			}
			else
			{
				throw FileError("output/unit_test_results.html") ;
			}
		}
	}
	catch (std::runtime_error& e)
	{
		KARMA_LOG(INFO) << "Unexpected exception encountered.\n";
		KARMA_LOG(INFO) << e.what() ;
		return EXIT_FAILURE;
	}
	return EXIT_SUCCESS;
}

