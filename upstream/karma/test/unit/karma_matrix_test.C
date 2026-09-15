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
#include "karma_config.h"
#include "karma_test_suite.h"
#ifdef KARMA_HAVE_PETSCEXT
#include "petscext_mat.h"
#endif
#include "karma_matrix.h"
#include "karma_vector.h"

class KARMAMatrixTest : public KARMATestSuite
{
public:
	KARMAMatrixTest (int argc, char** argv)
	: KARMATestSuite(argc, argv)
	{
#ifdef KARMA_HAVE_PETSC
		TEST_ADD(KARMAMatrixTest::petsc_mat_init)
#ifdef KARMA_HAVE_PETSCEXT
		TEST_ADD(KARMAMatrixTest::petsc_ext_mat_init)
#endif
#endif
	}

	virtual ~KARMAMatrixTest()
	{
	}

private:

#ifdef KARMA_HAVE_PETSC
	void petsc_mat_init() ; // simple test for MPI initialization
#ifdef KARMA_HAVE_PETSCEXT
	void petsc_ext_mat_init() ; // simple test for PETSc initialization
#endif
#endif
};

#include "petsc_vector.h"

#ifdef KARMA_HAVE_PETSC
void KARMAMatrixTest::petsc_mat_init()
{
	PetscErrorCode ierr ;
	const unsigned int n = 10 ;
	const unsigned int n_local = 10/KARMA::num_processors ;

	// FIRST: create a serial vector with 10 elements
	{
		KARMARealMatrix karma_mat1 ;
		karma_mat1.init(n, n, n_local, n_local) ;

		// first set vector 1 with multiples of 10
		for (int i=0; i < n; i++)
            for (int j=i-1; j <= i+1; j++)
            {
                if (j < 0 || j == n)
                    continue ;
                karma_mat1.insert(static_cast<double>(i*j*10.0+1), i, j) ;
            }
		karma_mat1.close() ;

		KARMA_LOG(INFO) << "KARMA Mat1 after setting." ;
		karma_mat1.print() ;

		PetscReal norm ;
		ierr = MatNorm(karma_mat1.mat(), NORM_FROBENIUS, &norm) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;

        double mynorm = karma_mat1.l2_norm() ;
		KARMA_LOG(INFO) << "KARMA Mat1 norm value = " << norm << ". Expected value = " << mynorm ;

		// Check if the L2_NORM of 1x10 is sqrt(10)
		TEST_ASSERT(norm == mynorm)
	}

	// Vec petsc_vec ;
	// PetscErrorCode ierr ;
	// ierr = VecCreate(KARMA_COMM_WORLD, &petsc_vec) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
	// // create a 10 element vector
	// ierr = VecSetSizes(petsc_vec,PETSC_DECIDE,10);	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
	// ierr = VecSetFromOptions(petsc_vec);	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;

	// ierr = VecSet(petsc_vec, 1.0) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
	// PetscReal  norm ;
	// ierr = VecNorm(petsc_vec, NORM_2, &norm) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;

	// // Check if the L2_NORM of 1x10 is sqrt(10)
    // TEST_ASSERT_DELTA(norm, std::sqrt(10.0), 1e-8)

	// ierr = VecDestroy(petsc_vec) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
}
#endif

#ifdef KARMA_HAVE_PETSCEXT
void KARMAMatrixTest::petsc_ext_mat_init()
{
	// In this test, create a block Matrix
	KARMAMatrix mybmat ;

	// Check if the L2_NORM of 1x10 is sqrt(10)
    TEST_ASSERT(!mybmat.initialized())

}
#endif

int main (int argc, char** argv)
{
	try
	{
		KARMAMatrixTest test(argc, argv) ;

		const KARMATestSuite::OutputType otype = KARMATestSuite::TextVerbose ;

		std::auto_ptr<Test::Output> output_format = KARMATestSuite::get_output_format(otype) ;
		test.run (*output_format) ;

		if (otype == KARMATestSuite::Html)
		{
			std::ofstream unit_test_output ;
			unit_test_output.open("output/unit_test_results.html") ;

			Test::HtmlOutput* html = dynamic_cast<Test::HtmlOutput*>(output_format.get());

			if (html && unit_test_output.is_open())
			{
				html->generate(unit_test_output, true, "KARMAMatrixTest");
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

