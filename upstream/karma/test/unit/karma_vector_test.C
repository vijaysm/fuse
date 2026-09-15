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
#include "petscext_vec.h"
#endif
#include "karma_vector.h"

class KARMAVectorTest : public KARMATestSuite
{
public:
	KARMAVectorTest (int argc, char** argv)
	: KARMATestSuite(argc, argv)
	{
#ifdef KARMA_HAVE_PETSC
		TEST_ADD(KARMAVectorTest::petsc_vec_init)
#ifdef KARMA_HAVE_PETSCEXT
		TEST_ADD(KARMAVectorTest::petsc_ext_vec_init)
#endif
#endif
	}
	
	virtual ~KARMAVectorTest()
	{
	}
	
private:

#ifdef KARMA_HAVE_PETSC
	void petsc_vec_init() ; // simple test for MPI initialization
#ifdef KARMA_HAVE_PETSCEXT
	void petsc_ext_vec_init() ; // simple test for PETSc initialization
#endif
#endif
};

#include "petsc_vector.h"

#ifdef KARMA_HAVE_PETSC
void KARMAVectorTest::petsc_vec_init()
{
	PetscErrorCode ierr ;
	const unsigned int n = 10 ;
	const unsigned int n_local = 10/KARMA::num_processors ;

	// FIRST: create a serial vector with 10 elements
	{
		KARMARealVector karma_vec1(n) ;
		
	// 	KARMA_LOG(INFO) << "KARMA Vec1 before setting." ;
	// 	karma_vec1.print() ;
		
		// first set vector 1 with multiples of 10
		for (unsigned int i=0; i < n; i++)
		{
			karma_vec1(i) = static_cast<double>(i*10.0) ;
		}
		karma_vec1.close() ;
		/*
		KARMA_LOG(INFO) << "KARMA Vec1 after setting." ;
		karma_vec1.print() ;*/
		
		PetscReal max ;
		ierr = VecMax(karma_vec1.vec(), 0, &max) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
		
		KARMA_LOG(INFO) << "KARMA Vec1 maximum value = " << max << ". Expected value = " << (n-1)*10.0 ;
	
		// Check if the L2_NORM of 1x10 is sqrt(10)
		TEST_ASSERT(max==(n-1)*10.0)
	}

	// NEXT, create a parallel vector with 5 local elements (2 procs)
	{
		KARMARealVector karma_vec2(n, n_local) ;
	
		// set vector 2 with random values
		ierr = VecSetRandom(karma_vec2.vec(), 0) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
		PetscReal  norm ;
		ierr = VecNorm(karma_vec2.vec(), NORM_2, &norm) ;	CHKERRABORT(KARMA_COMM_WORLD, ierr) ;
		
		KARMA_LOG(INFO) << "KARMA Vec2 l2_norm value = " << karma_vec2.l2_norm() << ". Expected value = " << norm ;
		
		// Check if the L2_NORM from the wrapper and native call yield same value
		TEST_ASSERT_DELTA(karma_vec2.l2_norm(), norm, 1e-8)
	}

	// THEN, create a ghosted vector with 5 local elements (2 procs)
	{
		KARMARealVector* karma_vec3 = new KARMARealVector(n, n_local) ;
		delete karma_vec3  ;
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
void KARMAVectorTest::petsc_ext_vec_init()
{
	PetscErrorCode ierr ;
	const unsigned int n = 10 ;
	const unsigned int n_local = 10/KARMA::num_processors ;
	
	Vec X, a,b;
	VecCreate( KARMA_COMM_WORLD, &X );
	VecSetSizes( X, 2, 2 );
	VecSetType( X, "block" );
	
	VecCreate( KARMA_COMM_WORLD, &a );
	VecSetSizes( a, 2*n, 2*n_local );
	VecSetFromOptions( a );
	VecSet(a, 19.0) ;
		
	VecCreate( KARMA_COMM_WORLD, &b );
	VecSetSizes( b, n, n_local );
	VecSetFromOptions( b );
	VecSet(b, 20.0) ;

		// Create the raw PETSc-EXT vector by assembling ourselves
	VecBlockSetValue( X, 0, a, INSERT_VALUES );
	VecBlockSetValue( X, 1, b, INSERT_VALUES );
	VecAssemblyBegin(X);
	VecAssemblyEnd(X);
	
	// Now initialize a Block vector with this X
	KARMAVector* karma_vec3 = new KARMAVector(X) ;
	karma_vec3->name() = "karma_vec3" ;
	karma_vec3->close() ;

	// The second method, create a block vector and assign a, b ourselves.
	KARMARealVector A(a) ;
	A.name() = "A" ;
	KARMARealVector B(b) ;
	B.name() = "B" ;

	KARMAVector* karma_vec4 = new KARMAVector(2) ;
	karma_vec4->name() = "karma_vec4" ;

	karma_vec4->insert(&A, 0) ;
	karma_vec4->insert(&B, 1) ;
	karma_vec4->close() ;

	karma_vec3->print() ;
	karma_vec4->print() ;

	KARMAVector* karma_vec5 = new KARMAVector(a) ;
	karma_vec5->name() = "karma_vec5" ;
	karma_vec5->print();

	// Check if the L2_NORM of the block vector is equal to L2 norm of individual blocks
	// KARMA_LOG(INFO) << "karma_vec4->l2_norm() = " << karma_vec4->l2_norm()*karma_vec4->l2_norm() << "\tA.l2_norm() = " << A.l2_norm()*A.l2_norm() << "\tB.l2_norm() = " << B.l2_norm()*B.l2_norm() << "\tTotal = " << A.l2_norm()*A.l2_norm()+B.l2_norm()*B.l2_norm();
	
	TEST_ASSERT_DELTA(karma_vec4->l2_norm(), std::sqrt(A.l2_norm()*A.l2_norm()+B.l2_norm()*B.l2_norm()), 1e-10)
	
	ierr = VecDestroy(a) ;
	ierr = VecDestroy(b) ;
	ierr = VecDestroy(X) ;
	delete karma_vec3 ;
	delete karma_vec5 ;
	delete karma_vec4 ;

	KARMA_LOG(INFO) << "Just before debug1 call." ;
	KARMA_LOG(DEBUG1) << "Exiting scope." ; 
	KARMA_LOG(ERROR) << "Just after debug1 call." ;
}	
#endif

int main (int argc, char** argv)
{
	try
	{
		KARMAVectorTest test(argc, argv) ;

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

