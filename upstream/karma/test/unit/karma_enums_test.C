// $Id: karma_enums_test.C 450 2011-09-19 20:55:34Z vijaysm $

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
#include "karma_test_suite.h"

// include all enumerator files from include/enums
#include "bc_type.h"
#include "partitioner_type.h"
#include "karma_nonlinear_operations.h"
#include "karma_linear_operations.h"
#include "runge_kutta_schemes.h"
#include "physics_properties.h"
#include "supporting_packages.h"
#include "primary_axis.h"
#include "numerical_flux_type.h"
#include "matrix_structure.h"
#include "linear_algebra_layout.h"

// Finally include our enum utility header
#include "enum_utils.h"

#include <fstream>

class KARMAEnumsTest : public KARMATestSuite
{
public:
	KARMAEnumsTest (int argc, char** argv)
	: KARMATestSuite(argc, argv)
	{
		TEST_ADD(KARMAEnumsTest::enum_tests)
	}
	
	virtual ~KARMAEnumsTest()
	{
	}
	
private:

	// create a test function for each enum type
	void enum_tests() ;
	
	// create a templated test function for enums
	template <typename TEnum>
	void enum_test_base() ;

};

void KARMAEnumsTest::enum_tests()
{
	enum_test_base<KARMA::PHYSICS::BCType>() ;
	enum_test_base<KARMA::PHYSICS::PropertyType>() ;
	enum_test_base<KARMA::PartitionerType>() ;
	enum_test_base<KARMA::KARMALinearOperations>() ;
	enum_test_base<KARMA::KARMANonlinearOperations>() ;
	enum_test_base<KARMA::TEMPORAL::RKCategory>() ;
	enum_test_base<KARMA::TEMPORAL::RKType>() ;
}

template <typename TEnum>
void KARMAEnumsTest::enum_test_base()
{
	using namespace KARMA::Utility ;
	
	KARMA_LOG(INFO) << "== Testing Enum Type: " << enum_traits<TEnum>::enum_type << " ==" << std::endl ;

	// loop over all elements in the enum
	for (unsigned int ienum=0; ienum < enum_traits<TEnum>::N; ienum++)
	{
		TEnum test ;
		
		// create a random number between 0 and N
		// and test the int_to_enum function
		test = int_to_enum<TEnum>(ienum) ;	// TODO: make the argument random
		
		// check if the re-conversion yields the same value
		TEST_ASSERT(ienum == enum_to_int<TEnum>(test) )
		
		std::string test_string = enum_to_string<TEnum>(test) ;	// get a string version of the enum
		
		// check if the re-conversion yields the same value
		TEST_ASSERT(test == string_to_enum<TEnum>(test_string) )
		
		KARMA_LOG(INFO) << "Enum [" << ienum << "]: Name: " << test_string << std::endl ;
	}
	
	// KARMAPackage will go out of scope now...
	return ;
}

int main (int argc, char** argv)
{
	try
	{
		KARMAEnumsTest test(argc, argv) ;

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
				html->generate(unit_test_output, true, "KARMAEnumsTest");
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

