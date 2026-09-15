// $Id: karma_test_suite.h 450 2011-09-19 20:55:34Z vijaysm $

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


#ifndef __karma_test_suite_h__
#define __karma_test_suite_h__

#include "karma_config.h"

#ifdef KARMA_HAVE_CPPTEST

#include <iostream>
#include "cpptest.h"


// The base test-suite for all KARMA unit tests
//
class KARMATestSuite : public Test::Suite
{
public:
	KARMATestSuite(int argc, char** argv)
	: _argc(argc), _argv(argv)
	{
		/** Sample implementation to add a function
		 *  to the test suite is as follows:
		 * 	TEST_ADD(KARMATestSuite::test1)
		 */

		// let us initialize and make sure KARMA along
		// with the contributing packages are initialized
		// correctly.
		init() ;
	}
	
	~KARMATestSuite()
	{
		// clean up the memory for interface and other auxillary packages
		clear() ;
	}
	
	enum OutputType
	{
		Compiler=0,
		Html,
		TextTerse,
		TextVerbose
	};

	static AutoPtr<Test::Output> get_output_format(KARMATestSuite::OutputType type = TextVerbose, std::ostream& stream=std::cout) ;

protected:

	// setup resources...
	virtual void init()	;

	// cleanup resources...
	virtual void clear() ;

	// provide access to the KARMA object and the command-line options
	int _argc ;
	char** _argv ;
	KARMA::KARMAPackage* _karma ;
	
private:

	// void test1() {} // simple example of a test function
};

inline
void KARMATestSuite::init()
{
	// Initialize KARMA and any dependent libaries including PETSc
	_karma = new KARMA::KARMAPackage(_argc, _argv);
}

inline
void KARMATestSuite::clear()
{
	delete _karma ;
}

inline
AutoPtr<Test::Output> KARMATestSuite::get_output_format(KARMATestSuite::OutputType type, std::ostream& stream)
{
	// if (argc > 2)
	// {
		// std::cerr << "usage: " << argv[0] << " [MODE] other_options\n"
				 // << "where MODE may be one of:\n"
				 // << "  --compiler[default]\n"
				 // << "  --html\n"
				 // << "  --text-terse\n"
				 // << "  --text-verbose\n";
		// exit(0);
	// }
	
	Test::Output* output = NULL;
	
	switch(type)
	{
		case Compiler:		// "--compiler"
			output = new Test::CompilerOutput;
			break ;
		case Html:			// "--html"
			output =  new Test::HtmlOutput;
			break ;
		case TextTerse:		// "--text-terse"
			output = new Test::TextOutput(Test::TextOutput::Terse, stream);
			break ;
		case TextVerbose:	// "--text-verbose") == 0)
			output = new Test::TextOutput(Test::TextOutput::Verbose, stream);
			break ;
	}
	
	return AutoPtr<Test::Output>(output);
}


#endif

#endif	// __karma_test_suite_h__