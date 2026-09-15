// Simple code to test if KARMA library links correctly.

// C++ include files that we need
#include <iostream>

// Basic include file needed for the mesh functionality.
#include "getpot.h"

#include "karma.h"

// Global objects
static char help[] = "";

// Begin the main program.
#undef __FUNCT__
#define __FUNCT__ "Hello-Physics-Test-Function"
int main (int argc, char** argv)
{
	// Initialize libMesh and any dependent libaries including PETSc
	KARMA::KARMAPackage karma (argc, argv);

	// Create a GetPot object to parse the command line
	GetPot command_line (argc, argv);
	
	// Check for proper calling arguments.
	if (argc > 1)
	{
		std::cout << "\nYou typed: " << argv[0];

		for (int i=1; i<argc; i++)
			std::cout << " " << argv[i];
		std::cout << std::endl << std::endl;
	}
	
	unsigned int dim ;
	if (karma.search_option("-d"))
	{
		dim = karma.get_option("-d", 1) ;
	
		std::cout << "The test problem dimension = " << dim << std::endl << std::endl;
	}
	else
	{
		std::cout << "You did not enter any arguments for -d" << std::endl << std::endl;
	}
	
	std::cout << "Thanks for testing KARMA Framework." << std::endl << std::endl;

	// All done.  
	return 0 ;	
}
