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


#include "ikarma.h"
#include "karma.h"
#include "karma_getpot.h"
#include "karma_utils.h"
#include "karma_log.h"

#ifdef KARMA_HAVE_LIBMESH
#include "libmesh.h"
#endif

#ifdef USE_PETSC_BLOCK_EXTENSIONS
#include "petscext.h"
#include "petscext_vec.h"
#include "petscext_mat.h"
#include "petscext_ksp.h"
#include "petscext_pc.h"
#include "petscext_snes.h"
#endif

MPI_Comm  KARMA_COMM_WORLD = MPI_COMM_NULL;
MPI_Comm  KARMA_COMM_SELF = MPI_COMM_NULL;

namespace KARMA
{
	bool verbose = true ;
	KARMALogLevel verbosity_level = INFO ;
	
	int num_processors ;
	int processor_id ;

	namespace KARMAPrivateData
	{
#ifdef KARMA_HAVE_LIBMESH
		LibMeshInit	*libFEMObj ;
#endif
	}


#ifdef KARMA_HAVE_LIBMESH
	PerfLog perflog ("KARMA",
#ifdef LIBMESH_ENABLE_PERFORMANCE_LOGGING
				     true
#else
				     false
#endif
				     ) ;
#endif


	KARMAPackage::KARMAPackage(int argc, char **argv, const std::string& s_option_file)
		:	_options_cmdline_db(),
			_options_file_db()
	{
#ifdef KARMA_HAVE_LIBMESH
		
		// First let us initialize libMesh, PETSc and MPI libraries, in that order
		KARMAPrivateData::libFEMObj = new LibMeshInit(argc, argv) ;

		// Duplicate the input communicator for internal use
		MPI_Comm_dup (MPI_COMM_WORLD, &KARMA_COMM_WORLD);
		MPI_Comm_dup (MPI_COMM_SELF, &KARMA_COMM_SELF);

		// get the total number of processors
		MPI_Comm_size(MPI_COMM_WORLD, &num_processors);

		// get the rank of the current processor
		MPI_Comm_rank(MPI_COMM_WORLD, &processor_id);
		
		_options_cmdline_db.reset (new GetPot (argc, argv));

		_options_file_db.reset (new GetPot (s_option_file));

		// initialize random-number-generator
		initialize_rng() ;
		
		// Now initialize anything specific to KARMA now that things are in place.
		
		// Update all options: Remember that command-line options override options file.

		// print detailed output during run
		bool verb_defined = this->get_option("-v", false) ;
		
		if (verb_defined)
		{
			// User specified a verbosity level
			verbosity_level = static_cast<KARMALogLevel>(this->get_option("-v", 7)) ;
			if (verbosity_level > 0)
				verbose = true ;
			else
				verbose = false ;
		}
		else
		{
			verbose = true ;
			verbosity_level = INFO ;	// Default = INFO
		}

		// print detailed output during run
		KARMALog<Log2Stream>::reporting_level = KARMALog<Log2Stream>::from_int(verbosity_level);
		
		std::string log_file = this->get_option("-log_file", "") ;
		if (log_file.size() > 0)
		{
			KARMALog<Log2File>::reporting_level = KARMALog<Log2File>::from_int(verbosity_level);
			// Append the processor id to file name so that there are no clashes.
			std::stringstream sstr ;
			sstr << log_file << "." << KARMA::processor_id ;
			std::ofstream& log_stream = Log2File::stream();
			log_stream.open(sstr.str().c_str()) ;
			if(!log_stream) {
				PetscPrintf(PETSC_COMM_SELF, "\nCannot open log file for processor_id = %D.\n", KARMA::processor_id) ;
				std::cin.get() ;
			} 
		}
		
		KARMA_LOG(INFO) << "KARMA: Verbosity level : " << verbosity_level ;
		KARMA_LOG(INFO) << "KARMA: Verbosity level : " << KARMALog<Log2Stream>::reporting_level ;
		
#endif


		// if using petsc block extensions, register all the BLOCK objects.
#ifdef USE_PETSC_BLOCK_EXTENSIONS
		// register all the extensions to Petsc
		PetscExtVecRegisterAll();
		PetscExtMatRegisterAll();
		PetscExtKSPRegisterAll();
		PetscExtPCRegisterAll();
		PetscExtSNESRegisterAll();
#endif

		// output to user all the options used in the current application run
		{
			PetscPrintf(KARMA_COMM_WORLD, "\nRunning Command: %s", argv[0]) ;

			for (int i=1; i<argc; i++)
				PetscPrintf(KARMA_COMM_WORLD, " %s", argv[i]) ;
			PetscPrintf(KARMA_COMM_WORLD, "\n\n") ;
		}

		// Use GetPot to read KARMA specific command line info here !
	}

	KARMAPackage::~KARMAPackage()
	{
		// Release any references to libMesh information

	#ifdef KARMA_HAVE_LIBMESH

		// Finally close libMesh handles, PETSc structures and MPI communication
		delete KARMA::KARMAPrivateData::libFEMObj ;
		
		// Print the perflog to the individual processor's files.
		KARMA::perflog.print_log();

		// Now clear the logging object, we don't want it to print
		// a second time during the PerfLog destructor.
		KARMA::perflog.clear();
		
	#endif
		
	}
	
	// helper functions for getting command-line and other file based options based on key
	unsigned int KARMAPackage::get_option(const std::string& option, unsigned int default_opt)
	{
		unsigned int opt = default_opt ;
		bool is_valid=false, is_cmd=false ;
		
		is_valid = search_option(option, is_cmd) ;
		
		if (is_valid)
		{
			if(is_cmd)
				opt = _options_cmdline_db->next(opt);
			else
				opt = _options_file_db->next(opt);
		}
			
		return opt ;
	}
	
	int KARMAPackage::get_option(const std::string& option, int default_opt)
	{
		int opt = default_opt ;
		bool is_valid=false, is_cmd=false ;
		
		is_valid = search_option(option, is_cmd) ;
		
		if (is_valid)
		{
			if(is_cmd)
				opt = _options_cmdline_db->next(opt);
			else
				opt = _options_file_db->next(opt);
		}
			
		return opt ;
	}
	
	double KARMAPackage::get_option(const std::string& option, double default_opt)
	{
		double opt = default_opt ;
		bool is_valid=false, is_cmd=false ;
		
		is_valid = search_option(option, is_cmd) ;
		
		if (is_valid)
		{
			if(is_cmd)
				opt = _options_cmdline_db->next(opt);
			else
				opt = _options_file_db->next(opt);
		}
			
		return opt ;
	}
	
	std::string KARMAPackage::get_option(const std::string& option, std::string default_opt)
	{
		std::string opt(default_opt) ;
		bool is_valid=false, is_cmd=false ;
		
		is_valid = search_option(option, is_cmd) ;
		
		if (is_valid)
		{
			if(is_cmd)
				opt = _options_cmdline_db->next(opt);
			else
				opt = _options_file_db->next(opt);
		}
			
		return opt ;
	}
	
	// helper function : returns true if an option with specified key exists
	bool KARMAPackage::search_option(const std::string& option)
	{
		bool is_cmd ;
		
		return this->search_option(option, is_cmd) ;
	}
	
	// helper function : returns true if an option with specified key exists and 
	// also checks if the option is specified in command-line.
	bool KARMAPackage::search_option(const std::string& option, bool& is_cmd_line)
	{
		if ( _options_cmdline_db->search(option) )
		{
			is_cmd_line = true ;
			return true ;
		}
		else if ( _options_file_db->search(option) )
		{
			is_cmd_line = false ;
			return true ;
		}
		else
		{
			is_cmd_line = false ;
			return false ;
		}
	}
	
}
