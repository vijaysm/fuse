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


//#include <cstdio>
#include <iostream>
#include "petsc.h"
#include "karma_log.h"

// define the static stream for Log2File
std::ofstream Log2File::_file_stream ;

template <typename T>
std::ostringstream& KARMALog<T>::get(KARMALogLevel level)
{
	// Maybe later, we can print the message number or something ?!
	// Do not spend finding the time, if you don't care about it.
	if (KARMALog<T>::use_time_info)
		os << "- " << current_time() << " " ;
	
    os << to_string(level) << ": ";

	// Indent extra depending on whether it is DEBUG1, DEBUG2 ...
	if(level > INFO)
		os << std::string(level - INFO, '\t');
    
	return os;
}

template <typename T>
std::ostringstream& KARMALog<T>::get(const std::string level)
{
	KARMALogLevel elevel = KARMALog<T>::from_string(level) ;
	
	return get(elevel) ;
}

template <typename T>
std::ostringstream& KARMALog<T>::get(unsigned int level)
{
	KARMALogLevel elevel = KARMALog<T>::from_int(level) ;
	
	return get(elevel) ;
}

template <typename T>
void KARMALog<T>::flush()
{
    os << std::endl;
    T::output(os.str(), KARMALog<T>::process_all_procs);
	os.clear() ;	os.str("") ;
}



// int log_main(int argc, char* argv[])
// {
    // try
    // {
        // FILELog::ReportingLevel() = FILELog::FromString(argv[1] ? argv[1] : "DEBUG1");
	    // const int count = 3;
        // FILE_LOG(DEBUG) << "A loop with " << count << " iterations";
        // for (int i = 0; i != count; ++i)
        // {
	        // FILE_LOG(DEBUG1) << "the counter i = " << i;
        // }
        // return 0;
    // }
    // catch(const std::exception& e)
    // {
        // FILE_LOG(ERROR) << e.what();
    // }
    // return -1;
// }


// specialize the logging to File and console stream
template class KARMALog<Log2File> ;
template class KARMALog<Log2Stream> ;
