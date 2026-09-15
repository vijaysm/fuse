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


#ifndef __karma_log_h__
#define __karma_log_h__

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/time.h>

#include "karma_base.h"

static std::string log_file = "application.log" ;

enum KARMALogLevel {
	NONE=0,
	ERROR=1,
	WARNING=2,
	INFO=3,
	DEBUG1=4,
	DEBUG2=5,
	DEBUG3=6,
	DEBUG4=7	};

namespace KARMA
{
	extern KARMALogLevel verbosity_level ;
}

template <typename T>
class KARMALog
{
public:

    KARMALog() {}
    virtual ~KARMALog();

	// member function to get the stream corresponding to the user request
    std::ostringstream& get(const std::string level);
    std::ostringstream& get(unsigned int level);
    std::ostringstream& get(KARMALogLevel level);

	// flush the current stream to the console or the log file
	void flush();
	
	// Static methods to set reporting levels dynamically
    static std::string to_string(KARMALogLevel level);
    static KARMALogLevel from_string(const std::string& level);
    static KARMALogLevel from_int(unsigned int level);

	// Static Member attributes
	static KARMALogLevel reporting_level ;
	static bool use_time_info, process_all_procs ;
	
protected:

	// TODO: Replace OS with a much faster, threaded stream
    std::ostringstream os;

private:

    // KARMALog(const KARMALog&);
    // KARMALog& operator =(const KARMALog&);

};

template <typename T>
KARMALogLevel KARMALog<T>::reporting_level = DEBUG4;
template <typename T>
bool KARMALog<T>::use_time_info = false ; 
template <typename T>
bool KARMALog<T>::process_all_procs = true ;

// Inline methods for KARMALog

template <typename T>
inline
KARMALog<T>::~KARMALog()
{
	os << std::endl;
	T::output(os.str(), KARMALog<T>::process_all_procs);
}


template <typename T>
inline
std::string KARMALog<T>::to_string(KARMALogLevel level)
{
	// the first buffer = "[NONE]"
	static const char* const buffer[] = {"", "[ERROR]", "[WARNING]", "[INFO]", "[DEBUG1]", "[DEBUG2]", "[DEBUG3]", "[DEBUG4]"};
	return buffer[level];
}

template <typename T>
inline
KARMALogLevel KARMALog<T>::from_int(unsigned int level)
{
	switch (level)
	{
		case 0:
			return NONE;
		case 1:
			return ERROR;
		case 2:
			return WARNING;
		case 3:
			return INFO;
		case 4:
			return DEBUG1;
		case 5:
			return DEBUG2;
		case 6:
			return DEBUG3;
		case 7:
			return DEBUG4;
		default:
			KARMALog<T>().get(WARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
			return INFO;
	}
}

template <typename T>
inline
KARMALogLevel KARMALog<T>::from_string(const std::string& level)
{
	if (!level.compare("NONE"))
		return NONE;
	if (!level.compare("ERROR"))
		return ERROR;
	if (!level.compare("WARNING"))
		return WARNING;
	if (!level.compare("INFO"))
		return INFO;
	if (!level.compare("DEBUG1"))
		return DEBUG1;
	if (!level.compare("DEBUG2"))
		return DEBUG2;
	if (!level.compare("DEBUG3"))
		return DEBUG3;
	if (!level.compare("DEBUG4"))
		return DEBUG4;

	KARMALog<T>().get(WARNING) << "Unknown logging level '" << level << "'. Using INFO level as default.";
	return INFO;
}

inline
std::string current_time()
{
	char result[100] = {0};
	time_t tim=time(NULL);
	tm *now=localtime(&tim);
	// std::sprintf(result, "%02d:%02d", now->tm_hour, now->tm_min);
    std::stringstream oss ;
	oss << result << now->tm_hour << ":" << now->tm_min ;

	return result;
}


// Definition of Log functions for a console stream
class Log2Stream 
{
public:
    static void output(const std::string& msg, bool p_process_all_procs);
};


// Definition of Log functions for a file stream
class Log2File 
{
public:
	static std::ofstream& stream();
    static void output(const std::string& msg, bool p_process_all_procs);
private:
	static std::ofstream _file_stream ;
};

inline
std::ofstream& Log2File::stream()
{
	if (!_file_stream)
	{
		_file_stream.open(log_file.c_str()) ;	// It is not necessary to close the stream
												// because the destructor closes the open file by default.
	}

	return _file_stream;
}


// //--------------------------------------------------------------
// // Explicit specialization
// //--------------------------------------------------------------

inline
void Log2File::output(const std::string& msg, bool p_process_all_procs)
{
	if (p_process_all_procs || !KARMA::processor_id)
	{
		std::ofstream& output_stream = Log2File::stream();
		if (output_stream.good())
		{
			output_stream << "[" << KARMA::processor_id << "]: " << msg ;
			output_stream.flush() ;
		}
	}
}

inline
void Log2Stream::output(const std::string& msg, bool p_process_all_procs)
{
	if (p_process_all_procs)
		PetscPrintf(KARMA_COMM_WORLD, "[%d]: %s", KARMA::processor_id, msg.c_str()) ;
	else
		PetscPrintf(KARMA_COMM_SELF, "[%d]: %s", KARMA::processor_id, msg.c_str()) ;
}


#ifndef LOG_MAX_LEVEL
#define LOG_MAX_LEVEL 7
#endif

#define KARMA_FILE_LOG(level) \
    if (level > KARMALog<Log2File>::reporting_level || level > LOG_MAX_LEVEL) ;\
    else if (!Log2File::stream()) ; \
    else KARMALog<Log2File>().get(level)

#define KARMA_LOG(level) \
    if (level > KARMALog<Log2Stream>::reporting_level || level > LOG_MAX_LEVEL) ;\
    else KARMALog<Log2Stream>().get(level)


#endif //__karma_log_h__
