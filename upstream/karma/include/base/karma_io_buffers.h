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



#include <iostream>
#include <fstream>

// standard_buffer forwards all output to cout
class cout_proxy : public std::streambuf 
{
protected:
	// write a string s of length n to standard
	// output
	virtual std::streamsize xsputn (const char_type* s, std::streamsize n) 
	{
		std::cout.write(s, n);
		return n;
	}

};

class static_logger
{
public:
	static_logger() : 
			logfile("error.log", std::ios::out)
	{	}
	
	~static_logger()
	{
		logfile.close();
	}
	
	// the output log file
	std::ofstream logfile;
	
};

// logbuf forwards all output to cerr and logfile
class cerr_proxy : public std::streambuf 
{
protected:

	// write a string s of length n to standard
	// error and a log file
	virtual std::streamsize xsputn (const char_type* s, std::streamsize n) 
	{
		std::cerr.write(s, n);
		proxy.logfile.write(s, n);
		return n;
	}
	
	static static_logger proxy ;

};

// cerr_proxy::static_logger proxy ;
