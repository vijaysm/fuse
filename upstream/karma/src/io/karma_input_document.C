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

// C++ includes

// KARMA includes
#include "karma.h"
#include "karma_error.h"
#include "karma_input_document.h"

// We need this definition so that std::string can be used  
// in place of TIXML_STRING.
#define TIXML_USE_STL

KARMAInputDocument::KARMAInputDocument (const std::string& filename) :
	_filename(filename)
{
	_doc = new TiXmlDocument() ;
	
	if (! _doc->LoadFile(filename.c_str()))
	{
		// Failed to load file successfully;
		// Either file could not be found or the format is not XML compliant
		std::cerr << "Can't find " << filename << " or it is not standard compliant XML document\n" ;
		karma_error() ;
	}
	
	// _doc->Print() ;
	
	_main_element = _doc -> RootElement () ;
}

KARMAInputDocument::~KARMAInputDocument ()
{
	_main_element = NULL ;
	
	delete _doc ;
}
	
TIXML_STRING KARMAInputDocument::get_string(const std::string& xpath_str)
{
	return TinyXPath::S_xpath_string (_main_element, xpath_str.c_str());
}

bool KARMAInputDocument::get_string(const std::string& xpath_str, TIXML_STRING& str_temp)
{
	bool result = TinyXPath::o_xpath_string (_main_element, xpath_str.c_str(), str_temp);
	
	if (!result)
	{
		std::cerr << "\nInvalid XPath string input. Target 'String' not found." ;
		std::cerr << "\nCheck XPath = " << xpath_str ;
		karma_error() ;
	}
	
	return result ;
	
}

double KARMAInputDocument::get_double(const std::string& xpath_str)
{
	return TinyXPath::d_xpath_double (_main_element, xpath_str.c_str());
}

bool KARMAInputDocument::get_double(const std::string& xpath_str, double& d_temp)
{
	bool result = TinyXPath::o_xpath_double (_main_element, xpath_str.c_str(), d_temp);
	
	if (!result)
	{
		std::cerr << "\nInvalid XPath string input. Target 'Double' not found." ;
		std::cerr << "\nCheck XPath = " << xpath_str ;
		karma_error() ;
	}
	
	return result ;
}

int KARMAInputDocument::get_int(const std::string& xpath_str)
{
	return TinyXPath::i_xpath_int (_main_element, xpath_str.c_str()) ;
}

bool KARMAInputDocument::get_int(const std::string& xpath_str, int& i_temp)
{
	bool result = TinyXPath::o_xpath_int (_main_element, xpath_str.c_str(), i_temp);
	
	if (result)
	{
		std::cerr << "\nInvalid XPath string input. Target 'Integer' not found." ;
		std::cerr << "\nCheck XPath = " << xpath_str ;
		karma_error() ;
	}
	
	return result ;
}

bool KARMAInputDocument::get_bool(const std::string& xpath_str)
{
	std::stringstream str_temp ;
	
	str_temp << xpath_str << "=\"true\" or " << xpath_str << "=\"TRUE\" or " << xpath_str << "=1"  ;
	
	return TinyXPath::o_xpath_bool (_main_element, str_temp.str().c_str()) ;
}

bool KARMAInputDocument::get_bool(const std::string& xpath_str, bool& b_temp)
{
	std::stringstream str_temp ;
	
	str_temp << xpath_str << "=\"true\" or " << xpath_str << "=\"TRUE\" or " << xpath_str << "=1"  ;
	
	bool result = TinyXPath::o_xpath_bool (_main_element, str_temp.str().c_str(), b_temp);
	
	if (result)
	{
		std::cerr << "\nInvalid XPath string input. Target 'Bool' not found." ;
		std::cerr << "\nCheck XPath = " << xpath_str ;
		karma_error() ;
	}
	
	return result ;
}

unsigned int KARMAInputDocument::get_count(const std::string& xpath_str)
{
	std::stringstream str_temp ;
	
	str_temp << "count( " << xpath_str << ")" ;
	
	return static_cast<unsigned int>(get_int(str_temp.str())) ;
}

bool KARMAInputDocument::get_count(const std::string& xpath_str, unsigned int& ui_temp)
{
	std::stringstream str_temp ;
	
	str_temp << "count( " << xpath_str << ")" ;
	
	int i_temp ;
	bool result = get_int(str_temp.str(), i_temp) ;
	
	ui_temp = static_cast<unsigned int>(i_temp) ;
	
	return result ;
}
