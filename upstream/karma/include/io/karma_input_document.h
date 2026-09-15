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



#ifndef __karma_input_document_h__
#define __karma_input_document_h__

// Standard C++ includes
#include <string>

// TinyXPath include files
#include "xpath_static.h"

class KARMAInputDocument
{
public:
	
	KARMAInputDocument (const std::string& filename) ;
	
	~KARMAInputDocument () ;
	
	TIXML_STRING get_string(const std::string& xpath_str) ;
	
	double get_double(const std::string& xpath_str) ;
	
	int get_int(const std::string& xpath_str) ;
	
	bool get_bool(const std::string& xpath_str) ;
	
	unsigned int get_count(const std::string& xpath_str) ;
	
	bool get_string(const std::string& xpath_str, TIXML_STRING& data) ;
	
	bool get_double(const std::string& xpath_str, double& data) ;
	
	bool get_int(const std::string& xpath_str, int& data) ;
	
	bool get_bool(const std::string& xpath_str, bool& data) ;
	
	bool get_count(const std::string& xpath_str, unsigned int& data) ;
	
private:

	std::string _filename ;
	
	TiXmlDocument* _doc ;
	const TiXmlElement* _main_element ;
};


#endif // #define __karma_input_document_h__
