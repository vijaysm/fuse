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



#ifndef __karma_parameters_h__
#define __karma_parameters_h__

// C++ includes
#include <cassert>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <typeinfo>

#ifndef libmesh_assert
#define libmesh_assert(condition) assert(condition)
#endif
#ifndef libmesh_error
#define libmesh_error() std::abort()
#endif


/**
 * This class provides the ability to map between
 * arbitrary, user-defined strings and several data
 * types.  This can be used to provide arbitrary
 * user-specified options.
 *
 * Consider this as a loosely typed cousin to std::map.
 *
 * \author $Author: vijaysm $
 * \date $Date: 2011-02-04 14:06:06 -0600 (Fri, 04 Feb 2011) $
 * \version $Revision: 384 $
 */

// ------------------------------------------------------------
// KARMAParameters class definition
class KARMAParameters 
{
public:

  /**
   * Default constructor.  Does nothing.
   */
  KARMAParameters () {};

  /**
   * Copy constructor.
   */
  KARMAParameters (const KARMAParameters&);

  /**
   * Destructor.  Clears any allocated memory.
   */
  ~KARMAParameters ();

  /**
   * Assignment operator.
   */
  KARMAParameters& operator= (const KARMAParameters&);
  
  /**
   * @returns \p true if a parameter of type \p T
   * with a specified name exists, \p false otherwise.
   */
  template <typename T>
  bool have_parameter (const std::string&) const;

  /**
   * @returns a constant reference to the specified parameter
   * value.  Requires, of course, that the parameter exists.
   */
  template <typename T>
  const T& get (const std::string&) const;

  /**
   * @returns a writeable reference to the specified parameter.
   * This method will create the parameter if it does not exist,
   * so it can be used to define parameters which will later be
   * accessed with the \p get() member.
   */
  template <typename T>
  T& set (const std::string&);

  /**
   * Removes the specified parameter from the list, if it exists.
   */
  void remove (const std::string&);

  /**
   * @returns the total number of parameters.
   */
  unsigned int n_parameters () const { return _values.size(); }

  /**
   * @returns the number of parameters of the requested type.
   */
  template <typename T>
  unsigned int n_parameters () const;
  
  /**
   * Clears internal data structures & frees any allocated memory.
   */
  void clear ();

  /**
   * Prints the contents to the specified stream.
   */
  void print (std::ostream& os=std::cout) const;
  
private:

  /**
   * Abstract definition of a parameter value.
   */
  class Value
  {
  public:

    /**
     * Destructor.
     */
    virtual ~Value() {};

    /**
     * String identifying the type of parameter stored.
     * Must be reimplemented in derived classes.
     */
    virtual std::string type () const = 0;

    /**
     * Prints the parameter value to the specified stream.
     * Must be reimplemented in derived classes.
     */
    virtual void print(std::ostream&) const = 0;

    /**
     * Clone this value.  Useful in copy-construction.
     * Must be reimplemented in derived classes.
     */
    virtual Value* clone () const = 0;
  };

public:

  /**
   * Concrete definition of a parameter value
   * for a specified type.
   */
  template <typename T>
  class Parameter : public Value
  {
  public:

    /**
     * @returns a read-only reference to the parameter value.
     */
    const T& get () const { return _value; }
    
    /**
     * @returns a writeable reference to the parameter value.
     */
    T& set () { return _value; }

    /**
     * String identifying the type of parameter stored.
     */
    virtual std::string type () const;

    /**
     * Prints the parameter value to the specified stream.
     */
    virtual void print(std::ostream&) const;

    /**
     * Clone this value.  Useful in copy-construction.
     */
    virtual Value* clone () const; 
    
  private:

    /**
     * Stored parameter value.
     */
    T _value;
  };

  /**
   * Parameter map iterator.
   */
  typedef std::map<std::string, Value*>::iterator iterator;

  /**
   * Constant parameter map iterator.
   */
  typedef std::map<std::string, Value*>::const_iterator const_iterator;

  /**
   * Iterator pointing to the beginning of the set of parameters.
   */
  iterator begin();

  /**
   * Iterator pointing to the beginning of the set of parameters.
   */
  const_iterator begin() const;

  /**
   * Iterator pointing to the end of the set of parameters
   */
  iterator end();

  /**
   * Iterator pointing to the end of the set of parameters
   */
  const_iterator end() const;

private:

  /**
   * Data structure to map names with values.
   */
  std::map<std::string, Value*> _values;

};



// ------------------------------------------------------------
// KARMAParameters::Parameter<> class inline methods
template <typename T>
inline
std::string KARMAParameters::Parameter<T>::type () const
{
  return typeid(T).name();
}



template <typename T>
inline
void KARMAParameters::Parameter<T>::print (std::ostream& os) const
{
  os << _value;
}



template <typename T>
inline
KARMAParameters::Value* KARMAParameters::Parameter<T>::clone () const
{
  // No good for Solaris C++! - BSK
//  KARMAParameters::Parameter<T>
//    *copy = new KARMAParameters::Parameter<T>;
  Parameter<T> *copy = new Parameter<T>;

  libmesh_assert (copy != NULL);
  
  copy->_value = _value;

  return copy;
}


// ------------------------------------------------------------
// KARMAParameters class inline methods
inline
void KARMAParameters::clear () // since this is inline we must define it
{                         // before its first use (for some compilers)
  while (!_values.empty())
    {
      KARMAParameters::iterator it = _values.begin();
      
      delete it->second;      
      it->second = NULL;
      
      _values.erase(it);
    }
}



inline
KARMAParameters& KARMAParameters::operator= (const KARMAParameters& rhs)
{
  this->clear();
  
  for (KARMAParameters::const_iterator it = rhs._values.begin();
       it != rhs._values.end(); ++it)
    _values[it->first] = it->second->clone();
  
  return *this;
}



inline
KARMAParameters::KARMAParameters (const KARMAParameters& p)
{
  *this = p;
}



inline
KARMAParameters::~KARMAParameters ()
{
  this->clear ();
}



inline
void KARMAParameters::print (std::ostream& os) const
{
	KARMAParameters::const_iterator it = _values.begin();

	os << "Name\t Type\t Value\n" 
	   << "---------------------\n";
	
	while (it != _values.end())
	{
		os << " "   << it->first
		<< "\t " << it->second->type()
		<< "\t ";   it->second->print(os);      
		os << '\n';

		++it;
	}
}



// Declare this now that Paramers::print() is defined.
// By declaring this early we can use it in subsequent
// methods.  Required for gcc-4.0.2 -- 11/30/2005, BSK
inline
std::ostream& operator << (std::ostream& os, const KARMAParameters& p)
{
	p.print(os);
	return os;
}



template <typename T>
inline
bool KARMAParameters::have_parameter (const std::string& name) const
{
	KARMAParameters::const_iterator it = _values.find(name);

	if (it != _values.end())
	if (dynamic_cast<const Parameter<T>*>(it->second) != NULL)
	  return true;

	return false;
}



template <typename T>
inline
const T& KARMAParameters::get (const std::string& name) const
{
	KARMAParameters::const_iterator it = _values.find(name);

	if (it == _values.end())
	{
      std::cerr << "ERROR: no "
		<< typeid(T).name()
		<< " parameter named \""
		<< name << "\":" << std::endl
		<< *this;
      
      libmesh_error();
    }

	libmesh_assert (it != _values.end());
	libmesh_assert (it->second != NULL);
	libmesh_assert (dynamic_cast<const Parameter<T>*>(it->second) != NULL);

	return dynamic_cast<Parameter<T>*>(it->second)->get();
}



template <typename T>
inline
T& KARMAParameters::set (const std::string& name)
{
  Parameter<T>* param = NULL;
  
	KARMAParameters::const_iterator it = _values.find(name);

	if (it == _values.end())
		_values[name] = new Parameter<T>;

	param = dynamic_cast<Parameter<T>*>(_values[name]);

	libmesh_assert (param != NULL);

	return param->set();
}



inline
void KARMAParameters::remove (const std::string& name)
{
  KARMAParameters::iterator it = _values.find(name);

  if (it != _values.end())
    {
      delete it->second;
      it->second = NULL;

      _values.erase(it);
    }
}



template <typename T>
inline
unsigned int KARMAParameters::n_parameters () const
{
	// TODO: Minor Optimization
	// Can't we just maintain a counter at every add, remove ?!
	// That should be a lot more efficient that this.
	
	unsigned int cnt = 0;  

	KARMAParameters::const_iterator       it  = _values.begin();
	const KARMAParameters::const_iterator end = _values.end();

	for (; it != end; ++it)
		if (dynamic_cast<Parameter<T>*>(it->second) != NULL)
			cnt++;

	return cnt;	 
}

inline
KARMAParameters::iterator KARMAParameters::begin()
{
  return _values.begin();
}

inline
KARMAParameters::const_iterator KARMAParameters::begin() const
{
  return _values.begin();
}

inline
KARMAParameters::iterator KARMAParameters::end()
{
  return _values.end();
}

inline
KARMAParameters::const_iterator KARMAParameters::end() const
{
  return _values.end();
}

#endif // #define __karma_parameters_h__
