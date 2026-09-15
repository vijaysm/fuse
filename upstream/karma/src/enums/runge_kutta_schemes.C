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


#include "runge_kutta_schemes.h"

#define X(a,b) #a,
const std::string enum_traits<KARMA::TEMPORAL::RKCategory>::names[] = { RK_CATEGORIES } ;
const std::string enum_traits<KARMA::TEMPORAL::RKType>::names[] = { RK_TYPES } ;
#undef X

#define X(a,b) b,
const int enum_traits<KARMA::TEMPORAL::RKCategory>::enumerators[] = { RK_CATEGORIES };
const int enum_traits<KARMA::TEMPORAL::RKType>::enumerators[] = { RK_TYPES };
#undef X

const std::string enum_traits<KARMA::TEMPORAL::RKCategory>::enum_type = "KARMA::TEMPORAL::RKCategory" ;
const std::string enum_traits<KARMA::TEMPORAL::RKType>::enum_type = "KARMA::TEMPORAL::RKType" ;
