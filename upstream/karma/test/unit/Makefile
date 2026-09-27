#// KARMA: K(C)ode for Analysis of Reactor and other Multiphysics Applications
#// Copyright (C) 2007-2011  Vijay S Mahadevan
#//
#// This program is free software; you can redistribute it and/or modify
#// it under the terms of the GNU Lesser General Public License as published
#// by the Free Software Foundation; either version 3 of the License, or
#// (at your option) any later version.
#//
#// This program is distributed in the hope that it will be useful,
#// but WITHOUT ANY WARRANTY; without even the implied warranty of
#// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#// GNU General Public License for more details.
#//
#// You should have received a copy of the GNU Lesser General Public License
#// along with this program; if not, write to the Free Software Foundation,
#// Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#//

# include the common options determined by configure for KARMA.
include ${KARMA_DIR}/KARMA.common

###############################################################################
# File management.  This is where the source, header, and object files are defined
# for all the test code.

TEST_DIRECTORY	:= unit


# Include and Source files in KARMA LIBRARY to test each physics;
KARMA_TEST_INCLUDE  =	-I${KARMA_TEST_DIR}/$(TEST_DIRECTORY)/ 

KARMA_TEST_SRC =	$(wildcard ${KARMA_TEST_DIR}/$(TEST_DIRECTORY)/*.C)

###############################################################################

.PHONY: clean distclean


###############################################################################
# INDIVIDUAL TEST SUITES FOR EACH PHYSICS
###############################################################################

target 	   := bin/tests-$(METHOD)

all:: hello_test package_test vector_test matrix_test

hello_test: hello_test.C $(KARMA_LIBRARY)
	@echo "Building test: "bin/$@"..."
	@$(KARMA_CXX) $(KARMA_INCLUDE) $(KARMA_TEST_INCLUDE) $(KARMA_CXXFLAGS) $< -o bin/$@ $(KARMA_LIBS)

package_test: karma_package_test.C $(KARMA_LIBRARY)
	@echo "Building test: "bin/$@"..."
	@$(KARMA_CXX) $(KARMA_INCLUDE) $(KARMA_TEST_INCLUDE) $(KARMA_CXXFLAGS) $< -o bin/$@ $(KARMA_LIBS)

vector_test: karma_vector_test.C $(KARMA_LIBRARY)
	@echo "Building test: "bin/$@"..."
	@$(KARMA_CXX) $(KARMA_INCLUDE) $(KARMA_TEST_INCLUDE) $(KARMA_CXXFLAGS) $< -o bin/$@ $(KARMA_LIBS)

matrix_test: karma_matrix_test.C $(KARMA_LIBRARY)
	@echo "Building test: "bin/$@"..."
	@$(KARMA_CXX) $(KARMA_INCLUDE) $(KARMA_TEST_INCLUDE) $(KARMA_CXXFLAGS) $< -o bin/$@ $(KARMA_LIBS)

# Useful rules.
clean:
	@rm -f bin/* $(KARMA_TEST_OBJS) $(subst -I,$, $(addsuffix *.h~, $(KARMA_TEST_INCLUDE))) $(KARMA_TEST_SRC:.C=.C~)

echo:
	@echo $(libmesh_CXXFLAGS)
	@echo ${KARMA_DIR}
	@echo ${KARMA_SRC}
	@echo $(KARMA_OBJS)
	@echo %.$(obj-suffix)
	@echo $(KARMA_LIBS)
	@echo $(KARMA_INCLUDE)
	@echo ${KARMA_TEST_DIR}
	@echo ${KARMA_TEST_SRC}
	@echo $(KARMA_TEST_INCLUDE)

%.$(obj-suffix) : %.C
	@echo "Compiling C++ (in "$(mode)" mode) "$<"..."
	@$(libmesh_CXX) $(libmesh_CXXFLAGS) $(libmesh_INCLUDE) -c $< -o $@

#
# In the contrib/bin directory, we run the test_headers.sh shell
# script.  This is a make rule for those tests.
#
$(KARMA_TEST_OBJS) : ${KARMA_TEST_SRC}
	$(libmesh_CXX) $(libmesh_CXXFLAGS) $(libmesh_INCLUDE) -c $< -o $@

#
# Dependencies
#
dependencies:
	touch Make.dep
	@rm Make.dep
	@$(perl) ${KARMA_DIR}/contrib/bin/make_dependencies.pl -I. $(foreach i, $(wildcard ${KARMA_DIR}/include/*), -I$(i)) "-S\$$(obj-suffix)" ${KARMA_TEST_SRC} > Make.dep
	@echo "Updated Dependencies."


###############################################################################
# Build rules

##################################
# C++ rules                      #
##################################

#
# How to compile C++
#
%.$(obj-suffix) : %.C
	@echo "Compiling C++ (in "$(mode)" mode) "$<"..."
ifeq ($(KARMA_TEST_FLAG),yes)
	@$(KARMA_CXX) $(KARMA_CXXFLAGS) $(KARMA_INCLUDE) -c $< -o $@
else
	@$(KARMA_CXX) $(KARMA_CXXFLAGS) $(KARMA_LIBCXXFLAGS) $(KARMA_INCLUDE) -c $< -o $@
endif

###############################################################################

# include the dependency list
include ${KARMA_DIR}/Make.dep

