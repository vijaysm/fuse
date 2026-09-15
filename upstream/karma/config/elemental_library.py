#! /usr/bin/env python
# encoding: utf-8

import os, sys, re, subprocess
import config_utils
import load_config_variables

from subprocess import PIPE, Popen, STDOUT

def configure(conf, petsc_dictionary):

	if 'ELEMENTAL_DIR' not in os.environ:
		config_utils.print_error(conf, 'ELEMENTAL is a necessary dependency (path not specified).')
		return

	config_utils.display_header('PETSc library for parallel data-structures and algorithm')
	
	petsc_arch = ''
	
	# a simple routine to add flags to both C/C++ configs.
	def append_cxx_flags(vals):
		conf.env.append_unique('CCFLAGS', vals.split())
		conf.env.append_unique('CXXFLAGS', vals.split())
	
	if not os.path.exists(os.environ['PETSC_DIR']):
		print_error(conf, 'PETSC_DIR environment variable (dir=%s) points to invalid location' % os.environ['PETSC_DIR'], 1)
	else:
		# check for PETSC_DIR and PETSC_ARCH environmental variables
		conf.env.PETSC_DIR = os.environ['PETSC_DIR']
		conf.env.PETSC_ARCH = os.environ['PETSC_ARCH']
		petsc_arch = conf.env.PETSC_ARCH 

		conf.msg( 'PETSc directory', conf.env.PETSC_DIR )
		conf.define('PETSC_DIR', conf.env.PETSC_DIR)
	
	petsc_version_file = Popen(['/bin/cat', '%s/include/petscversion.h' % conf.env.PETSC_DIR], stdout=PIPE, stderr=STDOUT).stdout.read()
	for test in re.finditer(r'\$*PETSC_VERSION_\w+\s+\d', petsc_version_file):
		version_line = re.sub(' +',' ',test.group(0))
		templst = re.split(' ', version_line)
		
		if version_line.startswith('PETSC_VERSION_RELEASE'):
			petsc_version_release = templst[1]
		elif version_line.startswith('PETSC_VERSION_MAJOR'):
			petsc_version_major = templst[1]
		elif version_line.startswith('PETSC_VERSION_MINOR'):
			petsc_version_minor = templst[1]
		elif version_line.startswith('PETSC_VERSION_SUBMINOR'):
			petsc_version_subminor = templst[1]
		else:
			petsc_version_patch = templst[1]
	
	if petsc_version_release == '0':
		petsc_version = "%s.%s.%s-p%s-dev" % (petsc_version_major, petsc_version_minor, petsc_version_subminor, petsc_version_patch)
	else:
		petsc_version = "%s.%s.%s-p%s" % (petsc_version_major, petsc_version_minor, petsc_version_subminor, petsc_version_patch)
	conf.msg( 'PETSc version', petsc_version )
	conf.define('PETSC_VERSION', petsc_version )

	petsc_installation = 0
	if len(os.environ['PETSC_ARCH']) == 0 or os.environ['PETSC_ARCH'] == 'installed-petsc':
		petsc_installation = 1
		if os.path.exists(os.path.join(conf.env.PETSC_DIR, 'installed-petsc')):
			petsc_arch = 'installed-petsc'
		else:
			petsc_arch = ''
		
		conf.msg( 'PETSc has been configured and installed', 'installed-petsc')
		conf.define('PETSC_ARCH', petsc_arch)
		#conf.env.PETSC_ARCH = petsc_arch
	else:
		if not os.path.exists(os.path.join(conf.env.PETSC_DIR, petsc_arch)):
			print_error(conf, 'PETSC_ARCH environment variable (%s) is invalid. Check PETSc configuration.' % petsc_arch, 1)
		else:
			conf.msg( 'PETSc configured architecture', petsc_arch)
			conf.define('PETSC_ARCH', petsc_arch)
	
	#sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config'))
	#sys.path.insert(0, os.path.join(os.environ['PETSC_DIR'], 'config', 'BuildSystem'))
	
	import glob

	petsc_exists = 0 
	if conf.find_file('petsc.h', os.path.join(conf.env.PETSC_DIR, 'include')):
		#if conf.find_file('libpetsc.*', os.path.join(conf.env.PETSC_DIR, petsc_arch, 'lib')):
		if glob.glob(os.path.join(conf.env.PETSC_DIR, petsc_arch, 'lib','libpetsc.*')):
			petsc_exists = 1

	conf.define('KARMA_HAVE_PETSC', petsc_exists)
	
	if petsc_exists:
		
		print 
		
		#petsc_dictionary = {}
		if conf.find_file('petscvariables', os.path.join(conf.env.PETSC_DIR,petsc_arch,'conf')):
			conf.msg( 'Reading PETSc configuration files', os.path.join(conf.env.PETSC_DIR,'conf', 'variables'))
			load_config_variables.load_config_file(os.path.join(conf.env.PETSC_DIR,'conf', 'variables'), petsc_dictionary)
			conf.msg( '', os.path.join(conf.env.PETSC_DIR,petsc_arch,'conf', 'petscvariables'))
			load_config_variables.load_config_file(os.path.join(conf.env.PETSC_DIR,petsc_arch,'conf', 'petscvariables'), petsc_dictionary)
			
			load_config_variables.unroll_variables(conf.env, petsc_dictionary)
		
		print 
		
		conf.msg('PETSc was configured with the command', petsc_dictionary['CONFIGURE_OPTIONS'])
		
		print 
		
		sys.path.insert(0, os.path.join(conf.env.PETSC_DIR, 'conf'))
		sys.path.insert(0, os.path.join(conf.env.PETSC_DIR, petsc_arch, 'conf'))
		
		#petsc_include_dirs = Popen(['make', '-d', '-f', os.path.join(conf.env.PETSC_DIR,'makefile'), 'getincludedirs'], stdout=PIPE).stdout.read()
		
		#conf.env.append_unique('INCLUDES', os.path.join(conf.env.PETSC_DIR, 'include'))
		#conf.env.append_unique('CPPPATH', os.path.join(conf.env.PETSC_DIR, 'include'))
		conf.env.append_unique('BINDIR', os.path.join(conf.env.PETSC_DIR,'bin'))
		#conf.env.append_unique('LIBPATH', os.path.join(conf.env.PETSC_DIR,petsc_arch,'lib'))
		#conf.env.append_unique('LINKFLAGS', petsc_dictionary['PETSC_LIB_BASIC'])
		
		# PACKAGES_LIBS exists only <= v3.1. Dev version uses PETSC_WITH_EXTERNAL_LIB
		if 'PACKAGES_LIBS' in petsc_dictionary:
			conf.env.append_unique('LINKFLAGS', petsc_dictionary['PACKAGES_LIBS'].split(' '))
		else:
			conf.env.append_unique('LINKFLAGS', petsc_dictionary['PETSC_LIB'].split(' '))
		
		#petsc_dictionary['PETSC_INCLUDE']
		#petsc_dictionary['MPI_INCLUDE']
		#petsc_dictionary['PETSCFLAGS']
		
		
		append_cxx_flags(petsc_dictionary['MPI_INCLUDE'])
		# PETSC_INCLUDE exists only <= v3.1. Dev version uses PETSC_CC_INCLUDES
		if 'PETSC_INCLUDE' in petsc_dictionary:
			conf.env.append_unique('INC_PATHS', petsc_dictionary['PETSC_INCLUDE']) # for the waf preprocessor
			append_cxx_flags(petsc_dictionary['PETSC_INCLUDE'])
		else:
			conf.env.append_unique('INC_PATHS', petsc_dictionary['PETSC_CC_INCLUDES']) # for the waf preprocessor
			append_cxx_flags(petsc_dictionary['PETSC_CC_INCLUDES'])
		
		
		conf.check(header_name='petsc.h')
			
		conf.check(msg='Checking for PETSc library', lib='petsc', uselib_store='PETSC', mandatory=True)
		
		# check if PETSc is built as a static library
		suffix_key = 'SL_LINKER_SUFFIX'
		#if 'SL_LINKER_SUFFIX' not in petsc_dictionary:
		#	suffix_key = 'AR_LIB_SUFFIX'
		if petsc_dictionary['BUILDSHAREDLIB'] == 'no':
			suffix_key = 'AR_LIB_SUFFIX'
		if conf.find_file('libpetsc.%s'%petsc_dictionary[suffix_key], os.path.join(conf.env.PETSC_DIR,petsc_arch,'lib')):
			conf.env['LIBPATH_PETSC'] = os.path.join(conf.env.PETSC_DIR,petsc_arch,'lib')
			conf.env['LIB_PETSC'] = 'petsc'
		else:
			conf.env['STATICLIB_PETSC'] = 'libpetsc.a'
			conf.env['STATICLIBPATH_PETSC'] = os.path.join(conf.env.PETSC_DIR,petsc_arch,'lib')
			conf.env.append_unique('LINKFLAGS', os.path.join(conf.env.PETSC_DIR,petsc_arch,'lib', 'libpetsc.a'))
			
		conf.check_cxx(fragment='''
				#include <iostream>
				#include "petsc.h"
				#include "petscvec.h"
				int main() { Vec a; std::cout << "1" ; return 0; } ''',
			#define_name = "KARMA_HAVE_PETSC",
			execute     = True,
			#define_ret  = True,
			mandatory=True,
			msg         = "Linking with PETSc library")
			
	else:
		conf.fatal('PETSC_DIR environment variable (dir=%s) points to invalid location' % os.environ['PETSC_DIR'] )

