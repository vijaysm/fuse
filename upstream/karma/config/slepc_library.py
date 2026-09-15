#! /usr/bin/env python
# encoding: utf-8

import os, subprocess, re
import config_utils
import load_config_variables

from subprocess import PIPE, Popen, STDOUT

def configure(conf, petsc_dictionary):

	from Options import options as opts 
	
	if len(opts.slepc_dir) == 0:
		config_utils.print_error(conf, 'SLEPc path not specified. Skipping ...')
		return

	config_utils.display_header('SLEPc library for eigenvalue computation')
	
	slepc_arch = ''
	
	# a simple routine to add flags to both C/C++ configs.
	def append_cxx_flags(vals):
		conf.env.append_unique('CCFLAGS', vals.split())
		conf.env.append_unique('CXXFLAGS', vals.split())
	
	
	if not os.path.exists(opts.slepc_dir):
		config_utils.print_error(conf, 'SLEPC_DIR environment variable (dir=%s) points to invalid location' % opts.slepc_dir, 1)
	else:
		# get and store the SLEPC_DIR variable from the environment
		conf.env.SLEPC_DIR = opts.slepc_dir

		conf.msg( 'SLEPc directory', conf.env.SLEPC_DIR)
		slepc_arch = conf.env.PETSC_ARCH
		
	
	slepc_version_file = Popen(['/bin/cat', '%s/include/slepcversion.h' % conf.env.SLEPC_DIR], stdout=PIPE, stderr=STDOUT).stdout.read()
	for test in re.finditer(r'\$*SLEPC_VERSION_\w+\s+\d', slepc_version_file):
		version_line = re.sub(' +',' ',test.group(0))
		templst = re.split(' ', version_line)
		
		if version_line.startswith('SLEPC_VERSION_RELEASE'):
			slepc_version_release = int(templst[1])
		elif version_line.startswith('SLEPC_VERSION_MAJOR'):
			slepc_version_major = int(templst[1])
		elif version_line.startswith('SLEPC_VERSION_MINOR'):
			slepc_version_minor = int(templst[1])
		elif version_line.startswith('SLEPC_VERSION_SUBMINOR'):
			slepc_version_subminor = int(templst[1])
		else:
			slepc_version_patch = int(templst[1])
	
	if slepc_version_release == '0':
		slepc_version = "%d.%d.%d-p%d-dev" % (slepc_version_major, slepc_version_minor, slepc_version_subminor, slepc_version_patch)
	else:
		slepc_version = "%d.%d.%d-p%d" % (slepc_version_major, slepc_version_minor, slepc_version_subminor, slepc_version_patch)
	conf.msg( 'SLEPc version', slepc_version )
	
	# check if slepc is installed
	slepc_installation = 0
	if len(os.environ['PETSC_ARCH']) == 0 or os.environ['PETSC_ARCH'] == 'installed-petsc':
		slepc_installation = 1
		if os.path.exists(os.path.join(conf.env.SLEPC_DIR, 'installed-petsc')):
			slepc_arch = 'installed-petsc'
			conf.msg( 'SLEPc has been configured with installed PETSc', os.environ['PETSC_DIR'])
			conf.msg( 'SLEPc configured architecture', slepc_arch)
		else:
			slepc_arch = ''
			conf.msg( 'SLEPc configured and installed architecture', 'installed-slepc')
		
		conf.env['SLEPC_ARCH'] = slepc_arch
	else:
		if not os.path.exists(os.path.join(conf.env.SLEPC_DIR, slepc_arch)):
			config_utils.print_error(conf, 'PETSC_ARCH environment variable (%s) is invalid. Check SLEPc configuration.' % slepc_arch, 1)
		else:
			conf.msg( 'SLEPc configured architecture', slepc_arch)
			
	import glob
	
	slepc_exists = 0 
	if conf.find_file('slepc.h', os.path.join(conf.env.SLEPC_DIR, 'include')):
		#if conf.find_file('libslepc.*', os.path.join(conf.env.PETSC_DIR, slepc_arch, 'lib')):
		if glob.glob(os.path.join(conf.env.SLEPC_DIR,slepc_arch, 'lib','libslepc.*')):
			slepc_exists = 1
		
	conf.define('KARMA_HAVE_SLEPC', slepc_exists)
	conf.define('SLEPC_DIR', conf.env.SLEPC_DIR)
	conf.define('SLEPC_ARCH', slepc_arch)
	conf.define('KARMA_DETECTED_SLEPC_VERSION', slepc_version )
	conf.define('KARMA_DETECTED_SLEPC_VERSION_MAJOR', slepc_version_major )
	conf.define('KARMA_DETECTED_SLEPC_VERSION_MINOR', slepc_version_minor )
	conf.define('KARMA_DETECTED_SLEPC_VERSION_SUBMINOR', slepc_version_subminor )
	conf.define('KARMA_DETECTED_SLEPC_VERSION_PATCH', slepc_version_patch )
	
	if slepc_exists:
	
		slepc_varfile = 'slepc_common_variables'
		if (slepc_version_major == 3 and slepc_version_minor > 1) or slepc_version_release == 0:
			slepc_varfile = 'slepc_variables'
		
		print 
		if slepc_installation:
			conf.env.append_unique('CPPPATH', os.path.join(conf.env.SLEPC_DIR,'include'))
			#conf.env.append_unique('LIBPATH', os.path.join(conf.env.SLEPC_DIR,'lib'))
		else:
			conf.env.append_unique('CPPPATH', os.path.join(conf.env.SLEPC_DIR,slepc_arch,'include'))
			#conf.env.append_unique('LIBPATH', os.path.join(conf.env.SLEPC_DIR,slepc_arch,'lib'))
			append_cxx_flags('-I%s' % os.path.join(conf.env.SLEPC_DIR,slepc_arch,'include'))
		
		slepc_dictionary = petsc_dictionary.copy()
		if conf.find_file('slepcvariables', os.path.join(conf.env.SLEPC_DIR,slepc_arch,'conf')):
			conf.msg( 'Reading SLEPc configuration files', os.path.join(conf.env.SLEPC_DIR,'conf', slepc_varfile))
			load_config_variables.load_config_file(os.path.join(conf.env.SLEPC_DIR,'conf', slepc_varfile), slepc_dictionary)
			if slepc_version_major == 3 and slepc_version_minor < 1:
				conf.msg( '', os.path.join(conf.env.SLEPC_DIR,slepc_arch,'conf', 'slepcvariables'))
				load_config_variables.load_config_file(os.path.join(conf.env.SLEPC_DIR,slepc_arch,'conf', 'slepcvariables'), slepc_dictionary)
			
			#print slepc_dictionary
			load_config_variables.unroll_variables(conf.env, slepc_dictionary)
			#print slepc_dictionary
		else:
			print_error('Could not find slepcvariables needed for configuring SLEPc. Please check path.')
		
		print 
		
		# unfortunately, SLEPc doesn't store its configure command as part of the variables
		# so lets poke into configure.log and extract this info
		slepc_configure_command = Popen(['grep', 'Configure Options:', os.path.join(conf.env.SLEPC_DIR,slepc_arch,'conf','configure.log')], stdout=PIPE).stdout.read()
		
		if "config/configure.py" in slepc_configure_command:
			slepc_configure_command = slepc_configure_command.replace("Configure Options: config/configure.py", '').strip()
			if len(slepc_configure_command) == 0:
				slepc_configure_command = "--default-options"
		else:
			slepc_configure_command = slepc_configure_command.replace('Configure Options: ./configure', '').strip()

		conf.msg('SLEPc was configured with the command', slepc_configure_command)
		
		conf.env['LINKFLAGS'] = slepc_dictionary['SLEPC_LIB'].split(' ') + conf.env['LINKFLAGS']
		conf.env['LINKFLAGS'] = config_utils.unique(conf.env['LINKFLAGS'])
		
		append_cxx_flags(slepc_dictionary['CCPPFLAGS'])
		#append_cxx_flags(slepc_dictionary['FC_INCLUDES'])
		
		conf.check(header_name='slepc.h', define_name='KARMA_HAVE_SLEPC_H')

		conf.check(msg='Checking for SLEPc library', lib='slepc', uselib_store='SLEPC', mandatory=True)
		
		suffix_key = 'SL_LINKER_SUFFIX'
		#if 'SL_LINKER_SUFFIX' not in petsc_dictionary:
		#	suffix_key = 'AR_LIB_SUFFIX'
		if petsc_dictionary['BUILDSHAREDLIB'] == 'no':
			suffix_key = 'AR_LIB_SUFFIX'
		
		# check if SLEPc is built as a static library
		if conf.find_file('libslepc.%s'%petsc_dictionary[suffix_key], os.path.join(conf.env.SLEPC_DIR,slepc_arch,'lib')):
			conf.env['LIBPATH_SLEPC'] = os.path.join(conf.env.SLEPC_DIR,slepc_arch,'lib')
			conf.env['LIB_SLEPC'] = 'slepc'
		else:
			conf.env['STATICLIB_SLEPC'] = slepc_dictionary['SLEPC_LIB']
			conf.env['STATICLIBPATH_SLEPC'] = os.path.join(conf.env.SLEPC_DIR,slepc_arch,'lib')
			conf.env.append_unique('LINKFLAGS', os.path.join(conf.env.SLEPC_DIR,slepc_arch,'lib', slepc_dictionary['SLEPC_LIB']))
		
		conf.check_cxx(fragment='''
				#include <iostream>
				#include "slepc.h"
				int main() { EPS a; std::cout << 1 ; return 0; } ''',
			#define_name = "KARMA_HAVE_SLEPC",
			execute     = True,
			#define_ret  = True,
			uselib_store='SLEPC', 
			msg         = "Linking with SLEPc library")
				
	else:
		conf.fatal('SLEPC_DIR environment variable (dir=%s) points to invalid location' % opts.slepc_dir )

