#! /usr/bin/env python
# encoding: utf-8

import os, subprocess, re
import config_utils
import load_config_variables

from subprocess import Popen, PIPE

def configure(conf, dictionary):

	from Options import options as opts 
	
	if len(opts.libmesh_dir) == 0:
		config_utils.print_error(conf, 'libMesh path not specified. Skipping ...')
		return

	config_utils.display_header('libMesh Finite Element library')
	
	dbg = conf.all_envs['debug']
	rel = conf.all_envs['release']
	# a simple routine to add flags to both C/C++ configs.
	def append_cxx_flags(vals, variant=0):
		if variant == 1:
			# only debug variant
			dbg.append_unique('CCFLAGS', vals.split())
			dbg.append_unique('CXXFLAGS', vals.split())
		elif variant == 2:
			# only release variant
			rel.append_unique('CCFLAGS', vals.split())
			rel.append_unique('CXXFLAGS', vals.split())
		else:
			# both variants
			dbg.append_unique('CCFLAGS', vals.split())
			dbg.append_unique('CXXFLAGS', vals.split())
			rel.append_unique('CCFLAGS', vals.split())
			rel.append_unique('CXXFLAGS', vals.split())
			conf.env.append_unique('CCFLAGS', vals.split())
			conf.env.append_unique('CXXFLAGS', vals.split())

	if not os.path.exists(opts.libmesh_dir):
		config_utils.print_error(conf, 'LIBMESH_DIR environment variable (dir=%s) points to invalid location' % opts.libmesh_dir, 1)
	else:
		# get and store the LIBMESH_DIR variable from the environment
		conf.env.LIBMESH_DIR = opts.libmesh_dir

		conf.msg( 'libMesh directory', conf.env.LIBMESH_DIR)
	
	import glob
	
	libmesh_exists = 0 
	if conf.find_file('libmesh.h', os.path.join(conf.env.LIBMESH_DIR, 'include', 'base')):
		libmesh_exists = 1

	conf.define('KARMA_HAVE_LIBMESH', libmesh_exists)
	conf.define('LIBMESH_DIR', conf.env.LIBMESH_DIR)
	
	print
	if libmesh_exists:
	
		if os.path.exists(os.path.join(conf.env.LIBMESH_DIR,'Make.common')):
		
			libmesh_dictionary = dictionary.copy()
			conf.msg( 'Reading libMesh configuration files', os.path.join(conf.env.LIBMESH_DIR,'Make.common'))
			#load_config_variables.load_config_file(os.path.join(conf.env.LIBMESH_DIR,'Make.common'), libmesh_dictionary)
			libmesh_dictionary_tmp = load_config_variables.get_all_makefile_variables(os.path.join(conf.env.LIBMESH_DIR,'Make.common'))
			
			libmesh_dictionary = config_utils.append_unique(dictionary, libmesh_dictionary_tmp)
			
			#print libmesh_dictionary
			
			conf.env['METHOD'] = os.getenv('METHOD', 'opt')
			conf.define('LIBMESH_ARCHITECTURE', '%s_%s'%(libmesh_dictionary['hosttype'], conf.env.METHOD))
			conf.msg('libMesh configured architecture', '%s_%s'%(libmesh_dictionary['hosttype'], conf.env.METHOD))

			print
			# unfortunately, libMesh doesn't store its configure command as part of the variables
			# so just like SLEPc, lets poke into configure.log and extract this info
			libmesh_configure_command = Popen(['grep', '$ ./configure', os.path.join(conf.env.LIBMESH_DIR,'config.log')], stdout=PIPE).stdout.read()
			conf.msg('libMesh was configured with the command', libmesh_configure_command.replace('$ ./configure', '').strip())

		else:
			print_error('libMesh has not been configured correctly. Please check path.')

		# unfortunately, libMesh doesn't store its the variables unrolled.
		# use libmesh-config to get the necessary info
		# so lets call the config command directly to extract this info
		libmesh_cppflags = Popen([os.path.join(conf.env.LIBMESH_DIR,'contrib','bin','libmesh-config'), '--cppflags'], stdout=PIPE).stdout.read()
		libmesh_cxxflags = Popen([os.path.join(conf.env.LIBMESH_DIR,'contrib','bin','libmesh-config'), '--cxxflags'], stdout=PIPE).stdout.read()
		libmesh_include = Popen([os.path.join(conf.env.LIBMESH_DIR,'contrib','bin','libmesh-config'), '--include'], stdout=PIPE).stdout.read()
		libmesh_ldflags = Popen([os.path.join(conf.env.LIBMESH_DIR,'contrib','bin','libmesh-config'), '--ldflags'], stdout=PIPE).stdout.read()
		
		libmesh_cppflags = libmesh_cppflags.strip()
		libmesh_cxxflags = libmesh_cxxflags.strip()
		libmesh_include_dirs = libmesh_include.replace('-I','').strip()
		libmesh_ldflags = libmesh_ldflags.strip()
		
		#print 'cppflags %s' % libmesh_cppflags
		#print 'cxxflags %s' % libmesh_cxxflags
		#print 'include %s' % libmesh_include
		#print 'include dirs %s' % libmesh_include_dirs
		#print 'ldflags %s' % libmesh_ldflags		
		
		print 
			
		# Read Make.common and get all the necessary variables
		conf.env.append_unique('INC_PATHS', libmesh_include_dirs) 
		
		append_cxx_flags(libmesh_include)
		
		conf.check(header_name='libmesh.h', define_name='KARMA_HAVE_LIBMESH_H')
		
		# TODO: Need to qualify based on the variant
		append_cxx_flags(libmesh_cppflags)
		append_cxx_flags(libmesh_cxxflags)
		
		#conf.env.append_unique('LINKFLAGS', libmesh_ldflags.split(' '))
		conf.env['LINKFLAGS'] = libmesh_ldflags.split(' ') + conf.env['LINKFLAGS'] 
		conf.env['LINKFLAGS'] = config_utils.unique(conf.env['LINKFLAGS'])
		conf.env.append_unique('LIBPATH', os.path.join(libmesh_dictionary['libmesh_dir'],'lib', '%s_%s'%(libmesh_dictionary['hosttype'],conf.env.METHOD)))
		
		# alternately, we can find hosttype used by libMesh by running their config.guess
		#libmesh_hosttype = Popen(['/bin/sh', os.path.join(conf.env.LIBMESH_DIR,'config.guess')], stdout=PIPE).stdout.read()
		#print 'libmesh_hosttype = %s' % libmesh_hosttype
		
		conf.check(msg='Checking for libMesh library', lib='mesh', uselib_store='LIBMESH', mandatory=True)
		
		# check if libMesh is built as a static library
		if libmesh_dictionary['enable-shared'] == 'no':
		#if conf.find_file('libmesh%s'%libmesh_dictionary['libext'], #os.path.join(libmesh_dictionary['libmesh_dir'],'lib','%s_%s'%(libmesh_dictionary['hosttype'],conf.env.METHOD)), mandatory=False):
			conf.env['STATICLIB_LIBMESH'] = os.path.join(libmesh_dictionary['libmesh_dir'],'lib','%s_%s'%(libmesh_dictionary['hosttype'],conf.env.METHOD), 'libmesh%s'%libmesh_dictionary['libext'])
			conf.msg('Found static libMesh library', 'libmesh%s'%libmesh_dictionary['libext'])
			conf.define('LIBMESH_STATIC', 1)
		else:
			# this is weird but works. the problem is that WAF assumes everything
			# specified in LIB_* can be translated to -l*. So for libmesh, in the shared
			# build case, we need to provide -lmesh instead of -llibmesh. 
			# Hacky, yes but inevitable due to naming of "libMesh".
			conf.env['LIB_LIBMESH'] = 'mesh'
			conf.define('LIBMESH_STATIC', 0)
		
		# FIXME: there is a problem when linking libMesh using Intel compilers
		# the linker errors are associated with whether the object file appears first or last
		# when it appears last, we get undefined references. GNU > 4.6 does not care about this.
		if conf.env['CC_NAME'] is not 'icc':
			conf.check_cxx(fragment='''
					#include "libmesh.h"
					#include "mesh.h"
					using namespace libMesh;
					int main(int argc, char** argv) { LibMeshInit init (argc, argv); Mesh mesh; return 0; } ''',
				#define_name = "KARMA_HAVE_LIBMESH",
				execute     = True,
				#define_ret  = True,
				uselib_store='LIBMESH SLEPC PETSC', 
				msg         = "Linking with libMesh library")
				
	else:
		conf.fatal('SLEPC_DIR environment variable (dir=%s) points to invalid location' % opts.libmesh_dir )
