#! /usr/bin/env python
# encoding: utf-8

import os, subprocess, re
import config_utils
import load_config_variables

from subprocess import Popen, PIPE

def configure(conf, dictionary):

	if 'DEALII_DIR' not in os.environ:
		config_utils.print_error(conf, 'Deal.II path not specified. Skipping ...')
		return

	config_utils.display_header('Deal.II Finite Element library')
	
	# a simple routine to add flags to both C/C++ configs.
	def append_cxx_flags(vals):
		conf.env.append_unique('CCFLAGS', vals.split())
		conf.env.append_unique('CXXFLAGS', vals.split())
	
	
	if not os.path.exists(os.environ['DEALII_DIR']):
		print_error(conf, 'DEALII_DIR environment variable (dir=%s) points to invalid location' % os.environ['DEALII_DIR'], 1)
	else:
		# get and store the DEALII_DIR variable from the environment
		conf.env.DEALII_DIR = os.environ['DEALII_DIR']

		conf.msg( 'Deal.II directory', conf.env.DEALII_DIR)
		conf.define('DEALII_DIR', conf.env.DEALII_DIR)
	
	import glob
	
	dealii_exists = 0 
	if conf.find_file('config.h', os.path.join(conf.env.DEALII_DIR, 'include', 'deal.ii', 'base')):
		#if glob.glob(os.path.join(conf.env.DEALII_DIR, 'lib','libslepc.*')):
			dealii_exists = 1
	
	#`uname -m`-
	
	conf.define('KARMA_HAVE_DEALII', dealii_exists)
	
	print
	if dealii_exists:
		global_options_file = os.path.join(conf.env.DEALII_DIR,'common', 'Make.global_options')
		if os.path.exists(global_options_file):
		
			#dealii_dictionary = {}
			dealii_dictionary = dictionary.copy()
			conf.msg( 'Reading DealII configuration files', global_options_file)
			load_config_variables.load_config_file(global_options_file, dealii_dictionary)
			#print dealii_dictionary
			#load_config_variables.unroll_variables(conf.env, dealii_dictionary)
			conf.define('DEALII_VERSION', dealii_dictionary['DEAL_II_VERSION'])
			conf.msg('DealII version', dealii_dictionary['DEAL_II_VERSION'])

			print
			# unfortunately, DealII doesn't store its configure command as part of the variables
			# so just like SLEPc, lets poke into configure.log and extract this info
			dealii_configure_command = Popen(['grep', '$ ./configure', os.path.join(conf.env.DEALII_DIR,'config.log')], stdout=PIPE).stdout.read()
			conf.msg('DealII was configured with the command', dealii_configure_command.replace('$ ./configure', '').strip())

		else:
			print_error('DealII has not been configured correctly. Please check path.')
		
		# unfortunately, DealII doesn't store its the variables unrolled.
		# use dealii-config to get the necessary info
		# so lets call the config command directly to extract this info
		#dealii_dictionary = {}
		#dealii_dictionary = load_config_variables.unroll_makefile(global_options_file)
		#print 'print deal.ii dictionary'
		#print dealii_dictionary
		dealii_ccflags = load_config_variables.get_makefile_variable(global_options_file, 'CC_FLAGS')
		#Popen(['/usr/bin/make', '-f', global_options_file, '-f', 'config/helper.mak', 'echo_INCLUDE'], stdout=PIPE).stdout.read()
		#print 'dealii_ccflags = %s' % dealii_ccflags 
		dealii_cxxflags = load_config_variables.get_makefile_variable(global_options_file, 'CXXFLAGS.o')
		#print 'dealii_cxxflags = %s' % dealii_cxxflags 
		dealii_include = load_config_variables.get_makefile_variable(global_options_file, 'INCLUDE')
		#print 'dealii_include = %s' % dealii_include 
		dealii_ldflags = load_config_variables.get_makefile_variable(global_options_file, 'LDFLAGS')
		#print 'dealii_ldflags = %s' % dealii_ldflags 
		dealii_libs = load_config_variables.get_makefile_variable(global_options_file, 'lib-deal2.o')
		#print 'dealii_libs = %s' % dealii_libs
		
		dealii_ccflags = dealii_ccflags.strip()
		dealii_cxxflags = dealii_cxxflags.strip()
		dealii_include_dirs = dealii_include.replace('-I','').strip()
		dealii_ldflags = dealii_ldflags.strip()
		dealii_libs = dealii_libs.strip()
		
		
		#print 'cppflags %s' % dealii_ccflags
		#print 'cxxflags %s' % dealii_cxxflags
		#print 'include %s' % dealii_include
		#print 'ldflags %s' % dealii_ldflags		
		
		print 
			
		# Read Make.common and get all the necessary variables
		conf.env.append_unique('INC_PATHS', dealii_include_dirs) 
		
		append_cxx_flags(dealii_cxxflags)
		append_cxx_flags(dealii_ccflags)
		
		# conf.env.append_unique('LINKFLAGS', slepc_dictionary['SLEPC_LIB'].split(' '))
		# append_cxx_flags(slepc_dictionary['CCPPFLAGS'])
		
		conf.check(header_name='base/config.h', define_name='KARMA_HAVE_DEALII_H')
		
		#conf.env.append_unique('LINKFLAGS', dealii_libs.split(' '))
		conf.env['LINKFLAGS'] = dealii_libs.split(' ') + conf.env['LINKFLAGS'] 
		conf.env['LINKFLAGS'] = dealii_ldflags.split(' ') + conf.env['LINKFLAGS'] 
		conf.env.append_unique('LIBPATH', os.path.join(dealii_dictionary['D'],'lib'))
		
		#first_entry = dealii_libs.split()
		#print first_entry
		#dealii_lib = re.findall(r'^.*dealii+\.(a|so|dylib)', first_entry[0])
		#print dealii_lib
		
		# alternately, we can find hosttype used by DealII by running their config.guess
		#dealii_hosttype = Popen(['/bin/sh', os.path.join(conf.env.DEALII_DIR,'config.guess')], stdout=PIPE).stdout.read()
		#print 'dealii_hosttype = %s' % dealii_hosttype
		
		conf.check(msg='Checking for DealII library', lib='deal_II.%s'%dealii_dictionary['DEAL_II_VERSION'], uselib_store='DEALII', mandatory=True)
		
		# check if Deal.II is built as a static library
		#if conf.find_file('dealii.%s'%dealii_dictionary['libext'], dealii_lib[0].replace(r'dealii+\.(a|so|dylib)', ''), mandatory=False):
		if conf.find_file('libdeal_II.%s%s'%(dealii_dictionary['DEAL_II_VERSION'], dealii_dictionary['lib-suffix']), os.path.join(dealii_dictionary['D'],'lib'), mandatory=False):
			conf.env['STATICLIB_DEALII'] = os.path.join(dealii_dictionary['D'],'lib', 'libdeal_II.%s%s'%(dealii_dictionary['DEAL_II_VERSION'], dealii_dictionary['lib-suffix']))
			conf.msg('Found static library', 'libdeal_II.%s%s'%(dealii_dictionary['DEAL_II_VERSION'], dealii_dictionary['lib-suffix']))
			conf.define('DEALII_STATIC', 1)
		else:	#SL_LINKER_SUFFIX
			# this is weird but works. the problem is that WAF assumes everything
			# specified in LIB_* can be translated to -l*. So for dealii, in the shared
			# build case, we need to provide -lmesh instead of -ldealii. 
			# Hacky, yes but inevitable due to naming of "DealII".
			conf.msg('Found shared library', 'libdeal_II.%s%s'%(dealii_dictionary['DEAL_II_VERSION'], dealii_dictionary['shared-lib-suffix']))
			conf.env['LIB_DEALII'] = 'deal_II.%s'%dealii_dictionary['DEAL_II_VERSION']
			conf.define('DEALII_STATIC', 0)
		
		conf.check_cxx(fragment='''
				#include <deal.II/grid/tria.h>
				#include <deal.II/grid/grid_generator.h>
				using namespace dealii;
				int main() { Triangulation<2> triangulation; GridGenerator::hyper_cube (triangulation); return 0; } ''',
			#define_name = "KARMA_HAVE_DEALII",
			execute     = True,
			#define_ret  = True,
			uselib_store='DEALII', 
			msg         = "Linking with DealII library")
				
	else:
		conf.fatal('DEALII_DIR environment variable (dir=%s) points to invalid location' % os.environ['DEALII_DIR'] )
