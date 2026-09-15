#!/usr/bin/env python
# Waf utilities for easily building standard unixey packages/libraries
# Licensed under the GNU GPL v2 or later, see COPYING file for details.
# Copyright (C) 2008 David Robillard
# Copyright (C) 2008 Nedko Arnaudov

import Configure
import Options
import os, re, sys
import subprocess
import glob
import config_utils

from TaskGen import feature, before, after
from waflib import Options, Utils
from subprocess import Popen, PIPE

global g_is_child
g_is_child = False

# Only run autowaf hooks once (even if sub projects call several times)
global g_step
g_step = 0

# Compute dependencies globally
#import preproc
#preproc.go_absolute = True

#@feature('cc', 'cxx')
#@after('apply_lib_vars')
#@before('apply_obj_vars_cc', 'apply_obj_vars_cxx')
#def include_config_h(self):
#	self.env.append_value('INC_PATHS', self.bld.srcnode)

def set_options(opt):
	# "Add standard autowaf options if they havn't been added yet"
	global g_step
	if g_step > 0:
		return
	
	opt.load('python')
	opt.tool_options('compiler_cc')
	opt.tool_options('compiler_cxx')
	
	g_step = 1

def check_header_wrapper(conf, name, define='', mandatory=False):
	"Check for a header iff it hasn't been checked for yet"
	if type(conf.env['AUTOWAF_HEADERS']) != dict:
		conf.env['AUTOWAF_HEADERS'] = {}

	checked = conf.env['AUTOWAF_HEADERS']
	if not name in checked:
		checked[name] = True
		includes = '' # search default system include paths
		if sys.platform == "darwin":
			includes = '/opt/local/include'
		if define != '':
			conf.check(header_name=name, includes=includes, define_name=define, mandatory=mandatory)
		else:
			conf.check(header_name=name, includes=includes, mandatory=mandatory)

def nameify(name):
	return name.replace('/', '_').replace('++', 'PP').replace('-', '_').replace('.', '_')

def check_pkg(conf, name, **args):
	if not 'mandatory' in args:
		args['mandatory'] = True
	"Check for a package iff it hasn't been checked for yet"
	var_name = 'KARMA_HAVE_' + nameify(args['uselib_store'])
	check = not var_name in conf.env
	if not check and 'atleast_version' in args:
		# Re-check if version is newer than previous check
		checked_version = conf.env['VERSION_' + name]
		if checked_version and checked_version < args['atleast_version']:
			check = True;
	if check:
		conf.check_cfg(package=name, args="--cflags --libs", **args)
		found = bool(conf.env[var_name])
		if found:
			conf.define(var_name, int(found))
			if 'atleast_version' in args:
				conf.env['VERSION_' + name] = args['atleast_version']
		else:
			conf.undefine(var_name)
			if args['mandatory'] == True:
				conf.fatal("Required package " + name + " not found")


# useful regex tester: http://regexpal.com/

from Configure import conf

@conf
def check_version(self, t, **kw):
	if 'define_name' not in kw:
		kw['define_name'] = '%s_VERSION' % t.upper().replace(' ', '_').replace('*', 'P')

	s = Popen([t, '--version'], stdout=PIPE).stdout.read()
	#print s
	#vnum = re.findall(r'^.*?\d+(?:\.\d+)(?:\.\d+)(?:\.\d+)?', s)
	vnum = re.findall(r'^.*?\d+\.\d+(?:\.\d+|(?:\.\d+\.\d+))?',s)
	#print vnum
	#print 'finding double dot'
	#vnum = re.findall(r'^.*?\d+(\.\d+){2}?', s)	# version number = major.minor.rev
	#if len(vnum) == 0:
	#	print 'finding single dot'
	#	vnum = re.findall('^.*?\d+(\.\d+){1}?', s)	# version number = major.minor
		#^.*?\d+(\.\d+){0,2}?
	
	#print vnum

	self.msg('Version of %s ' % t, vnum)

# useful regex tester: http://regexpal.com/

# get a list of types and then compute the size of each in one shot
@conf
def check_sizeof_types(self, typekeys, lo = 1, hi=17, **kw):

	if 'define_name' in kw:
		del kw['define_name'] 

	if 'guess' in kw:
		del kw['guess']

	# by default, the language used = C++
	prog_lang = 'cxx'
	if 'language' in kw:
		prog_lang = kw['language']
		
		# validate the language specified by user
		if prog_lang not in ['c', 'cxx']:
			prog_lang = 'cxx'
	
	if prog_lang == 'cxx':
		fragment_code = '#include <iostream>\n'
		if 'header_name' in kw:
			for header in kw['header_name']:
				fragment_code = fragment_code + '#include <%s>\n' % header
		fragment_code = fragment_code + 'int main() {\nstd::cout '
		for typename in typekeys:
			fragment_code = fragment_code + '<< sizeof(%s) << ";" ' % typename
		fragment_code = fragment_code + ';\nreturn 0;\n}'
	else:
		fragment_code = '#include <stdio.h>\n'
		if 'header_name' in kw:
			for header in kw['header_name']:
				fragment_code = fragment_code + '#include <%s>\n' % header
		fragment_code = fragment_code + 'int main() {\nprintf("'
		for typename in typekeys:
			fragment_code = fragment_code + '%d;'
		fragment_code = fragment_code + '"'
		for typename in typekeys:
			fragment_code = fragment_code + ', sizeof(%s)' % typename
		fragment_code = fragment_code + ');\nreturn 0;\n}'
	
	kw['compiler'] = prog_lang
	#print fragment_code
	
	kw.update({'fragment': fragment_code, 
			   'execute':True, 
			   'errmsg':'Unknown',
			   'define_ret':True,
			   'quote':False})

	try:
		self.validate_c(kw)
	except:
		self.start_msg('Checking for size of types: %s' % (', '.join(typekeys)))
		self.end_msg('Unknown', 'YELLOW')
		raise
		if 'mandatory' not in kw or kw['mandatory']:
			self.fatal("Can't determine size of types: %s" % (', '.join(typekeys)))
	else:
		ret = None
	
		try:
			ret = self.run_c_code(**kw)
		except:
			self.start_msg('Checking for size of types: %s' % (', '.join(typekeys)))
			self.end_msg('Unknown', 'YELLOW')
			raise
			if 'mandatory' not in kw or kw['mandatory']:
				self.fatal("Can't determine size of types: %s" % (', '.join(typekeys)))
		else:
			kw['success'] = ret
			self.post_check(**kw)

	#print ret
	typesizes = ret.split(';')
	index = 0
	for typename in typekeys:
		self.msg('Checking for sizeof(%s)'%typename, typesizes[index])
		self.define('KARMA_SIZEOF_' + typename.upper().replace(' ', '_').replace('*', 'PTR').replace('::', '_'), 
				int(typesizes[index]))
		index = index+1
	
	return typesizes
	#return index

@conf
def custom_find_file(self, filename, path_list=[], mandatory=True):
	"""
	Find a file in a list of paths

	:param filename: name of the file to search for
	:param path_list: list of directories to search
	:return: the first occurrence filename or '' if filename could not be found
	"""

	# add the standard include paths to the list
	path_list = '/usr/include ' + path_list
	path_list = '/usr/local/include ' + path_list
	path_list = '/opt/include ' + path_list
	path_list = '/opt/local/include ' + path_list
	
	for n in Utils.to_list(filename):
		for d in Utils.to_list(path_list):
			p = os.path.join(d, n)
			if os.path.exists(p):
				return p
				
	if mandatory:
		self.fatal('Could not find %r' % filename)

def configure(conf):
	global g_step
	if g_step > 1:
		return
	
	def append_cxx_flags(vals):
		conf.env.append_value('CCFLAGS', vals.split())
		conf.env.append_value('CXXFLAGS', vals.split())
	
	config_utils.display_header('Global Configuration')
	
	#conf.check_tool('misc')
	
	conf.load('python')
	conf.check_python_version((2,4))
	print
	
	use_mpi_compilers = 0
	# check for MPI compiler wrappers first. At some point will have to check
	# if these are the same as the ones used by PETSc/SLEPc etc..
	if conf.check_cfg(path='mpicxx', args='--version', package='', uselib_store='MPI'):
		os.environ.setdefault('CXX', 'mpicxx')
		os.environ.setdefault('CXXNAME', 'mpicxx')
		if conf.check_cfg(path='mpicc', args='--version', package='', uselib_store='MPI'):
			os.environ.setdefault('CC', 'mpicc') 
			os.environ.setdefault('CCNAME', 'mpicc')
			if conf.check_cfg(path='mpif90', args='--version', package='', uselib_store='MPI'):
				os.environ.setdefault('FC', 'mpif90') 
				os.environ.setdefault('FCNAME', 'mpif90')
				use_mpi_compilers = 1
			
	conf.msg('Using MPI compilers', config_utils.yesno(use_mpi_compilers))

	#conf.check_tool('compiler_fortran')
	conf.check_tool('compiler_cc')
	conf.check_tool('compiler_cxx')
	
	conf.check_version(os.environ['CC'], define_name='CC_VERSION')
	conf.check_version(os.environ['CXX'], define_name='CXX_VERSION')
	conf.check_version(os.environ['FC'], define_name='FC_VERSION')
	
	print
	
	conf.env['BUILD_DOCS'] = Options.options.build_docs
	#conf.env['DEBUG'] = Options.options.debug
	conf.env['STRICT'] = Options.options.strict
	conf.env['PREFIX'] = os.path.abspath(os.path.expanduser(os.path.normpath(conf.env['PREFIX'])))
	
	conf.env['BINDIR'] = os.path.join(conf.env['PREFIX'], 'bin')
	conf.env['INCLUDEDIR'] = os.path.join(conf.env['PREFIX'], 'include')
	conf.env['LIBDIR'] = os.path.join(conf.env['PREFIX'], 'lib')
	conf.env['DATADIR'] = os.path.join(conf.env['PREFIX'], 'share')
	conf.env['CONFIGDIR'] = os.path.join(conf.env['PREFIX'], 'etc')
	conf.env['HTMLDIR'] = os.path.join(conf.env['DATADIR'], 'doc', 'html')
	conf.env['MANDIR'] = os.path.join(conf.env['DATADIR'], 'doc')

	conf.env['BINDIRNAME'] = os.path.basename(conf.env['BINDIR'])
	conf.env['LIBDIRNAME'] = os.path.basename(conf.env['LIBDIR'])
	conf.env['DATADIRNAME'] = os.path.basename(conf.env['DATADIR'])
	conf.env['CONFIGDIRNAME'] = os.path.basename(conf.env['CONFIGDIR'])

	if Options.options.build_docs:
		doxygen = conf.find_program('doxygen')
		if not doxygen:
			conf.fatal("Doxygen is required to build documentation, configure without --docs")

		dot = conf.find_program('dot')
		if not dot:
			conf.fatal("Graphviz (dot) is required to build documentation, configure without --docs")
	
	if Options.options.strict:
		conf.env.append_value('CCFLAGS', [ '-std=c99', '-pedantic' ])
		conf.env.append_value('CXXFLAGS', [ '-ansi', '-Woverloaded-virtual', '-Wnon-virtual-dtor'])
		append_cxx_flags('-Wall -Wextra -Wno-unused-parameter')

	if Options.options.enable_shared:
		append_cxx_flags('-fPIC -DPIC')
		# -fshow-column is GNU specific; intel compilers do not understand
		if conf.env['CC_NAME'] is not 'icc':
			append_cxx_flags('-fshow-column')
	try:
	
		s = Popen(['which', 'mpiexec'], stdout=PIPE).stdout.read()
		
		# check if mpiexec is available for running the codes.
		conf.find_file('mpiexec', s.replace('mpiexec', ''))
		
		conf.msg( "Checking for 'mpiexec'", s)
	
	except conf.errors.ConfigurationError:
		config_utils.print_error(conf, 'mpiexec was not found in the path - fix the issue')
		#print '\n\t***** mpiexec was not found in the path - please correct this issue *****\n'
		pass

	config_utils.display_msg(conf, "Install prefix", conf.env['PREFIX'])
	config_utils.display_msg(conf, "Debuggable build", config_utils.yesno(conf.env['DEBUG']))
	config_utils.display_msg(conf, "Strict compiler flags", config_utils.yesno(conf.env['STRICT']))
	config_utils.display_msg(conf, "Build documentation", config_utils.yesno(conf.env['DOCS']))

	g_step = 2
	
def set_local_lib(conf, name, has_objects):
	conf.define('KARMA_HAVE_' + nameify(name.upper()), 1)
	if has_objects:
		if type(conf.env['AUTOWAF_LOCAL_LIBS']) != dict:
			conf.env['AUTOWAF_LOCAL_LIBS'] = {}
		conf.env['AUTOWAF_LOCAL_LIBS'][name.lower()] = True
	else:
		if type(conf.env['AUTOWAF_LOCAL_HEADERS']) != dict:
			conf.env['AUTOWAF_LOCAL_HEADERS'] = {}
		conf.env['AUTOWAF_LOCAL_HEADERS'][name.lower()] = True

def use_lib(bld, obj, libs):
	abssrcdir = os.path.abspath('.')
	libs_list = libs.split()
	for l in libs_list:
		in_headers = l.lower() in bld.env['AUTOWAF_LOCAL_HEADERS']
		in_libs    = l.lower() in bld.env['AUTOWAF_LOCAL_LIBS']
		if in_libs:
			if hasattr(obj, 'uselib_local'):
				obj.uselib_local += ' lib' + l.lower() + ' '
			else:
				obj.uselib_local = 'lib' + l.lower() + ' '
		
		if in_headers or in_libs:
			inc_flag = '-iquote ' + os.path.join(abssrcdir, l.lower())
			for f in ['CCFLAGS', 'CXXFLAGS']:
				if not inc_flag in bld.env[f]:
					bld.env.append_value(f, inc_flag)
		else:
			if hasattr(obj, 'uselib'):
				obj.uselib += ' ' + l
			else:
				obj.uselib = l


def link_flags(env, lib):
	return ' '.join(map(lambda x: env['LIB_ST'] % x, env['LIB_' + lib]))

def compile_flags(env, lib):
	return ' '.join(map(lambda x: env['CPPPATH_ST'] % x, env['CPPPATH_' + lib]))

def set_recursive():
	global g_is_child
	g_is_child = True

def is_child():
	global g_is_child
	return g_is_child

# Pkg-config file
def build_pc(bld, name, version, libs):
	'''Build a pkg-config file for a library.
	name    -- uppercase variable name     (e.g. 'SOMENAME')
	version -- version string              (e.g. '1.2.3')
	libs    -- string/list of dependencies (e.g. 'LIBFOO GLIB')
	'''

	obj              = bld.new_task_gen('subst')
	obj.source       = name.lower() + '.pc.in'
	obj.target       = name.lower() + '.pc'
	obj.install_path = '${PREFIX}/${LIBDIRNAME}/pkgconfig'
	pkg_prefix       = bld.env['PREFIX'] 
	if pkg_prefix[-1] == '/':
		pkg_prefix = pkg_prefix[:-1]
	obj.dict = {
		'prefix'           : pkg_prefix,
		'exec_prefix'      : '${prefix}',
		'libdir'           : '${prefix}/' + bld.env['LIBDIRNAME'],
		'includedir'       : '${prefix}/include',
		name + '_VERSION'  : version,
	}
	if type(libs) != list:
		libs = libs.split()
	for i in libs:
		obj.dict[i + '_LIBS']   = link_flags(bld.env, i)
		obj.dict[i + '_CFLAGS'] = compile_flags(bld.env, i)

# Doxygen API documentation
def build_dox(bld, name, version, srcdir, blddir):
	if not bld.env['DOCS']:
		return
	obj = bld.new_task_gen('subst')
	obj.source = 'doc/reference.doxygen.in'
	obj.target = 'doc/reference.doxygen'
	if is_child():
		src_dir = os.path.join(srcdir, name.lower())
		doc_dir = os.path.join(blddir, 'default', name.lower(), 'doc')
	else:
		src_dir = srcdir
		doc_dir = os.path.join(blddir, 'default', 'doc')
	obj.dict = {
		name + '_VERSION' : version,
		name + '_SRCDIR'  : os.path.abspath(src_dir),
		name + '_DOC_DIR' : os.path.abspath(doc_dir)
	}
	obj.install_path = ''
	out1 = bld.new_task_gen('command-output')
	out1.dependencies = [obj]
	out1.stdout = '/doc/doxygen.out'
	out1.stdin = '/doc/reference.doxygen' # whatever..
	out1.command = 'doxygen'
	out1.argv = [os.path.abspath(doc_dir) + '/reference.doxygen']
	out1.command_is_external = True

# Version code file generation
def build_version_files(header_path, source_path, domain, major, minor, micro):
	header_path = os.path.abspath(header_path)
	source_path = os.path.abspath(source_path)
	text  = "int " + domain + "_major_version = " + str(major) + ";\n"
	text += "int " + domain + "_minor_version = " + str(minor) + ";\n"
	text += "int " + domain + "_micro_version = " + str(micro) + ";\n"
	try:
		o = open(source_path, 'w')
		o.write(text)
		o.close()
	except IOError:
		print("Could not open %s for writing\n", source_path)
		sys.exit(-1)

	text  = "#ifndef __" + domain + "_version_h__\n"
	text += "#define __" + domain + "_version_h__\n"
	text += "extern const char* " + domain + "_revision;\n"
	text += "extern int " + domain + "_major_version;\n"
	text += "extern int " + domain + "_minor_version;\n"
	text += "extern int " + domain + "_micro_version;\n"
	text += "#endif /* __" + domain + "_version_h__ */\n"
	try:
		o = open(header_path, 'w')
		o.write(text)
		o.close()
	except IOError:
		print("Could not open %s for writing\n", header_path)
		sys.exit(-1)
		
	return None

def run_tests(ctx, appname, tests):
	orig_dir = os.path.abspath(os.curdir)
	failures = 0
	base = '..'

	top_level = os.path.abspath(ctx.curdir) != os.path.abspath(os.curdir)
	if top_level:
		os.chdir('./build/default/' + appname)
		base = '../..'
	else:
		os.chdir('./build/default')

	lcov = True
	lcov_log = open('lcov.log', 'w')
	try:
		# Clear coverage data
		subprocess.call('lcov -d ./src -z'.split(),
				stdout=lcov_log, stderr=lcov_log)
	except:
		lcov = False
		print("Failed to run lcov, no coverage report will be generated")


	# Run all tests
	for i in tests:
		print()
		Logs.pprint('BOLD', 'Running %s test %s' % (appname, i))
		if subprocess.call(i) == 0:
			Logs.pprint('GREEN', 'Passed %s %s' % (appname, i))
		else:
			failures += 1
			Logs.pprint('RED', 'Failed %s %s' % (appname, i))

	if lcov:
		# Generate coverage data
		coverage_lcov = open('coverage.lcov', 'w')
		subprocess.call(('lcov -c -d ./src -d ./test -b ' + base).split(),
				stdout=coverage_lcov, stderr=lcov_log)
		coverage_lcov.close()
		
		# Strip out unwanted stuff
		coverage_stripped_lcov = open('coverage-stripped.lcov', 'w')
		subprocess.call('lcov --remove coverage.lcov *boost* c++*'.split(),
				stdout=coverage_stripped_lcov, stderr=lcov_log)
		coverage_stripped_lcov.close()
	
		# Generate HTML coverage output
		if not os.path.isdir('./coverage'):
			os.makedirs('./coverage')
		subprocess.call('genhtml -o coverage coverage-stripped.lcov'.split(),
				stdout=lcov_log, stderr=lcov_log)
	
	lcov_log.close()

	print()
	Logs.pprint('BOLD', 'Summary:', sep=''),
	if failures == 0:
		Logs.pprint('GREEN', 'All ' + appname + ' tests passed')
	else:
		Logs.pprint('RED', str(failures) + ' ' + appname + ' test(s) failed')

	Logs.pprint('BOLD', 'Coverage:', sep='')
	print(os.path.abspath('coverage/index.html'))

	os.chdir(orig_dir)

def shutdown():
	# This isn't really correct (for packaging), but people asking is annoying
	if Options.commands['install']:
		try: os.popen("/sbin/ldconfig")
		except: pass

def build_i18n(bld,srcdir,dir,name,sources):
	pwd = os.getcwd()
	os.chdir(os.path.join (srcdir, dir))

	pot_file = '%s.pot' % name

	args = [ 'xgettext',
		 '--keyword=_',
		 '--keyword=N_',
		 '--from-code=UTF-8',
		 '-o', pot_file,
		 '--copyright-holder="Paul Davis"' ]
	args += sources
	print 'Updating ', pot_file
	os.spawnvp (os.P_WAIT, 'xgettext', args)
	
	po_files = glob.glob ('po/*.po')
	languages = [ po.replace ('.po', '') for po in po_files ]
	
	for po_file in po_files:
		args = [ 'msgmerge',
			 '--update',
			 po_file,
			 pot_file ]
		print 'Updating ', po_file
		os.spawnvp (os.P_WAIT, 'msgmerge', args)
		
	for po_file in po_files:
		mo_file = po_file.replace ('.po', '.mo')
		args = [ 'msgfmt',
			 '-c',
			 '-f',
			 '-o',
			 mo_file,
			 po_file ]
		print 'Generating ', po_file
		os.spawnvp (os.P_WAIT, 'msgfmt', args)
	os.chdir (pwd)
