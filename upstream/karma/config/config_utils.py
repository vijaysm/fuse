#!/usr/bin/env python

import types, subprocess

from waflib import Options, Utils, Logs
from subprocess import PIPE, Popen, STDOUT

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    
def maximum(a, b):
	#return (a if a >= b else b)
	return (b, a)[a>=b]

def minimum(a, b):
	#return (a if a <= b else b)
	return (a, b)[a>=b]

def print_summary(msg, result=-1):
	if result < 0:
		print bcolors.OKBLUE + msg + bcolors.ENDC
	else:
		print bcolors.OKBLUE + msg + ' ' * (41-len(msg)) + '= %s'%result + bcolors.ENDC
	#conf.check_message_1(msg)
	#conf.check_message_2(result, color)
	
def print_error(conf, msg, fatal=0):
	error_msg = '\n\t*****  ' + msg + '  *****\n'
	conf.to_log(error_msg)
	print bcolors.FAIL + error_msg + bcolors.ENDC
	if fatal:
		conf.fatal('Aborting.')
	
def chop_prefix(conf, var):
	name = conf.env[var][len(conf.env['PREFIX']):]
	if len(name) > 0 and name[0] == '/':
		name = name[1:]
	if name == "":
		name = "/"
	return name;

def display_header(title):
	print_summary('\n\t\t ' + '=' * (len(title)+12) + ' ')
	print_summary('\t\t\t'+title)
	print_summary('\t\t ' + '=' * (len(title)+12) + ' ')
	print

def display_bold_msg(title):
	Logs.pprint('BOLD', title)

def display_msg(conf, msg, status = None, color = None):
	color = 'CYAN'
	if type(status) == bool and status or status == "True":
		color = 'GREEN'
	elif type(status) == bool and not status or status == "False":
		color = 'YELLOW'
	Logs.pprint('BOLD', "%s :" % msg.ljust(conf.line_just), sep='')
	Logs.pprint(color, status)

def yesno(bool):
	#return str(bool)
	if bool:
		return "yes"
	else:
		return "no"
	
def tolist(x):
	if type(x) is types.ListType:
		return x
	return [x]

def unique(seq):
	#
	# Returns a new list with all duplicates in seq removed.
	#
	already_seen = set()
	result = []
	for item in seq:
		if item not in already_seen:
			already_seen.add(item)
			result.append(item)
	return result

def append_unique(dict1, dict2):
	#
	# Returns a new list with all duplicates in seq removed.
	#
	already_seen = dict1.keys()
	result = []
	for key, value in dict2.items():
		if key not in already_seen:
			already_seen.append(key)
			dict1[key] = value
	return dict1

def fetch_git_revision (path):
    cmd = "LANG= git log --abbrev HEAD^..HEAD " + path
    output = subprocess.Popen(cmd, shell=True, stderr=subprocess.STDOUT, stdout=subprocess.PIPE).communicate()[0].splitlines()
    o = output[0].decode('utf-8')
    rev = o.replace ("commit", "git")[0:50]
    for line in output:
        try:
            if "git-svn-id" in line:
                line = line.split('@')[1].split(' ')
                rev = line[0]
        except:
            pass
    return rev.replace('git ', '')

def compute_revision():
	global REVISION

# use the pickle module to serialize the global dictionary for future reference
def serialize_dictionary(dictionary, filename):
	import pickle
	global_config = open(filename, 'w')
	pickle.dump(dictionary, global_config)
	global_config.close()
	

def make_flags_unique(env):
	env.LINKFLAGS = unique(env.LINKFLAGS)
	env.CCFLAGS = unique(env.CCFLAGS)
	env.CCPPFLAGS = unique(env.CCPPFLAGS)
	env.CPPPATH = unique(env.CPPPATH)
	env.LIBPATH = unique(env.LIBPATH)

def set_compiler_flags (conf,opt):
	#
	# Compiler flags and other system-dependent stuff
	#
	
	build_host_supports_sse = False
	optimization_flags = []
	debug_flags = []
	
	# guess at the platform, used to define compiler flags
	
	config_guess = Popen(['/usr/bin/uname', '-msr'], stdout=PIPE).stdout.read()
	config_guess = config_guess.lower()
	
	config_os = 0
	config_kernel = 1
	config_cpu = 2
	config = config_guess.lower().split (" ")
	
	print config_guess

	if config[config_arch] != 'apple':
		debug_flags = [ '-rdynamic' ] # waf adds -O0 -g itself. thanks waf!
	
	# Autodetect
	if config[config_arch] == 'apple':
		# The [.] matches to the dot after the major version, "." would match any character
		if re.search ("darwin[0-7][.]", config[config_kernel]) != None:
			conf.define ('build_target', 'panther')
		elif re.search ("darwin8[.]", config[config_kernel]) != None:
			conf.define ('build_target', 'tiger')
		else:
			conf.define ('build_target', 'leopard')
	else:
		if re.search ("x86_64", config[config_cpu]) != None:
			conf.define ('build_target', 'x86_64')
		elif re.search("i[0-5]86", config[config_cpu]) != None:
			conf.define ('build_target', 'i386')
		elif re.search("powerpc", config[config_cpu]) != None:
			conf.define ('build_target', 'powerpc')
		else:
			conf.define ('build_target', 'i686')
	
	if config[config_cpu] == 'powerpc' and conf.env['build_target'] != 'none':
		#
		# Apple/PowerPC optimization options
		#
		# -mcpu=7450 does not reliably work with gcc 3.*
		#
		if opt.dist_target == 'panther' or opt.dist_target == 'tiger':
			if config[config_arch] == 'apple':
				# optimization_flags.extend ([ "-mcpu=7450", "-faltivec"])
				# to support g3s but still have some optimization for above
				optimization_flags.extend ([ "-mcpu=G3", "-mtune=7450"])
			else:
				optimization_flags.extend ([ "-mcpu=7400", "-maltivec", "-mabi=altivec"])
		else:
			optimization_flags.extend([ "-mcpu=750", "-mmultiple" ])
		optimization_flags.extend (["-mhard-float", "-mpowerpc-gfxopt"])
		optimization_flags.extend (["-Os"])
	
	elif ((re.search ("i[0-9]86", config[config_cpu]) != None) or (re.search ("x86_64", config[config_cpu]) != None)) and conf.env['build_target'] != 'none':
	
	
		#
		# ARCH_X86 means anything in the x86 family from i386 to x86_64
		# USE_X86_64_ASM is used to distingush 32 and 64 bit assembler
		#
	
		if (re.search ("(i[0-9]86|x86_64)", config[config_cpu]) != None):
			debug_flags.append ("-DARCH_X86")
			optimization_flags.append ("-DARCH_X86")
	
		if config[config_kernel] == 'linux' :
	
			#
			# determine processor flags via /proc/cpuinfo
			#
	
			if conf.env['build_target'] != 'i386':
	
				flag_line = os.popen ("cat /proc/cpuinfo | grep '^flags'").read()[:-1]
				x86_flags = flag_line.split (": ")[1:][0].split ()
	
				if "mmx" in x86_flags:
					optimization_flags.append ("-mmmx")
				if "sse" in x86_flags:
					build_host_supports_sse = True
				if "3dnow" in x86_flags:
					optimization_flags.append ("-m3dnow")
	
			if config[config_cpu] == "i586":
				optimization_flags.append ("-march=i586")
			elif config[config_cpu] == "i686":
				optimization_flags.append ("-march=i686")
	
		if ((conf.env['build_target'] == 'i686') or (conf.env['build_target'] == 'x86_64')) and build_host_supports_sse:
			optimization_flags.extend (["-msse", "-mfpmath=sse", "-DUSE_XMMINTRIN"])
			debug_flags.extend (["-msse", "-mfpmath=sse", "-DUSE_XMMINTRIN"])
	
	# end of processor-specific section
	
	# optimization section
	if conf.env['FPU_OPTIMIZATION']:
		if conf.env['build_target'] == 'tiger' or conf.env['build_target'] == 'leopard':
			optimization_flags.append ("-DBUILD_VECLIB_OPTIMIZATIONS");
			debug_flags.append ("-DBUILD_VECLIB_OPTIMIZATIONS");
			conf.env.append_value('LINKFLAGS', "-framework Accelerate")
		elif conf.env['build_target'] == 'i686' or conf.env['build_target'] == 'x86_64':
			optimization_flags.append ("-DBUILD_SSE_OPTIMIZATIONS")
			debug_flags.append ("-DBUILD_SSE_OPTIMIZATIONS")
		elif conf.env['build_target'] == 'x86_64':
			optimization_flags.append ("-DUSE_X86_64_ASM")
			debug_flags.append ("-DUSE_X86_64_ASM")
		if not build_host_supports_sse:
			print("\nWarning: you are building Ardour with SSE support even though your system does not support these instructions. (This may not be an error, especially if you are a package maintainer)")
	
	# check this even if we aren't using FPU optimization
	if not conf.env['HAVE_POSIX_MEMALIGN']:
		optimization_flags.append("-DNO_POSIX_MEMALIGN")
	
	# end optimization section
	
	#
	# no VST on x86_64
	#
	
	if conf.env['build_target'] == 'x86_64' and opt.vst:
		print("\n\n==================================================")
		print("You cannot use VST plugins with a 64 bit host. Please run waf with --vst=0")
		print("\nIt is theoretically possible to build a 32 bit host on a 64 bit system.")
		print("However, this is tricky and not recommended for beginners.")
		sys.exit (-1)
	
	#
	# a single way to test if we're on OS X
	#
	
	if conf.env['build_target'] in ['panther', 'tiger', 'leopard' ]:
		conf.define ('IS_OSX', 1)
		# force tiger or later, to avoid issues on PPC which defaults
		# back to 10.1 if we don't tell it otherwise.
		conf.env.append_value('CCFLAGS', "-DMAC_OS_X_VERSION_MIN_REQUIRED=1040")
	
	else:
		conf.define ('IS_OSX', 0)
	
	#
	# save off guessed arch element in an env
	#
	conf.define ('CONFIG_ARCH', config[config_arch])
	
	#
	# ARCH="..." overrides all
	#
	
	if opt.arch != None:
		optimization_flags = opt.arch.split()
	
	#
	# prepend boiler plate optimization flags that work on all architectures
	#
	
	optimization_flags[:0] = [
			"-O3",
			"-fomit-frame-pointer",
			"-ffast-math",
			"-fstrength-reduce",
			"-pipe"
			]
	
	if opt.debug:
		conf.env.append_value('CCFLAGS', debug_flags)
		conf.env.append_value('CXXFLAGS', debug_flags)
		conf.env.append_value('LINKFLAGS', debug_flags)
	else:
		conf.env.append_value('CCFLAGS', optimization_flags)
		conf.env.append_value('CXXFLAGS', optimization_flags)
		conf.env.append_value('LINKFLAGS', optimization_flags)
	
	if opt.stl_debug:
		conf.env.append_value('CXXFLAGS', "-D_GLIBCXX_DEBUG")
	
	if conf.env['DEBUG_RT_ALLOC']:
		conf.env.append_value('CCFLAGS', '-DDEBUG_RT_ALLOC')
		conf.env.append_value('CXXFLAGS', '-DDEBUG_RT_ALLOC')
		conf.env.append_value('LINKFLAGS', '-ldl')
	
	if opt.universal:
		conf.env.append_value('CCFLAGS', "-arch i386 -arch ppc")
		conf.env.append_value('CXXFLAGS', "-arch i386 -arch ppc")
		conf.env.append_value('LINKFLAGS', "-arch i386 -arch ppc")
	
	#
	# warnings flags
	#
	
	conf.env.append_value('CCFLAGS', "-Wall")
	conf.env.append_value('CXXFLAGS', [ '-Wall', '-Woverloaded-virtual'])
	
	if opt.extra_warn:
		flags = [ '-Wextra' ]
		conf.env.append_value('CCFLAGS', flags )
		conf.env.append_value('CXXFLAGS', flags )
	
	
	#
	# more boilerplate
	#
	
	conf.env.append_value('CCFLAGS', '-D_LARGEFILE64_SOURCE')
	conf.env.append_value('CCFLAGS', '-D_FILE_OFFSET_BITS=64')
	conf.env.append_value('CXXFLAGS', '-D_LARGEFILE64_SOURCE')
	conf.env.append_value('CXXFLAGS', '-D_FILE_OFFSET_BITS=64')
	
	conf.env.append_value('CXXFLAGS', '-D__STDC_LIMIT_MACROS')
	conf.env.append_value('CXXFLAGS', '-D__STDC_FORMAT_MACROS')
	
	if opt.nls:
		conf.env.append_value('CXXFLAGS', '-DENABLE_NLS')
		conf.env.append_value('CCFLAGS', '-DENABLE_NLS')
	
