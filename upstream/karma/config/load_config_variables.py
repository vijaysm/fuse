#! /usr/bin/env python
# encoding: utf-8

import subprocess, re

from subprocess import Popen, PIPE

def partition(x, sep):
	try:
		func = getattr(x, "partition")
		return func(sep)
	except AttributeError :	# catch the AttributeError (Python version < 2.5)
		s = x.split(sep)
		vars = {}
		vars[0] = s[0]
		
		s.remove(s[0])
		if len(sep.join(s)):
			vars[1] = sep
			vars[2] = sep.join(s)
		else:
			vars[1] = ''
			vars[2] = ''
		
		return vars

def load_config_file(file, cvars):
	
	configfile = open(file,'r')
	configlist = configfile.readlines()
	
	for x in configlist:
		#print x
		x = x.strip()
		if x[:1] == '#' or x[:1] == '%' or x[:1] == '@' or \
		   x[:4] == 'ifeq' or x[:4] == 'else' or x[:5] == 'endif' or \
		   x[:3] == 'inc' or len(x) == 1: 	# x[:1] == '\n':
			pass
		else:
			#vars = x.partition(':=')
			vars = partition(x, ':=')
			
			# add if it really is a key-value pair
			if len(vars) == 3 and len(vars[0].strip()) > 0 and len(vars[1].strip()) == 0:
				#vars = x.partition('=')
				vars = {}
				vars = partition(x, '=')
			
			
			if len(vars) > 1 and len(vars[0].strip()) > 0 and len(vars[1].strip()) > 0:
				if len(vars) > 2:
					cvars[vars[0].strip()] = vars[2].strip()
				else:
					cvars[vars[0].strip()] = ''	# empty value

def unroll_variables(env, dict):
	#unroll_variables_elegant(env, dict)
	unroll_variables_recursive(env, dict)

def unroll_variables_elegant(env, dict):
	print dict
	for k,v in dict.iteritems():
		dict[k] = expand_variables(k, v, dict, env)
	print dict


def unroll_variables_recursive(env, dict):

	import string
	
	#print dict	
	#print env
	
	index = 1
	while True:
		masked_variables = 0
		for key, value in dict.iteritems():
			#print 'Before %s=%s' % (key, value)
			#print p.match(value)
			confvars_env = re.findall('\$\{([^}]+)\}', value)
			confvars = re.findall('\$\(([^}]+)\)', value)
			
			masked_variables = masked_variables+len(confvars)+len(confvars_env)
			
			if len(confvars_env) > 0:
				for var in confvars_env:
					#print 'replacing %s' % var
					if var in env:
						value = value.replace('${%s}'%var, "".join(env[var]))
					else:
						if dict.has_key(var):
							#print 'replacing ${%s} with %s in %s' % (var, dict[var], value)
							#print string.find(value, '${%s}'%var)
							value = value.replace('${%s}'%var, dict[var])
							#print 'value after replacement: %s' % value
						else:
							#print 'Cant find variable = {%s}' % var
							#print 'trying to replace %s in %s' % (var, value)
							value = value.replace('${%s}'%var, '')
								
				dict[key] = value.strip()
				
			if len(confvars) > 0:
				for var in confvars:
					#print 'replacing %s' % var
					if var in env:
						value = value.replace('$(%s)'%var, "".join(env[var]))
					else:
						if dict.has_key(var):
							#print 'replacing $(%s) with %s in %s' % (var, dict[var], value)
							#print string.find(value, '${%s}'%var)
							value = value.replace('$(%s)'%var, dict[var])
							#print 'value after replacement: %s' % value
						else:
							#print 'Cant find variable = (%s)' % var
							#print 'trying to replace %s in %s' % (var, value)
							value = value.replace('$(%s)'%var, '')
				
				dict[key] = value.strip()
					
			#print 'After  %s=%s \n%s' % (key, dict[key], value)
	
		#print 'Index %d - Num variables %d' % (index, masked_variables)
		if not masked_variables:
			break
		
		if index > 20:
			print 'Error in unrolling variables; maximum cycles reached.'
			break
		index = index+1
	
	#print dict
	

regex_expr = re.compile(r'\${(.*?)}')
def expand_variables (var, val, src, env):
	if var in env:
		val = val.replace('${%s}'%var, "".join(env[var]))
	else:
		#print 'trying to replace %s in %s' % (var, value)
		val = val.replace('${%s}'%var, '')
	return regex_expr.sub(lambda m: expand_variables(var, src.get(m.group(1), ''), src, env), val)


def unroll_makefile(file):
	# make -f Makefile -f config/helper.mak printvars
	makefile_vars = Popen(['/usr/bin/make', '-f', file, '-f', 'config/helper.mak', 'printvars'], stdout=PIPE).stdout.read()
	#print makefile_vars
	return makefile_vars

def get_makefile_variable(file, var_name):
	# make -f Makefile -f config/helper.mak echo_VARNAME
	var_value = Popen(['/usr/bin/make', '-f', file, '-f', 'config/helper.mak', 'echo_%s'%var_name], stdout=PIPE).stdout.read()
	#print var_value
	return var_value

def get_all_makefile_variables(file):
	# make -f Makefile -f config/helper.mak echo_VARNAME
	var_value = Popen(['/usr/bin/make', '-f', file, '-f', 'config/helper.mak', 'echo_all'], stdout=PIPE).stdout.read()
	#print var_value
	makefile_list = var_value.split('!! ')
	makefile_vars = {}
	for item in makefile_list:
		entry = item.split('=')
		makefile_vars[entry[0]] = entry[1]
	return makefile_vars
