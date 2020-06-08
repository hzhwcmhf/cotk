'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import sys
from . import import_local_resources, config

def show_command():
	'''show help'''
	print(r"""usage: cotk <command> [...]

command list:
   import       Import local files to cotk cache.
   config       Set variables at the config file (at ~/.cotk_config).

You can type `cotk <command>` for details of each command.
""")

def dispatch(sub_entrance, args):
	'''Dispatcher of sub-entrances'''
	# if sub_entrance == 'run':
	# 	report.run(args)
	# elif sub_entrance == 'download':
	# 	from . import download
	# 	download.download(args)
	if sub_entrance == 'import':
		import_local_resources.entry(args)
	elif sub_entrance == 'config':
		config.entry(args)
	else:
		print("Unknown command.\n")
		show_command()

def main():
	'''Entry of command line'''
	sys.path.append(".")
	if len(sys.argv) > 2 and sys.argv[1] == "debug":
		dispatch(sys.argv[2], sys.argv[3:]) # do not use "try" to show full stacktrace
	elif len(sys.argv) > 1:
		try:
			dispatch(sys.argv[1], sys.argv[2:])
		except Exception as err: #pylint: disable=broad-except
			print("%s: %s" % (type(err).__name__, err))
	else:
		show_command()

if __name__ == "__main__":
	dispatch(sys.argv[1], sys.argv[2:])
