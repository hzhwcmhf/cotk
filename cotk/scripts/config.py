'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import json
import argparse

from . import cli_constant as cli

def load_config():
	try:
		config_dict = json.load(open(cli.CONFIG_FILE, 'r', encoding='utf-8'))
	except (FileNotFoundError, json.JSONDecodeError):
		config_dict = {}
	return config_dict


def config_set(variable, value):
	config_dict = load_config()
	config_dict[variable] = value
	json.dump(config_dict, open(cli.CONFIG_FILE, 'w', encoding='utf-8'))
	cli.LOGGER.info("Save your configuration locally at {}".format(cli.CONFIG_FILE))


def config_load(variable):
	config_dict = load_config()
	return config_dict.get(variable, None)

def entry(args):
	'''Entrance of configuration'''
	parser = argparse.ArgumentParser(prog="cotk config", \
		description='Configuration (e.g. token)')
	parser.add_argument("action", type=str, choices=["set", "show"])
	parser.add_argument("variable", type=str)
	parser.add_argument("value", type=str, nargs="*")
	cargs = parser.parse_args(args)

	if cargs.action == "set":
		config_set(cargs.variable, " ".join(cargs.value))
		cli.LOGGER.info("%s = %s", cargs.variable, " ".join(cargs.value))
	elif cargs.action == "show":
		value = config_load(cargs.variable)
		cli.LOGGER.info("%s = %s", cargs.variable, value)
	else:
		raise RuntimeError("Unkown action.")
