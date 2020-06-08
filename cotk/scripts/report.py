'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import os
import os.path
import json
import sys
import argparse
import importlib
import traceback
import textwrap


import requests
from . import _utils
from . import cli_constant as cli
from ..hooks import start_recorder, close_recorder

LOGGER = cli.LOGGER
DASHBOARD_URL = cli.DASHBOARD_URL
REPORT_URL = DASHBOARD_URL + "/upload"
SHOW_URL = DASHBOARD_URL + "/show?id=%d"
BACKUP_FILE = '.cotk_upload_backup'
FILL = lambda str: textwrap.fill(textwrap.dedent(str))

def run_model(entry, args, result_path):
	'''Run the model and record the info of library'''
	# before run model
	# cotk recorder start
	start_recorder()
	sys.path.insert(0, os.getcwd())
	model = importlib.import_module(entry)

	if args is None:
		args = []

	try:
		model.run(*args)
	except Exception as _: #pylint: disable=broad-except
		traceback.print_exc()
		sys.exit(1)

	# after run model
	# cotk recorder end
	results = read_and_validate_result(result_path)
	return close_recorder(results)

def read_and_validate_result(result_path):
	if not os.path.isfile(result_path):
		raise ValueError("Result file ({}) is not found.".format(result_path))
	try:
		result = json.load(open(result_path, "r", encoding='utf-8'))
	except json.JSONDecodeError as err:
		raise json.JSONDecodeError("{} is not a valid json. {}".format(result_path, err.msg),\
				err.doc, err.pos)
	return result


def upload_report(result_path, entry, args, working_dir, \
	git_user, git_repo, git_commit, \
	cotk_record_information, token):
	'''Upload report to dashboard. Return id of the new record.'''
	# check result file existence
	# get git link
	result = read_and_validate_result(result_path)

	upload_information = { \
		"entry": entry, \
		"args": args, \
		"working_dir": working_dir, \
		"git_user": git_user, \
		"git_repo": git_repo, \
		"git_commit": git_commit, \
		"record_information": cotk_record_information, \
		"result": result \
	}
	#LOGGER.info("Save your report locally at {}".format(BACKUP_FILE))
	#json.dump(upload_information, open(BACKUP_FILE, 'w'))
	LOGGER.info("Uploading your report...")
	res = requests.post(REPORT_URL, {"data": json.dumps(upload_information), "token": token})
	res = json.loads(res.text)
	if res['code'] != "ok":
		if res['code'] == 'wrong format' or res['code'] == 'invalid form':
			raise RuntimeError(FILL("""
				The upload information is unrecognized. If this persists,
				upgrade cotk to a newer version. Detailed error: %s""" % res['err']))
		elif res['code'] == 'bad token':
			raise RuntimeError(FILL("""
				The token is invalid. Please reset your token using
				`cotk config set token <TOKEN>`, where your <TOKEN> can be find
				at %s/profile.""" % DASHBOARD_URL))
		else:
			raise RuntimeError("Unknown error. Code: %s" % res['code'])
	return res['id']

def get_local_token():
	'''Read locally-saved token'''
	if os.path.exists(cli.CONFIG_FILE):
		return json.load(open(cli.CONFIG_FILE, 'r', encoding='utf-8'))['token']
	else:
		raise RuntimeError("Please set your token. \n" + \
						   "    Either using \"--token\" to specify token temporarily\n" + \
						   "    or set token in global using `cotk config set token <TOKEN>`.\n" + \
						   "If you don't have a token, sign up at Dashboard and find your token at user profile." + \
						   "Cotk dashboard: %s" % DASHBOARD_URL)

def verify_token_online(token):
	res = requests.post(REPORT_URL, {"token": token})
	res = json.loads(res.text)
	return res['code'] != 'bad token'

def run(args):
	'''Entrance of run'''
	parser = argparse.ArgumentParser(prog="cotk run", \
		description='Run model and report performance to cotk dashboard.\
**More args can added at end of the command to pass to your model**.')
	parser.add_argument('--token', type=str, default=None, \
		help='Use a temporary token. (Specified your accounts on dashboard.)')
	parser.add_argument('--result', type=str, default="result.json", \
		help='Path to result file that your model generated. Default: result.json')
	parser.add_argument("--only-run", action="store_true", \
		help="Just run my model, save running information to local folder but do not upload anything.")
	parser.add_argument('--only-upload', action="store_true", \
		help="Don't run my model, just upload the existing result.")
	parser.add_argument('--entry', type=str, default="main", nargs='?',\
		help="Entry file of your model, suffix name '.py' should not be included. \
			  Default: main")
	#parser.add_argument('args', nargs='*', help="args passed to your model")

	cargs, extra_args = parser.parse_known_args(args)

	if not cargs.only_run:
		if cargs.token:
			token = cargs.token
		else:
			token = get_local_token()
		if not verify_token_online(token):
			raise RuntimeError(FILL("""
				The token is invalid. Please reset your token using
				`cotk config set token <TOKEN>`, where your <TOKEN> can be find
				at %s/profile.""" % DASHBOARD_URL))

		_utils.assert_repo_exist()
		if not _utils.check_repo_clean():
			raise RuntimeError("Your changes of code hasn't been committed. Use \"git status\" \
to check your changes.")

		git_user, git_repo = _utils.get_repo_remote()
		git_commit = _utils.get_repo_commit()
		#_utils.assert_commit_exist(git_user, git_repo, git_commit)
		LOGGER.info("git information detected.")
		LOGGER.info("user: %s, repo: %s, commit sha1: %s", git_user, git_repo, git_commit)

	try:
		_utils.assert_repo_exist()
		root_path = _utils.get_repo_root_path()
		config_file = os.path.join(root_path, ".model_config.json")
		in_git = True
	except RuntimeError:
		config_file = ".model_config.json"
		in_git = False


	if not cargs.only_upload:
		LOGGER.info("Running your model at '%s' with arguments: %s.", cargs.entry, extra_args)
		cotk_record_information = run_model(cargs.entry, extra_args, cargs.result)
		LOGGER.info("Your model has exited.")

		if in_git:
			working_dir = _utils.get_repo_workingdir()
			data = {"entry": cargs.entry, "args": extra_args,\
			"cotk_record_information": cotk_record_information, "working_dir": working_dir}
		else:
			data = {"entry": cargs.entry, "args": extra_args,\
			"cotk_record_information": cotk_record_information}

		json.dump(data, open(config_file, 'w', encoding='utf-8'))
		LOGGER.info("Runtime information has dumped into %s.", config_file)
	else:

		if not os.path.isfile(config_file):
			raise RuntimeError(".model_config.json not found. It seems you have not run your \
model with cotk. You can't use \"only-upload\" before \"only-run\".")
		data = json.load(open(config_file, 'r', encoding='utf-8'))

	if not cargs.only_run:
		LOGGER.info("Collecting info for upload...")
		upload_id = upload_report(cargs.result, data["entry"], data["args"], data.get("working_dir", ''),\
			git_user, git_repo, git_commit, \
			data["cotk_record_information"], token)
		LOGGER.info("Upload complete. Check %s for your report.", SHOW_URL % upload_id)
