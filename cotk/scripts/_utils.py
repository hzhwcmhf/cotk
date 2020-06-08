'''
A command library help user upload their results to dashboard.
'''
#!/usr/bin/env python
import subprocess
from subprocess import PIPE
import re
import sys

import requests

def assert_repo_exist():
	'''Assert cwd is in a git repo.'''
	try:
		in_git = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], stdout=PIPE, stderr=PIPE)
	except FileNotFoundError as _:
		raise RuntimeError("Git is not found. You must install git and \
make sure git command can be used.")

	if in_git.stdout.decode(sys.stdout.encoding).strip() != "true":
		raise RuntimeError("You have to create a git repo and make a commit first.")

def check_repo_clean():
	'''Check whether repo is clean.
	Return True if clean, False if dirty.'''
	git_diff = subprocess.run(["git", "diff", "HEAD"], stdout=PIPE, stderr=PIPE)
	if git_diff.stdout.decode(sys.stdout.encoding):
		return False
	else:
		return True

def get_repo_workingdir():
	'''Get relative path of cwd from git repo root.'''
	git_prefix = subprocess.run(["git", "rev-parse", "--show-prefix"], stdout=PIPE, stderr=PIPE)
	return git_prefix.stdout.decode(sys.stdout.encoding).strip()

def get_repo_root_path():
	git_toplevel = subprocess.run(["git", "rev-parse", "--show-toplevel"], stdout=PIPE, stderr=PIPE)
	return git_toplevel.stdout.decode(sys.stdout.encoding).strip()

def get_repo_remote():
	'''Get remote repo name on github'''
	git_upstream = subprocess.run(["git", "rev-parse", "--symbolic-full-name", "--abbrev-ref", \
			"@{upstream}"], stdout=PIPE, stderr=PIPE)
	err = git_upstream.stderr.decode(sys.stdout.encoding)
	if err:
		if re.match(r"fatal: no upstream configured for branch", err):
			raise RuntimeError("No upstream branch, you have to set upstream branch for your repo. \
E.g. git push -u origin master. ")
		else:
			raise RuntimeError("Unkown error when getting upstream branch: %s" % err)

	upstream_out = git_upstream.stdout.decode(sys.stdout.encoding).split('/')
	remote_name, remote_branch = upstream_out[0], upstream_out[1] #pylint: disable=unused-variable

	git_remote = subprocess.run(["git", "remote", "-v"], stdout=PIPE, stderr=PIPE)
	ssh_reg = re.search(r"%s\s+git@github.com:(\S+?)/(\S+?)\.git\s+\(push\)" % \
			remote_name, git_remote.stdout.decode(sys.stdout.encoding))
	http_reg = re.search(r"%s\s+https?://github\.com/(\S+?)/(\S+?)\.git\s+\(push\)" % \
			remote_name, git_remote.stdout.decode(sys.stdout.encoding))
	if ssh_reg:
		git_user = ssh_reg.group(1)
		git_repo = ssh_reg.group(2)
	elif http_reg:
		git_user = http_reg.group(1)
		git_repo = http_reg.group(2)
	else:
		raise RuntimeError("No remote named %s, please use 'git remote add' to identify \
your remote repo on github." % git_remote)

	return git_user, git_repo

def get_repo_commit():
	'''Return the commit sha of HEAD'''
	git_head = subprocess.run(["git", "rev-parse", "HEAD"], stdout=PIPE, stderr=PIPE)
	if git_head.stdout.decode(sys.stdout.encoding).find("fatal: Needed a single revision") >= 0:
		raise RuntimeError("You have to make a commit in your git repo first.")
	return git_head.stdout.decode(sys.stdout.encoding).strip()

def assert_commit_exist(git_user, git_repo, git_commit):
	'''Assert commit is available'''
	url = "https://github.com/{}/{}/archive/{}.zip".format(git_user, git_repo, git_commit)
	nonexist = True
	err_msg = None
	trial = 3
	while trial:
		try:
			res = requests.head(url)
			nonexist = not res.ok
			break
		except Exception as err:
			err_msg = err
			trial -= 1
	if not trial:
		raise RuntimeError("3 failed trials to query {}.\n{}".format(url, str(err_msg)))
	if nonexist:
		raise RuntimeError("Commit {} does not exist on github:{}/{}. \
It should be public.".format( \
			git_commit, git_repo, git_user \
		))

def git_clone(git_user, git_repo):
	url = "https://github.com/{}/{}.git".format(git_user, git_repo)
	git_clone_msg = subprocess.run(["git", "clone", url], stdout=PIPE, stderr=PIPE)
	if "fatal:" in git_clone_msg.stderr.decode(sys.stdout.encoding):
		raise RuntimeError(git_clone_msg.stderr.decode(sys.stdout.encoding))

def git_checkout_commit(git_commit):
	git_fetch_msg = subprocess.run(["git", "fetch", "origin", git_commit], stdout=PIPE, stderr=PIPE)
	if "fatal:" in git_fetch_msg.stderr.decode(sys.stdout.encoding):
		raise RuntimeError(git_fetch_msg.stderr.decode(sys.stdout.encoding))
	git_checkout_msg = subprocess.run(["git", "checkout", git_commit], stdout=PIPE, stderr=PIPE)
	if "fatal:" in git_checkout_msg.stderr.decode(sys.stdout.encoding):
		raise RuntimeError(git_checkout_msg.stderr.decode(sys.stdout.encoding))
