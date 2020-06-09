import os
from pathlib import Path
import shutil
from contextlib import suppress
import sys

import pytest

from cotk.scripts import main, config, cli_constant
from cotk.file_utils import file_utils

sys.path.insert(0, str(Path(__file__).parent.joinpath('../share').resolve()))
import cache_dir
from cache_dir import CACHE_DIR, CONFIG_DIR, CONFIG_FILE

def setup_module():
	if os.path.isdir(CACHE_DIR):
		shutil.rmtree(CACHE_DIR)
	if os.path.isdir(CONFIG_DIR):
		shutil.rmtree(CONFIG_DIR)
	if os.path.isfile(CONFIG_FILE):
		os.remove(CONFIG_FILE)

	file_utils.CACHE_DIR = CACHE_DIR
	file_utils.CONFIG_DIR = CONFIG_DIR
	os.makedirs(CONFIG_DIR)
	cli_constant.CONFIG_FILE = CONFIG_FILE

def teardown_module():
	if os.path.isdir(CACHE_DIR):
		shutil.rmtree(CACHE_DIR)
	if os.path.isdir(CONFIG_DIR):
		shutil.rmtree(CONFIG_DIR)
	if os.path.isfile(CONFIG_FILE):
		os.remove(CONFIG_FILE)

class TestScripts():
	# disable the download module

	# i = 0

	# def setup(self):
	# 	try:
	# 		shutil.rmtree("cotk-test-CVAE")
	# 	except FileNotFoundError:
	# 		pass
	# 	except PermissionError:
	# 		os.rename("cotk-test-CVAE", "cotk-test-CVAE" + str(TestScripts.i))
	# 		TestScripts.i += 1

	# @pytest.mark.parametrize('url, error, match', \
	# 	[("http://wrong_url", ValueError, "can't match any pattern"),\
	# 	('user/repo/commit/wrong_commit', RuntimeError, "fatal"),\
	# 	#('http://github.com/thu-coai/cotk-test-CVAE/no_result_file/', FileNotFoundError, r"Config file .* is not found."),\  // no config file is acceptable
	# 	('https://github.com/thu-coai/cotk-test-CVAE/tree/invalid_json', json.JSONDecodeError, ""),\
	# 	])
	# def test_download_error(self, url, error, match):
	# 	with pytest.raises(error, match=match):
	# 		dispatch('download', [url])

	# def test_download(self):
	# 	# with pytest.raises(FileNotFoundError) as excinfo:
	# 	# 	report.dispatch('download', \
	# 	# 					['--zip_url', 'https://github.com/thu-coai/cotk-test-CVAE/archive/no_output.zip'])
	# 	# assert "New result file not found." == str(excinfo.value)
	# 	dispatch('download', ['https://github.com/thu-coai/cotk-test-CVAE/tree/run_and_test'])

	def test_config(self):
		assert config.config_load("test_variable") is None

		main.dispatch('config', ["set", 'test_variable', "123"])
		main.dispatch('config', ["show", 'test_variable'])
		assert config.config_load("test_variable") == "123"

		main.dispatch('config', ["set", 'test_variable', "123", "456"])
		main.dispatch('config', ["show", 'test_variable'])
		assert config.config_load("test_variable") == "123 456"

	def test_import_local_resources(self):
		shutil.copyfile('./tests/file_utils/dummy_coai/test.json', CONFIG_DIR + '/test.json')
		main.dispatch('import', ['resources://test', './tests/file_utils/data/test.zip'])

	def test_unknown_dispatch(self):
		main.dispatch('unknown', [])
