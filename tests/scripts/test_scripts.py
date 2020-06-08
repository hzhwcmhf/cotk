import os
from pathlib import Path
import shutil
from contextlib import suppress

import pytest

from cotk.scripts import main, config, cli_constant

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
		backup = False
		if Path(cli_constant.CONFIG_FILE).is_file():
			backup = True
			shutil.move(cli_constant.CONFIG_FILE, cli_constant.CONFIG_FILE + ".bak")

		assert config.config_load("test_variable") is None

		main.dispatch('config', ["set", 'test_variable', "123"])
		main.dispatch('config', ["show", 'test_variable'])
		assert config.config_load("test_variable") == "123"

		main.dispatch('config', ["set", 'test_variable', "123", "456"])
		main.dispatch('config', ["show", 'test_variable'])
		assert config.config_load("test_variable") == "123 456"

		if backup:
			shutil.copyfile(cli_constant.CONFIG_FILE + ".bak", cli_constant.CONFIG_FILE)

	def test_import_local_resources(self):
		with suppress(FileNotFoundError):
			os.remove('./cotk/resource_config/test.json')
		with suppress(FileNotFoundError):
			os.remove(str(Path.home()) + '/.cotk_cache/9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08.json')
		with suppress(FileNotFoundError):
			os.remove(str(Path.home()) + '/.cotk_cache/9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08')

		shutil.copyfile('./tests/file_utils/dummy_coai/test.json', './cotk/resource_config/test.json')
		main.dispatch('import', ['resources://test', './tests/file_utils/data/test.zip'])
		os.remove('./cotk/resource_config/test.json')
		os.remove(str(Path.home()) + '/.cotk_cache/9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08.json')
		os.remove(str(Path.home()) + '/.cotk_cache/9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08')

	def test_unknown_dispatch(self):
		main.dispatch('unknown', [])
