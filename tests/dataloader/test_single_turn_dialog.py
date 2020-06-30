import copy
import os
import shutil
import pytest
from pytest_mock import mocker
import random
import operator
import sys
from pathlib import Path

from cotk.dataloader import PretrainedTokenizer, SingleTurnDialog, Field, Vocab, Tokenizer, OpenSubtitles
from cotk.metric import MetricBase
from cotk.file_utils import file_utils

from version_test_base import base_test_version

sys.path.insert(0, str(Path(__file__).parent.joinpath('../share').resolve()))
from cache_dir import CACHE_DIR

def setup_module():
	import random
	random.seed(0)
	import numpy as np
	np.random.seed(0)
	file_utils.CACHE_DIR = CACHE_DIR

def teardown_module():
	if os.path.isdir(CACHE_DIR):
		shutil.rmtree(CACHE_DIR)


class TestSingleTurnDialog():
	def base_test_init(self, dl):
		with pytest.raises(ValueError):
			SingleTurnDialog("./tests/dataloader/dummy_opensubtitles#OpenSubtitles", pretrained='none')
		with pytest.raises(ValueError):
			SingleTurnDialog("./tests/dataloader/dummy_opensubtitles#OpenSubtitles", pretrained='gpt2')
		with pytest.raises(ValueError):
			SingleTurnDialog("./tests/dataloader/dummy_opensubtitles#OpenSubtitles", pretrained='bert')

		assert isinstance(dl, SingleTurnDialog)
		assert isinstance(dl.file_id, str)
		assert isinstance(dl.file_path, str)
		for set_name, fields in dl.fields.items():
			assert isinstance(set_name, str)
			assert isinstance(fields, dict)
			for field_name, field in fields.items():
				assert isinstance(field_name, str)
				assert isinstance(field, Field)

		assert isinstance(dl.vocabs, list)
		for vocab in dl.vocabs:
			assert isinstance(vocab, Vocab)
		assert isinstance(dl.tokenizers, list)
		for toker in dl.tokenizers:
			assert isinstance(toker, Tokenizer)

		for (_, data), (_, index) in zip(dl.data.items(), dl.index.items()):
			assert isinstance(data, dict)
			assert isinstance(index, list)
			for field_name, content in data.items():
				assert isinstance(content, dict)
				for _, each_content in content.items():
					assert isinstance(each_content, list)
					assert len(index) == len(each_content)
		for _, batch_id in dl.batch_id.items():
			assert batch_id == 0
		for _, batch_size in dl.batch_size.items():
			assert batch_size is None

		assert isinstance(dl.frequent_vocab_list, list)
		assert dl.frequent_vocab_size == len(dl.frequent_vocab_list)
		assert isinstance(dl.all_vocab_list, list)
		assert dl.all_vocab_size == len(dl.all_vocab_list)
		assert dl.all_vocab_size >= dl.frequent_vocab_size

		for _, data in dl.data.items():
			post = data['post']
			post_ids = post['id']
			assert isinstance(post_ids, list)
			assert isinstance(post_ids[0], list)
			if dl._pretrained is None or dl._pretrained == "gpt2":
				assert post_ids[0][0] == dl.go_id
				assert post_ids[0][-1] == dl.eos_id
			else:  # dl._pretrained == "bert"
				assert post_ids[0][0] == dl.get_special_tokens_id("cls")
				assert post_ids[0][-1] == dl.get_special_tokens_id("sep")
			post_strs = post['str']
			assert isinstance(post_strs, list)
			assert isinstance(post_strs[0], str)

			resp = data['resp']
			resp_ids = resp['id']
			assert isinstance(resp_ids, list)
			assert isinstance(resp_ids[0], list)
			if dl._pretrained is None or dl._pretrained == "gpt2":
				assert post_ids[0][0] == dl.go_id
				assert post_ids[0][-1] == dl.eos_id
			else:  # dl._pretrained == "bert"
				assert post_ids[0][0] == dl.get_special_tokens_id("cls")
				assert post_ids[0][-1] == dl.get_special_tokens_id("sep")
			resp_strs = resp['str']
			assert isinstance(resp_strs, list)
			assert isinstance(resp_strs[0], str)

			assert len(post) == len(resp)

		with pytest.raises(TypeError):
			SingleTurnDialog()

	def base_test_all_unknown(self, dl):
		# if invalid_vocab_times very big, there is no invalid words.
		assert dl.frequent_vocab_size == dl.frequent_vocab_size

	def base_test_restart(self, dl):
		with pytest.raises(ValueError):
			dl.restart("unknown set")
		for set_name in dl.data.keys():
			with pytest.raises(ValueError):
				dl.restart(set_name)
			record_index = copy.copy(dl.index[set_name])
			dl.restart(set_name, batch_size=3, shuffle=False)
			assert record_index == dl.index[set_name]
			assert dl.batch_id[set_name] == 0
			assert dl.batch_size[set_name] == 3
			#rng_state_st = random.getstate()
			dl.restart(set_name, shuffle=True)
			#rng_state_ed = random.getstate()
			#assert operator.eq(rng_state_st, rng_state_ed)
			assert dl.batch_id[set_name] == 0
			record_index = copy.copy(dl.index[set_name])
			dl.restart(set_name, shuffle=False)
			assert record_index == dl.index[set_name]
			assert dl.batch_id[set_name] == 0

	def base_test_get_batch(self, dl):
		with pytest.raises(ValueError):
			dl.get_batch("unknown set", [0, 1])
		for set_name in dl.data.keys():
			with pytest.raises(IndexError):
				length = len(dl.index[set_name])
				dl.get_batch(set_name, [length-1, length])
			assert len(dl.index[set_name]) >= 2
			batch = dl.get_batch(set_name, [0, 1])

			assert len(batch["post_length"]) == 2
			assert len(batch["resp_length"]) == 2
			assert batch["post"].shape[0] == 2
			assert batch["resp"].shape[0] == 2

			for sent, length in [("post", "post_length"), ("resp", "resp_length")]:
				for idx in [0, 1]:
					if batch[length][idx] < batch[sent].shape[1]:
						if dl._pretrained is None or dl._pretrained == "gpt2":
							assert batch[sent][idx][batch[length][idx]-1] == dl.eos_id
							assert batch[sent][idx][0] == dl.go_id
						else:  # dl._pretrained == "bert":
							assert batch[sent][idx][batch[length][idx]-1] == dl.get_special_tokens_id("sep")
							assert batch[sent][idx][0] == dl.get_special_tokens_id("cls")

		if not dl._pretrained: # test only when not pretrained tokenizer
			# this is true, only when there is no unknown words in dl
			# (Only valid & invalid words)
			flag = False
			for set_name in dl.data.keys():
				length = len(dl.data[set_name]['post'])
				for i in range(length):
					batch = dl.get_batch(set_name, [i])
					assert dl.unk_id not in batch["post_allvocabs"]
					assert dl.unk_id not in batch["resp_allvocabs"]
					batch = dl.get_batch(set_name, [i])
					if dl.unk_id in batch["post"] or \
						dl.unk_id in batch["resp"]:
						flag = True
			assert flag

	def base_test_get_next_batch(self, dl):
		with pytest.raises(ValueError):
			dl.get_next_batch("unknown set")

		for set_name in dl.data.keys():
			with pytest.raises(RuntimeError):
				dl.get_next_batch(set_name)

			dl.restart(set_name, 7)
			sample_num = 0
			while True:
				batch = dl.get_next_batch(set_name, ignore_left_samples=True)
				if not batch:
					break
				assert batch["post"].shape[0] == 7
				sample_num += batch["post"].shape[0]
			assert sample_num + 7 >= len(dl.data[set_name]["post"]['id'])

			dl.restart(set_name, 7)
			sample_num = 0
			while True:
				batch = dl.get_next_batch(set_name)
				assert batch is not None # dummy dataset must not be multiple of 7
				if batch["post"].shape[0] == 7:
					sample_num += 7
				else:
					sample_num += batch["post"].shape[0]
					batch = dl.get_next_batch(set_name)
					assert not batch
					break
			assert sample_num == len(dl.data[set_name]["post"]['id'])

	def base_test_convert(self, dl): # test only when not pretrained tokenizer
		sent_id = [0, 1, 2]
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.convert_ids_to_tokens(sent_id)
		assert sent_id == dl.convert_tokens_to_ids(sent)

		sent = ["<unk>", "<go>", "<pad>", "<unkownword>", "<pad>", "<go>"]
		sent_id = [1, 2, 0, 1, 0, 2]
		assert sent_id == dl.convert_tokens_to_ids(sent)
		assert sent_id == dl.convert_tokens_to_ids(sent, only_frequent_word=True)

		sent = [dl.all_vocab_list[dl.frequent_vocab_size]]
		assert [1] == dl.convert_tokens_to_ids(sent, only_frequent_word=True)
		assert [dl.frequent_vocab_size] == dl.convert_tokens_to_ids(sent)


		sent_id = [0, 1, 2, 0, 0, 3, 1, 0, 0]
		sent = ["<pad>", "<unk>", "<go>", "<pad>", "<pad>", "<eos>", "<unk>", "<pad>", "<pad>"]
		assert sent == dl.convert_ids_to_tokens(sent_id, trim=False)
		sent = ["<pad>", "<unk>", "<go>"]
		assert sent == dl.convert_ids_to_tokens(sent_id)

		sent_id = [0, 0, 3]
		sent = ["<pad>", "<pad>", "<eos>"]
		assert sent == dl.convert_ids_to_tokens(sent_id, remove_special=False, trim=False)
		assert not dl.convert_ids_to_tokens(sent_id)

		sent_id = [3, 3, 3]
		sent = ["<eos>", "<eos>", "<eos>"]
		assert sent == dl.convert_ids_to_tokens(sent_id, remove_special=False, trim=False)
		assert not dl.convert_ids_to_tokens(sent_id)

		sent_id = [0, 0, 0]
		sent = ["<pad>", "<pad>", "<pad>"]
		assert sent == dl.convert_ids_to_tokens(sent_id, trim=False)
		assert not dl.convert_ids_to_tokens(sent_id)

	def base_test_teacher_forcing_metric(self, dl):
		assert isinstance(dl.get_teacher_forcing_metric(), MetricBase)

	def base_test_teacher_inference_metric(self, dl):
		assert isinstance(dl.get_inference_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.all_vocab_list == dl_list[0].all_vocab_list for x in dl_list)

def load_opensubtitles():
	def _load_opensubtitles(invalid_vocab_times=0):
		return OpenSubtitles("./tests/dataloader/dummy_opensubtitles#OpenSubtitles", min_rare_vocab_times=invalid_vocab_times)
	return _load_opensubtitles

def load_opensubtitles_gpt2():
	def _load_opensubtitles(invalid_vocab_times=0):
		from transformers import GPT2Tokenizer
		toker = PretrainedTokenizer(GPT2Tokenizer('./tests/dataloader/dummy_gpt2vocab/vocab.json', './tests/dataloader/dummy_gpt2vocab/merges.txt'))
		return OpenSubtitles("./tests/dataloader/dummy_opensubtitles#OpenSubtitles", tokenizer=toker, pretrained='gpt2', min_rare_vocab_times=invalid_vocab_times)
	return _load_opensubtitles

def load_opensubtitles_bert():
	def _load_opensubtitles(invalid_vocab_times=0):
		from transformers import BertTokenizer
		toker = PretrainedTokenizer(BertTokenizer('./tests/dataloader/dummy_bertvocab/vocab.txt'))
		return OpenSubtitles("./tests/dataloader/dummy_opensubtitles#OpenSubtitles", tokenizer=toker, pretrained='bert', min_rare_vocab_times=invalid_vocab_times)
	return _load_opensubtitles


all_load_dataloaders = [load_opensubtitles(), load_opensubtitles_gpt2(), load_opensubtitles_bert()]

class TestOpenSubtitles(TestSingleTurnDialog):
	def test_version(self):
		base_test_version(OpenSubtitles)

	def test_init_multi_runs(self, load_opensubtitles):
		super().base_test_multi_runs([load_opensubtitles() for i in range(3)])

	@pytest.mark.parametrize('load_dataloader', all_load_dataloaders)
	def test_init(self, load_dataloader):
		super().base_test_init(load_dataloader())
		super().base_test_all_unknown(load_dataloader(10000))
	
	@pytest.mark.parametrize('load_dataloader', all_load_dataloaders)
	def test_restart(self, load_dataloader):
		super().base_test_restart(load_dataloader())

	@pytest.mark.parametrize('load_dataloader', all_load_dataloaders)
	def test_get_batch(self, load_dataloader):
		super().base_test_get_batch(load_dataloader())

	@pytest.mark.parametrize('load_dataloader', all_load_dataloaders)
	def test_get_next_batch(self, load_dataloader):
		super().base_test_get_next_batch(load_dataloader())

	@pytest.mark.parametrize('load_dataloader', all_load_dataloaders[:1])
	def test_convert(self, load_dataloader):
		super().base_test_convert(load_dataloader())
	
	@pytest.mark.parametrize('load_dataloader', all_load_dataloaders)
	def test_teacher_forcing_metric(self, load_dataloader):
		super().base_test_teacher_forcing_metric(load_dataloader())
	
	@pytest.mark.parametrize('load_dataloader', all_load_dataloaders)
	def test_teacher_inference_metric(self, load_dataloader):
		super().base_test_teacher_inference_metric(load_dataloader())
	
	@pytest.mark.parametrize('load_dataloader', all_load_dataloaders)
	def test_init_multi_runs(self, load_dataloader):
		super().base_test_multi_runs([load_dataloader() for _ in range(3)])
