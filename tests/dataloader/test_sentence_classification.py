import os
import shutil
import pytest
import numpy as np
import sys
from pathlib import Path

from cotk.dataloader import PretrainedTokenizer, SentenceClassification, SST, Sentence
from cotk.metric import MetricBase
from cotk.file_utils import file_utils



from test_dataloader import BaseTestLanguageProcessing
from version_test_base import base_test_version
from test_field import CheckGetBatch

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

class TestSentenceClassification(BaseTestLanguageProcessing):
	def base_test_init(self, dl: SentenceClassification):
		super().base_test_init(dl)
		assert isinstance(dl, SentenceClassification)
		if dl._pretrained is None:
			assert list(dl.get_special_tokens_mapping().values()) == ["<pad>", "<unk>", "<go>", "<eos>"]
			assert [dl.pad_id, dl.unk_id, dl.go_id, dl.eos_id] == [0, 1, 2, 3]
		elif dl._pretrained == "gpt2":
			assert list(dl.get_special_tokens_mapping().values()) == ["<|endoftext|>", "<|endoftext|>", "<|endoftext|>"]
			assert [dl.unk_id, dl.go_id, dl.eos_id] == [413, 413, 413]
		else:  # dl._pretraiend == "bert"
			assert list(dl.get_special_tokens_mapping().values()) == ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]
			assert dl.get_special_tokens_id("pad") == 0
			assert dl.get_special_tokens_id("unk") == 100
			assert dl.get_special_tokens_id("cls") == 101
			assert dl.get_special_tokens_id("sep") == 102
			assert dl.get_special_tokens_id("mask") == 103

		assert isinstance(dl.all_vocab_list, list)
		assert dl.all_vocab_size == len(dl.all_vocab_list)
		for i, word in enumerate(dl.all_vocab_list):
			assert isinstance(word, str)
			assert dl.convert_tokens_to_ids([word])[0] == i
		assert dl.all_vocab_size == len(dl.all_vocab_list)

	def base_test_all_unknown(self, dl: SentenceClassification):
		# if min_rare_vocab_times very big, there is no rare words.
		assert dl.frequent_vocab_size == dl.all_vocab_size

	def base_test_metric(self, dl):
		assert isinstance(dl.get_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.all_vocab_list == dl_list[0].all_vocab_list for x in dl_list)

	def _test_get_batch_of_sentence_field(self, batch_data: dict, field_name, indexes:list):
		"""Help function for testing :meth:`Dataloader.get_batch.` The dataloader must have a sentence field.

		Args:
			batch_data: return value of get_batch.
			field_name: argument of Sentence.get_batch
			indexes: argument of Sentence.get_batch
		"""
		assert isinstance(batch_data.get(field_name, None), np.ndarray)
		assert len(batch_data[field_name].shape) == 2 and batch_data[field_name].shape[0] == len(indexes)
		assert isinstance(batch_data.get(field_name + '_length', None), np.ndarray)
		assert batch_data[field_name + '_length'].shape == (len(indexes), )
		assert isinstance(batch_data.get(field_name + '_allvocabs', None), np.ndarray)
		assert batch_data[field_name + '_allvocabs'].shape[0] == len(indexes) and len(batch_data[field_name + '_allvocabs'].shape) == 2
		assert isinstance(batch_data.get(field_name + '_str', None), list)
		assert isinstance(batch_data[field_name + '_str'][0], str)

		shape = batch_data[field_name].shape
		assert shape[1] == max(batch_data[field_name + '_length'])

	def base_test_get_batch(self, dl: SentenceClassification):
		super().base_test_get_batch(dl)
		for set_name in dl.fields:
			dl.restart(set_name, shuffle=True, batch_size=1)
			size = len(next(iter(dl.data[set_name].values())))
			for i in range(size):
				indexes = [i]
				batch_data = dl.get_batch(set_name, indexes)
				for field_name in dl.fields[set_name]:
					if isinstance(dl.fields[set_name][field_name], Sentence):
						CheckGetBatch.check_result_of_get_batch(dl.fields[set_name][field_name], field_name, dl.data[set_name][field_name], indexes, batch_data)

def load_sst():
	def _load_sst(min_rare_vocab_times=0):
		return SST("./tests/dataloader/dummy_sst#SST", min_rare_vocab_times=min_rare_vocab_times)
	return _load_sst

def load_sst_gpt2():
	def _load_sst(min_rare_vocab_times=0):
		from transformers import GPT2Tokenizer
		toker = PretrainedTokenizer(GPT2Tokenizer('./tests/dataloader/dummy_gpt2vocab/vocab.json', './tests/dataloader/dummy_gpt2vocab/merges.txt'))
		return SST("./tests/dataloader/dummy_sst#SST", tokenizer=toker, min_rare_vocab_times=min_rare_vocab_times, pretrained="gpt2")
	return _load_sst

def load_sst_bert():
	def _load_sst(min_rare_vocab_times=0):
		from transformers import BertTokenizer
		toker = PretrainedTokenizer(BertTokenizer('./tests/dataloader/dummy_bertvocab/vocab.txt'))
		return SST("./tests/dataloader/dummy_sst#SST", tokenizer=toker, min_rare_vocab_times=min_rare_vocab_times, pretrained="bert")
	return _load_sst

all_load_dataloaders = [load_sst(), load_sst_gpt2(), load_sst_bert()]

class TestSST(TestSentenceClassification):
	def test_version(self):
		base_test_version(SST)

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_init(self, load_dataloader):
		super().base_test_init(load_dataloader())
		super().base_test_all_unknown(load_dataloader(1000000))

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_restart(self, load_dataloader):
		super().base_test_restart(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_get_batch(self, load_dataloader):
		super().base_test_get_batch(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_get_next_batch(self, load_dataloader):
		super().base_test_get_next_batch(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders[:1])
	def test_convert(self, load_dataloader):
		 # test only when not pretrained tokenizer
		super().base_test_convert(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_metric(self, load_dataloader):
		super().base_test_metric(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_init_multi_runs(self, load_dataloader):
		super().base_test_multi_runs([load_dataloader() for i in range(3)])
