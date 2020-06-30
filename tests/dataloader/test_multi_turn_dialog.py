import random
from itertools import chain
from collections import OrderedDict
import os
import shutil
import pytest
import numpy as np
import sys
from pathlib import Path


from cotk.dataloader import MultiTurnDialog, Session, SwitchboardCorpus, UbuntuCorpus, PretrainedTokenizer
from cotk.metric import MetricBase
from cotk.dataloader.field import SentenceCandidateDefault, SentenceCandidateGPT2, SentenceCandidateBERT
from cotk.file_utils import file_utils
from cotk.wordvector import Glove
from test_dataloader import BaseTestLanguageProcessing
from version_test_base import base_test_version
from test_field import CheckGetBatch

sys.path.insert(0, str(Path(__file__).parent.joinpath('../share').resolve()))
from cache_dir import CACHE_DIR


def setup_module():
	random.seed(0)
	np.random.seed(0)
	file_utils.CACHE_DIR = CACHE_DIR

def teardown_module():
	if os.path.isdir(CACHE_DIR):
		shutil.rmtree(CACHE_DIR)


class TestMultiTurnDialog(BaseTestLanguageProcessing):
	def base_test_init(self, dl: MultiTurnDialog):
		with pytest.raises(ValueError):
			MultiTurnDialog("./tests/dataloader/dummy_switchboardcorpus#SwitchboardCorpus", pretrained='none')
		with pytest.raises(ValueError):
			MultiTurnDialog("./tests/dataloader/dummy_switchboardcorpus#SwitchboardCorpus", pretrained='gpt2')
		with pytest.raises(ValueError):
			MultiTurnDialog("./tests/dataloader/dummy_switchboardcorpus#SwitchboardCorpus", pretrained='bert')

		assert isinstance(dl, MultiTurnDialog)
		super().base_test_init(dl)
		assert dl.default_field_name is not None and dl.default_field_set_name is not None
		default_field = dl.get_default_field()
		assert isinstance(default_field, Session)

		assert isinstance(dl.frequent_vocab_list, list)
		assert dl.frequent_vocab_size == len(dl.frequent_vocab_list)
		assert isinstance(dl.all_vocab_list, list)
		assert dl.all_vocab_size == len(dl.all_vocab_list)
		assert dl.all_vocab_size > 4
		assert dl.all_vocab_size >= dl.frequent_vocab_size


	def base_test_get_batch(self, dl):
		super().base_test_get_batch(dl)
		for set_name in dl.fields:
			dl.restart(set_name, shuffle=True, batch_size=1)
			size = len(next(iter(dl.data[set_name].values())))
			for i in range(size):
				indexes = [i]
				batch_data = dl.get_batch(set_name, indexes)
				for field_name in dl.fields[set_name]:
					if isinstance(dl.fields[set_name][field_name], Session):
						CheckGetBatch.check_result_of_get_batch(dl.fields[set_name][field_name], field_name, dl.data[set_name][field_name], indexes, batch_data)

	def base_test_multi_turn_convert(self, dl: MultiTurnDialog):
		sent_id = [[0, 1, 2], [2, 1, 1]]
		sent = [["<pad>", "<unk>", "<go>"], ["<go>", "<unk>", "<unk>"]]
		assert sent == dl.convert_multi_turn_ids_to_tokens(sent_id, remove_special=False)
		assert sent_id == dl.convert_multi_turn_tokens_to_ids(sent)

		sent = [["<unk>", "<go>", "<pad>", "<unkownword>", "<pad>", "<go>"], ["<go>", "<eos>"]]
		sent_id = [[1, 2, 0, 1, 0, 2], [2, 3]]
		assert sent_id == dl.convert_multi_turn_tokens_to_ids(sent)

		sent_id = [[0, 1, 2, 2, 0, 3, 1, 0, 0], [0, 3, 2], [1, 2, 2, 0], [1, 2, 2, 3]]
		sent = [["<pad>", "<unk>", "<go>", "<go>", "<pad>", "<eos>", "<unk>", "<pad>", "<pad>"], \
				["<pad>", "<eos>", "<go>"], \
				["<unk>", "<go>", "<go>", "<pad>"], \
				["<unk>", "<go>", "<go>", "<eos>"]]
		assert sent == dl.convert_multi_turn_ids_to_tokens(sent_id, remove_special=False, trim=False)
		sent = [["<pad>", "<unk>", "<go>", "<go>"]]
		sent_id = [[0, 1, 2, 2]]
		assert sent == dl.convert_multi_turn_ids_to_tokens(sent_id)

		sent = [[dl.all_vocab_list[dl.unk_id]]]
		assert [[dl.unk_id]] == dl.convert_multi_turn_tokens_to_ids(sent)
		assert [[dl.unk_id]] == dl.convert_multi_turn_tokens_to_ids(sent, only_frequent_word=True)

	def base_test_teacher_forcing_metric(self, dl):
		assert isinstance(dl.get_teacher_forcing_metric(), MetricBase)

	def base_test_teacher_inference_metric(self, dl):
		assert isinstance(dl.get_inference_metric(), MetricBase)

	def base_test_multi_runs(self, dl_list):
		assert all(x.all_vocab_list == dl_list[0].all_vocab_list for x in dl_list)

def load_ubuntucorpus():
	def _load_ubuntucorpus(min_rare_vocab_times=0):
		return UbuntuCorpus("./tests/dataloader/dummy_ubuntucorpus#Ubuntu", min_rare_vocab_times=min_rare_vocab_times)
	return _load_ubuntucorpus

def load_ubuntucorpus_gpt2():
	def _load_ubuntucorpus(min_rare_vocab_times=0):
		from transformers import GPT2Tokenizer
		toker = PretrainedTokenizer(GPT2Tokenizer('./tests/dataloader/dummy_gpt2vocab/vocab.json', './tests/dataloader/dummy_gpt2vocab/merges.txt'))
		return UbuntuCorpus("./tests/dataloader/dummy_ubuntucorpus#Ubuntu", min_rare_vocab_times=min_rare_vocab_times, tokenizer=toker, pretrained="gpt2")
	return _load_ubuntucorpus


def load_ubuntucorpus_bert():
	def _load_ubuntucorpus(min_rare_vocab_times=0):
		from transformers import BertTokenizer
		toker = PretrainedTokenizer(BertTokenizer('./tests/dataloader/dummy_bertvocab/vocab.txt'))
		return UbuntuCorpus("./tests/dataloader/dummy_ubuntucorpus#Ubuntu", min_rare_vocab_times=min_rare_vocab_times, tokenizer=toker, pretrained="bert")
	return _load_ubuntucorpus


all_load_dataloaders = [load_ubuntucorpus(), load_ubuntucorpus_gpt2(), load_ubuntucorpus_bert()]

class TestUbuntuCorpus(TestMultiTurnDialog):
	def test_version(self):
		base_test_version(UbuntuCorpus)

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_init(self, load_dataloader):
		dl = load_dataloader()
		super().base_test_init(dl)
		assert isinstance(dl, UbuntuCorpus)
		assert set(dl.fields.keys()) == set(dl.data.keys()) == {'train', 'test', 'dev'}
		for fields_of_one_set in dl.fields.values():
			assert isinstance(fields_of_one_set, OrderedDict)
			assert len(fields_of_one_set) == 1
			assert isinstance(fields_of_one_set.get('session', None), Session)
		super().base_test_restart(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_get_batch(self, load_dataloader):
		super().base_test_get_batch(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_get_next_batch(self, load_dataloader):
		super().base_test_get_next_batch(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders[:1])
	def test_convert(self, load_dataloader):
		super().base_test_convert(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders[:1])
	def test_multi_turn_convert(self, load_dataloader):
		super().base_test_multi_turn_convert(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_teacher_forcing_metric(self, load_dataloader):
		super().base_test_teacher_forcing_metric(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_teacher_inference_metric(self, load_dataloader):
		super().base_test_teacher_inference_metric(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_init_multi_runs(self, load_dataloader):
		super().base_test_multi_runs([load_dataloader() for i in range(3)])


def load_switchboardcorpus():
	def _load_switchboardcorpus(min_rare_vocab_times=0):
		return SwitchboardCorpus("./tests/dataloader/dummy_switchboardcorpus#SwitchboardCorpus",
								 min_rare_vocab_times=min_rare_vocab_times)

	return _load_switchboardcorpus

def load_switchboardcorpus_gpt2():
	def _load_switchboardcorpus(min_rare_vocab_times=0):
		from transformers import GPT2Tokenizer
		toker = PretrainedTokenizer(GPT2Tokenizer('./tests/dataloader/dummy_gpt2vocab/vocab.json', './tests/dataloader/dummy_gpt2vocab/merges.txt'))
		return SwitchboardCorpus("./tests/dataloader/dummy_switchboardcorpus#SwitchboardCorpus",
								 min_rare_vocab_times=min_rare_vocab_times, tokenizer=toker, pretrained="gpt2")

	return _load_switchboardcorpus

def load_switchboardcorpus_bert():
	def _load_switchboardcorpus(min_rare_vocab_times=0):
		from transformers import BertTokenizer
		toker = PretrainedTokenizer(BertTokenizer('./tests/dataloader/dummy_bertvocab/vocab.txt'))
		return SwitchboardCorpus("./tests/dataloader/dummy_switchboardcorpus#SwitchboardCorpus",
								 min_rare_vocab_times=min_rare_vocab_times, tokenizer=toker, pretrained="bert")

	return _load_switchboardcorpus

all_load_dataloaders = [load_switchboardcorpus(), load_switchboardcorpus_gpt2(), load_switchboardcorpus_bert()]

class TestSwitchboardCorpus(TestMultiTurnDialog):
	def test_version(self):
		base_test_version(SwitchboardCorpus)

	SwitchboardCorpusSetNames = ('train', 'test', 'dev', 'multi_ref')

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_init(self, load_dataloader):
		dl = load_dataloader()
		super().base_test_init(dl)
		assert isinstance(dl, SwitchboardCorpus)
		set_names = self.SwitchboardCorpusSetNames
		assert set(dl.fields.keys()) == set(set_names)
		for set_name in set_names[:-1]:
			assert isinstance(dl.fields[set_name], OrderedDict)
			assert len(dl.fields[set_name]) == 1
			assert isinstance(dl.fields[set_name].get('session', None), Session)
		assert isinstance(dl.fields['multi_ref'], OrderedDict)
		assert len(dl.fields['multi_ref']) == 2
		assert all(a == b for a, b in zip(dl.fields['multi_ref'].keys(), ['session', 'candidate']))
		f1, f2 = dl.fields['multi_ref'].values()
		assert isinstance(f1, Session)
		if dl._pretrained is None:
			assert isinstance(f2, SentenceCandidateDefault)
		elif dl._pretrained == "gpt2" :
			assert isinstance(f2, SentenceCandidateGPT2)
		else:  # dl._pretrained == "bert"
			assert isinstance(f2, SentenceCandidateBERT)

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_restart(self, load_dataloader):
		super().base_test_restart(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_get_batch(self, load_dataloader):
		dl: SwitchboardCorpus = load_dataloader()
		super().base_test_get_batch(dl)

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_get_next_batch(self, load_dataloader):
		super().base_test_get_next_batch(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders[:1])
	def test_convert(self, load_dataloader):
		super().base_test_convert(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_teacher_forcing_metric(self, load_dataloader):
		super().base_test_teacher_forcing_metric(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_teacher_inference_metric(self, load_dataloader):
		super().base_test_teacher_inference_metric(load_dataloader())

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_teacher_precision_recall_metric(self, load_dataloader):
		dl = load_dataloader()
		glove = Glove("./tests/wordvector/dummy_glove/300d/")
		embed = glove.load_dict(dl.all_vocab_list)
		assert isinstance(dl.get_multi_ref_metric(generated_num_per_context=3, word2vec=embed), MetricBase)

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders)
	def test_init_multi_runs(self, load_dataloader):
		super().base_test_multi_runs([load_dataloader() for i in range(3)])

	@pytest.mark.parametrize("load_dataloader", all_load_dataloaders[:1])
	def test_multi_turn_convert(self, load_dataloader):
		super().base_test_multi_turn_convert(load_dataloader())
