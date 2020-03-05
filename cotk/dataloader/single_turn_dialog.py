'''
A module for single turn dialog.
'''
from collections import Counter, OrderedDict
from itertools import chain
import multiprocessing
from multiprocessing import Pool
import tqdm
from typing import Optional, Any, List, Tuple, Dict

import numpy as np

from nltk.tokenize import WordPunctTokenizer
from ..file_utils import get_resource_file_path
from ..hooks import hooks
from .dataloader import LanguageProcessing
from .tokenizer import PretrainedTokenizer
from .vocab import PretrainedVocab
# from .bert_dataloader import BERTLanguageProcessing
from ..metric import MetricChain, PerplexityMetric, BleuCorpusMetric, SingleTurnDialogRecorder
from .context import FieldContext, VocabContext

# pylint: disable=W0223
class SingleTurnDialog(LanguageProcessing):
	"""Bases: :class:`.dataloader.LanguageProcessing`

	This class is supported for sequence to sequence generation tasks, especially
	single turn dialog tasks.

	Arguments:{ARGUMENTS}
	"""

	_version = 1

	ARGUMENTS = r'''
			file_id (str): A string indicating the source of single turn dialog dataset. {FILE_ID_DEFAULT}
			frequent_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
				not less than ``min_vocab_times`` in **training set** will be marked as valid words.
				{VALID_VOCAB_TIMES_DEFAULT}
			max_sent_length (int): All sentences longer than ``max_sent_length`` will be shortened
				to first ``max_sent_length`` tokens. {MAX_SENT_LENGTH}
			rare_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
				not less than ``rare_vocab_times`` in the **whole dataset** (except valid words) will be
				marked as invalid words. Otherwise, they are unknown words, which are ignored both for
				model or metrics. {rare_vocab_TIMES_DEFAULT}
			tokenizer (str): How to tokenize sentence. ``nltk.tokenize.WordPunctTokenizer`` is used if ``nltk`` is specified,
				python built-in ``str.split`` is used if ``space`` is specified. {TOKENIZER_DEFAULT}
			remains_capital(bool): Whether remaining capital letter in data or converting them to lower case. {REMAINS_CAPITAL_DEFAULT}
		'''
	FILE_ID_DEFAULT = ''
	VALID_VOCAB_TIMES_DEFAULT = ''
	MAX_SENT_LENGTH = ''
	RARE_VOCAB_TIMES_DEFAULT = ''
	TOKENIZER_DEFAULT = ''
	REMAINS_CAPITAL_DEFAULT = ''

	@hooks.hook_dataloader
	def __init__(self, file_id, *, tokenizer=None, \
			max_sent_length=None, \
			convert_to_lower_letter=None, \
			min_frequent_vocab_times=None, \
			min_rare_vocab_times=None, \
			pretrained=None
		):

		self._pretrained = pretrained
		if pretrained is None:
			with FieldContext.set_parameters(tokenizer=tokenizer,\
				max_sent_length=max_sent_length,
				convert_to_lower_letter=convert_to_lower_letter):
				with VocabContext.set_parameters(min_frequent_vocab_times=min_frequent_vocab_times, \
						min_rare_vocab_times=min_rare_vocab_times):
					super().__init__(file_id, OrderedDict([("post", "SentenceDefault"), ('resp', 'SentenceDefault')]))
			self.set_default_field("train", "post")

		elif pretrained == "gpt2":
			if not isinstance(tokenizer, PretrainedTokenizer):
				raise ValueError("tokenize should be loaded first if you want a gpt2 dataloader")
			vocab = PretrainedVocab(tokenizer.tokenizer)
			with FieldContext.set_parameters(tokenizer=tokenizer,\
					vocab=vocab, \
					max_sent_length=max_sent_length, \
					convert_to_lower_letter=convert_to_lower_letter):
				super().__init__(file_id, OrderedDict([("post", "SentenceGPT2"), ("resp", "SentenceGPT2")]))
			self.set_default_field("train", "post")

		else:
			raise ValueError("No pretrained name %s" % pretrained)

	_GET_BATCH_MORE_DOC = '''Return a dict contains:

			* **post_length** (:class:`numpy.ndarray`): A 1-d array, the length of post in each batch. \
				Size: ``[batch_size]``
			* **post** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form in posts. \
				Only provide valid words. ``unk_id`` will be used if a word is not valid. \
				Size: ``[batch_size, max(sent_length)]``
			* **post_allvocabs** (:class:`numpy.ndarray`): A 2-d padded array containing words of id \
				form in posts. Provide both valid and invalid vocabs. \
				Size: ``[batch_size, max(sent_length)]``
			* **resp_length** (:class:`numpy.ndarray`): A 1-d array, the length of response in each batch. \
				Size: ``[batch_size]``
			* **resp** (:class:`numpy.ndarray`): A 2-d padded array containing words of id form \
				in responses. Only provide valid vocabs. ``unk_id`` will be used if a word is not valid. \
				Size: ``[batch_size, max(sent_length)]``
			* **resp_allvocabs** (:class:`numpy.ndarray`): \
				A 2-d padded array containing words of id form in responses. \
				Provide both valid and invalid vocabs. \
				Size: ``[batch_size, max(sent_length)]``'''

	_GET_BATCH_EXAMPLE = '''
		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # vocab_size = 9
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1])
			{
				"post_allvocabs": numpy.array([
					[2, 5, 6, 10, 3],  # first post:  <go> are you fine <eos>
					[2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
				]),
				"post": numpy.array([
					[2, 5, 6, 1, 3],   # first post:  <go> are you <unk> <eos>
					[2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
				]),
				"resp_allvocabs": numpy.array([
					[2, 8, 9, 10, 3],  # first response:  <go> i am fine <eos>
					[2, 7, 3, 0, 0],   # second response: <go> hello <eos> <pad> <pad>
				]),
				"resp": numpy.array([
					[2, 8, 1, 1, 3],   # first response:  <go> i <unk> <unk> <eos>
					[2, 7, 3, 0, 0],   # second response: <go> hello <eos> <pad> <pad>
				]),
				"post_length": numpy.array([5, 3]), # length of posts
				"resp_length": numpy.array([5, 3]), # length of responses
			}'''
	def get_batch(self, set_name: str, indexes: List[int] #pylint: disable=useless-super-delegation
			) -> Dict[str, Any]:
		return super().get_batch(set_name, indexes)

	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob",\
					   generate_rare_vocab=False) -> MetricChain:
		'''Get metrics for teacher-forcing.

		It contains:

		* :class:`.metric.PerplexityMetric`

		Arguments:
			gen_log_prob_key (str):  The key of predicted log probability over words.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.
			generate_rare_vocab (bool): Whether ``gen_log_prob`` contains invalid vocab.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``False``.
		'''
		metric = MetricChain()
		metric.add_metric(PerplexityMetric(self,\
			reference_allvocabs_key="resp_allvocabs",\
			reference_len_key="resp_length",\
			gen_log_prob_key=gen_log_prob_key,\
			generate_rare_vocab=generate_rare_vocab))
		return metric

	def get_inference_metric(self, gen_key="gen") -> MetricChain:
		'''Get metrics for inference.

		It contains:

		* :class:`.metric.BleuCorpusMetric`
		* :class:`.metric.SingleTurnDialogRecorder`

		Arguments:
			gen_key (str): The key of generated sentences in index form.
				Refer to :class:`.metric.BleuCorpusMetric` or
				:class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.
		'''
		metric = MetricChain()
		metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, \
			reference_allvocabs_key="resp_allvocabs", reference_str_key="resp_str"))
		metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
		return metric

class OpenSubtitles(SingleTurnDialog):
	'''Bases: :class:`.dataloader.SingleTurnDialog`

	A dataloader for OpenSubtitles dataset.
	Refer to :class:`.SingleTurnDialog`, :class:`.LanguageProcessing` for attributes and methods.

	Arguments:{ARGUMENTS}

	References:
		[1] http://opus.nlpl.eu/OpenSubtitles.php

		[2] P. Lison and J. Tiedemann, OpenSubtitles2016: Extracting Large Parallel Corpora from
		Movie and TV Subtitles. LREC 2016.
	'''

	ARGUMENTS = SingleTurnDialog.ARGUMENTS
	FILE_ID_DEFAULT = r'''Default: ``resources://OpenSubtitles``.'''
	FREQUENT_VOCAB_TIMES_DEFAULT = r'''Default: ``10``.'''
	MAX_SENT_LENGTH = r'''Default: ``50``.'''
	RARE_VOCAB_TIMES_DEFAULT = r'''Default: ``0`` (No unknown words).'''
	TOKENIZER_DEFAULT = r'''Default: ``nltk``'''
	REMAINS_CAPITAL_DEFAULT = r'''Default: ``False``'''

	@hooks.hook_dataloader
	def __init__(self, file_id="resources://OpenSubtitles", *, \
			tokenizer="nltk", \
			max_sent_length=50, \
			convert_to_lower_letter=False, \
			min_frequent_vocab_times=10, \
			min_rare_vocab_times=0, \
			pretrained=None
		):
		super().__init__(file_id, tokenizer=tokenizer, max_sent_length=max_sent_length,\
			convert_to_lower_letter=convert_to_lower_letter, min_frequent_vocab_times=min_frequent_vocab_times,\
			min_rare_vocab_times=min_rare_vocab_times, pretrained=pretrained)
