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
from .dataloader import LanguageProcessing
from .tokenizer import PretrainedTokenizer
from .vocab import PretrainedVocab
from .context import FieldContext, VocabContext

if False: # for type check # pylint: disable=using-constant-test
	from ..metric import MetricChain #pylint: disable=unused-import

# pylint: disable=W0223
class SingleTurnDialog(LanguageProcessing):
	"""Bases: :class:`.dataloader.LanguageProcessing`

	This class is supported for sequence to sequence generation tasks, especially
	single turn dialog tasks.

	Arguments:{SHARED_ARGUMENTS}
	"""

	_version = 2

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
			* **post** (:class:`numpy.ndarray`): A 2-d padded array containing tokens of id form in posts. \
				Only provide frequent tokens. ``unk_id`` will be used for a rare token. \
				Size: ``[batch_size, max(sent_length)]``
			* **post_allvocabs** (:class:`numpy.ndarray`): A 2-d padded array containing tokens of id \
				form in posts. Provide both frequent and rare vocabs. \
				Size: ``[batch_size, max(sent_length)]``
			* **post_str** (:class:`List[str]`): A list containing raw posts \
				before tokenizing, converting to ids, or padding. \
				Do not contain any special tokens. \
				Size: ``[batch_size]``
			* **resp_length** (:class:`numpy.ndarray`): A 1-d array, the length of response in each batch. \
				Size: ``[batch_size]``
			* **resp** (:class:`numpy.ndarray`): A 2-d padded array containing tokens of id form \
				in responses. Only provide valid vocabs. ``unk_id`` will be used for a rare token. \
				Size: ``[batch_size, max(sent_length)]``
			* **resp_allvocabs** (:class:`numpy.ndarray`): \
				A 2-d padded array containing tokens of id form in responses. \
				Provide both valid and invalid vocabs. \
				Size: ``[batch_size, max(sent_length)]``
			* **post_str** (:class:`List[str]`): A list containing raw responses \
				before tokenizing, converting to ids, or padding. \
				Do not contain any special tokens. \
				Size: ``[batch_size]`` '''

	_GET_BATCH_EXAMPLE = '''
		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # frequent_vocab_size = 9
			>>> # frequent_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1])
			{
				"post_str": [
					"are you fine",
					"hello",
				],
				"post_allvocabs": numpy.array([
					[2, 5, 6, 10, 3],  # first post:  <go> are you fine <eos>
					[2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
				]),
				"post": numpy.array([
					[2, 5, 6, 1, 3],   # first post:  <go> are you <unk> <eos>
					[2, 7, 3, 0, 0],   # second post: <go> hello <eos> <pad> <pad>
				]),
				"resp_str": [
					"i am fine",
					"hello"
				],
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
					   generate_rare_vocab=False) -> "MetricChain":
		'''Get metrics for teacher-forcing.

		It contains:

		* :class:`.metric.PerplexityMetric`

		Arguments:
			gen_log_prob_key (str):  The key of predicted log probability over words.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``gen_log_prob``.
			generate_rare_vocab (bool): Whether ``gen_log_prob`` contains invalid vocab.
				Refer to :class:`.metric.PerplexityMetric`. Default: ``False``.
		'''
		from ..metric import MetricChain, PerplexityMetric
		metric = MetricChain()
		metric.add_metric(PerplexityMetric(self,\
			reference_allvocabs_key="resp_allvocabs",\
			reference_len_key="resp_length",\
			gen_log_prob_key=gen_log_prob_key,\
			generate_rare_vocab=generate_rare_vocab))
		return metric

	def get_inference_metric(self, gen_key="gen") -> "MetricChain":
		'''Get metrics for inference.

		It contains:

		* :class:`.metric.BleuCorpusMetric`
		* :class:`.metric.SingleTurnDialogRecorder`

		Arguments:
			gen_key (str): The key of generated sentences in index form.
				Refer to :class:`.metric.BleuCorpusMetric` or
				:class:`.metric.SingleTurnDialogRecorder`. Default: ``gen``.
		'''
		from ..metric import MetricChain, BleuCorpusMetric, SingleTurnDialogRecorder
		metric = MetricChain()
		metric.add_metric(BleuCorpusMetric(self, gen_key=gen_key, \
			reference_allvocabs_key="resp_allvocabs", reference_str_key="resp_str"))
		metric.add_metric(SingleTurnDialogRecorder(self, gen_key=gen_key))
		return metric

class OpenSubtitles(SingleTurnDialog):
	'''Bases: :class:`.dataloader.SingleTurnDialog`

	A dataloader for OpenSubtitles dataset.
	Refer to :class:`.SingleTurnDialog`, :class:`.LanguageProcessing` for attributes and methods.

	Arguments:{SHARED_ARGUMENTS}

	References:
		[1] http://opus.nlpl.eu/OpenSubtitles.php

		[2] P. Lison and J. Tiedemann, OpenSubtitles2016: Extracting Large Parallel Corpora from
		Movie and TV Subtitles. LREC 2016.
	'''

	_FILE_ID_DEFAULT = r'''Default: ``resources://OpenSubtitles``.'''
	_TOKENIZER_DEFAULT = r'''Default: ``nltk``'''
	_CONVERT_TO_LOWER_LETTER_DEFAULT = r'''Default: ``False``'''
	_MAX_SENT_LENGTH_DEFAULT = r'''Default: ``50``.'''
	_MIN_FREQUENT_VOCAB_TIMES_DEFAULT = r'''Default: ``10``.'''
	_MIN_RARE_VOCAB_TIMES_DEFAULT = r'''Default: ``0``'''

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
