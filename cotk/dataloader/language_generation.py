"""Dataloader for language generation"""
import numpy as np
from typing import List, Any, Tuple, Optional, Dict
from collections import OrderedDict

from .dataloader import LanguageProcessing
from .context import FieldContext, VocabContext
from .tokenizer import PretrainedTokenizer
from .vocab import GeneralVocab, PretrainedVocab
from ..metric.metric import MetricChain, MetricBase

# pylint: disable=W0223
class LanguageGeneration(LanguageProcessing):
	"""Bases: :class:`.dataloader.LanguageProcessing`

	This class is supported for language modeling tasks or language generation tasks
	without any inputs.

	Arguments:{SHARED_ARGUMENTS}
	"""

	_version = 2

	def __init__(self, file_id, *, tokenizer=None, \
			max_sent_length=None, \
			convert_to_lower_letter=None, \
			min_frequent_vocab_times=None, \
			min_rare_vocab_times=None, \
			pretrained=None):

		self._pretrained = pretrained
		if pretrained is None:
			with FieldContext.set_parameters(tokenizer=tokenizer,\
					max_sent_length=max_sent_length,
					convert_to_lower_letter=convert_to_lower_letter):
				with VocabContext.set_parameters(min_frequent_vocab_times=min_frequent_vocab_times, \
						min_rare_vocab_times=min_rare_vocab_times):
					super().__init__(file_id, OrderedDict([("sent", "SentenceDefault")]))
			self.set_default_field("train", "sent")

		elif pretrained == "gpt2":
			if not isinstance(tokenizer, PretrainedTokenizer):
				raise ValueError("tokenize should be loaded first if you want a gpt2 dataloader")
			vocab = PretrainedVocab(tokenizer.tokenizer)
			with FieldContext.set_parameters(tokenizer=tokenizer,\
					vocab=vocab, \
					max_sent_length=max_sent_length, \
					convert_to_lower_letter=convert_to_lower_letter):
				super().__init__(file_id, OrderedDict([("sent", "SentenceGPT2")]))
			self.set_default_field("train", "sent")
		else:
			raise ValueError("No pretrained name %s" % pretrained)

	_GET_BATCH_MORE_DOC = '''Returns a dict at least contains:

			* **sent_length** (:class:`numpy.ndarray`): A 1-d array, the length of sentence in each batch.
			  Size: ``[batch_size]``
			* **sent** (:class:`numpy.ndarray`): A 2-d padding array containing id of tokens.
			  Only provide frequent tokens. ``unk_id`` will be used for a rare token.
			  Size: ``[batch_size, max(sent_length)]``
			* **sent_allvocabs** (:class:`numpy.ndarray`): A 2-d padding array containing id of tokens.
			  Provide both frequent and rare tokens.
			  Size: ``[batch_size, max(sent_length)]``
			* **sent_str** (:class:`List[str]`): A list containing raw sentences
			  before tokenizing, converting to ids, or padding.
			  Do not contain any special tokens.
			  Size: ``[batch_size]``'''

	_GET_BATCH_EXAMPLE = '''
		Examples:

			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # frequent_vocab_size = 9
			>>> # frequent_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1, 2])
			{
				"sent": numpy.array([
					[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
					[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
					[2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
				]),
				"sent_length": numpy.array([5, 3, 6]), # length of sentences
				"sent_allvocabs": numpy.array([
					[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
					[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
					[2, 7, 8, 9, 10, 3]   # third sentence: <go> hello i am fine <eos>
				]),
				"sent_str": [
					"how are you",
					"hello",
					"hello i am fine"
				],
			}'''

	def get_batch(self, set_name: str, indexes: List[int] #pylint: disable=useless-super-delegation
			) -> Dict[str, Any]:
		return super().get_batch(set_name, indexes)


	GEN_LOG_PROB_KEY_ARGUMENTS = MetricBase.GEN_LOG_PROB_KEY_ARGUMENTS
	def get_teacher_forcing_metric(self, gen_log_prob_key="gen_log_prob") -> "MetricChain":
		'''Get metrics for teacher-forcing. In other words, this function
		provides metrics for language modelling task.

		It contains:

		* :class:`.metric.PerplexityMetric`

		See the above class for details of arguments.

		Arguments:
			{GEN_LOG_PROB_KEY_ARGUMENTS}
		'''
		from ..metric import MetricChain, PerplexityMetric
		metric = MetricChain()
		metric.add_metric(PerplexityMetric(self, \
					reference_allvocabs_key='sent_allvocabs', \
					reference_len_key='sent_length', \
					gen_log_prob_key=gen_log_prob_key))
		return metric

	GEN_KEY_ARGUMENTS = MetricBase.GEN_KEY_ARGUMENTS
	SAMPLE_ARGUMENTS_IN_BLEU = MetricBase.SAMPLE_ARGUMENTS_IN_BLEU.\
			replace("sample (int, optional)", "sample_in_bleu (int, optional)")
	SAMPLE_ARGUMENTS_IN_NGRAM_PERPLEXITY = MetricBase.SAMPLE_ARGUMENTS_IN_NGRAM_PERPLEXITY.\
			replace("sample (int, optional)", "sample_in_ngram_perplexity (int, optional)")
	SEED_ARGUMENTS = MetricBase.SEED_ARGUMENTS
	CPU_COUNT_ARGUMENTS = MetricBase.CPU_COUNT_ARGUMENTS
	def get_inference_metric(self, gen_key="gen", sample_in_bleu=1000, \
			sample_in_ngram_perplexity=10000, seed=1229, cpu_count=None) -> "MetricChain":
		'''Get metrics for inference. In other words, this function provides metrics for
		language generation tasks.

		It contains:

		* :class:`.metric.SelfBleuCorpusMetric`
		* :class:`.metric.FwBwBleuCorpusMetric`
		* :class:`.metric.NgramFwBwPerplexityMetric`
		* :class:`.metric.LanguageGenerationRecorder`

		See the above class for details of arguments.

		Arguments:
			{GEN_KEY_ARGUMENTS}
			{SAMPLE_ARGUMENTS_IN_BLEU}
			{SAMPLE_ARGUMENTS_IN_NGRAM_PERPLEXITY}
			{SEED_ARGUMENTS}
			{CPU_COUNT_ARGUMENTS}
		'''
		from ..metric import MetricChain, LanguageGenerationRecorder, \
			FwBwBleuCorpusMetric, SelfBleuCorpusMetric, NgramFwBwPerplexityMetric
		metric = MetricChain()
		metric.add_metric(SelfBleuCorpusMetric(self, \
					gen_key=gen_key, \
					sample=sample_in_bleu, \
					seed=seed, \
					cpu_count=cpu_count))
		metric.add_metric(FwBwBleuCorpusMetric(self, \
					reference_test_list=self.get_all_batch("test")["sent"], \
					gen_key=gen_key, \
					sample=sample_in_bleu, \
					seed=seed, \
					cpu_count=cpu_count))
		metric.add_metric(FwBwBleuCorpusMetric(self, \
					reference_test_list=self.get_all_batch("test")["sent"], \
					gen_key=gen_key, \
					sample=sample_in_ngram_perplexity, \
					seed=seed, \
					cpu_count=cpu_count))
		metric.add_metric(LanguageGenerationRecorder(self, gen_key=gen_key))
		return metric

class MSCOCO(LanguageGeneration):
	'''Bases: :class:`.dataloader.LanguageGeneration`

	A dataloader for preprocessed MSCOCO dataset.
	Refer to :class:`.LanguageGeneration` and :class:`.LanguageProcessing` for attributes and methods.

	Arguments:{SHARED_ARGUMENTS}

	References:
		[1] http://images.cocodataset.org/annotations/annotations_trainval2017.zip

		[2] Chen X, Fang H, Lin T Y, et al. Microsoft COCO Captions:
		Data Collection and Evaluation Server. arXiv:1504.00325, 2015.
	'''

	_FILE_ID_DEFAULT = r'''Default: ``resources://MSCOCO``.'''
	_TOKENIZER_DEFAULT = r'''Default: ``nltk``'''
	_MAX_SENT_LENGTH = r'''Default: ``50``.'''
	_CONVERT_TO_LOWER_LETTER_DEFAULT = r'''Default: ``True``'''
	_MIN_FREQUENT_VOCAB_TIMES_DEFAULT = r'''Default: ``10``.'''
	_MIN_RARE_VOCAB_TIMES_DEFAULT = r'''Default: ``0``.'''

	def __init__(self, file_id, *, tokenizer="nltk", \
			max_sent_length=50, \
			convert_to_lower_letter=False, \
			min_frequent_vocab_times=10, \
			min_rare_vocab_times=0, \
			pretrained=None):
		super().__init__(file_id, tokenizer=tokenizer, max_sent_length=max_sent_length,\
			convert_to_lower_letter=convert_to_lower_letter, min_frequent_vocab_times=min_frequent_vocab_times,\
			min_rare_vocab_times=min_rare_vocab_times, pretrained=pretrained)
