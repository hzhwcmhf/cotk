"""Dataloader for language generation"""
import warnings
from collections import OrderedDict

from .field import Sentence, SentenceGPT2
from .dataloader import LanguageProcessing
from .context import FieldContext, VocabContext
from .vocab import GeneralVocab, PretrainedVocab
from .tokenizer import PretrainedTokenizer

if False: # for type check # pylint: disable=using-constant-test
	from ..metric import MetricChain #pylint: disable=unused-import

# pylint: disable=W0223
class SentenceClassification(LanguageProcessing):
	r"""Base class for sentence classification datasets. This is an abstract class.

	Arguments:

	Notes:
		A :class:`Sentence` field must be set as default field. When invoking :meth:`__init__` of :class:`SentenceClassification`,
		the default field, which may be reset in subclass, is set as self.fields['train']['sent'].
	"""

	_version = 2


	def __init__(self, file_id: str,
				 tokenizer=None,
				 max_sent_length=None,
				 convert_to_lower_letter=None,
				 min_frequent_vocab_times=None,
				 min_rare_vocab_times=None,
				 fields=None,
				 pretrained=None):
		self._pretrained = pretrained

		if pretrained is None:
			if fields is None:
				fields = OrderedDict([('sent', 'SentenceDefault')])
			with FieldContext.set_parameters(tokenizer=tokenizer,
											 max_sent_length=max_sent_length,
											 convert_to_lower_letter=convert_to_lower_letter):
				with VocabContext.set_parameters(min_rare_vocab_times=min_rare_vocab_times,
												 min_frequent_vocab_times=min_frequent_vocab_times):
					super().__init__(file_id, fields)
		elif pretrained == 'gpt2':
			if fields is None:
				fields = OrderedDict(['sent', 'SentenceGPT2'])
			if not isinstance(tokenizer, PretrainedTokenizer):
				raise ValueError("tokenize should be loaded first if you want a gpt2 dataloader")
			vocab = PretrainedVocab(tokenizer.tokenizer)
			with FieldContext.set_parameters(tokenizer=tokenizer,
											 vocab=vocab,
											 max_sent_length=max_sent_length,
											 convert_to_lower_letter=convert_to_lower_letter):
				super().__init__(file_id, fields)
		else:
			raise ValueError("No pretrained name %s" % pretrained)

		self.set_default_field('train', 'sent')

		if pretrained == 'gpt2':
			# check whether SentenceGPT2 is used.
			for set_name, set_fields in self.fields.items():
				for field_name, field in set_fields.items():
					if isinstance(field, Sentence) and not isinstance(field, SentenceGPT2):
						warnings.warn("If you want to use a gpt2 multi_turn_dialog, you'd better use %s instead of %s."
									  % (SentenceGPT2.__name__, type(field).__name__))


	def get_batch(self, set_name, indexes):
		'''Get a batch of specified `indexes`.

		Arguments:
			set_name (str): must be contained in `key_name`
			indexes (list): a list of specified indexes

		Returns:
			(dict): A dict at least contains:

				* sent_length(:class:`numpy.array`): A 1-d array, the length of sentence in each batch.
				  Size: `[batch_size]`
				* sent(:class:`numpy.array`): A 2-d padding array containing id of words.
				  Only provide valid words. `unk_id` will be used if a word is not valid.
				  Size: `[batch_size, max(sent_length)]`
				* label(:class:`numpy.array`): A 1-d array, the label of sentence in each batch.
				* sent_allvocabs(:class:`numpy.array`): A 2-d padding array containing id of words.
				  Provide both valid and invalid words.
				  Size: `[batch_size, max(sent_length)]`

		Examples:
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # vocab_size = 9
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1, 2])
			{
				"sent": numpy.array([
						[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
						[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
						[2, 7, 8, 1, 1, 3]    # third sentence: <go> hello i <unk> <unk> <eos>
					]),
				"label": numpy.array([1, 2, 0]) # label of sentences
				"sent_length": numpy.array([5, 3, 6]), # length of sentences
				"sent_allvocabs": numpy.array([
						[2, 4, 5, 6, 3, 0],   # first sentence: <go> how are you <eos> <pad>
						[2, 7, 3, 0, 0, 0],   # second sentence:  <go> hello <eos> <pad> <pad> <pad>
						[2, 7, 8, 9, 10, 3]   # third sentence: <go> hello i am fine <eos>
					]),
			}
		'''
		return super().get_batch(set_name, indexes)

	def get_metric(self, prediction_key="prediction"):
		'''Get metrics for accuracy. In other words, this function
		provides metrics for sentence classification task.

		It contains:

			* :class:`.metric.AccuracyMetric`

		Arguments:
			prediction_key (str): The key of prediction over sentences.
				Refer to :class:`.metric.AccuracyMetric`. Default: ``prediction``.

		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		from ..metric import MetricChain, AccuracyMetric
		metric = MetricChain()
		metric.add_metric(AccuracyMetric(self, \
										 label_key='label', \
										 prediction_key=prediction_key))
		return metric


class SST(SentenceClassification):
	'''A dataloader for preprocessed SST dataset.

	Arguments:
			file_id (str): a str indicates the source of SST dataset.
			min_frequent_vocab_times (int): A cut-off threshold of valid tokens. All tokens appear
					not less than `min_frequent_vocab_times` in **training set** will be marked as frequent words.
					Default: 10.
			max_sent_length (int): All sentences longer than `max_sent_length` will be shortened
					to first `max_sent_length` tokens. Default: 50.
			min_rare_vocab_times (int):  A cut-off threshold of invalid tokens. All tokens appear
					not less than `min_rare_vocab_times` in the **whole dataset** (except valid words) will be
					marked as rare words. Otherwise, they are unknown words, both in training or
					testing stages. Default: 0 (No unknown words).

	Refer to :class:`.SentenceClassification` for attributes and methods.

	References:
		[1] http://images.cocodataset.org/annotations/annotations_trainval2017.zip

		[2] Lin T Y, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context. ECCV 2014.

	'''

	def __init__(self, file_id, min_frequent_vocab_times=10, \
				 max_sent_length=50, min_rare_vocab_times=0, tokenizer='space', pretrained=None):
		fields = OrderedDict([['sent', 'SentenceDefault'], ['label', 'DenseLabel']])
		super().__init__(file_id,
						 tokenizer=tokenizer,
						 max_sent_length=max_sent_length,
						 convert_to_lower_letter=False,
						 min_frequent_vocab_times=min_frequent_vocab_times,
						 min_rare_vocab_times=min_rare_vocab_times,
						 fields=fields,
						 pretrained=pretrained)
