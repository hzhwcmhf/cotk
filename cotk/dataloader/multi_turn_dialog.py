"""
A module for multi turn dialog.
"""
import warnings
from collections import OrderedDict

from .._utils.metaclass import copy_func
from .dataloader import LanguageProcessing
from .field import Session, SessionGPT2, Field
from .tokenizer import PretrainedTokenizer
from .vocab import PretrainedVocab
from .context import FieldContext, VocabContext
from ..wordvector import Glove

if False:  # for type check # pylint: disable=using-constant-test
	from ..metric import MetricChain  # pylint: disable=unused-import


# pylint: disable=W0223
class MultiTurnDialog(LanguageProcessing):
	r"""Base class for multi-turn dialog datasets. This is an abstract class.

	Arguments:

	Attributes:{ATTRIBUTES}

	Notes:
		A :class:`Session` field must be set as default field. When invoking :meth:`__init__` of :class:`MultiTurnDialog`,
		the default field, which may be reset in subclass, is set as self.fields['train']['session'].
	"""

	_version = 2

	# TODO: fill ATTRIBUTES
	ATTRIBUTES = ''
	# ATTRIBUTES = LanguageProcessing.ATTRIBUTES
	# ARGUMENTS = LanguageProcessing.ARGUMENTS
	GET_BATCH_RETURNS_DICT = r'''
			* turn_length(:class:`numpy.ndarray`): A 1-d list, the number of turns in sessions.
			  Size: ``[batch_size]``
			* sent_length(:class:`numpy.ndarray`): A 2-d non-padded list, the length of sentence in turns.
			  The second dimension is various in different session.
			  Length of outer list: ``[batch_size]``
			* sent(:class:`numpy.ndarray`): A 3-d padding array containing words of index form.
			  Only provide valid words. `unk_id` will be used if a word is not valid.
			  Size: ``[batch_size, max(turn_length[i]), max(sent_length)]``
			* sent_allvocabs(:class:`numpy.ndarray`): A 3-d padding array containing words of index form.
			  Provide both valid and invalid vocabs.
			  Size: ``[batch_size, max(turn_length[i]), max(sent_length)]``
	'''

	GET_BATCH_EXAMPLES_PART = r'''
			>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you",
			>>> #	"hello", "i", "am", "fine"]
			>>> # vocab_size = 9
			>>> # vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "how", "are", "you", "hello", "i"]
			>>> dataloader.get_batch('train', [0, 1])
			{
				"sent_allvocabs": numpy.array([
					[[2, 7, 3, 0, 0, 0],   # 1st sentence in 1st session: <go> hello <eos> <pad> <pad> <pad>
					[2, 7, 3, 0, 0, 0],    # 2nd sentence in 1st session: <go> hello <eos> <pad> <pad> <pad>
					[2, 4, 5, 6, 3, 0],    # 3rd sentence in 1st session: <go> how are you <eos> <pad>
					[2, 8, 9, 10, 3, 0]],  # 4th sentence in 1st session: <go> i am fine <eos> <pad>
					[[2, 7, 4, 5, 6, 3],   # 1st sentence in 2nd session: <go> hello how are you <eos>
					[2, 8, 9, 10, 3, 0],   # 2nd sentence in 2nd session: <go> i am fine <eos> <pad>
					[0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0]]
				]),
				"sent": numpy.array([
					[[2, 7, 3, 0, 0, 0],  # 1st sentence in 1st session: <go> hello <eos> <pad> <pad> <pad>
					[2, 7, 3, 0, 0, 0],   # 2nd sentence in 1st session: <go> hello <eos> <pad> <pad> <pad>
					[2, 4, 5, 6, 3, 0],   # 3rd sentence in 1st session: <go> how are you <eos> <pad>
					[2, 8, 1, 1, 3, 0]],  # 4th sentence in 1st session: <go> i <unk> <unk> <eos> <pad>
					[[2, 7, 4, 5, 6, 3],  # 1st sentence in 2nd session: <go> hello how are you <eos>
					[2, 8, 1, 1, 3, 0],   # 2nd sentence in 2nd session: <go> i <unk> <unk> <eos> <pad>
					[0, 0, 0, 0, 0, 0],
					[0, 0, 0, 0, 0, 0]]
				]),
				"turn_length": np.array([4, 2]), # the number of turns in each session
				"sent_length": np.array([np.array([3, 3, 5, 5]), np.array([6, 5])]), # length of sentences'''

	def __init__(self, file_id: str,
				 tokenizer=None,
				 max_sent_length=None,
				 max_turn_length=None,
				 convert_to_lower_letter=None,
				 min_frequent_vocab_times=None,
				 min_rare_vocab_times=None,
				 fields=None,
				 pretrained=None):
		self._pretrained = pretrained

		if pretrained is None:
			if fields is None:
				fields = OrderedDict([('session', 'SessionDefault')])
			with FieldContext.set_parameters(tokenizer=tokenizer,
											 max_sent_length=max_sent_length,
											 convert_to_lower_letter=convert_to_lower_letter,
											 max_turn_length=max_turn_length):
				with VocabContext.set_parameters(min_rare_vocab_times=min_rare_vocab_times,
											 	min_frequent_vocab_times=min_frequent_vocab_times):
					super().__init__(file_id, fields)
		elif pretrained == 'gpt2':
			if fields is None:
				fields = OrderedDict(['session', 'SessionGPT2'])
			if not isinstance(tokenizer, PretrainedTokenizer):
				raise ValueError("tokenize should be loaded first if you want a gpt2 dataloader")
			vocab = PretrainedVocab(tokenizer.tokenizer)
			with FieldContext.set_parameters(tokenizer=tokenizer,
											 vocab=vocab,
											 max_sent_length=max_sent_length,
											 max_turn_length=max_turn_length,
											 convert_to_lower_letter=convert_to_lower_letter):
				super().__init__(file_id, fields)
		else:
			raise ValueError("No pretrained name %s" % pretrained)

		self.set_default_field('train', 'session')

		if pretrained == 'gpt2':
			# check whether SessionGPT2 is used.
			for set_name, set_fields in self.fields.items():
				for field_name, field in set_fields.items():
					if isinstance(field, Session) and not isinstance(field, SessionGPT2):
						warnings.warn("If you want to use a gpt2 multi_turn_dialog, you'd better use %s instead of %s."
									  % (SessionGPT2.__name__, type(field).__name__))

	_SESSION_MORE_DOCSTRING = '''It calls the identical method of the :class:`Session` instance ``session``,\
		from :meth:`.get_default_field()`.'''


	multi_turn_trim_in_ids = copy_func(LanguageProcessing.get_default_field, Session, 'multi_turn_trim_in_ids')
	convert_multi_turn_tokens_to_ids = copy_func(LanguageProcessing.get_default_field, Session,
												 'convert_multi_turn_tokens_to_ids')
	convert_multi_turn_ids_to_tokens = copy_func(LanguageProcessing.get_default_field, Session,
												 'convert_multi_turn_ids_to_tokens')

	def get_teacher_forcing_metric(self, multi_turn_gen_log_prob_key="multi_turn_gen_log_prob"):
		'''Get metric for teacher-forcing.

		It contains:

		* :class:`.metric.MultiTurnPerplexityMetric`

		Arguments:
			gen_log_prob_key (str): The key of predicted log probability over words.
				Refer to :class:`.metric.MultiTurnPerplexityMetric`. Default: ``gen_log_prob``.

		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		from ..metric import MetricChain, MultiTurnPerplexityMetric
		metric = MetricChain()
		metric.add_metric(MultiTurnPerplexityMetric(self, \
													multi_turn_gen_log_prob_key=multi_turn_gen_log_prob_key, \
													multi_turn_reference_len_key="sent_length", \
													multi_turn_reference_allvocabs_key="sent_allvocabs"))
		return metric

	def get_inference_metric(self, multi_turn_gen_key="multi_turn_gen"):
		'''Get metric for inference.

		It contains:

		* :class:`.metric.BleuCorpusMetric`
		* :class:`.metric.MultiTurnDialogRecorder`

		Arguments:
			gen_key (str): The key of generated sentences in index form.
				Refer to :class:`.metric.BleuCorpusMetric` or :class:`.metric.MultiTurnDialogRecorder`.
				Default: ``gen``.

		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		from ..metric import MetricChain, MultiTurnBleuCorpusMetric, MultiTurnDialogRecorder
		metric = MetricChain()
		metric.add_metric(MultiTurnBleuCorpusMetric(self, multi_turn_gen_key=multi_turn_gen_key, \
													multi_turn_reference_allvocabs_key="sent_allvocabs",
													turn_len_key="turn_length"))
		metric.add_metric(MultiTurnDialogRecorder(self, multi_turn_gen_key=multi_turn_gen_key, \
												  multi_turn_reference_allvocabs_key="sent_allvocabs",
												  turn_len_key="turn_length"))
		return metric


# TODO: doc
class UbuntuCorpus(MultiTurnDialog):
	'''A dataloader for Ubuntu dataset.

	Arguments:
		file_id (str): a str indicates the source of UbuntuCorpus dataset.
			Default: ``resources://Ubuntu``. A preset dataset is downloaded and cached.{ARGUMENTS}

	Refer to :class:`.MultiTurnDialog` for attributes and methods.

	References:
		[1] https://github.com/rkadlec/ubuntu-ranking-dataset-creator

		[2] Lowe R, Pow N, Serban I, et al. The Ubuntu Dialogue Corpus: A Large Dataset
		for Research in Unstructured Multi-Turn Dialogue Systems. SIGDIAL 2015.
	'''

	ARGUMENTS_FORMATTER = r'''
		min_frequent_vocab_times (int):  A cut-off threshold of valid tokens. All tokens appear
			not less than `min_vocab_times` in **training set** will be marked as frequent words.
			Default: ``{default_min_frequent_vocab_times}``.
		max_sent_length (int): All sentences longer than ``max_sent_length`` will be shortened
			to first ``max_sent_length`` tokens. Default: ``{default_max_sent_length}``.
		max_turn_length (int): All sessions longer than ``max_turn_length`` will be shortened
			to first ``max_turn_length`` sentences. Default: ``{default_max_turn_length}``.
		min_rare_vocab_times (int):  A cut-off threshold of rare tokens. All tokens appear
			not less than ``invalid_vocab_times`` in the **whole dataset** (except valid words) will be
			marked as rare words. Otherwise, they are unknown words, both in training or
			testing stages. Default: ``{default_min_rare_vocab_times}`` (No unknown words).'''

	ARGUMENTS = ARGUMENTS_FORMATTER.format(
		default_min_frequent_vocab_times=10,
		default_max_sent_length=50,
		default_max_turn_length=20,
		default_min_rare_vocab_times=0
	)

	def __init__(self, file_id="resources://Ubuntu", min_frequent_vocab_times=10,
				 max_sent_length=50, max_turn_length=20, min_rare_vocab_times=0,
				 tokenizer='nltk',
				 pretrained=None):
		super().__init__(file_id,
						 tokenizer=tokenizer,
						 max_sent_length=max_sent_length,
						 max_turn_length=max_turn_length,
						 convert_to_lower_letter=True,
						 min_frequent_vocab_times=min_frequent_vocab_times,
						 min_rare_vocab_times=min_rare_vocab_times,
						 pretrained=pretrained)


class SwitchboardCorpus(MultiTurnDialog):
	'''A dataloader for Switchboard dataset.

	In this dataset, all sessions start with a ``<d>`` representing empty context.

	Arguments:
		file_id (str): a string indicating the source of SwitchboardCorpus dataset.
			Default: ``resources://SwitchboardCorpus``. A preset dataset is downloaded and cached.

	Refer to :class:`.MultiTurnDialog` for attributes and methods.

	References:
		[1] https://catalog.ldc.upenn.edu/LDC97S62

		[2] John J G and Edward H. Switchboard-1 release 2. Linguistic Data Consortium, Philadelphia 1997.
	'''

	ARGUMENTS = UbuntuCorpus.ARGUMENTS_FORMATTER.format(
		default_min_frequent_vocab_times=5,
		default_max_sent_length=50,
		default_max_turn_length=1000,
		default_min_rare_vocab_times=0
	)


	def __init__(self, file_id="resources://SwitchboardCorpus", min_frequent_vocab_times=5, \
				 max_sent_length=50, max_turn_length=1000, min_rare_vocab_times=0, tokenizer='nltk',
				 pretrained=None):
		if pretrained is None:
			fields = {
				**{k: OrderedDict([['session', 'SessionDefault']]) for k in ['train', 'dev', 'test']},
				'multi_ref': OrderedDict([['session', 'SessionDefault'], ['candidate', "SentenceCandidateDefault"]])
			}
		elif pretrained == 'gpt2':
			fields = {
				**{k: OrderedDict([['session', 'SessionGPT2']]) for k in ['train', 'dev', 'test']},
				'multi_ref': OrderedDict([['session', 'SessionGPT2'], ['candidate', "SentenceCandidateGPT2"]])
			}
		else:
			raise ValueError("No pretrained name %s" % pretrained)

		with FieldContext.set_parameters(
			vocab_from_mappings={**Field.DEFAULT_VOCAB_FROM_MAPPINGS, 'multi_ref': 'test'}):
			super().__init__(file_id,
							 tokenizer=tokenizer,
							 max_sent_length=max_sent_length,
							 max_turn_length=max_turn_length,
							 convert_to_lower_letter=False,
							 min_frequent_vocab_times=min_frequent_vocab_times,
							 min_rare_vocab_times=min_rare_vocab_times,
							 fields=fields,
							 pretrained=pretrained)

	def get_batch(self, set_name, indexes):
		# '''{LanguageProcessing.GET_BATCH_DOC_WITHOUT_RETURNS}
		'''

		Returns:
			(dict): A dict contains what is in the return of MultiTurnDialog.get_batch.
			{MultiTurnDialog.GET_BATCH_RETURNS_DICT}

				It additionally contains:

				* candidate_allvocabs (list): A 3-d list, multiple responses for a session.
				  Size: ``[batch_size, ~reference_num, ~sent_length]``, where "~" means different sizes
				  in this dimension is allowed.

			See the example belows.

		Examples:
			{MultiTurnDialog.GET_BATCH_EXAMPLES_PART}
			"candidate_allvocabs":[
				[[2, 7, 3],                   # two responses to 1st session: <go> hello <eos>
				 [2, 6, 5, 10, 3]],           #                               <go> you are fine <eos>
				[[2, 6, 5, 10, 3]]]           # one response to 2nd session:  <go> you are fine <eos>
			}
		'''
		return super().get_batch(set_name, indexes)

	def get_multi_ref_metric(self, generated_num_per_context=20, word2vec=None, \
							 multiple_gen_key="multiple_gen_key"):
		'''Get metrics for multiple references.

		It contains:

		* :class:`.metric.BleuPrecisionRecallMetric`
		* :class:`.metric.EmbSimilarityPrecisionRecallMetric`

		Arguments:
			generated_num_per_context (int): The number of sentences generated per context.
			word2vec (dict): Maps words to word embeddings for embedding similarity.
				Default: if ``None``, using glove word embedding from ``resources://Glove300d``.

		Returns:
			A :class:`.metric.MetricChain` object.
		'''
		from ..metric import MetricChain, BleuPrecisionRecallMetric, EmbSimilarityPrecisionRecallMetric
		metric = MetricChain()
		if word2vec is None:
			glove = Glove("resources://Glove300d")
			word2vec = glove.load_dict(self.frequent_vocab_list)
		for ngram in range(1, 5):
			metric.add_metric(BleuPrecisionRecallMetric(self, ngram, generated_num_per_context, \
														multiple_gen_key=multiple_gen_key))
		metric.add_metric(EmbSimilarityPrecisionRecallMetric(self, word2vec, \
															 'avg', generated_num_per_context,
															 multiple_gen_key=multiple_gen_key))
		metric.add_metric(EmbSimilarityPrecisionRecallMetric(self, word2vec, \
															 'extrema', generated_num_per_context,
															 multiple_gen_key=multiple_gen_key))
		return metric
