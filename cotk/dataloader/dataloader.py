'''A module for dataloader'''
import random
from typing import Optional, Any, Union, Sequence, Dict, Tuple, Iterable, List
from collections import Counter, OrderedDict
from itertools import chain
import logging
from hashlib import sha256

import numpy as np

from .._utils import trim_before_target
from .._utils.unordered_hash import UnorderedSha256, dumps
from .._utils.metaclass import DocStringInheritor, LoadClassInterface, copy_func, copy_property
from .._utils.typehint import OrderedDictType
from ..file_utils import get_resource_file_path
from .tokenizer import Tokenizer
from .field import Field, SentenceDefault, _FieldContent, Sentence
from .vocab import Vocab, GeneralVocab
from .context import FieldContext, VocabContext

class Dataloader(LoadClassInterface, metaclass=DocStringInheritor):
	'''Base class of Dataloader.
	'''



class LanguageProcessing(Dataloader):
	"""Bases: :class:`.dataloader.Dataloader`

	Base class for all language processing tasks. This is an abstract class.

	During the initialization of a dataloader, :class:`Vocab`, :class:`Tokenizer` or :class:`Field` may be created.
	See :ref:`how to create a dataloader<customized_tasks_ref>`.

	Arguments:{FILE_ID_DOCS}{FIELD_DETAILS}

	"""

	FILE_ID_DOCS = r"""
			file_id (str): A string indicating the source (path) of the dataset. It can be local path (``"./data"``), a resource name
				(``"resources://dataset"``), or an url (``"http://test.com/dataset.zip"``).
				See :ref:`the details of file id<file_id>`."""

	FIELD_DETAILS = r"""
			fields (List, OrderedDict, Dict):
				This arguments supports multiple input types:

				* If ``OrderDict`` or ``List``, it specify ``data format`` of the ``"train"``, ``"dev"``, ``"test"`` set.

						* A ``data format`` should be an ``OrderedDict`` or a ``List[Tuple]`` can be converted to ``OrderedDict``.
						* The ``key`` of ``data format`` is the name of a Field (used by :meth:`.get_batch`),
						  and the ``value`` is either a class name of a Field or a :class:`Field` object.
						* Examples:

							>>> postField = SentenceDefault(...)
							>>> respField = SentenceDefault(...)
							>>> data_format = [("post", postField), ("resp", respField)]

							or

							>>> data_format = [("post", "SentenceDefault"), ("resp", "SentenceDefault")]
						* Examples:

							>>> fields = data_format

							equals to

							>>> fields = {"train": data_format, "dev": data_format, "test": data_format}

				* If ``Dict``, ``fields[key]`` describes ``data format`` of the set named ``key``. Examples:

					>>> fields = {"train": data_format, "extra": data_format}

				* See :ref:`how to create a dataloader<customized_tasks_ref>`."""

	FIELD_REF = r"""
			fields (List, OrderedDict, Dict): See initialization of :class:`LanguageProcessing` for explanation. """

	SHARED_ARGUMENTS = r'''{LanguageProcessing.FILE_ID_DOCS} {_FILE_ID_DEFAULT}
			{LanguageProcessing.TOKENIZER_DOCS} {_TOKENIZER_DEFAULT}
			{LanguageProcessing.MAX_SENT_LENGTH_DOCS} {_MAX_SENT_LENGTH_DEFAULT}
			{LanguageProcessing.CONVERT_TO_LOWER_LETTER_DOCS} {_CONVERT_TO_LOWER_LETTER_DEFAULT}
			{LanguageProcessing.MIN_FREQUENT_VOCAB_TIMES_DOCS} {_MIN_FREQUENT_VOCAB_TIMES_DEFAULT}
			{LanguageProcessing.MIN_RARE_VOCAB_TIMES_DOCS} {_MIN_RARE_VOCAB_TIMES_DEFAULT}
			{LanguageProcessing.PRETRAINED_DOCS} {_PRETAINED_DEFAULT}'''
	_FILE_ID_DEFAULT = ""
	TOKENIZER_DOCS = Sentence.TOKENIZER_DOCS
	_TOKENIZER_DEFAULT = Sentence.TOKENIZER_DEFAULT
	MAX_SENT_LENGTH_DOCS = Sentence.MAX_SENT_LENGTH_DOCS
	_MAX_SENT_LENGTH_DEFAULT = Sentence.MAX_SENT_LENGTH_DEFAULT
	CONVERT_TO_LOWER_LETTER_DOCS = Sentence.CONVERT_TO_LOWER_LETTER_DOCS
	_CONVERT_TO_LOWER_LETTER_DEFAULT = Sentence.CONVERT_TO_LOWER_LETTER_DEFAULT
	MIN_FREQUENT_VOCAB_TIMES_DOCS = GeneralVocab.MIN_FREQUENT_VOCAB_TIMES_DOCS
	_MIN_FREQUENT_VOCAB_TIMES_DEFAULT = GeneralVocab.MIN_FREQUENT_VOCAB_TIMES_DEFAULT
	MIN_RARE_VOCAB_TIMES_DOCS = GeneralVocab.MIN_RARE_VOCAB_TIMES_DOCS
	_MIN_RARE_VOCAB_TIMES_DEFAULT = GeneralVocab.MIN_RARE_VOCAB_TIMES_DEFAULT
	PRETRAINED_DOCS = r'''
			pretrained (str, optional): Use :ref:`pretrained field<pretrained_field_ref>` instead of :class:`SentenceDefault`.'''
	_PRETAINED_DEFAULT = "Default: If ``None``, no pretrained field used."

	# for docstring
	fields: Dict[str, "OrderedDict[str, Union[str, Field]]"] = {}
	'''This instance attribute shows fields of the dataloader (See the initialization of :class:`LanguageProcessing`).
		For example, the fields can be printed as follows:

		.. code-block:: python

		    {
		        'train': OrderedDict([('sent', <cotk.dataloader.field.SentenceDefault object at 0x000001E170F8B588>)]),
		        'dev': OrderedDict([('sent', <cotk.dataloader.field.SentenceDefault object at 0x000001E170F8BB48>)]),
		        'test': OrderedDict([('sent', <cotk.dataloader.field.SentenceDefault object at 0x000001E170F8BEC8>)])}
		    }
	'''

	def __init__(self, file_id: str, \
				 fields: Union["OrderedDict[str, Union[str, Field]]", List[Tuple[str, Union[str, Field]]],\
					 		   Dict[str, Union["OrderedDict[str, Union[str, Field]]", List[Tuple[str, Union[str, Field]]]]]], \
				 ):
		self.file_id = file_id
		self.file_path = get_resource_file_path(file_id)

		with FieldContext.set_parameters(vocab=GeneralVocab(), weak=True) as field_context:

			fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]] = {}
			self.fields: Dict[str, OrderedDictType[str, Field]] = {}
			if isinstance(fields, OrderedDict) or isinstance(fields, list):
				fields = {set_name: fields for set_name in ["train", "dev", "test"]}
			if isinstance(fields, dict):
				for set_name, fields_in_one_set in fields.items():
					one_fields, one_fieldcontents = self._fill_field_and_create_content(set_name, fields_in_one_set)
					self.fields[set_name] = one_fields
					fieldcontents[set_name] = one_fieldcontents
			else:
				raise TypeError("Unknown type for fields")

			self._load_data(fieldcontents)

			self.vocabs = self._collect_vocabs_from_fields(self.fields)
			# self.default_vocab_id = 0 if len(self.vocabs) == 1 else None
			self.tokenizers = self._collect_tokenizers_from_fields(self.fields)
			# self.default_tokenizer_id = 0 if len(self.tokenizers) == 1 else None
			self.default_field_set_name: Optional[str] = None
			self.default_field_name: Optional[str] = None
			self._build_vocabs()

			self._setting_hash = self._create_setting_hash()
			self._vocab_hash = self._create_vocab_hash()
			self.data = self._get_data(fieldcontents)
			self._raw_data_hash, self._data_hash = self._create_data_hash(fieldcontents)
			self.index, self.batch_id, self.batch_size = self._init_batch(fieldcontents)

	@staticmethod
	def simple_create(file_id: str, \
				fields: Union[OrderedDictType[str, Union[str, Field]],\
					 		   Dict[str, OrderedDictType[str, Union[str, Field]]]], \
				**kwargs) -> "LanguageProcessing":
		'''A simple way to create a dataloader. Instead of using :class:`VocabContext`
		and :class:`FieldContext`, specifying all the possible parameters here.

		Arguments:{FILE_ID_DOCS}{FIELD_REF}
			**kwargs: Arguments passed to created :class:`Vocab` and :class:`Field`.
		'''
		with VocabContext.set_parameters(**kwargs):
			with FieldContext.set_parameters(**kwargs):
				with FieldContext.set_parameters(tokenizer="space", weak=True):
					return LanguageProcessing(file_id, fields)

	def _load_data(self, fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]]):
		'''Load data from file.
		Arguments:
			fieldcontents (Dict[str, OrderedDictType[str, _FieldContent]]): fieldcontents for each set
		'''
		for set_name, fieldcontents_in_one_set in fieldcontents.items():
			if not fieldcontents_in_one_set:
				raise RuntimeError("no field specified")
			with open("%s/%s.txt" % (self.file_path, set_name), encoding='utf-8') as f_file:
				line_cnt = 0
				file_iterator = iter(f_file)
				while True:
					try:
						for _, fieldcontent in fieldcontents_in_one_set.items():
							line_add = fieldcontent.read_next(file_iterator)
							if line_add == 0:
								while True:
									if next(file_iterator):
										raise RuntimeError("the file %s corrupted at line %d" % (set_name, line_cnt))
							line_cnt += line_add
					except StopIteration:
						break

			sample_nums = [fieldcontent.get_data_number() for _, fieldcontent in fieldcontents_in_one_set.items()]
			if not all([sample_num == sample_nums[0] for sample_num in sample_nums]):
				raise RuntimeError("the file %s corrupted at end of the file")

		for _, fieldcontents_in_one_set in fieldcontents.items():
			for _, fieldcontent in fieldcontents_in_one_set.items():
				fieldcontent.process_before_vocab()

	def _init_batch(self, fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]]) -> \
			Tuple[Dict[str, List[int]], Dict[str, int], Dict[str, Optional[int]]]:
		'''Initialize the batches. Return a tuple contains
		``index``, ``batch_id``, ``batch_size`` for each set.
		Arguments:
			fieldcontents (Dict[str, OrderedDictType[str, _FieldContent]]): fieldcontents for each set.
		'''
		index: Dict[str, List[int]] = {}
		batch_id: Dict[str, int] = {}
		batch_size: Dict[str, Optional[int]] = {}

		for set_name, fieldcontents_in_one_set in fieldcontents.items():
			sample_nums = [fieldcontent.get_data_number() \
					for _, fieldcontent in fieldcontents_in_one_set.items()]
			batch_id[set_name] = 0
			batch_size[set_name] = None
			index[set_name] = list(range(sample_nums[0]))

		return index, batch_id, batch_size

	def _get_data(self, fieldcontents: Dict[str, OrderedDictType[str, _FieldContent]]) -> \
			Dict[str, Dict[str, Any]]:
		'''Get the data from fieldcontents.
		Arguments:
			fieldcontents (Dict[str, OrderedDict[str, _FieldContent]]): fieldcontents for each set.
		'''
		data: Dict[str, Dict[str, Any]] = {}
		for set_name, fieldcontents_in_one_set in sorted(fieldcontents.items()):
			data[set_name] = {}
			for field_name, fieldcontent in fieldcontents_in_one_set.items():
				data[set_name][field_name] = fieldcontent.get_data()
		return data

	def _build_vocabs(self):
		'''Invoke build vocab for each vocabulary'''
		for vocab in self.vocabs:
			vocab.build_vocab()

	def _collect_vocabs_from_fields(self, fields: Dict[str, OrderedDictType[str, Field]])\
			-> List[Vocab]:
		'''Collect all vocabulary instances (deduplicated).
		Arguments:
			fieldcontents (Dict[str, OrderedDict[str, Field]]): field for each set.
		'''
		vocabs: List[Vocab] = []
		for _, fields_in_one_set in sorted(fields.items()): # sort to keep order
			for _, field in fields_in_one_set.items():
				vocab = field.get_vocab()
				if vocab is not None and vocab not in vocabs:
					vocabs.append(vocab)
		return vocabs

	def _collect_tokenizers_from_fields(self, fields: Dict[str, OrderedDictType[str, Field]])\
			-> List[Tokenizer]:
		'''Collect all tokenizer instances (deduplicated).
		Arguments:
			fieldcontents (Dict[str, OrderedDict[str, Field]]): field for each set.
		'''
		tokenizers: List[Tokenizer] = []
		tokenizers_setting_hash: List[str] = []
		for _, fields_in_one_set in sorted(fields.items()): # sort to keep order
			for _, field in fields_in_one_set.items():
				tokenizer = field.get_tokenizer()
				if tokenizer is not None and tokenizer.get_setting_hash() not in tokenizers_setting_hash:
					tokenizers.append(tokenizer)
					tokenizers_setting_hash.append(tokenizer.get_setting_hash())
		return tokenizers

	def _fill_field_and_create_content(self, set_name: str, fields: \
				Union[OrderedDictType[str, Union[str, Field]], List[Tuple[str, Union[str, Field]]]], \
				) -> \
					Tuple[OrderedDictType[str, Field], OrderedDictType[str, _FieldContent]]:
		'''Create and return fields and field contexts.
		Arguments:
			set_name(str): name of the set
			field (OrderedDictType[str, Union[str, Field]]): fields for the set.
		'''

		fieldcontents: OrderedDictType[str, _FieldContent] = OrderedDict()
		new_fields: OrderedDictType[str, Field] = OrderedDict()

		fields_iter: Iterable[Tuple[str, Union[str, Field]]]
		if isinstance(fields, OrderedDict):
			fields_iter = fields.items()
		elif isinstance(fields, list):
			fields_iter = fields
		else:
			raise TypeError("Unexpected Type for fields")

		for name, field_name in fields_iter:
			if isinstance(field_name, str):
				field = Field.load_class(field_name)()
			elif isinstance(field_name, Field):
				field = field_name
			else:
				raise TypeError("Each value of `fields` must be a Field object or a string indicating the name of a Field class.")
			fieldcontent = field._create(set_name) #pylint: disable=protected-access
			fieldcontents[name] = fieldcontent
			new_fields[name] = field
		return new_fields, fieldcontents

	def _create_data_hash(self, fieldcontents):
		raw_data_hash = sha256()
		data_hash = sha256()
		for _, fieldcontents_in_one_set in sorted(fieldcontents.items()):
			for _, fieldcontent in fieldcontents_in_one_set.items():
				raw_data_hash.update(dumps(fieldcontent.get_raw_data_hash()))
				data_hash.update(dumps(fieldcontent.get_data_hash()))
		return raw_data_hash.hexdigest(), data_hash.hexdigest()

	def _create_setting_hash(self):
		setting_hash = sha256()
		for _, fields_in_one_set in sorted(self.fields.items()):
			for _, field in fields_in_one_set.items():
				setting_hash.update(dumps(field._get_setting_hash(self.vocabs))) #pylint: disable=protected-access
		for vocab in self.vocabs:
			setting_hash.update(dumps(vocab.get_setting_hash()))
		for tokenizer in self.tokenizers:
			setting_hash.update(dumps(tokenizer.get_setting_hash()))
		return setting_hash.hexdigest()

	def _create_vocab_hash(self):
		vocab_hash = sha256()
		for vocab in self.vocabs:
			vocab_hash.update(dumps(vocab.get_vocab_hash()))
		return vocab_hash.hexdigest()

	def get_default_vocab(self) -> Vocab:
		'''Get the default :class:`Vocab` in this dataloader.
		It can be set by :meth:`.set_default_field`.
		'''
		vocab = self.get_default_field().get_vocab()
		if vocab is None:
			raise ValueError("This field do not have vocab")
		return vocab

	def get_default_tokenizer(self) -> Tokenizer:
		'''Get the default :class:`Tokenizer` in this dataloader.
		It can be set by :meth:`.set_default_field`.
		'''
		tokenizer = self.get_default_field().get_tokenizer()
		if tokenizer is None:
			raise ValueError("This field do not have tokenizer")
		return tokenizer

	def get_default_field(self) -> Field:
		'''Get the default :class:`Field` in this dataloader.
		It can be set by :meth:`.set_default_field`.
		'''
		if self.default_field_name is None or self.default_field_set_name is None:
			raise RuntimeError("No default field. \
				Specify the default field by set_default_field.")
		return self.fields[self.default_field_set_name][self.default_field_name]

	SET_NAME_DESCRIPTION = '''set_name (str): The name of set. For example: ``"train"``, ``"dev"``, ``"test"``.'''
	FIELD_NAME_DESCRIPTION = '''field_name (str): The name of field.'''

	def set_default_field(self, set_name: str, field_name: str):
		'''Set the default :class:`Field` in this dataloader. In the meanwhile,
		the default :class:`Vocab` and :class:`Tokenizer` is also set according
		to the field (if the field have vocab and tokenizer).

		The default field will affect the action in the following methods:

		* :meth:`get_default_field`
		* :meth:`tokenize`
		* :meth:`tokenize_sentences`
		* :meth:`convert_tokens_to_ids`
		* :meth:`convert_ids_to_tokens`
		* :meth:`convert_ids_to_sentence`
		* :meth:`convert_sentence_to_ids`
		* :meth:`add_special_to_ids`
		* :meth:`remove_special_in_ids`
		* :meth:`process_sentences`
		* :meth:`trim_in_ids`
		* :meth:`get_default_vocab`
		* :meth:`get_special_tokens_mapping`
		* :meth:`get_special_tokens_id`
		* :meth:`get_default_tokenizer`

		Arguments:
			{SET_NAME_DESCRIPTION}
			{FIELD_NAME_DESCRIPTION}
		'''
		if set_name not in self.fields:
			raise KeyError("No such set named %s" % set_name)
		elif field_name not in self.fields[set_name]:
			raise KeyError("No such field named %s" % field_name)
		self.default_field_set_name = set_name
		self.default_field_name = field_name

		# tokenizer = self.fields[set_name][field_name].get_tokenizer()
		# if tokenizer:
		# 	self.set_default_tokenizer(tokenizer)
		# vocab = self.fields[set_name][field_name].get_vocab()
		# if vocab:
		# 	self.set_default_vocab(vocab)

	def get_field(self, set_name: str, field_name: str) -> Field:
		'''Get :class:`Field` according to name of set and field.

		Arguments:
			{SET_NAME_DESCRIPTION}
			{FIELD_NAME_DESCRIPTION}
		'''
		return self.fields[set_name][field_name]

	def get_general_hash(self) -> str:
		'''General hash. Identifying all details in dataloader,
		including raw data before processed, tokenized data, vocabulary, and settings.

		See :ref:`dataloader hash<dataloader_hash_ref>` for explaination.
		'''
		general_hash = sha256()
		general_hash.update(dumps(self._raw_data_hash))
		general_hash.update(dumps(self._data_hash))
		general_hash.update(dumps(self._vocab_hash))
		general_hash.update(dumps(self._setting_hash))
		return general_hash.hexdigest()

	def get_raw_data_hash(self) -> str:
		'''Raw data hash. Identifying raw data before processed.

		See :ref:`dataloader hash<dataloader_hash_ref>` for explaination.
		'''
		return self._raw_data_hash

	def get_data_hash(self) -> str:
		'''Data hash. Identifying data after processed (tokenized).

		See :ref:`dataloader hash<dataloader_hash_ref>` for explaination.
		'''
		return self._data_hash

	def get_vocab_hash(self) -> str:
		'''Vocab hash. Identifying vocabulary.

		See :ref:`dataloader hash<dataloader_hash_ref>` for explaination.
		'''
		return self._vocab_hash

	def get_setting_hash(self) -> str:
		'''Setting hash, identifying settings to create the data loader.

		See :ref:`dataloader hash<dataloader_hash_ref>` for explaination.
		'''
		return self._setting_hash

	def restart(self, set_name, batch_size=None, shuffle=True):
		'''Initialize batches. This function be called before :func:`get_next_batch`
		or an epoch is end. See :meth:`get_next_batch` for examples.

		Arguments:
			{SET_NAME_DESCRIPTION}
			batch_size (int): the number of sample in a batch.
				default: if ``None``, last ``batch_size`` is used.
			shuffle (bool): whether to shuffle the data. Default: ``True``.
		'''
		if set_name not in self.fields:
			raise ValueError("No set named %s." % set_name)
		if batch_size is None and self.batch_size[set_name] is None:
			raise ValueError("You need batch_size to initialize.")
		if shuffle:
			# rng_state = random.getstate()
			random.shuffle(self.index[set_name])
			# random.setstate(rng_state)

		self.batch_id[set_name] = 0
		if batch_size is not None:
			self.batch_size[set_name] = batch_size
		batch_size_div = self.batch_size[set_name]
		assert batch_size_div is not None
		print("%s set restart, %d batches and %d left" % (set_name, \
						len(self.index[set_name]) // batch_size_div, \
						len(self.index[set_name]) % batch_size_div))

	_GET_BATCH_MORE_DOC = "Return a merged dict containing all the data from each field by calling :meth:`.field.get_batch`. " \
		"See examples in subclasses for the return value of predefined tasks."
	_GET_BATCH_EXAMPLE = ""
	def get_batch(self, set_name: str, indexes: List[int]) -> Dict[str, Any]:
		'''Get a batch of data with specified ``indexes``.
		{_GET_BATCH_MORE_DOC}

		:meth:`get_next_batch`, :meth:`get_batches`, :meth:`get_all_batch` provide other methods to get batched data,
		Their return values are consistent with this methods.

		Arguments:
			{SET_NAME_DESCRIPTION}
			indexes (list): a list of specified indexes of batched data.

		{_GET_BATCH_EXAMPLE}
		'''
		if set_name not in self.fields:
			raise ValueError("No set named %s." % set_name)
		res: Dict[str, Any] = {}
		for field_name, field_obj in self.fields[set_name].items():
			res.update(field_obj.get_batch(field_name, self.data[set_name][field_name], indexes)) #pylint: disable=protected-access
		return res

	IGNORE_LEFT_SAMPLES = "ignore_left_samples (bool): If the number of the samples is not divisible by ``batch_size``, " \
			"ignore the left samples less than ``batch_size`` " \
			"Setting it to ``True`` make that every batch will have the same number of samples. " \
			"Default: ``False``."
	def get_next_batch(self, set_name, ignore_left_samples=False) -> Optional[Dict[str, Any]]:
		'''Get next batch. It can be called only after Initializing batches (:func:`restart`).
		Return a dict like :func:`get_batch`, or None if the epoch is end.

		Arguments:
			{SET_NAME_DESCRIPTION}
			{IGNORE_LEFT_SAMPLES}

		Examples:

			>>> dataloader.restart("train")
			>>> while True:
			>>>     data = dataloader.get_next_batch("train")
			>>>     if data:
			>>>         break
            >>>     print(data)

		'''
		if set_name not in self.fields:
			raise ValueError("No set named %s." % set_name)
		batch_size = self.batch_size[set_name]
		if batch_size is None:
			raise RuntimeError( \
				"Please run restart before calling this function.")
		batch_id = self.batch_id[set_name]

		start, end = batch_id * \
					 	batch_size, (batch_id + 1) * batch_size
		if start >= len(self.index[set_name]):
			return None
		if ignore_left_samples and end > len(self.index[set_name]):
			return None
		index = self.index[set_name][start:end]
		res = self.get_batch(set_name, index)
		self.batch_id[set_name] += 1
		return res

	def get_batches(self, set_name, batch_size=None, shuffle=True,
			ignore_left_samples=False) -> Iterable[Dict[str, Any]]:
		'''An iterable generator over batches. It first call :func:`restart`, and then :func:`get_next_batch`
		until no more data is available. Returns an iterable generator where each element is like :func:`get_batch`.

		Arguments:
			{SET_NAME_DESCRIPTION}
			batch_size (int, optional): default: ``None``.  Use ``batch_size`` by default.
			shuffle (bool): whether to shuffle the data. Default: ``True``.
			{IGNORE_LEFT_SAMPLES}
		'''
		self.restart(set_name, batch_size, shuffle)
		while True:
			res = self.get_next_batch(set_name, ignore_left_samples)
			if res is None:
				break
			yield res

	def get_all_batch(self, set_name) -> Dict[str, List[Any]]:
		r'''Concatenate all batches to a single dict, where padding will not be applied.

		Returns a dict like :func:`get_batch` with all valid ``indexes``,
		but all the sentences are not padded and their type will be converted to list.
		Exactly, this function called :func:`get_batch` where ``len(indexes)==1`` multiple times
		and concatenate all the values in the returned dicts.

		Arguments:
			{SET_NAME_DESCRIPTION}
		'''
		res: Dict[str, List[Any]] = {}
		for idx in self.index[set_name]:
			batch = self.get_batch(set_name, [idx])
			for attr, val in batch.items():
				if attr not in res:
					res[attr] = []
				if not isinstance(val, (list, np.ndarray)):
					val = [val]
				res[attr].extend(val)
		return res

	# copy some functions from vocab
	_VOCAB_MORE_DOCSTRING = '''It calls the identical method of the :class:`Vocab` instance ``vocab``,\
		from :meth:`.get_default_vocab()`.'''
	frequent_vocab_size = copy_property(get_default_vocab, Vocab, "frequent_vocab_size")
	all_vocab_size = copy_property(get_default_vocab, Vocab, "all_vocab_size")
	frequent_vocab_list = copy_property(get_default_vocab, Vocab, "frequent_vocab_list")
	all_vocab_list = copy_property(get_default_vocab, Vocab, "all_vocab_list")
	get_special_tokens_mapping = copy_func(get_default_vocab, Vocab, "get_special_tokens_mapping")
	get_special_tokens_id = copy_func(get_default_vocab, Vocab, "get_special_tokens_id")
	pad_id = copy_property(get_default_vocab, Vocab, "pad_id")
	unk_id = copy_property(get_default_vocab, Vocab, "unk_id")
	go_id = copy_property(get_default_vocab, Vocab, "go_id")
	eos_id = copy_property(get_default_vocab, Vocab, "eos_id")

	_SENTENCE_MORE_DOCSTRING = '''It calls the identical method of the :class:`Sentence` instance ``sentence``,\
		from :meth:`.get_default_field()`.'''
	_SESSION_MORE_DOCSTRING = '''It calls the identical method of the :class:`Session` instance ``session``,\
		from :meth:`.get_default_field()`.'''
	tokenize = copy_func(get_default_field, Sentence, "tokenize")
	tokenize_sentences = copy_func(get_default_field, Sentence, "tokenize_sentences")
	convert_tokens_to_ids = copy_func(get_default_field, Sentence, "convert_tokens_to_ids")
	convert_ids_to_tokens = copy_func(get_default_field, Sentence, "convert_ids_to_tokens")
	convert_ids_to_sentence = copy_func(get_default_field, Sentence, "convert_ids_to_sentence")
	convert_sentence_to_ids = copy_func(get_default_field, Sentence, "convert_sentence_to_ids")
	add_special_to_ids = copy_func(get_default_field, Sentence, "add_special_to_ids")
	remove_special_in_ids = copy_func(get_default_field, Sentence, "remove_special_in_ids")
	process_sentences = copy_func(get_default_field, Sentence, "process_sentences")
	trim_in_ids = copy_func(get_default_field, Sentence, "trim_in_ids")
