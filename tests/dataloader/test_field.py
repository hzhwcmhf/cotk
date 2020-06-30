try:
	from collections import Iterator  # it may stop working in py3.8
except ImportError:
	from collections.abc import Iterator

from itertools import chain
import os
import shutil
import random
import numpy as np
import pytest
import sys
from pathlib import Path

from cotk.file_utils import file_utils
from cotk.dataloader.field import Field, _FieldContent, Sentence, Session, DenseLabel, SparseLabel, SessionDefault, \
	SentenceDefault, SentenceGPT2, SentenceBERT, SessionGPT2, SessionBERT
from cotk.dataloader import SimpleVocab
from cotk.dataloader import Vocab, Tokenizer, GeneralVocab

sys.path.insert(0, str(Path(__file__).parent.joinpath('../share').resolve()))
from cache_dir import CACHE_DIR


def setup_module():
	random.seed(7)
	np.random.seed(7)
	file_utils.CACHE_DIR = CACHE_DIR

def teardown_module():
	if os.path.isdir(CACHE_DIR):
		shutil.rmtree(CACHE_DIR)


class DummyDataset:
	sentence_file = ["I love NLP.\\n", "Yes I do\\n",
					 "I love deep learning\\n"]  # lines in a text file, if the file contains sentences.
	session_file = ['How are you?\n', "I'm fine. Thank you! And you?\n", "I'm fine, too.\n", "\n",
					"How to install CoTk?\n", "pip install cotk.\n", "\n"]
	dense_label_file = ["1\n", "0\n", "1\n", "2\n"]
	sparse_label_file = ["Java\n", "Python\n", "Cpp\n", "Java\n"]

	sentences = [line.rstrip() for line in sentence_file]

	def __process_ssessions(session_file: list):
		sessions = []
		tmp_list = []
		for line in session_file:
			if line != '\n':
				tmp_list.append(line.rstrip())
			else:
				assert tmp_list
				sessions.append(tmp_list)
				tmp_list = []
		if tmp_list:
			sessions.append(tmp_list)
		return sessions

	sessions = __process_ssessions(session_file)
	del __process_ssessions
	sparse_labels = [label.rstrip() for label in sparse_label_file]
	dense_labels = [int(label.rstrip()) for label in dense_label_file]

	@classmethod
	def get_sentence_iterator(cls) -> Iterator:
		return iter(cls.sentence_file)

	@classmethod
	def get_session_iterator(cls) -> Iterator:
		return iter(cls.session_file)

	@classmethod
	def get_dense_label_iterator(cls) -> Iterator:
		return iter(cls.dense_label_file)

	@classmethod
	def get_sparse_label_iterator(cls) -> Iterator:
		return iter(cls.sparse_label_file)


def load_dataset(field: Field, dataset: Iterator, set_name='train') -> _FieldContent:
	"""Load dataset and fill in the field content."""
	field_content = field._create(set_name)
	while True:
		try:
			field_content.read_next(dataset)
		except StopIteration:
			break
	field_content.process_before_vocab()
	vocab = field.get_vocab()
	if vocab is not None:
		vocab.build_vocab()
	return field_content


class CheckGetBatch:
	UNK_ID = 1  # the same with vocab.py

	@classmethod
	def check_result_of_sentence_get_batch(cls, sentence_field, field_name, data, indexes, result):
		assert isinstance(result, dict)
		assert isinstance(result.get(field_name, None), np.ndarray)
		assert len(result[field_name].shape) == 2 and result[field_name].shape[0] == len(indexes)
		assert isinstance(result.get(field_name + '_length', None), np.ndarray)
		assert result[field_name + '_length'].shape == (len(indexes),)
		assert result[field_name].shape[1] == max(result[field_name + '_length'])
		assert isinstance(result.get(field_name + '_str', None), list)
		assert len(result[field_name + '_str']) == len(indexes)
		if indexes:
			assert isinstance(result[field_name + '_str'][0], str) and result[field_name + '_str'][0]
		assert isinstance(result.get(field_name + '_allvocabs', None), np.ndarray)
		assert len(result[field_name + '_allvocabs'].shape) == 2 and result[field_name + '_allvocabs'].shape[0] == len(
			indexes)
		assert (result[field_name + '_allvocabs'] == cls.UNK_ID).sum() == 0

	@classmethod
	def check_result_of_session_get_batch(cls, session_field, field_name, data, indexes, result):
		assert isinstance(result, dict)
		assert isinstance(result.get(field_name, None), np.ndarray)
		assert len(result[field_name].shape) == 3 and result[field_name].shape[0] == len(indexes)
		assert isinstance(result.get(field_name + '_turn_length'), np.ndarray)
		assert len(result[field_name + '_turn_length'].shape) == 1 and \
			   result[field_name + '_turn_length'].shape[0] == len(indexes)
		assert isinstance(result.get(field_name + '_sent_length', None), list)
		assert len(result[field_name + '_sent_length']) == len(indexes) and isinstance(
			result[field_name + '_sent_length'][0], list) and not isinstance(result[field_name + '_sent_length'][0][0],
																			 list)
		assert isinstance(result.get(field_name + '_allvocabs', None), np.ndarray)
		assert len(result[field_name + '_allvocabs'].shape) == 3 and result[field_name + '_allvocabs'].shape[
			0] == len(indexes)
		assert (result[field_name + '_allvocabs'] == cls.UNK_ID).sum() == 0
		assert isinstance(result.get(field_name + '_str', None), list)
		assert isinstance(result[field_name + '_str'][0], list)
		assert isinstance(result[field_name + '_str'][0][0], str)

		shape = result[field_name].shape
		assert max(result[field_name + '_turn_length']) == shape[1]
		assert max(chain.from_iterable(result[field_name + '_sent_length'])) == shape[2]

	@classmethod
	def check_result_of_dense_label_get_batch(cls, dense_label_field, field_name, data, indexes, result):
		assert isinstance(result, dict)
		assert isinstance(result.get(field_name, None), np.ndarray)
		assert result[field_name].tolist() == [data['label'][i] for i in indexes]

	@classmethod
	def check_result_of_sparse_label_get_batch(cls, sparse_label_field, field_name, data, indexes, result):
		assert isinstance(result, dict)
		assert isinstance(result.get(field_name + '_id', None), np.ndarray)
		assert result[field_name + '_id'].shape == (len(indexes),)
		assert result[field_name + '_str'] == [data['str'][i] for i in indexes]

	@classmethod
	def check_result_of_get_batch(cls, field, field_name, data, indexes, result):
		"""Check whether the returned value of :meth:`Field.get_batch` has correct format.

		Args:
			field: An instance of Field.
			field_name: argument of :meth:`Field.get_batch`
			data: argument of :meth:`Field.get_batch`
			indexes: argument of :meth:`Field.get_batch`
			result: returned by :meth:`Field.get_batch`
		"""
		if isinstance(field, DenseLabel):
			cls.check_result_of_dense_label_get_batch(field, field_name, data, indexes, result)
		elif isinstance(field, SparseLabel):
			cls.check_result_of_sparse_label_get_batch(field, field_name, data, indexes, result)
		elif isinstance(field, Session):
			cls.check_result_of_session_get_batch(field, field_name, data, indexes, result)
		elif isinstance(field, Sentence):
			cls.check_result_of_sentence_get_batch(field, field_name, data, indexes, result)
		else:
			raise TypeError('invalid type %s' % type(field))


class BaseTestField:
	"""Test subclass of :class:`Field`"""

	def base_test_get_vocab(self, field: Field):
		vocab = field.get_vocab()
		assert vocab is None or isinstance(vocab, Vocab)

	def base_test_get_tokenizer(self, field: Field):
		tokenizer = field.get_tokenizer()
		assert tokenizer is None or isinstance(tokenizer, Tokenizer)

	def base_test_create(self, field: Field):
		for set_name in ['train', 'test', 'dev']:
			assert isinstance(field._create(set_name), _FieldContent)

	def base_test_load_dataset(self, field: Field, dataset: Iterator, set_name='train'):
		field_content = load_dataset(field, dataset, set_name)
		with pytest.raises(StopIteration):
			next(dataset)
		return field_content

	def base_get_setting_hash(self, field: Field, vocabs):
		setting_hash = field._get_setting_hash(vocabs)
		assert isinstance(setting_hash, str)

	@classmethod
	def check_result_of_get_batch(cls, field: Field, field_name: str, data: dict, indexes: list, result: dict):
		"""Check whether the returned value of :meth:`Field.get_batch` has correct format.

		Args:
			field: An instance of Field.
			field_name: argument of :meth:`Field.get_batch`
			data: argument of :meth:`Field.get_batch`
			indexes: argument of :meth:`Field.get_batch`
			result: returned by :meth:`Field.get_batch`
		"""
		CheckGetBatch.check_result_of_get_batch(field, field_name, data, indexes, result)

	def base_test_get_batch(self, field, field_name, get_dataset):
		field_content = load_dataset(field, get_dataset())
		data = field_content.get_data()
		for indexes in self.get_indexes_iterator(data):
			batch_data = field.get_batch(field_name, data, indexes)
			self.check_result_of_get_batch(field, field_name, data, indexes, batch_data)

	@staticmethod
	def get_indexes_iterator(data: dict) -> Iterator:
		"""
		Generate indexes, which is used for an argument :meth:`Field.get_batch`.

		Args:
			data: The returned value of :meth:`_FieldContent.get_data`. The parameter `data` of :meth:`Field.get_batch`.
		"""
		size = len(next(iter(data.values())))
		all_indexes = list(range(size))
		random.shuffle(all_indexes)
		for i in all_indexes:
			yield [i]


@pytest.fixture
def get_field():
	return lambda: Field()


class TestField(BaseTestField):
	"""Test Field itself."""

	def test_get_vocab(self, get_field):
		assert get_field().get_vocab() is None

	def test_get_tokenizer(self, get_field):
		assert get_field().get_tokenizer() is None

	def test_create(self, get_field):
		with pytest.raises(NotImplementedError):
			get_field()._create('train')

	def test_get_setting_hash(self, get_field):
		with pytest.raises(NotImplementedError):
			get_field()._get_setting_hash([])

	def test_get_batch(self, get_field):
		with pytest.raises(NotImplementedError):
			get_field().get_batch('field_name', {}, [])


@pytest.fixture
def get_sentence_field():
	def _get_sentence_field():
		return SentenceDefault('space', GeneralVocab(), convert_to_lower_letter=True)

	return _get_sentence_field


class BaseTestSentence(BaseTestField):
	def base_test_init(self, sentence_field):
		assert isinstance(sentence_field, Sentence)
		assert isinstance(getattr(sentence_field, 'vocab', None), Vocab)
		assert isinstance(getattr(sentence_field, 'vocab_from_mappings', None), dict)
		for k, v in sentence_field.vocab_from_mappings.items():
			assert isinstance(k, str) and isinstance(v, str)
		assert hasattr(sentence_field, 'max_sent_length')
		assert sentence_field.max_sent_length is None or isinstance(sentence_field.max_sent_length, int)
		assert isinstance(getattr(sentence_field, 'convert_to_lower_letter', None), bool)
		assert isinstance(getattr(sentence_field, 'tokenizer', None), Tokenizer)

		super().base_test_create(sentence_field)
		super().base_test_get_vocab(sentence_field)
		super().base_test_get_tokenizer(sentence_field)


class TestSentence(BaseTestSentence):
	def test_init(self, get_sentence_field):
		super().base_test_init(get_sentence_field())

	def test_load_dataset(self, get_sentence_field):
		sentence_field = get_sentence_field()
		sentence_field_content = super().base_test_load_dataset(sentence_field, DummyDataset.get_sentence_iterator())
		assert sentence_field_content._original_data == DummyDataset.sentences

	def test_get_setting_hash(self, get_sentence_field):
		sentence_field: Sentence = get_sentence_field()
		super().base_get_setting_hash(sentence_field, [sentence_field.get_vocab()])
		setting_hash = sentence_field._get_setting_hash([sentence_field.get_vocab()])
		load_dataset(sentence_field, DummyDataset.get_sentence_iterator())
		assert setting_hash == sentence_field._get_setting_hash([sentence_field.get_vocab()])

	def test_get_batch(self, get_sentence_field):
		super().base_test_get_batch(get_sentence_field(), 'sent', DummyDataset.get_sentence_iterator)


def get_session_field(tokenizer='space', min_frequent_vocab_times=0, min_rare_vocab_times=0):
	def _get_session_field():
		return SessionDefault(tokenizer,
							  GeneralVocab(min_frequent_vocab_times=min_frequent_vocab_times, min_rare_vocab_times=min_rare_vocab_times),
							  convert_to_lower_letter=True)

	return _get_session_field


all_get_session_fields = [get_session_field(), get_session_field(tokenizer='nltk', min_frequent_vocab_times=2),
					  get_session_field(min_rare_vocab_times=2)]


class TestSession(BaseTestSentence):
	@pytest.mark.parametrize('get_session_field', all_get_session_fields)
	def test_init(self, get_session_field):
		session_field: Session = get_session_field()
		super().base_test_init(session_field)
		assert isinstance(session_field, Session)
		assert hasattr(session_field, 'max_turn_length')
		assert session_field.max_turn_length is None or isinstance(session_field.max_turn_length, int)

	@pytest.mark.parametrize('get_session_field', all_get_session_fields)
	def test_load_dataset(self, get_session_field):
		session_field = get_session_field()
		session_field_content = super().base_test_load_dataset(session_field, DummyDataset.get_session_iterator())
		assert session_field_content._original_data == DummyDataset.sessions

	@pytest.mark.parametrize('get_session_field', all_get_session_fields)
	def test_get_setting_hash(self, get_session_field):
		session_field: Session = get_session_field()
		super().base_get_setting_hash(session_field, [session_field.get_vocab()])
		setting_hash = session_field._get_setting_hash([session_field.get_vocab()])
		load_dataset(session_field, DummyDataset.get_session_iterator())
		assert setting_hash == session_field._get_setting_hash([session_field.get_vocab()])

	@pytest.mark.parametrize('get_session_field', all_get_session_fields)
	def test_get_batch(self, get_session_field):
		super().base_test_get_batch(get_session_field(), 'session', DummyDataset.get_session_iterator)


@pytest.fixture
def get_dense_label():
	def _get_dense_label():
		return DenseLabel()

	return _get_dense_label


class TestDenseLabel(BaseTestField):
	def test_init(self, get_dense_label):
		dense_label = get_dense_label()
		super().base_test_create(dense_label)
		super().base_test_get_vocab(dense_label)
		super().base_test_get_tokenizer(dense_label)

	def test_load_dataset(self, get_dense_label):
		dense_label_field = get_dense_label()
		dense_label_field_content = super().base_test_load_dataset(dense_label_field,
																   DummyDataset.get_dense_label_iterator())
		assert dense_label_field_content._original_data == DummyDataset.dense_labels

	def test_get_setting_hash(self, get_dense_label):
		dense_label_field: DenseLabel = get_dense_label()
		super().base_get_setting_hash(dense_label_field, [dense_label_field.get_vocab()])
		setting_hash = dense_label_field._get_setting_hash([dense_label_field.get_vocab()])
		load_dataset(dense_label_field, DummyDataset.get_dense_label_iterator())
		assert setting_hash == dense_label_field._get_setting_hash([dense_label_field.get_vocab()])

	def test_get_batch(self, get_dense_label):
		super().base_test_get_batch(get_dense_label(), 'dense_label', DummyDataset.get_dense_label_iterator)


@pytest.fixture
def get_sparse_label_field():
	def _get_sparse_label_field():
		return SparseLabel(SimpleVocab())

	return _get_sparse_label_field


class TestSparseLabel(BaseTestField):

	def test_init(self, get_sparse_label_field):
		sparse_label_field = get_sparse_label_field()
		assert isinstance(sparse_label_field, SparseLabel)
		super().base_test_get_tokenizer(sparse_label_field)
		super().base_test_get_vocab(sparse_label_field)
		super().base_test_create(sparse_label_field)

	def test_load_dataset(self, get_sparse_label_field):
		sparse_label_field = get_sparse_label_field()
		sparse_label_field_content = super().base_test_load_dataset(sparse_label_field,
																	DummyDataset.get_sparse_label_iterator())
		assert sparse_label_field_content._original_data == DummyDataset.sparse_labels

	def test_get_batch(self, get_sparse_label_field):
		super().base_test_get_batch(get_sparse_label_field(), 'sparse_label', DummyDataset.get_sparse_label_iterator)

	def test_get_setting_hash(self, get_sparse_label_field):
		sparse_label_field: SparseLabel = get_sparse_label_field()
		super().base_get_setting_hash(sparse_label_field, [])
		setting_hash = sparse_label_field._get_setting_hash([])
		load_dataset(sparse_label_field, DummyDataset.get_sparse_label_iterator())
		assert setting_hash == sparse_label_field._get_setting_hash([])  # setting hash doesn't change.
