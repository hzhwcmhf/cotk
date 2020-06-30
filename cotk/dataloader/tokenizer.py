"""A module for Tokenizer"""
from typing import Any, List, Callable
import hashlib
import tempfile
import re

from nltk.tokenize import WordPunctTokenizer
from checksumdir import dirhash

from .._utils.unordered_hash import dumps
from .._utils.metaclass import DocStringInheritor
from .._utils import chain_sessions, restore_sessions

class Tokenizer(metaclass=DocStringInheritor):
	"""Tokenizer is used for spliting sentence to tokens.
	This is an abstract base class.
	It often works as a part of :class:`Field`"""

	def tokenize(self, sentence: str) -> List[str]:
		'''Tokenize a sentence to a list of tokens.

		Arguments:
			sentence (str): a sentence to tokenize.
		'''
		raise NotImplementedError

	def tokenize_sentences(self, sentences: List[str]) -> List[List[str]]:
		'''Tokenize a list of sentences to a list of lists of tokens.

		Arguments:
			sentences (List[str]): sentences to tokenize.
		'''
		return [self.tokenize(sentence) for sentence in sentences]

	def tokenize_sessions(self, sessions: List[List[str]]) -> List[List[List[str]]]:
		'''Tokenize sessions to a 3-d list of tokens.

		Arguments:
			sessions (List[List[str]]): sessions to tokenize.
		'''
		sentences, session_lengths = chain_sessions(sessions)
		tokenized_sentences = self.tokenize_sentences(sentences)
		return restore_sessions(tokenized_sentences, session_lengths)

	def convert_tokens_to_sentence(self, tokens: List[str]) -> str:
		'''Convert tokens to sentence.
		It usually works like the reverse operation of :meth:`tokenize`, but it is not gauranteed.
		It may like ``" ".join(tokens)``, but some special condition and tokens will be took care.

		Arguments:
			tokens(List[str]): tokenized sentence
		'''
		raise NotImplementedError

	def get_setting_hash(self) -> str:
		'''Return the setting hash of this tokenizer instance.
		See :ref:`here <dataloader_hash_ref>` for the explaination of ``setting hash``.
		'''
		raise NotImplementedError

class SimpleTokenizer(Tokenizer):
	'''Bases: :class:`.dataloader.Tokenizer`

	A simple tokenizer. ``method`` can either be ``nltk`` or ``space``.
	If ``nltk``, use ``WordPunctTokenizer`` from ``nltk.tokenize``.
	If ``space``, use ``str.split(" ")``.

	Arguments:
		method (str): the tokenization method, ``nltk`` or ``space``.
		special_tokens (List[str]): special tokens not to tokenize, such as ``<go>``.
	'''
	def __init__(self, method: str, special_tokens: List[str] = None):
		self.method = method
		self.special_tokens = special_tokens

		if method == "nltk":
			self._callable_tokenizer = WordPunctTokenizer().tokenize
		elif method == "space":
			self._callable_tokenizer = str.split
		else:
			raise ValueError('`method` is invalid value {}, should be "nltk" or "space" '.format(method))
		self._setting_hash = hashlib.sha256(dumps(["adapter", method, special_tokens])).hexdigest()

	def tokenize(self, sentence: str) -> List[str]:
		if self.special_tokens is None:
			return self._callable_tokenizer(sentence)
		regexPattern = '(' + '|'.join(map(re.escape, self.special_tokens)) + ')'
		segments = re.split(regexPattern, sentence)
		sent = []
		for seg in segments:
			if seg not in self.special_tokens:
				sent += self._callable_tokenizer(seg.strip())
			else:
				sent += [seg]
		return sent

	def convert_tokens_to_sentence(self, tokens: List[str]) -> str:
		if self.method == "nltk":
			sent = " ".join(tokens)
			out_string = sent.replace(' .', '.').replace(' ?', '?'). \
				replace(' !', '!').replace(' ,', ',').replace(" ' ", "'"). \
				replace(" n't", "n't").replace(" 'm", "'m"). \
				replace(" 's", "'s"). \
				replace(" 've", "'ve").replace(" 're", "'re")
			return out_string
		elif self.method == "space":
			return " ".join(tokens)
		else:
			raise RuntimeError("No such tokenizer %s" % self.method)

	def get_setting_hash(self) -> str:
		return self._setting_hash

class PretrainedTokenizer(Tokenizer):
	'''Bases: :class:`.dataloader.Tokenizer`

	A wrapper for ``Pretrainedtokenizer`` from ``transformers`` package.
	If you don't want to do tokenization on some special tokens, see
	``transformers.Pretrainedtokenizer.add_special_tokens``.

	Arguments:
		tokenizer (transformers.Pretrainedtokenizer): An
			instance of ``transformers.Pretrainedtokenizer``.
	'''
	def __init__(self, tokenizer):
		self.tokenizer = tokenizer
		self._tokenizer_class_name = tokenizer.__class__.__name__
		with tempfile.TemporaryDirectory() as tmp_dir:
			tokenizer.save_vocabulary(str(tmp_dir))
			tokenizer_hash = dirhash(str(tmp_dir), "sha256")
		self._setting_hash = hashlib.sha256(dumps(["pretrained", self._tokenizer_class_name, tokenizer_hash])).hexdigest()

	def tokenize(self, sentence: str) -> List[str]:
		return self.tokenizer.tokenize(sentence)

	def convert_tokens_to_sentence(self, tokens: List[str]) -> str:
		return self.tokenizer.convert_tokens_to_string(tokens)

	def get_setting_hash(self) -> str:
		return self._setting_hash

	def get_tokenizer_class(self) -> str:
		'''Get the class name of pretrained tokenizer.
		'''
		return self._tokenizer_class_name
