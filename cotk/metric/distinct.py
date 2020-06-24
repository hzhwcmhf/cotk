from typing import Union, List, Any, Optional, Iterable
import random
from itertools import chain
import numpy as np
from nltk import ngrams

from .metric import MetricBase
from ..dataloader import LanguageProcessing, Tokenizer, SimpleTokenizer
from .._utils import replace_unk

class DistinctNgramsCorpus(MetricBase):
	'''Metric for calculating BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}

		reference_num can be none, which means uncertain multiple reference number, it will be determined by the argument of forward

	'''

	_name = 'DistinctNgrams'
	_version = 1

	def __init__(self, dataloader: "LanguageProcessing", ngram=3, *, \
			tokenizer: Union[None, Tokenizer, str] = None, \
			sample=10000, \
			seed=1234, \
			gen_key="gen"):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.ngram = ngram
		self.tokenizer = tokenizer
		self.gen_key = gen_key
		self.sample = sample
		self.seed = seed
		self.hyps: List[Any] = []
		self._hash_ordered_data(self.ngram)

	def forward(self, data):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_REFERENCE_ALLVOCABS_ARGUMENTS}
				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...     reference_allvocabs_key: [[2,4,3], [2,5,6,3]],
					...	    gen_key: [[4,5,3], [6,7,8,3]]
					... }
		'''
		super().forward(data)
		if self.tokenizer is None:
			self._direct_forward(data)
		else:
			self._re_tokenize_forward(data)

	def _direct_forward(self, data):
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")

		for gen_sen in gen:
			hyp = self.dataloader.convert_ids_to_tokens(gen_sen, remove_special=True, trim=True)
			self.hyps.append(hyp)

	def _re_tokenize_forward(self, data):
		gen = data[self.gen_key]
		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")
		#fill more typeerror hints

		for gen_sen in gen:
			hyp = self.dataloader.convert_ids_to_sentence(gen_sen, remove_special=True, trim=True)
			self.hyps.append(hyp)

	def close(self):
		'''
		Returns:
			(dict): Return a dict which contains

			* **bleu**: bleu value.
			* **bleu hashvalue**: hash value for bleu metric, same hash value stands
			  for same evaluation settings.
		'''
		result = super().close()
		if not self.hyps:
			raise RuntimeError("The metric has not been forwarded data correctly.")

		if self.sample > len(self.hyps):
			sample = len(self.hyps)
		else:
			sample = self.sample
		self._hash_ordered_data(sample)

		rng_state = random.getstate()
		random.seed(self.seed)
		random.shuffle(self.hyps)
		random.setstate(rng_state)
		self.hyps = self.hyps[:sample]

		if self.tokenizer:
			self._do_tokenize()

		if "unk" in self.dataloader.get_special_tokens_mapping():
			self.hyps = replace_unk(self.hyps, unk = self.dataloader.get_special_tokens_mapping().get("unk", None))

		ngram_list = list(chain(*[ngrams(sentence, self.ngram, pad_left=True, pad_right=True) for sentence in self.hyps]))
		ngram_set = set(ngram_list)

		result.update({"distinct": len(ngram_set) / len(ngram_list), \
			"distinct hashvalue": self._hashvalue()})
		return result

	def _do_tokenize(self):
		tokenizer: Tokenizer
		if isinstance(self.tokenizer, str):
			tokenizer = SimpleTokenizer(self.tokenizer)
		elif isinstance(self.tokenizer, Tokenizer):
			tokenizer = self.tokenizer
		else:
			raise TypeError("Unknown type of tokenizer")

		self.hyps = tokenizer.tokenize_sentences(self.hyps)
