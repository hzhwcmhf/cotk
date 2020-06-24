r"""
Containing some classes and functions about bleu evaluating results of models.
"""
from typing import Union, List, Any, Optional, Iterable, Dict
import random
import os
import multiprocessing
from multiprocessing import Pool
import numpy as np
import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from .metric import MetricBase
from ..dataloader.tokenizer import Tokenizer, SimpleTokenizer
from .._utils import replace_unk

if False: # for type check # disable: using-constant-test
	from ..dataloader.dataloader import LanguageProcessing

def _sentence_bleu(ele):
	'''Auxiliary function for computing sentence bleu:

	Arguments:
		ele (tuple): A tuple (`reference sentences`, `a hypothesis sentence`).

	Returns:

		* int: **sentence-bleu** value.
	'''

	return sentence_bleu(ele[0], ele[1], weights=ele[2], smoothing_function=SmoothingFunction().method1)

class BleuCorpusMetric(MetricBase):
	'''Metric for calculating BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.NGRAM_ARGUMENTS}
		{MetricBase.TOKENIZER_ARGUMENTS}
		reference_num (int, None, optional): The number of references used to calculate BLEU. If ``None``, the number of references \
		is uncertain and will be determined by the argument of :meth:`.forward`. Default: ``1``.
		{MetricBase.IGNORE_SMOOTHING_ERROR_ARGUMENTS}
		{MetricBase.REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}
		reference_str_key (str, optional): The key of reference sentences in the string form. Default: ``ref_str``.

	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> reference_allvocabs_key = "ref_allvocabs"
		>>> gen_key = "gen"
		>>> metric = cotk.metric.BleuCorpusMetric(dl,
		...	    reference_allvocabs_key=reference_allvocabs_key,
		...	    gen_key=gen_key)
		>>> data = {
		...     reference_allvocabs_key: [[2, 10, 64, 851, 3], [2, 10, 48, 851, 3]],
		...     # reference_allvocabs_key: [["<go>", "I", "like", "python", "<eos>"], ["<go>", "I", "use", "python", "<eos>"]],
		...
		...	    gen_key: [[10, 1028, 479, 285, 220, 3], [851, 17, 2451, 3]]
		...	    # gen_key: [["I", "love", "java", "very", "much", "<eos>"], ["python", "is", "excellent", "<eos>"]],
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'bleu': 0.08582363099612991,
		'bleu hashvalue': '70e019630fef24d9477034a3d941a5349fcbff5a3dc6978a13ea3d85290114fb'}

	'''

	_name = 'BleuCorpusMetric'
	_version = 2

	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], ngram: int =4, *, \
			tokenizer: Union[None, Tokenizer, str] = None, reference_num: Optional[int] = 1, \
			ignore_smoothing_error: bool = False, reference_allvocabs_key: str = "ref_allvocabs", \
			gen_key: str = "gen", reference_str_key: str = "ref_str"):
		super().__init__(self._name, self._version)
		#self._hash_ordered_data(self.ngram)
		self.dataloader = dataloader
		self.ngram = ngram
		self.tokenizer = tokenizer
		self.reference_num = reference_num
		self.ignore_smoothing_error = ignore_smoothing_error
		self.reference_allvocabs_key = reference_allvocabs_key
		self.reference_str_key = reference_str_key
		self.gen_key = gen_key
		self.hyps: List[Any] = []
		self.refs: List[List[Any]] = []

	def forward(self, data: Dict[str, Any]):
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

	def _direct_forward(self, data: Dict[str, Any]):
		gen = data[self.gen_key]
		resp = data[self.reference_allvocabs_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")
		if not isinstance(resp, (np.ndarray, list)):
			raise TypeError("Unknown type for resp")

		if len(resp) != len(gen):
			raise ValueError("Batch num is not matched.")

		relevant_data = []
		for gen_sen, resp_sen in zip(gen, resp):
			hyp = self.dataloader.convert_ids_to_tokens(gen_sen, remove_special=True, trim=True)
			if self.reference_num == 1:
				refs = [self.dataloader.convert_ids_to_tokens(resp_sen, remove_special=True, trim=True)]
			else:
				if self.reference_num is not None and len(resp_sen) != self.reference_num:
					raise RuntimeError("Require %d references but get %d" % (self.reference_num, len(resp_sen)))
				refs = [self.dataloader.convert_ids_to_tokens(resp_single_sen, remove_special=True, trim=True) for resp_single_sen in resp_sen]
			self.hyps.append(hyp)
			self.refs.append(refs)
			relevant_data.append(refs)
		self._hash_unordered_list(relevant_data)

	def _re_tokenize_forward(self, data: Dict[str, Any]):
		gen = data[self.gen_key]
		resp = data.get(self.reference_allvocabs_key, None)
		resp_str = data.get(self.reference_str_key, None)

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")
		#fill more typeerror hints

		relevant_data = []
		for i, gen_sen in enumerate(gen):
			hyp = self.dataloader.convert_ids_to_sentence(gen_sen, remove_special=True, trim=True)
			if resp_str:
				if self.reference_num == 1:
					refs = [resp_str[i]]
				else:
					if self.reference_num is not None and len(resp_str[i]) != self.reference_num:
						raise RuntimeError("Require %d references but get %d" % (self.reference_num, len(resp_str[i])))
					refs = resp_str[i]
			else:
				if self.reference_num == 1:
					refs = [self.dataloader.convert_ids_to_sentence(resp[i], remove_special=None, trim=True)]
				else:
					if self.reference_num is not None and len(resp[i]) != self.reference_num:
						raise RuntimeError("Require %d references but get %d" % (self.reference_num, len(resp[i])))
					refs = [self.dataloader.convert_ids_to_sentence(resp_single_sen, remove_special=True, trim=True) for resp_single_sen in resp[i]]
			self.hyps.append(hyp)
			self.refs.append(refs)
			relevant_data.append(refs)
		self._hash_unordered_list(relevant_data)

	def close(self) -> Dict[str, Any]:
		'''Return a dict which contains

			* **bleu**: bleu value.
			* **bleu hashvalue**: hash value for bleu metric, same hash value stands
			  for same evaluation settings.
		'''
		result = super().close()
		if (not self.hyps) or (not self.refs):
			raise RuntimeError("The metric has not been forwarded data correctly.")

		if self.tokenizer:
			self._do_tokenize()

		if "unk" in self.dataloader.get_special_tokens_mapping():
			self.hyps = replace_unk(self.hyps, self.dataloader.get_special_tokens_mapping()["unk"])
		try:
			weights = np.ones(self.ngram) / self.ngram
			result.update({"bleu": \
				corpus_bleu(self.refs, self.hyps, weights=weights, smoothing_function=SmoothingFunction().method3), \
				"bleu hashvalue": self._hashvalue()})
		except ZeroDivisionError as _:
			if not self.ignore_smoothing_error:
				raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
				usually caused when there is only one sample and the sample length is 1.") from None
			result.update({"bleu": \
					0, \
					"bleu hashvalue": self._hashvalue()})
		return result

	def _do_tokenize(self):
		tokenizer: Tokenizer
		if isinstance(self.tokenizer, str):
			tokenizer = SimpleTokenizer(self.tokenizer)
		elif isinstance(self.tokenizer, Tokenizer):
			tokenizer = self.tokenizer
		else:
			raise TypeError("Unknown type of tokenizer")

		self.refs = tokenizer.tokenize_sessions(self.refs)
		self.hyps = tokenizer.tokenize_sentences(self.hyps)


class SelfBleuCorpusMetric(MetricBase):
	r'''Metric for calculating Self-BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.NGRAM_ARGUMENTS}
		{MetricBase.TOKENIZER_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}
		{MetricBase.SAMPLE_ARGUMENTS_IN_BLEU}
		{MetricBase.SEED_ARGUMENTS}
		{MetricBase.CPU_COUNT_ARGUMENTS}

	Warning:
		the calculation of ``hashvalue`` considers the actual sample size of hypotheses which
		will be less than ``sample`` if the size of hypotheses is smaller than ``sample``.

	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> gen_key = 'gen'
		>>> metric = cotk.metric.SelfBleuCorpusMetric(dl, gen_key=gen_key)
		>>> data = {
		...	    gen_key: [[10, 64, 851, 3], [10, 48, 851, 3]],
		...	    # gen_key: [["I", "like", "python", "<eos>"], ["I", "use", "python", "<eos>"]],
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'self-bleu': 0.13512001548070346,
		'self-bleu hashvalue': '53cf55829c1b080c86c392c846a5d39a54340c70d838ec953f952aa6731118fb'}
	'''

	_name = 'SelfBleuCorpusMetric'
	_version = 2

	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], ngram: int = 4, *, \
		tokenizer: Union[None, Tokenizer, str] = None, \
		gen_key: str = "gen", \
		sample: int = 1000, \
		seed: int = 1229, \
		cpu_count: Optional[int] = None):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.ngram = ngram
		self.tokenizer = tokenizer
		self.gen_key = gen_key
		self.sample = sample
		self.hyps: List[Any] = []
		self.seed = seed
		if cpu_count is not None:
			self.cpu_count = cpu_count
		elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
			self.cpu_count = int(os.environ["CPU_COUNT"])
		else:
			self.cpu_count = multiprocessing.cpu_count()

	def forward(self, data: Dict[str, Any]):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...	    gen_key: [[4,5,3], [6,7,8,3]]
					... }
		'''
		super().forward(data)
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")

		self.hyps.extend(gen)

	def close(self) -> Dict[str, Any]:
		'''Return a dict which contains

			* **self-bleu**: self-bleu value.
			* **self-bleu hashvalue**: hash value for self-bleu metric, same hash value stands
			  for same evaluation settings.
		'''
		res = super().close()
		if not self.hyps:
			raise RuntimeError("The metric has not been forwarded data correctly.")
		if len(self.hyps) == 1:
			raise RuntimeError("Self-Bleu can't be computed because there is only 1 generated sentence.")
		if self.sample <= 1:
			raise RuntimeError('`self.sample` should be more than 1, \
				whose value is `{}`'.format(self.sample))

		if self.sample > len(self.hyps):
			self.sample = len(self.hyps)

		rng_state = random.getstate()
		random.seed(self.seed)
		random.shuffle(self.hyps)
		random.setstate(rng_state)

		ref = self.hyps[:self.sample]

		if self.tokenizer:
			tokenizer: Tokenizer
			if isinstance(self.tokenizer, str):
				tokenizer = SimpleTokenizer(self.tokenizer)
			else:
				tokenizer = tokenizer
			ref = [self.dataloader.convert_ids_to_sentence(ids, remove_special=True, trim=True) for ids in ref]
			ref = tokenizer.tokenize_sentences(ref)
		else:
			ref = [self.dataloader.convert_ids_to_tokens(ids, remove_special=True, trim=True) for ids in ref]

		if "unk" in self.dataloader.get_special_tokens_mapping():
			_ref = replace_unk(ref, self.dataloader.get_special_tokens_mapping()["unk"])
		else:
			_ref = ref

		bleu_irl = []

		weights = np.ones(self.ngram) / self.ngram
		tasks = ((ref[:i]+ref[i+1:self.sample], _ref[i], weights) for i in range(self.sample))

		pool: Optional[Any]
		values: Iterable[Any]
		if self.sample >= 1000 and self.cpu_count > 1:
			# use multiprocessing
			pool = Pool(self.cpu_count)
			values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
		else:
			pool = None
			values = map(_sentence_bleu, tasks)
		if self.sample >= 1000:
			# use tqdm
			values = tqdm.tqdm(values, total=self.sample)
		for ans in values:
			bleu_irl.append(ans)
		if pool is not None:
			pool.close()
			pool.join()

		self._hash_ordered_data((self.seed, self.sample))
		res.update({"self-bleu" : 1.0 * sum(bleu_irl) / len(bleu_irl),\
					"self-bleu hashvalue": self._hashvalue()})
		return res

class FwBwBleuCorpusMetric(MetricBase):
	r'''Metric for calculating FwBw-BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.REFERENCE_TEST_LIST_ARGUMENTS}
		{MetricBase.NGRAM_ARGUMENTS}
		{MetricBase.TOKENIZER_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}
		{MetricBase.SAMPLE_ARGUMENTS_IN_BLEU}
		{MetricBase.SEED_ARGUMENTS}
		{MetricBase.CPU_COUNT_ARGUMENTS}
	Warning:
		The calculation of ``hashvalue`` considers the actual sample size of hypotheses and
		references. Therefore ``hashvalue`` may vary with the size of hypothesis or references
		if the size of them is smaller than ``sample``.

	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> gen_key = 'gen'
		>>> metric = cotk.metric.FwBwBleuCorpusMetric(dl,
		...	    reference_test_list=dl.get_all_batch('test')['session'][0].tolist(),
		...	    gen_key=gen_key)
		>>> data = {
		...	    gen_key: [[10, 64, 851, 3], [10, 48, 851, 3]],
		...	    # gen_key: [["I", "like", "python", "<eos>"], ["I", "use", "python", "<eos>"]],
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'fw-bleu': 0.007688528488990184,
 		 'bw-bleu': 0.0012482612634667945,
 		 'fw-bw-bleu': 0.002147816509441494,
 		 'fw-bw-bleu hashvalue': '0e3f58a90225af615ff780f04c91613759e04a3c7b4329670b1d03b679adf8cd'}
	'''

	_name = 'FwBwBleuCorpusMetric'
	_version = 2

	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], \
			reference_test_list: List[Any], ngram: int = 4, *, \
			tokenizer: Union[None, Tokenizer, str] = None, \
			gen_key: str = "gen", \
			sample: int = 1000, \
			seed: int = 1229, \
			cpu_count: Optional[int] = None):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.tokenizer = tokenizer
		self.reference_test_list = reference_test_list
		self.gen_key = gen_key
		self.sample = sample
		self.seed = seed
		self.ngram=ngram
		if cpu_count is not None:
			self.cpu_count = cpu_count
		elif "CPU_COUNT" in os.environ and os.environ["CPU_COUNT"] is not None:
			self.cpu_count = int(os.environ["CPU_COUNT"])
		else:
			self.cpu_count = multiprocessing.cpu_count()
		self.hyps: List[Any] = []

	def forward(self, data: Dict[str, Any]):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_GEN_ARGUMENTS}

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...	    gen_key: [[4,5,3], [6,7,8,3]]
					... }
		'''
		gen = data[self.gen_key]

		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen.")

		for gen_sen in gen:
			self.hyps.append(list(self.dataloader.trim_in_ids(gen_sen)))

	def close(self) -> Dict[str, Any]:
		'''Return a dict which contains

			* **fw-bleu**: fw bleu value.
			* **bw-bleu**: bw bleu value.
			* **fw-bw-bleu**: harmony mean of fw/bw bleu value.
			* **fw-bw-bleu hashvalue**: hash value for fwbwbleu metric, same hash value stands
			  for same evaluation settings.
		'''
		res = super().close()
		if not self.hyps:
			raise RuntimeError("The metric has not been forwarded data correctly.")
		if not self.reference_test_list:
			raise RuntimeError("Reference cannot be empty")

		sample_hyps_num = self.sample if self.sample < len(self.hyps) else len(self.hyps)
		sample_refs_num = self.sample if self.sample < len(self.reference_test_list) else len(self.reference_test_list)

		if sample_hyps_num <= 1:
			raise RuntimeError('`sample_hyps` should be more than 1, \
				whose value is `{}`'.format(sample_hyps_num))
		if sample_refs_num <= 1:
			raise RuntimeError('`sample_refs` should be more than 1, \
				whose value is `{}`'.format(sample_refs_num))

		sample_hyps = self.hyps[:sample_hyps_num]
		sample_refs = self.reference_test_list[:sample_refs_num]

		refs: List[Any]
		hyps: List[Any]
		if self.tokenizer:
			tokenizer: Tokenizer
			if isinstance(self.tokenizer, str):
				tokenizer = SimpleTokenizer(self.tokenizer)
			else:
				tokenizer = tokenizer
			if isinstance(sample_refs[0], List):
				ref_sents = [self.dataloader.convert_ids_to_sentence(ids, remove_special=True, trim=True) for ids in sample_refs]
			else:
				ref_sents = sample_refs
			refs = tokenizer.tokenize_sentences(ref_sents)

			hyp_sents = [self.dataloader.convert_ids_to_sentence(ids, remove_special=True, trim=True) for ids in sample_hyps]
			hyps = tokenizer.tokenize_sentences(hyp_sents)
		else:
			refs = [self.dataloader.convert_ids_to_tokens(ids, remove_special=True, trim=True) for ids in sample_refs]
			hyps = [self.dataloader.convert_ids_to_tokens(ids, remove_special=True, trim=True) for ids in sample_hyps]

		rng_state = random.getstate()
		random.seed(self.seed)
		random.shuffle(hyps)
		random.shuffle(refs)
		random.setstate(rng_state)

		if "unk" in self.dataloader.get_special_tokens_mapping():
			refs = replace_unk(refs, self.dataloader.get_special_tokens_mapping()["unk"])


		bleu_irl_fw, bleu_irl_bw = [], []
		weights = np.ones(self.ngram) / self.ngram

		tasks = ((refs, hyps[i], weights) for i in range(sample_hyps_num))
		pool: Optional[Any]
		values: Iterable[Any]
		if sample_hyps_num >= 1000 and self.cpu_count > 1:
			pool = Pool(self.cpu_count)
			values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
		else:
			pool = None
			values = map(_sentence_bleu, tasks)
		if sample_hyps_num >= 1000:
			values = tqdm.tqdm(values, total=sample_hyps_num)
		for ans in values:
			bleu_irl_fw.append(ans)
		if pool is not None:
			pool.close()
			pool.join()

		tasks = ((hyps, refs[i], weights) for i in range(sample_refs_num))
		if sample_refs_num >= 1000 and self.cpu_count > 1:
			pool = Pool(self.cpu_count)
			values = pool.imap_unordered(_sentence_bleu, tasks, chunksize=20)
		else:
			pool = None
			values = map(_sentence_bleu, tasks)
		if sample_refs_num >= 1000:
			values = tqdm.tqdm(values, total=sample_refs_num)
		for ans in values:
			bleu_irl_bw.append(ans)
		if pool is not None:
			pool.close()
			pool.join()

		fw_bleu = (1.0 * sum(bleu_irl_fw) / len(bleu_irl_fw))
		bw_bleu = (1.0 * sum(bleu_irl_bw) / len(bleu_irl_bw))
		if fw_bleu + bw_bleu > 0:
			fw_bw_bleu = 2.0 * bw_bleu * fw_bleu / (fw_bleu + bw_bleu)
		else:
			fw_bw_bleu = 0

		res.update({"fw-bleu" : fw_bleu, \
			"bw-bleu" : bw_bleu, \
			"fw-bw-bleu" : fw_bw_bleu \
		})

		self._hash_unordered_list(refs)
		self._hash_ordered_data((self.ngram, self.seed, sample_hyps_num, sample_refs_num))
		res.update({"fw-bw-bleu hashvalue" : self._hashvalue()})
		return res

class MultiTurnBleuCorpusMetric(MetricBase):
	'''Metric for calculating multi-turn BLEU.

	Arguments:
		{MetricBase.DATALOADER_ARGUMENTS}
		{MetricBase.IGNORE_SMOOTHING_ERROR_ARGUMENTS}
		{MetricBase.MULTI_TURN_REFERENCE_ALLVOCABS_KEY_ARGUMENTS}
		{MetricBase.MULTI_TURN_GEN_KEY_ARGUMENTS}
		{MetricBase.MULTI_TURN_LENGTH_KEY_ARGUMENTS}

	Here is an example:

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> multi_turn_reference_allvocabs_key = "reference_allvocabs"
		>>> multi_turn_gen_key = "multi_turn_gen"
		>>> turn_len_key = "turn_length"
		>>> metric = cotk.metric.MultiTurnBleuCorpusMetric(dl,
		>>>     multi_turn_reference_allvocabs_key=multi_turn_reference_allvocabs_key,
		>>>     multi_turn_gen_key=multi_turn_gen_key,
		>>>     turn_len_key=turn_len_key)
		>>> data = {
		...	    multi_turn_reference_allvocabs_key: [[[2, 10, 64, 851, 3], [2, 10, 64, 479, 3]], [[2, 10, 64, 279, 1460, 3]]],
		...     # multi_turn_reference_allvocabs_key = [[["<go>", "I", "like", "python", "<eos>"], ["<go>", "I", "like", "java", "<eos>"]],
		...     # 	[["<go>", "I", "like", "machine", "learning", "<eos>"]]]
		...
		...	    turn_len_key: [2, 1],
		...     # turn_len_key: [len(multi_turn_reference_allvocabs_key[0]), len(multi_turn_reference_allvocabs_key[1])]
		...
		...	    multi_turn_gen_key: [[[851, 17, 2451, 3], [2019, 17, 393, 3]], [[10, 64, 34058, 805, 2601, 3]]]
		...     # multi_turn_gen_key = [[["python", "is", "excellent", "<eos>"], ["PHP", "is", "best", "<eos>"]],
		...     # 	[["I", "like", "natural", "language", "processing", "<eos>"]]]
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'bleu': 0.12081744577265555,
		'bleu hashvalue': 'c65b44c454dee5a8d393901644c7f1acfdb847bae3ab03823cb5b9f643958960'}
	'''

	_name = 'MultiTurnBleuCorpusMetric'
	_version = 2

	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], \
					ignore_smoothing_error: bool = False,\
					multi_turn_reference_allvocabs_key: str = "reference_allvocabs", \
					multi_turn_gen_key: str = "multi_turn_gen", \
					turn_len_key: str = "turn_length" \
			  ):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.ignore_smoothing_error = ignore_smoothing_error
		self.multi_turn_reference_allvocabs_key = multi_turn_reference_allvocabs_key
		self.turn_len_key = turn_len_key
		self.multi_turn_gen_key = multi_turn_gen_key
		self.refs = []
		self.hyps = []

	def forward(self, data: Dict[str, Any]):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_MULTI_TURN_REFERENCE_ALLVOCABS_ARGUMENTS}
				{MetricBase.FORWARD_MULTI_TURN_GEN_ARGUMENTS}
				{MetricBase.FORWARD_MULTI_TURN_LENGTH_ARGUMENTS}

				Here is an example for data:

					>>> # all_vocab_list = ["<pad>", "<unk>", "<go>", "<eos>", "I", "have",
					>>> #   "been", "to", "China"]
					>>> data = {
					...	    multi_turn_reference_allvocabs_key: [[[2,4,3], [2,5,6,3]], [[2,7,6,8,3]]],
					...	    turn_len_key: [2, 1],
					...	    gen_key: [[[6,7,8,3], [4,5,3]], [[7,3]]]
					... }
		'''
		super().forward(data)
		reference_allvocabs = data[self.multi_turn_reference_allvocabs_key]
		length = data[self.turn_len_key]
		gen = data[self.multi_turn_gen_key]

		if not isinstance(reference_allvocabs, (np.ndarray, list)):
			raise TypeError("Unknown type for reference_allvocabs.")
		if not isinstance(length, (np.ndarray, list)):
			raise TypeError("Unknown type for length")
		if not isinstance(gen, (np.ndarray, list)):
			raise TypeError("Unknown type for gen")

		if len(length) != len(reference_allvocabs) or len(length) != len(gen):
			raise ValueError("Batch num is not matched.")

		for i, turn_length in enumerate(length):
			gen_session = gen[i]
			ref_session = reference_allvocabs[i]
			for j in range(turn_length):
				self.hyps.append(list(self.dataloader.trim_in_ids(gen_session[j])))
				self.refs.append([list(self.dataloader.trim_in_ids(ref_session[j])[1:])])

	def close(self) -> Dict[str, Any]:
		'''Return a dict which contains

			* **bleu**: bleu value.
			* **bleu hashvalue**: hash value for bleu metric, same hash value stands
			  for same evaluation settings.
		'''
		result = super().close()
		if (not self.hyps) or (not self.refs):
			raise RuntimeError("The metric has not been forwarded data correctly.")
		self.hyps = replace_unk(self.hyps, self.dataloader.unk_id)

		self._hash_unordered_list(self.refs)

		try:
			result.update({"bleu": \
				corpus_bleu(self.refs, self.hyps, smoothing_function=SmoothingFunction().method3), \
				"bleu hashvalue": self._hashvalue()})
		except ZeroDivisionError as _:
			if not self.ignore_smoothing_error:
				raise ZeroDivisionError("Bleu smoothing divided by zero. This is a known bug of corpus_bleu, \
				usually caused when there is only one sample and the sample length is 1.")
			result.update({"bleu": \
					0, \
					"bleu hashvalue": self._hashvalue()})
		return result
