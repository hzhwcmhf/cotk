'''Containing NgramFwBwPerplexityMetric'''
from typing import Optional, List, Any, Union, Dict
import logging

from ..dataloader import Tokenizer, SimpleTokenizer
from .metric import MetricBase
from ..models.ngram_language_model import KneserNeyInterpolated

class NgramFwBwPerplexityMetric(MetricBase):
	'''Metric for calculating n gram forward perplexity and backward perplexity.

	Arguments:
	    {MetricBase.DATALOADER_ARGUMENTS}
	    {MetricBase.REFERENCE_TEST_LIST_ARGUMENTS}
	    {MetricBase.NGRAM_ARGUMENTS}
	    {MetricBase.TOKENIZER_ARGUMENTS}
		{MetricBase.GEN_KEY_ARGUMENTS}
		{MetricBase.SAMPLE_ARGUMENTS_IN_NGRAM_PERPLEXITY}
		{MetricBase.SEED_ARGUMENTS}
		{MetricBase.CPU_COUNT_ARGUMENTS}

	Here is an example (to only show the format but not the exact value of results):

		>>> dl = cotk.dataloader.UbuntuCorpus('resources://Ubuntu_small')
		>>> gen_key = "gen"
		>>> metric = cotk.metric.NgramFwBwPerplexityMetric(dl, dl.get_all_batch('test')['session'][0].tolist(), 2, gen_key=gen_key)
		>>> data = {
		...	    gen_key: [[10, 1028, 479, 285, 220, 3], [851, 17, 2451, 3]]
		...	    # gen_key: [["I", "love", "java", "very", "much", "<eos>"], ["python", "is", "excellent", "<eos>"]],
		... }
		>>> metric.forward(data)
		>>> metric.close()
		{'fwppl': 51.44751843841384,
 		 'bwppl': 138.954327895075,
 		 'fwppl hashvalue': '2ea52377084692953f602e4ebad23e8a46e1c4bb527947d29a03c14b426efe67',
 		 'bwppl hashvalue': '2ea52377084692953f602e4ebad23e8a46e1c4bb527947d29a03c14b426efe67'}
	'''

	_name = 'NgramFwBwPerplexityMetric'
	_version = 2

	def __init__(self, dataloader: Union["LanguageProcessing", "Sentence", "Session"], \
			reference_test_list: List[Any], ngram: int = 4, *, \
			tokenizer: Union[None, Tokenizer, str] = None, gen_key: str = "gen", \
			sample: int = 10000, seed: int = 1229, cpu_count: Optional[int] = None):
		super().__init__(self._name, self._version)
		self.dataloader = dataloader
		self.ngram = ngram
		self.reference_test_list = reference_test_list
		self.tokenizer = tokenizer
		self.gen_key = gen_key
		self.hyps: List[Any] = []
		self.cpu_count = cpu_count
		self.sample = sample
		self.seed = seed

	def forward(self, data: Dict[str, Any]):
		'''Processing a batch of data.

		Arguments:
			data (dict): A dict at least contains the following keys:

				{MetricBase.FORWARD_GEN_ARGUMENTS}
		'''
		gen = data[self.gen_key]
		self.hyps.extend(gen)

	def close(self) -> Dict[str, Any]:
		'''Return a dict which contains:

			* **fwppl**: fw ppl value.
			* **bwppl**: bw ppl value.
			* **fwppl hashvalue**: hash value of fw ppl.
			* **bwppl hashvalue**: hash value of bw ppl.
		'''
		res = super().close()

		sample_num = self.sample
		if sample_num > len(self.reference_test_list):
			sample_num = len(self.reference_test_list)
		if sample_num > len(self.hyps):
			sample_num = len(self.hyps)

		origin_refs = self.reference_test_list[:sample_num]
		origin_hyps = self.hyps[:sample_num]

		refs: List[Any]
		hyps: List[Any]
		if self.tokenizer:
			tokenizer: Tokenizer
			if isinstance(self.tokenizer, str):
				tokenizer = SimpleTokenizer(self.tokenizer)
			else:
				tokenizer = self.tokenizer
			if isinstance(origin_refs[0], List):
				ref_sents = [self.dataloader.convert_ids_to_sentence(ids, remove_special=True, trim=True) for ids in origin_refs]
			else:
				ref_sents = origin_refs
			refs = tokenizer.tokenize_sentences(ref_sents)

			hyp_sents = [self.dataloader.convert_ids_to_sentence(ids, remove_special=True, trim=True) for ids in origin_hyps]
			hyps = tokenizer.tokenize_sentences(hyp_sents)
		else:
			refs = [self.dataloader.convert_ids_to_tokens(ids, remove_special=True, trim=True) for ids in origin_refs]
			hyps = [self.dataloader.convert_ids_to_tokens(ids, remove_special=True, trim=True) for ids in origin_hyps]

		left_pad, right_pad = None, None
		unk = self.dataloader.get_special_tokens_mapping().get("unk", None)

		model = KneserNeyInterpolated(self.ngram, \
					left_pad, right_pad, \
					unk, cpu_count=self.cpu_count)
		logging.info("training forward")
		model.fit(refs)
		logging.info("scoring forward")
		fwppl = model.perplexity(hyps)

		model = KneserNeyInterpolated(self.ngram, \
					left_pad, right_pad, \
					unk, cpu_count=self.cpu_count)
		logging.info("training backward")
		model.fit(hyps)
		logging.info("scoring backward")
		bwppl = model.perplexity(refs)

		res.update({"fwppl": fwppl, "bwppl": bwppl})

		self._hash_unordered_list(refs)
		self._hash_ordered_data((self.ngram,))
		res["fwppl hashvalue"] = res["bwppl hashvalue"] = self._hashvalue()
		return res
