r""" A implemention of KneserNey Interpolated Language Model.
"""
from functools import partial
from itertools import chain
from collections import Counter
from multiprocessing import Pool
import tqdm

from nltk.lm.preprocessing import pad_both_ends
from nltk.util import everygrams
import numpy as np

from .._utils import get_cpu_count


class KneserNeyInterpolated:
	r'''Language model with modified Kneser-Ney smoothing.

	Arguments:
	    order (int): The order of language model
	    left_pad_symbol (str): Symbol prepended to each sentence (e.g., <go>)
	    right_pad_symbol (str): Symbol appended to each sentence (e.g., <eos>)
	    unk_symbol (str): Symbol for unknown words
	    cutoff (int): A word will be replaced with unk_symbol if its count is less than ``cutoff``.
	    default_delta_1 (float): Discount for words with exactly one occurence
	    	Default: ``0.1``.
	    default_delta_2 (float): Discount for words with exactly two occurences
	    	Default: ``0.1``
	    default_delta_3 (float): Discount for words with more than three occurences
	    	Default: ``0.1``

	Warning:
	    Default discounts will be adopted only when failing to estimate optimal discounts from the
	    	training corpus
	'''
	def __init__(self, order, left_pad_symbol, right_pad_symbol, unk_symbol, cutoff=1, \
				 default_delta_1=0.1, default_delta_2=0.1, default_delta_3=0.1, \
				 cpu_count=None):
		self.order = order
		self.left_pad_symbol = left_pad_symbol
		self.right_pad_symbol = right_pad_symbol
		self.unk_symbol = unk_symbol
		self.cutoff = cutoff
		self.default_delta_1 = default_delta_1
		self.default_delta_2 = default_delta_2
		self.default_delta_3 = default_delta_3
		self.word2cnt = {}
		self.n2ngram2cnt = {n: {} for n in range(1, self.order + 1)}
		self.suffix2concnt = {}
		self.midseq2concnt = {}
		self.n2cnt2discount = {n: {} for n in range(1, self.order + 1)}
		self.cpu_count = get_cpu_count(cpu_count)

	@property
	def vocab_size(self):
		r"""the size of vocabulary"""
		res = len(self.n2ngram2cnt[1])
		if (self.unk_symbol, ) not in self.n2ngram2cnt[1]:
			res += 1
		return res

	def _mask_oov(self, ngram):
		r'''Replace infrequent words (with counts less than ``cutoff``) with ``unk_symbol``
		'''
		res = []
		for word in ngram:
			if self.word2cnt.get(word, 0) < self.cutoff and \
					word not in [self.left_pad_symbol, self.right_pad_symbol]:
				res.append(self.unk_symbol)
			else:
				res.append(word)
		return tuple(res)

	def fit(self, corpus):
		r'''Train language model on ``corpus``

		Arguments:
		    corpus (list of list): Each inner list is a sentence (a sequence of words)
		'''
		self.word2cnt = Counter(chain(*corpus))
		self.word2cnt[self.left_pad_symbol] = len(corpus) * (self.order - 1)
		self.word2cnt[self.right_pad_symbol] = len(corpus) * (self.order - 1)
		padding_fn = partial(pad_both_ends, n=self.order, \
							 left_pad_symbol=self.left_pad_symbol, \
							 right_pad_symbol=self.right_pad_symbol)
		train_data = (everygrams(list(padding_fn(sent)), max_len=self.order) for sent in corpus)

		for sent in train_data:
			for ngram in sent:
				ngram = self._mask_oov(ngram)
				order = len(ngram)
				self.n2ngram2cnt[order][ngram] = self.n2ngram2cnt[order].get(ngram, 0) + 1

				if len(ngram) >= 2:
					suffix = ngram[1:]
					midseq = ngram[1:-1]
					if suffix not in self.suffix2concnt:
						self.suffix2concnt[suffix] = set()
					self.suffix2concnt[suffix].add(ngram[0])

					if midseq not in self.midseq2concnt:
						self.midseq2concnt[midseq] = set()
					self.midseq2concnt[midseq].add((ngram[0], ngram[-1]))

		for stat in [self.suffix2concnt, self.midseq2concnt]:
			for key, val in stat.items():
				stat[key] = len(val)

		self._compute_discount()

	def _compute_discount(self):
		r'''Compute discounts for each ngram-level
		'''
		#pylint: disable=invalid-name
		for order in range(1, self.order + 1):
			n = [0] * 4
			for _, cnt in self.n2ngram2cnt[order].items():
				if cnt < 5:
					n[cnt - 1] += 1
			self.n2cnt2discount[order][0] = 0
			try:
				Y = n[0] / (n[0] + 2 * n[1])
				self.n2cnt2discount[order][1] = max(1 - 2 * Y * n[1] / n[0], 0)
				self.n2cnt2discount[order][2] = max(2 - 3 * Y * n[2] / n[1], 0)
				self.n2cnt2discount[order][3] = max(3 - 4 * Y * n[3] / n[2], 0)
			except ZeroDivisionError:
				self.n2cnt2discount[order][1] = self.default_delta_1
				self.n2cnt2discount[order][2] = self.default_delta_2
				self.n2cnt2discount[order][3] = self.default_delta_3

	def _get_discount(self, order, cnt):
		r'''Return discount for the given ngram-level (``order``) and word ``cnt``
		'''
		if cnt >= 3:
			return self.n2cnt2discount[order][3]
		else:
			return self.n2cnt2discount[order][cnt]

	def _word_prob(self, word, context):
		r'''Compute the probability of omitting ``word`` given ``context``
		'''
		ngram = context + (word,)
		if len(ngram) == self.order:
			if self.order == 1:
				return 1 / self.vocab_size
			denominator_stat = self.n2ngram2cnt[self.order]
			numerator_stat = self.n2ngram2cnt[self.order - 1]
		else:
			denominator_stat = self.suffix2concnt
			numerator_stat = self.midseq2concnt
		denominator = denominator_stat.get(ngram, 0)
		numerator = numerator_stat.get(context, 0)
		if numerator == 0:
			return self._word_prob(word, context[1:])
		else:
			alpha = max(0, denominator - self._get_discount(len(ngram), denominator)) / numerator
			gamma = 0
			for unigram in self.n2ngram2cnt[1]:
				cnt = denominator_stat.get(context + unigram, 0)
				discount = self._get_discount(len(ngram), cnt)
				if cnt >= discount:
					gamma += discount
			gamma /= numerator
			if gamma == 0:
				gamma = 1
			return alpha + gamma * (self._word_prob(word, context[1:]) \
									if context else 1 / self.vocab_size)

	def score(self, word, context):
		r'''Compute the probability of omitting ``word`` given ``context``

		Arguments:
		    word (str): The omitted word
		    context (tuple of str): Context of the ``word``
		    	Size: ``order - 1``

		Returns:
		    (float): P(word|context)
		'''
		if self.word2cnt.get(word, 0) < self.cutoff:
			word = self.unk_symbol
		context = self._mask_oov(context)
		ngram = context + (word, )
		if len(ngram) != self.order:
			raise RuntimeError('Provided context should be {}-gram.'.format(self.order - 1))
		return self._word_prob(word, context)

	def sent_log_prob(self, sent):
		r'''Compute the log probability of omitting sentence ``sent``

 		Arguments:
 		    sent (list of str): A sentence

		Returns:
		    (float): \log{P(sent)}
		'''
		log_prob = 0
		sent_now = [self.left_pad_symbol] * self.order + sent + [self.right_pad_symbol]
		for i in range(len(sent) + 1):
			log_prob += np.log(self.score(sent_now[i + self.order], tuple(sent_now[i + 1:i + self.order])))
		return log_prob

	@classmethod
	def _set_language_model(cls, language_model):
		cls.language_model = language_model

	@classmethod
	def _compute_sent_log_prob(cls, sent):
		return cls.language_model.sent_log_prob(sent)

	def perplexity(self, corpus):
		r'''Compute perplexity when generating the given ``corpus``

		Arguments:
		    corpus (list of list): Each inner list is a sentence (a sequence of words)

		Returns:
		    (float): Perplexity when generating ``corpus``
		'''
		log_probs = []
		if len(corpus) > 100 and self.cpu_count > 0:
			pool = Pool(self.cpu_count, initializer=self._set_language_model, initargs=(self,))
			for lp in tqdm.tqdm(pool.imap_unordered(self._compute_sent_log_prob, corpus, chunksize=40), \
								total=len(corpus)):
				log_probs.append(lp)
			pool.close()
			pool.join()
		else:
			if len(corpus) > 100:
				tasks = tqdm.tqdm(corpus, total=len(corpus))
			else:
				tasks = corpus
			for sent in tasks:
				log_probs.append(self.sent_log_prob(sent))
		return np.exp(-sum(log_probs) / sum(map(len, corpus)))
