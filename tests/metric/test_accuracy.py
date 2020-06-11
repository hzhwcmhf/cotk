import random
import numpy as np
import json
import pytest
from metric_base import *
from cotk.metric import AccuracyMetric
from cotk.dataloader import UbuntuCorpus, LanguageProcessing

def setup_module():
	random.seed(0)
	np.random.seed(0)

class TestAccuracyMetric:
	def test_init(self):
		dl = None
		label_key = "label"
		prediction_key = "prediction"
		metric = AccuracyMetric(dl, label_key, prediction_key)
		assert isinstance(metric, AccuracyMetric)
		#assert isinstance(metric.dataloader, LanguageProcessing)
		assert isinstance(metric.label_key, str)
		assert isinstance(metric.prediction_key, str)

	def test_close(self):
		dl = None
		label_key = "label"
		prediction_key = "prediction"

		metric = AccuracyMetric(dl, label_key, prediction_key)
		data = {label_key:[1, 2, 1, 2], prediction_key: [1, 2, 2, 1]}
		metric.forward(data)
		assert len(metric.refs) == len(data[prediction_key])
		assert len(metric.hyps) == len(data[label_key])
		res = metric.close()
		assert isinstance(res, dict)
		assert res['accuracy'] == 0.5
		assert res['accuracy hashvalue'][-1] == 'a'
		metric = AccuracyMetric(dl, label_key, prediction_key)
		data = {label_key:[1, 2, 2, 1], prediction_key: [1, 2, 1, 2]}
		metric.forward(data)
		res2 = metric.close()
		assert(res2 == res)
		metric = AccuracyMetric(dl, label_key, prediction_key)
		data = {label_key:[1, 2, 2, 1], prediction_key: [1, 2, 1]}
		with pytest.raises(Exception):
			metric.forward(data)
		metric = AccuracyMetric(dl, label_key, prediction_key)
		with pytest.raises(Exception):
			metric.close()
		


	def test_version(self):
		version_test(AccuracyMetric, dataloader=None)
