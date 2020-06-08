Extending Cotk: More Data, More Metrics!
============================================

We hope ``cotk`` can be adapted to more datasets
and more tasks. Therefore, we have a repository (TO BE ONLINE)
to collect any contribution to cotk (**regardless of its quality**).
Our maintainer will choose modules with high-quality
and merge them to the main repository.

Add A New Dataset
----------------------------------------------

For local use
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Now you have a new dataset and want to load it with ``cotk`` and
the task is similar with the existed tasks, like :class:`.dataloader.LanguageGeneration`,
:class:`.dataloader.SingleTurnDialog` or :class:`.dataloader.MultiTurnDialog`.
The simplest way is to just adapt the format of your dataset.

For example, if you want to adapt your data to :class:`.dataloader.LanguageGeneration`,
you just have to generate 3 text files named ``train.txt``,
``dev.txt``, ``test.txt``.
Each file contains several lines and each line is a sentence.
The structure of your directory should be like:

.. code-block:: none

    mydata
    ├── train.txt
    ├── dev.txt
    └── test.txt

Then you can load your data using :class:`.dataloader.LanguageGeneration` with a local path.

.. code-block:: python

    dataloader = LanguageGeneration("./path/to/mydata", min_vocab_times, max_sent_length, invalid_vocab_times, \
			tokenizer, remains_capital)

.. note ::

    If you want to write your own ``dataloader`` with a complex data format,
    see `Add A New Task`_.

Download Dataset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you want to publish your dataset and make them can be
downloaded automatically. You can zip your data and upload to an server.
The url path should be accessible for every one.

Using :class:`.dataloader.LanguageGeneration` with a url path is adequate for
the requirements.

.. code-block:: python

    dataloader = MSCOCO("http://url/to/new_data.zip")

.. note ::

    The zip file is downloaded then processed by
    :class:`._utils.resource_processor.DefaultResourcesProcessor`.
    For more about ``ResourcesProcessor``, refer to :ref:`this <resources_reference>`.

Use a resources name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _resources_desc:

This time, you have to change some codes in ``cotk``.
First you have to understand the usage of resources at
:ref:`here <resources_reference>`. Then see the json file under
``/cotk/resource_config/``, you will find each predefined resource
corresponds to a json file, like

.. code-block:: javascript

    {
        "name": "MSCOCO_small",
        "type": "MSCOCO",
        "hashtag": "68f6b8d764bff6f5440a63f87aeea97049a1d2e89942a7e524b7dabd475ffd79",
        "link": {
            "default":"https://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco_small.zip",
            "amazon": "https://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco_small.zip"
        }
    }

There are some places you have to pay attention:

    * ``type`` is the prefix of its ``ResourcesProcessor``.
    * ``link.default`` is necessary when no source is specified.
    * ``hashtag`` is required for checksum.

We use the following codes to hash the zip file.

.. code-block :: python

    def _get_file_sha256(file_path):
        '''Get sha256 of given file'''
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as fin:
            for chunk in iter(lambda: fin.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

After accomplishment the config file, make a **Pull Request** and
wait for update! And soon you can use the following code to load the data.

.. code-block:: python

    dataloader = MSCOCO("resouces://new_name")

Add A New Task
----------------------------------------------

Sometimes you want to deal with a totally different task
from the predefined ones.
In that case, you have to implement a subclass of :class:`.LanguageProcessingBase`,
and some necessary function is necessary for your dataloader. You can click
on the following links for its input and outputs.

* :class:`.LanguageProcessingBase`
* :meth:`.LanguageProcessingBase._general_load_data`
* :meth:`.LanguageProcessingBase._load_data`
* :meth:`.LanguageProcessingBase.get_batch`
* (Optional) some function like ``get_metric()`` to define the standard metric.

For example, we implement a new dataloader for sentence classification.

.. code-block:: python

    class SentenceClassification(LanguageProcessingBase):

        def __init__(self, file_id, invalid_vocab_times, min_vocab_times, max_sent_length):
            self._file_id = file_id
            self._file_path = get_resource_file_path(file_id)
            self._invalid_vocab_times = invalid_vocab_times
            self._min_vocab_times = min_vocab_times
            self._max_sent_length = max_sent_length
            super().__init__()

        def _load_data(self):
            r'''Loading dataset, invoked by `LanguageProcessingBase.__init__`
            '''
            return super()._general_load_data(self._file_path, [['sent', 'Sentence'], ['label', 'Label']], \
			    self._min_vocab_times, self._max_sent_length, None, self._invalid_vocab_times)

        def get_batch(self, key, index):
            '''Get a batch of specified `index`.'''
            res = {"sent": [], "label": []}

            # use the "self.data" you have stored
            for i in index:
                res["sent"].append(self.data[key]['sent'][i])
                res["label"].append(self.data[key]['label'][i])

            # the return value is exactly what you will get when ``get_batches`` is called
            # may be you want to do padding before return
            return res

Add A New Metric
---------------------------------------------

If you have a new way to evaluate the model, you should construct a
metric class inheriting the :class:`.metric.MetricBase`.

Here are some necessary functions you must implement. You can click on
the link to find more details.

* ``__init__()``
* :meth:`.MetricBase.forward`
* :meth:`.MetricBase.close`

Here we give an example for calculating the average length of generated
sentences.

.. code-block:: python

    class AverageLengthMetric(MetricBase):
        def __init__(self, dataloader, gen_key="gen"):
            super().__init__()
            self.dataloader = dataloader
            self.gen_key = gen_key
            self.token_num = 0
            self.sent_num = 0
        
        def forward(self, data):
            gen = data[gen_key]
            for sent in gen:
                self.token_num += len(self.dataloader.trim_index(sent))
                self.sent_num += 1
        
        def close(self):
            return {"len_avg": self.token_num / self.sent_num}

There is some regulations to design an metric.

* Using :ref:`allvocabs <vocabulary_ref>` for reference.
* Dealing with ``<unk>``, which should be regarded as error or
  using some methods to do smoothing. Pay atention the difference
  between ``<unk>`` and
  :ref:`unknown vocabularies <vocabulary_ref>`.
* Record hash value. Hash value equal if and only if the metric is tested
  under the same settings. (In the example, there is no hash value
  because we don't have input and the setting is always the same)

