## Quick Start

Let's skim through the whole package to find what you want. 

### Dataloader

Load common used dataset and do preprocessing:

* Download online resources or import from local path
* Split training set, development set and test set
* Construct vocabulary list

```python
    >>> import cotk.dataloader
    >>> # automatically download online resources
    >>> dataloader = cotk.dataloader.MSCOCO("resources://MSCOCO_small")
    >>> # or download from a url
    >>> dl_url = cotk.dataloader.MSCOCO("http://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco_small.zip#MSCOCO")
    >>> # or import from local file
    >>> dl_zip = cotk.dataloader.MSCOCO("./MSCOCO.zip#MSCOCO")

    >>> print("Dataset is split into:", dataloader.key_name)
    ["train", "dev", "test"]
```

Inspect vocabulary list

```python
    >>> print("Vocabulary size:", dataloader.vocab_size)
    Vocabulary size: 2588
    >>> print("First 10 tokens in vocabulary:", dataloader.vocab_list[:10])
    First 10 tokens in vocabulary: ['<pad>', '<unk>', '<go>', '<eos>', '.', 'a', 'A', 'on', 'of', 'in']
```

Convert between ids and strings

```python
    >>> print("Convert string to ids", \
    ...           dataloader.convert_tokens_to_ids(["<go>", "hello", "world", "<eos>"]))
    Convert string to string [2, 1379, 1897, 3]
    >>> print("Convert ids to string", \
    ...           dataloader.convert_ids_to_tokens([2, 1379, 1897, 3]))
```

Iterate over batches

```python
    >>> for data in dataloader.get_batch("train", batch_size=1):
    ...     print(data)
    {'sent':
        array([[ 2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 1, 1099, 4, 3]]),
        # <go> This is an old photo of people and a <unk> wagon.
     'sent_allvocabs':
        array([[ 2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3]]),
        # <go> This is an old photo of people and a horse-drawn wagon.
     'sent_length': array([14])}
    ......
```

Or using ``while`` if you like

```python
    >>> dataloader.restart("train", batch_size=1):
    >>> while True:
    ...    data = dataloader.get_next_batch("train")
    ...    if data is None: break
    ...    print(data)
```

!! ifndef readme
.. note::

   If you want to know more about ``Dataloader``, please refer to :mod:`docs of dataloader <cotk.dataloader>`.
!! endif

!! ifdef readme
**note**: If you want to know more about ``Dataloader``, please refer to [docs of dataloader](https://thu-coai.github.io/cotk_docs/index.html#model-zoo).
!! endif

### Metrics

We found there are different versions of the same metric in different papers,
which leads to **unfair comparison between models**. For example, whether considering
``unk``, calculating the mean of NLL across sentences or tokens in
``perplexity`` may introduce huge differences.

We provide a unified implementation for metrics, where ``hashvalue`` is provided for
checking whether the same data is used. The metric object receives mini-batches.

```python
    >>> import cotk.metric
    >>> metric = cotk.metric.SelfBleuCorpusMetric(dataloader, gen_key="gen")
    >>> metric.forward({
    ...    "gen":
    ...        [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
    ...         [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    ... })
    >>> print(metric.close())
    {'self-bleu': 0.02206768072402293,
     'self-bleu hashvalue': 'c206893c2272af489147b80df306ee703e71d9eb178f6bb06c73cb935f474452'}
```

!! ifndef readme
You can merge multiple metrics together by :class:`cotk.metric.MetricChain`.
!! endif
!! ifdef readme
You can merge multiple metrics together by cotk.metric.MetricChain.
!! endif


```python
    >>> metric = cotk.metric.MetricChain()
    >>> metric.add_metric(cotk.metric.SelfBleuCorpusMetric(dataloader, gen_key="gen"))
    >>> metric.add_metric(cotk.metric.FwBwBleuCorpusMetric(dataloader, reference_test_list=dataloader.get_all_batch()['sent_allvocabs'], gen_key="gen"))
    >>> metric.forward({
    ...    "gen":
    ...        [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
    ...         [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    ... })
    >>> print(metric.close())
    {'self-bleu': 0.02206768072402293,
     'self-bleu hashvalue': 'c206893c2272af489147b80df306ee703e71d9eb178f6bb06c73cb935f474452',
     'fw-bleu': 0.3831004349785445, 'bw-bleu': 0.025958979254273006, 'fw-bw-bleu': 0.04862323612604027,
     'fw-bw-bleu hashvalue': '530d449a096671d13705e514be13c7ecffafd80deb7519aa7792950a5468549e'}
```

We also provide recommended metrics for selected dataloader.

```python
    >>> metric = dataloader.get_inference_metric(gen_key="gen")
    >>> metric.forward({
    ...    "gen":
    ...        [[2, 181, 13, 26, 145, 177, 8, 22, 12, 5, 3755, 1099, 4, 3],
    ...         [2, 46, 145, 500, 1764, 207, 11, 5, 93, 7, 31, 4, 3]]
    ... })
    >>> print(metric.close())
    {'self-bleu': 0.02206768072402293,
     'self-bleu hashvalue': 'c206893c2272af489147b80df306ee703e71d9eb178f6bb06c73cb935f474452',
     'fw-bleu': 0.3831004349785445, 'bw-bleu': 0.025958979254273006, 'fw-bw-bleu': 0.04862323612604027,
     'fw-bw-bleu hashvalue': '530d449a096671d13705e514be13c7ecffafd80deb7519aa7792950a5468549e',
     'gen': [
         ['<go>', 'This', 'is', 'an', 'old', 'photo', 'of', 'people', 'and', 'a', 'horse-drawn', 'wagon', '.'],
         ['<go>', 'An', 'old', 'stone', 'castle', 'tower', 'with', 'a', 'clock', 'on', 'it', '.']
     ]}
```

!! ifndef readme
.. note::

   If you want to know more about metrics, please refer to :mod:`docs of metrics <cotk.metric>`.
!! endif

!! ifdef readme
**note**: If you want to know more about metrics, please refer to [docs of metrics](https://thu-coai.github.io/cotk_docs/metric.html).
!! endif

### Predefined Models

!! ifndef readme
We have provided some baselines for the classical tasks, see :ref:`Model Zoo <model_zoo>` in docs for details.
!! endif

!! ifdef readme
We have provided some baselines for the classical tasks, see [Model Zoo](https://thu-coai.github.io/cotk_docs/index.html#model-zoo) in docs for details.
!! endif

You can also use ``cotk download thu-coai/MODEL_NAME/master`` to get the codes.
