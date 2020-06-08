.. _resources_reference:

Resources
===================================


File ID
-----------------------------------

Classes like :class:`.dataloder.Dataloader` usually need ``file_id`` to locate
resources. The string will be passed to :meth:`.file_utils.file_utils.get_resource_file_path` or
:meth:`.file_utils.file_utils.import_local_resources`.

The format  of ``file_id`` is ``name[@source][#processor]``,
where ``source`` and ``processor`` are optional.

* ``name`` can be:

    * A string start with "resources://", indicating predefined resources.
    * A string start with "https://" indicating online resources.
    * A string indicating local path, absolute or relative to cwd.

* ``source`` only works when ``name`` indicates a predefined resources,
  where ``source`` means where to download the dataset.
  See the following sections of a specific resource for all availiable ``source``.
  If not specified, a default source will be chosen
  (the first one in the list showing source).

* ``preprocessor`` is necessary when ``name`` is not a predefined resources.
  It has to be one of the subclass of :class:`.file_utils.file_utils.ResourceProcessor`.

Examples:

============================================================================  =======  ===============  ===================================
file_id                                                                       name     source           processor  
============================================================================  =======  ===============  ===================================
resources://MSCOCO                                                            MSCOCO   default(amazon)  Default(MSCOCOResourceProcessor)
resources://MSCOCO@tsinghua                                                   MSCOCO   tsinghua         Default(MSCOCOResourceProcessor)
https://cotk-data.s3-ap-northeast-1.amazonaws.com/mscoco.zip#MSCOCO           MSCOCO   None             MSCOCOResourceProcessor
./mscoco.zip#MSCOCO                                                           MSCOCO   None             MSCOCOResourceProcessor
============================================================================  =======  ===============  ===================================

Download / Import Resources Manually
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to use CoTK resources in an offline environment,
you can download resources and import them to cotk manually.

* The download urls are placed under ``./cotk/resource_config``, see :ref:`Resource Configs<resources_desc>`.

* The following command can import a local file of resources into cache:

.. code-block:: none

    cotk import <file_id> <file_path>

where ``file_id`` should start with ``resources://``, and ``file_path`` is the path to the local resource. For example:

.. code-block:: none

    cotk import resources://MSCOCO ./MSCOCO.zip

More :ref:`CLI commands here<cli_usage>`.


Predefined Resources


Word Vector
----------------------------------

Glove50d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove50d``
    * processor: :class:`._utils.resource_processor.Glove50dResourceProcessor`
    * source: ``stanford``
    * used by: :class:`.wordvector.Glove`
    
    Introduction
        This is a file containing vector representation for words pretrained by `GloVe`, which is an unsupervised learning
        algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word
        co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures
        of the word vector space.
        
        This file is a 50 dimension version from ``glove.6B``, which is trained on `Wikipedia2014` and `Gigaword5`.

    Statistic
        ===================== ==============
        Property
        ===================== ==============
        Dimension             50
        Vocabulary Size       400,000
        ===================== ==============

    Reference
        [1] GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/projects/glove/

        [2] Pennington J, Socher R, Manning C. `Glove: Global vectors for word representation <https://www.aclweb.org/anthology/D14-1162>`_
        //Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 2014: 1532-1543.

Glove100d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove100d``
    * processor: :class:`._utils.resource_processor.Glove100dResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.wordvector.Glove`

    Introduction
        * This is a file containing vector representation for words pretrained by `GloVe`.
        * This file is a 100 dimension version from ``glove.6B``.
        * Refer to `Glove50d`_ for more information and references.

    Statistic
        ===================== ==============
        Property
        ===================== ==============
        Dimension             100
        Vocabulary Size       400,000
        ===================== ==============

Glove200d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove200d``
    * processor: :class:`._utils.resource_processor.Glove200dResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.wordvector.Glove`

    Introduction
        * This is a file containing vector representation for words pretrained by `GloVe`.
        * This file is a 200 dimension version from ``glove.6B``.
        * Refer to `Glove50d`_ for more information and references.

    Statistic
        ===================== ==============
        Property
        ===================== ==============
        Dimension             200
        Vocabulary Size       400,000
        ===================== ==============

Glove300d
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove300d``
    * processor: :class:`._utils.resource_processor.Glove300dResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.wordvector.Glove`
    
    Introduction
        * This is a file containing vector representation for words pretrained by `GloVe`.
        * This file is a 300 dimension version from ``glove.6B``.
        * Refer to `Glove50d`_ for more information and references.

    Statistic
        ===================== ==============
        Property
        ===================== ==============
        Dimension             300
        Vocabulary Size       40,000
        ===================== ==============

Glove50d_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Glove50d_small``
    * processor: :class:`._utils.resource_processor.Glove50dResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.wordvector.Glove`

    Introduction
        * This is a file containing vector representation for words pretrained by `GloVe`.
        * This file is a 50 dimension version from ``glove.6B`` and only contains the most frequency 40,000 words.
        * Refer to `Glove50d`_ for more information and references.

    Statistic
        ===================== ==============
        Property
        ===================== ==============
        Dimension             50
        Vocabulary Size       40,000
        ===================== ==============


Datasets
----------------------------------

MSCOCO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``MSCOCO``
    * processor: :class:`._utils.resource_processor.MSCOCOResourceProcessor`
    * source: ``amazon``, ``tsinghua``
    * usage: :class:`.dataloader.MSCOCO`
    
    Introduction
        MSCOCO is a new dataset gathering images of complex everyday scenes containing common objects
        in their natural context. We neglect the images and just employ the corresponding caption.

        The original data is `2017 Train/Val annotations [241MB] <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>`_ .
        We use the same train set as original data, but split the val set into dev(odd-numbered sentences) and test set(even-numbered sentences).
        We extract the caption and use ``nltk.tokenize.word_tokenize`` for tokenization.
        We also capitalize each sentence and add full stop to it if it does not have one.

    Statistic
        ======================================  =======  ======  ======
        Property                                Train    Dev     Test  
        ======================================  =======  ======  ======
        Numbers of Sentences                    591,753  12,507  12,507
        Minimum length of Sentences             8        10      10    
        Maximum length of Sentences             50       48      50    
        Average length of Sentences             13.55    13.55   12.52 
        Std Deviation of Length of Sentences    2.51     2.44    2.44  
        Vocabulary Size                         33,241   5,950   5,869
        Frequency Vocabulary Size (times>=10)   8,064    1,050   1,034
        ======================================  =======  ======  ======

    Reference
        [1] COCO: Common Objects in Context. http://cocodataset.org

        [2] Chen X, Fang H, Lin T Y, et al. `Microsoft COCO Captions: Data Collection and Evaluation Server <https://arxiv.org/pdf/1504.00325.pdf>`_.
        arXiv:1504.00325, 2015.
        

MSCOCO_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``MSCOCO_small``
    * processor: :class:`._utils.resource_processor.MSCOCOResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.MSCOCO`
    
    Introduction
        * The data is random uniformed sampled from `MSCOCO`_.
        * Refer to `MSCOCO`_ for more information and references.

    Statistic
        ======================================  =========  =========  =========
        Property                                Train      Dev        Test 
        ======================================  =========  =========  =========
        Numbers of Sentences                    59,175     1,250      1,250
        Vocabulary Size                         12,212     1,877      1,841
        Frequency Vocabulary Size (times>=10)   2,588      219        219
        ======================================  =========  =========  =========


OpenSubtitles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``OpenSubtitles``
    * processor: :class:`._utils.resource_processor.OpenSubtitlesResourceProcessor`
    * source: ``amazon``, ``tsinghua``
    * usage: :class:`.dataloader.OpenSubtitles`



    Introduction
        Opensubtitle dataset is collected from movie subtitles.
        To construct this dataset for single-turn dialogue generation,
        we follow [3] and regard a pair of adjacent sentences as one dialogue turn.
        We set the former sentence as a post and the latter one as the corresponding response.
        We remove the pairs whose lengths are extremely long or short,
        and randomly sample 1,144,949/20,000/10,000 pairs as our train/dev/test set.
        The dataset is tokenized and all the tokens are in lower-case.

    Statistic
        ======================================  =========  =========  =========
        Property                                Train      Dev        Test 
        ======================================  =========  =========  =========
        Numbers of Post/Response Pairs          1,144,949  20,000     10,000
        Average Length (Post/Response)          9.08/9.10  9.06/9.13  9.04/9.05
        Vocabulary Size                         104,683    18,653     12,769
        Frequency Vocabulary Size (times>=10)   31,039     2,030      1,241
        ======================================  =========  =========  =========

    Reference
        [1] OpenSubtitles. http://opus.nlpl.eu/OpenSubtitles.php

        [2] J. Tiedemann, 2016, `Finding Alternative Translations in a Large Corpus of Movie Subtitles <http://www.lrec-conf.org/proceedings/lrec2016/pdf/62_Paper.pdf>`_.
        In Proceedings of the 10th International Conference on Language Resources and Evaluation (LREC 2016)
		
        [3] Vinyals O, Le Q. `A Neural Conversational Model <https://arxiv.org/pdf/1506.05869.pdf>`_. arXiv preprint arXiv:1506.05869, 2015.

OpenSubtitles_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``OpenSubtitles_small``
    * processor: :class:`._utils.resource_processor.OpenSubtitlesResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.OpenSubtitles`

    Introduction
        * The data is random uniformed sampled from `OpenSubtitles`_.
        * Refer to `OpenSubtitles`_ for more information and references.

    Statistic
        ======================================  =========  =========  =========
        Property                                Train      Dev        Test 
        ======================================  =========  =========  =========
        Numbers of Post/Response Pairs          11,449     2,000      1,000
        Vocabulary Size                         13,984     4,661      3,079
        Frequency Vocabulary Size (times>=10)   1,333      367        211
        ======================================  =========  =========  =========



SST
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``SST``
    * processor: :class:`._utils.resource_processor.SSTResourceProcessor`
    * source: ``stanford``
    * usage: :class:`.dataloader.SST`
    
    Introduction
        Stanford Sentiment Treebank is the first corpus with fully labeled
        parse trees that allows for a complete analysis of the compositional
        effects of sentiment in language.

        We remain the original split of dataset. The dataset is tokenized and
        contains capitalized letters.
    
    Statistic
        =====================================  =========  =========  =========
        Property                               Train      Dev        Test 
        =====================================  =========  =========  =========
        Numbers of Sentences                   8,544      1,101      2,210
        Average Length                         19.14      19.32      19.19
        Vocabulary Size                        16,586     5,043      7,934
        Frequency Vocabulary Size (times>=10)  1,685      232        441
        =====================================  =========  =========  =========
    
    Reference
        [1] Recursive Deep Model for Sematic Compositionality over a Sentiment Treebank. https://nlp.stanford.edu/sentiment/

        [2] Socher R, Perelygin A, Wu J, et al. 
        `Recursive deep models for semantic compositionality over a sentiment treebank <https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf>`_
        //Proceedings of the 2013 conference on empirical methods in natural language processing. 2013: 1631-1642.

SwitchboardCorpus
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``SwitchboardCorpus``
    * processor: :class:`._utils.resource_processor.SwitchboardCorpusResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.SwitchboardCorpus`
    
    Introduction
        Switchboard is a collection of about 2,400 two-sided telephone conversations among
        543 speakers (302 male, 241 female) from all areas of the United States.
        A computer-driven robot operator system handled the calls, giving the caller
        appropriate recorded prompts, selecting and dialing another person (the callee) to
        take part in a conversation, introducing a topic for discussion and recording the
        speech from the two subjects into separate channels until the conversation was finished.
        About 70 topics were provided, of which about 50 were used frequently. Selection of topics
        and callees was constrained so that: (1) no two speakers would converse together more than
        once and (2) no one spoke more than once on a given topic.

        We introduce the data processed by Zhao, Ran and Eskenazi[4], which construct the set
        ``multi_ref`` for one-to-many dialog evaluation (One context, multiple responses).
        ``multi_ref`` was constructed by extracting multiple responses for single context
        with retrieval method and annotation on the other test set. For the details, please refer
        to [4].
        
        We used their training set, dev set and test set. That is, capitalization, 
        tokenization, splits of dataset and any other aspects of our data are the
        same as theirs (in their version, utterances are lowercase and are not tokenized).

        However, there are two differences between our data with theirs:

        * We ensure that any two consecutive utterances come from different speakers,
          by concatenating the original consecutive utterances from the same speakers
          during pre-processing of :class:`.dataloader.SwitchboardCorpus`.
          (This is because we want to be compatible with other multi-turn dialog set.)
        * To avoid the gap between training and test, we have to remove some samples
          from ``multi_ref``, where the target speaker is the same as the last one in the context.

    Statistic
        =========================================  ======  =====  =====
        Property                                   Train   Dev    Test 
        =========================================  ======  =====  =====
        Numbers of Sessions                        2,316   60     62   
        Minimum Number of Turns (per session)      3       19     8    
        Maximum Number of Turns (per session)      190     144    148  
        Average Number of Turns (per session)      59.47   58.92  58.95
        Std Deviation of Number of Turns           27.50   26.91  32.43
        Minimum Length of Sentences                3       3      3    
        Maximum Length of Sentences                401     185    333  
        Average Length of Sentences                19.03   19.12  20.15
        Std Deviation of Number of Sentences       20.25   19.65  21.59
        Vocabulary Size                            24,719  7,704  6,203
        Frequency Vocabulary Size (times>=10)      9,508   1,021  1,095
        =========================================  ======  =====  =====

    Reference
        [1] Switchboard-1 release 2. https://catalog.ldc.upenn.edu/LDC97S62

        [2] John J G and Edward H. `Switchboard-1 release 2 <https://catalog.ldc.upenn.edu/LDC97S62>`_. Linguistic Data Consortium, Philadelphia 1997.

        [3] NeuralDialog-CVAE. https://github.com/snakeztc/NeuralDialog-CVAE

        [4] Zhao, Tiancheng and Zhao, Ran and Eskenazi, Maxine. Learning Discourse-level Diversity for Neural Dialog Models using Conditional Variational Autoencoders. ACL 2017.

SwitchboardCorpus_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``SwitchboardCorpus_small``
    * processor: :class:`._utils.resource_processor.SwitchboardCorpusResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.SwitchboardCorpus`
    
    Introduction
        * The data is random uniformed sampled from `SwitchboardCorpus`_.
        * Refer to `SwitchboardCorpus`_ for details and references.
    
    Statistic
        ======================================  =========  =========  =========
        Property                                Train      Dev        Test 
        ======================================  =========  =========  =========
        Numbers of Sessions                     463        12         12
        Vocabulary Size                         13,235     5,189      4,729
        Frequency Vocabulary Size (times>=10)   3,923      341        317
        ======================================  =========  =========  =========

Ubuntu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Ubuntu``
    * processor: :class:`._utils.resource_processor.UbuntuResourceProcessor`
    * source: ``amazon``, ``tsinghua``
    * usage: :class:`.dataloader.UbuntuCorpus`
    
    Introduction
        Ubuntu Dialogue Corpus 2.0 is a dataset containing a mass of multi-turn dialogues.
        The dataset has both the multi-turn property of conversations in the Dialog State Tracking Challenge datasets,
        and the unstructured nature of interactions from microblog services such as Twitter.

        We build the dataset using the ubuntu-ranking-dataset-creator[1], without tokenization, lemmatization or stemming. 
		The dataset contains capital character. The positive example probability is set to 1.0 for training set. 
		The examples for training/dev/test sets are set to 1,000,000/19,560/18,920 as default.
		Other settings are default.

    Statistic
        =====================================  ============  ============  =======
        Property                               Train         Dev           Test
        =====================================  ============  ============  =======
        Numbers of Sessions                    1,000,000     19,560        18,920
        Minimum Number of Turns (per session)  3             3             3     
        Maximum Number of Turns (per session)  19            19            19    
        Average Number of Turns (per session)  4.95          4.79          4.85  
        Std Deviation of Number of Turns	   2.97          2.79          2.85  
        Minimum Length of Sentences            2             2             2     
        Maximum Length of Sentences            977           343           817   
        Average Length of Sentences            17.98         19.40         19.61 
        Std Deviation of Number of Sentences   16.26         17.25         17.94 
        Vocabulary Size                        576,693       47,288        47,847
        Frequency Vocabulary Size (times>=10)  23,498        2,568         2,565
        =====================================  ============  ============  =======
    
    Reference
        [1] Ubuntu Dialogue Corpus v2.0. https://github.com/rkadlec/ubuntu-ranking-dataset-creator

        [2] R. Lowe, N. Pow, I. Serban, and J. Pineau.
        `The ubuntu dialogue corpus: A large dataset for research in unstructured multi-turn dialogue systems <https://arxiv.org/pdf/1506.08909.pdf>`_.
        In Special Interest Group on Discourse and Dialogue (SIGDIAL), 2015a.

Ubuntu_small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    * name: ``Ubuntu_small``
    * processor: :class:`._utils.resource_processor.UbuntuResourceProcessor`
    * source: ``amazon``
    * usage: :class:`.dataloader.UbuntuCorpus`
    
    Introduction
        * The data is random uniformed sampled from `Ubuntu`_.
        * Refer to `Ubuntu`_ for details and references.

    Statistic
        ======================================  =========  =========  =========
        Property                                Train      Dev        Test 
        ======================================  =========  =========  =========
        Numbers of Sessions                     10,001     1,957      1,893
        Vocabulary Size                         32,941     12,348     12,090
        Frequency Vocabulary Size (times>=10)   1,590      501        475
        ======================================  =========  =========  =========