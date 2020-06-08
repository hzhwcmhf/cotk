.. _cli_usage:

CLI Usage
============================================

``cotk`` provides command line interface.


.. code-block:: none

    Usage: cotk <command> [...]

    command list:
    import       Import local files to cotk cache.
    config       Set variables at the config file (at ~/.cotk_config).

Import
--------------------------------------------

The following command can import a local file of resources into cache:

.. code-block:: python

    cotk import <file_id> <file_path>

where ``file_id`` should start with ``resources://``, and ``file_path`` is the path to the local resource. For example:

.. code-block:: python

    cotk import resources://MSCOCO ./MSCOCO.zip


Config
--------------------------------------------

It can set global variables of the cotk. (Not used yet.)
An variable should be a string without space.

Set Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The variable will be set according to the value.

.. code-block:: none

    cotk set <variable> <value>

Show Variable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The variable will be shown

.. code-block:: none

    cotk show <variable>
