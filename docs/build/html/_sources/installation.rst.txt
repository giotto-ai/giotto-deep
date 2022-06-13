############
Installation
############

.. _installation:

************
Dependencies
************

The latest stable version of ``giotto-deep`` requires:

- python (>= 3.7)
- tensorboard (=> 2.9.1)
- torch (>= 1.11.0)
- torchdata (>= 0.3.0)
- torchtext (>) 0.12.0)
- torchvision (>= 0.12.0)

For all the required dependencies, please refer to the ``requirements.txt`` file.

To run the examples, ``jupyter`` is required.

To run the test, both ``pytest`` and ``mypy`` are required.


*****************
User installation
*****************

The simplest way to install ``giotto-deep`` is using ``pip``   ::

    python -m pip install -U giotto-deep

If necessary, this command will also automatically install all the above dependencies. 
Note: we recommend upgrading ``pip`` to a recent version as the above may fail on very old versions.

**********************
Developer installation
**********************

.. _dev_installation:

Installing both the PyPI release and source of ``giotto-deep`` in the same environment is not recommended.

Source code
===========

You can obtain the latest state of the source code with the command::

    git clone https://github.com/giotto-ai/giotto-deep.git


To install:
===========

.. code-block:: bash

   cd giotto-deep
   python -m pip install -e .

This way, you can pull the library's latest changes and make them immediately available on your machine.
Note: we recommend upgrading ``pip`` and ``setuptools`` to recent versions before installing in this way.

Testing
=======

After installation, you can launch the test suite from inside the
``/gdeep`` directory::

    pytest .
    
You can also run the typing checks from ``/gdeep`` using::

   mypy --ignore-missing-imports .

You can also run the bash script for local tests (from the root folder!) like this::

   bash local_test.bh



