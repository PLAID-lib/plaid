.. plaid documentation master file, created by
   sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

    <br>

.. image:: https://plaid-lib.github.io/assets/images/PLAID-large-logo.png
   :align: center
   :width: 300px

+-------------+-----------------------------------------------------------------------------------------------+
| **Testing** | |CI Status| |Docs| |Coverage| |Last Commit|                                                   |
+-------------+-----------------------------------------------------------------------------------------------+
| **Package** | |Conda Version| |Conda Downloads| |PyPI Version| |PyPi Downloads| |Platform| |Python Version| |
+-------------+-----------------------------------------------------------------------------------------------+
| **Meta**    | |License| |GitHub Stars| |JOSS paper|                                                         |
+-------------+-----------------------------------------------------------------------------------------------+


.. |CI Status| image:: https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml/badge.svg
   :target: https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml

.. |Docs| image:: https://readthedocs.org/projects/plaid-lib/badge/?version=latest
   :target: https://plaid-lib.readthedocs.io/en/latest/?badge=latest

.. |Coverage| image:: https://codecov.io/gh/plaid-lib/plaid/branch/main/graph/badge.svg
   :target: https://app.codecov.io/gh/plaid-lib/plaid/tree/main?search=&displayType=list

.. |Last Commit| image:: https://img.shields.io/github/last-commit/PLAID-lib/plaid/main
   :target: https://github.com/PLAID-lib/plaid/commits/main

.. |PyPI Version| image:: https://img.shields.io/pypi/v/pyplaid.svg
   :target: https://pypi.org/project/pyplaid/

.. |Conda Version| image:: https://anaconda.org/conda-forge/plaid/badges/version.svg
   :target: https://anaconda.org/conda-forge/plaid

.. |Platform| image:: https://img.shields.io/badge/platform-any-blue
   :target: https://github.com/PLAID-lib/plaid

.. |Python Version| image:: https://img.shields.io/pypi/pyversions/pyplaid
   :target: https://github.com/PLAID-lib/plaid

.. |Conda Downloads| image:: https://img.shields.io/conda/dn/conda-forge/plaid.svg?label=Conda%20downloads
   :target: https://anaconda.org/conda-forge/plaid

.. |PyPi Downloads| image:: https://static.pepy.tech/badge/pyplaid
   :target: https://pepy.tech/projects/pyplaid

.. |License| image:: https://anaconda.org/conda-forge/plaid/badges/license.svg
   :target: https://github.com/PLAID-lib/plaid/blob/main/LICENSE.txt

.. |GitHub Stars| image:: https://img.shields.io/github/stars/PLAID-lib/plaid?style=social
   :target: https://github.com/PLAID-lib/plaid

.. |JOSS paper| image:: https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e/status.svg
   :target: https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e

PLAID (Physics Learning AI Datamodel) is a library proposing an implementation for a datamodel tailored for AI and ML learning of physics problems.
It has been developped at SafranTech, the research center of `Safran group <https://www.safran-group.com/>`_

The code is hosted on `GitHub <https://github.com/PLAID-lib/plaid>`_

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Overview

   source/quickstart.md
   source/citation.md
   source/contributing.md

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: User guide

   source/core_concepts.rst
   Data Conversion Guide <source/notebooks/convert_users_data_example>
   Examples & Tutorials <source/examples_tutorials.rst>
   API Reference <autoapi/plaid/index>

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Ecosystem & Benchmarks

   CGNS Standard <https://cgns.org/>
   Muscat library <https://muscat.readthedocs.io/>
   PLAID Benchmarks <source/plaid_benchmarks.rst>

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
