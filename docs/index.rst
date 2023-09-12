.. plaid documentation master file, created by
   sphinx-quickstart
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

PLAID documentation
=================

This library proposes an implementation for a datamodel tailored for AI and ML learning of physics problems.
It has been developped at SafranTech, the research center of `Safran group <https://www.safran-group.com/>`_

The code is hosted on `Safran Gitlab <https://gitlab.com/drti/plaid>`_

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Getting Started

   source/getting_started.md
   source/tutorial.md
   .. source/notebooks.rst

.. _Advanced:

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Advanced

   .. source/impl_details.md

   source/contributing.md
   .. source/diagrams.rst

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: API

   plaid.Dataset <autoapi/plaid/containers/dataset/index>
   plaid.Sample <autoapi/plaid/containers/sample/index>
   autoapi/plaid/index
..    autoapi/examples/index
..    autoapi/tests/index

.. `Coverage Report <coverage/index.html>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
