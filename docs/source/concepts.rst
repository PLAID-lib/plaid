Concepts
========

PLAID is a data model and library for organizing samples of heterogeneous physics data and defining
learning problems on physics datasets.

The current public API is centered on :py:class:`~plaid.containers.sample.Sample`,
:py:class:`~plaid.problem_definition.ProblemDefinition`, and the storage helpers
from :mod:`plaid.storage`.  Datasets are loaded and saved through backend-specific
collections plus converters.

PLAID relies on the CGNS standard for representing complex physics meshes and
uses human-readable metadata files such as YAML for dataset schemas, problem
definitions, and dataset information.

For practical examples, see the :doc:`examples_tutorials` pages.

.. toctree::
   :glob:
   :maxdepth: 1

   core_concepts/sample
   core_concepts/dataset
   core_concepts/problem_definition
   core_concepts/defaults
   core_concepts/disk_format
   core_concepts/interoperability
   core_concepts/viewer
