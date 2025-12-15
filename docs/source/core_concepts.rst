Core concepts
=============

PLAID is a datamodel and library for organizing physics datasets and defining learning problems on these datasets.

It provides high-level classes such as :py:class:`~plaid.containers.dataset.Dataset`, :py:class:`~plaid.containers.sample.Sample`, and :py:class:`~plaid.problem_definition.ProblemDefinition`, with features addressed via :doc:`core_concepts/feature_identifiers`.

PLAID relies on the CGNS standard for representing complex physics meshes and uses human-readable formats like `.yaml`, `.csv` for other features.

For more details and examples, see the :doc:`core_concepts` and :doc:`examples_tutorials` pages.

.. toctree::
   :glob:
   :maxdepth: 1

   core_concepts/sample
   core_concepts/dataset
   core_concepts/problem_definition
   core_concepts/feature_identifiers
   core_concepts/defaults
   core_concepts/disk_format
   core_concepts/interoperability
