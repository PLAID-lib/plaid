Core concepts
=============

PLAID is both a datamodel and a library implementing it. This page introduces the main concepts you’ll encounter when using PLAID to build datasets and learning problems.

This page, previously referred to as Description, provides elements on PLAID functionalities and a high-level overview of the entities manipulated by PLAID. It introduces {py:class}`~plaid.containers.dataset.Dataset`, {py:class}`~plaid.containers.sample.Sample`, features and the learning {py:class}`~plaid.problem_definition.ProblemDefinition`, as well as how features are addressed using {doc}`core_concepts/feature_identifiers`.

PLAID is the name of a datamodel, described in this section, and also the name of the present library, which is an implementation of this datamodel, and the format when the data is saved on disk.

PLAID aims to formalize, as generally as possible, a set of physics problems configurations, named dataset, and a learning problem defined on these configurations, named problem_definition. The provided implementation allows to populate the datamodel in memory, and read from / write to disk in a corresponding format using provided io routines.

For the on-disk layout and a guided tour of the entities ({py:class}`~plaid.containers.dataset.Dataset`, {py:class}`~plaid.containers.sample.Sample`, features, {py:class}`~plaid.problem_definition.ProblemDefinition`), see {doc}`core_concepts`.

The datamodel heavily relies on CGNS, see `Seven keys for practical understanding and use of CGNS <https://ntrs.nasa.gov/api/citations/20180006202/downloads/20180006202.pdf>`_, where a very large number of possible physics configurations have already been formalized and standardized (like multiblock configuration, time evolution, etc...). The format is human-readable: the ``.yaml`` and ``.csv`` files can be opened with any text editor, and the physics configurations contained in ``.cgns`` files can be explored using `paraview <https://www.paraview.org/>`_, for instance.

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
