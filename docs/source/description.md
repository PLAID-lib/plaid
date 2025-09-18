# Description

This page, still under construction, provides elements on PLAID functionalities.

---

- [1. Datamodel](#1-datamodel)
- [2. How to use it ?](#2-how-to-use-it-)

---

## 1. Datamodel

For a high-level overview of the entities manipulated by PLAID, see the Core Concepts page. It introduces `Dataset`, `Sample`, features and the learning `ProblemDefinition`, as well as how features are addressed using feature identifiers.

- Core concepts: {doc}`core_concepts`
- Feature identifiers: {doc}`feature_identifiers`


PLAID is the name of a datamodel, described in this section, and also the name of the present library, which is an implementation of this datamodel, and the format when the data is saved on disk.

PLAID aims to formalize, as generally as possible, a set of physics problems configurations, named dataset, and a learning problem defined on these configurations, named problem_definition. The provided implementation allows to populate the datamodel in memory, and read from / write to disk in a corresponding format using provided io routines.

For the on-disk layout and a guided tour of the entities (`Dataset`, `Sample`, features, `ProblemDefinition`), see {doc}`core_concepts`.

The datamodel heavily relies on CGNS, see [Seven keys for practical understanding and use of CGNS](https://ntrs.nasa.gov/api/citations/20180006202/downloads/20180006202.pdf), where a very large number of possible physics configurations have already been formalized and standardized (like multiblock configuration, time evolution, etc...). The format is human-readable: the ``yaml`` and ``csv`` files can be opened with any text editor, and the physics configurations contained in ``cgns`` files can be explored using [``paraview``](https://www.paraview.org/), for instance.



## 2. How to use it ?


PLAID proposes high-level functions to construct and handling datasets.
In practice, the user should only use the classes {py:class}`plaid.containers.dataset.Dataset` and {py:class}`plaid.containers.sample.Sample` when handling a database of physical solutions. Inputs/outputs for learning tasks should be expressed with feature identifiers. See {doc}`feature_identifiers` for details.

Example usage of each class are available and documented in the {doc}`notebooks`.
