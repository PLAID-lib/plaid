# Description

This page, still under construction, provides elements on PLAID functionalities.

---

- [Description](#description)
  - [1. Datamodel](#1-datamodel)
  - [2. How to use it ?](#2-how-to-use-it-)

---

## 1. Datamodel


PLAID is the name of a datamodel, described in this section, and also the name of the present library, which is an implementation of this datamodel, and the format when the data is saved on disk.

PLAID aims to formalize, as generally as possible, a set of physics problems configurations, named dataset, and a learning problem defined on these configurations, named problem_definition. The provided implementation allows to populate the datamodel in memory, and read from / write to disk in a corresponding format using provided io routines. Let us take a look at the characteristics of the format, from which we will be able to defined our objects. The plaid format looks like this:

```
folder
├───dataset
│   ├───samples
│   │   ├───sample_000000000
│   │   │   ├───meshes
│   │   │   │   ├───mesh_000000000.cgns
│   │   │   │   ├───mesh_000000001.cgns
│   │   │   │   ⋮
│   │   │   │   └───mesh_xxxxxxxxx.cgns
│   │   │   └───scalars.csv
│   │   ├───sample_000000001
│   │   ⋮
│   │   └───sample_yyyyyyyyy
│   └───infos.yaml
└───problem_definition
    ├───problem_infos.yaml
    └───split.csv
```

where ``folder`` is the name of the folder on disk contaning the data. The folder ``dataset`` contains a set of physics problems configurations in the folder ``samples``, and ``infos.yaml`` containing descriptive text information on the dataset. The folder ``problem_definition`` contains ``problem_infos.yaml``, where the learning task is defined (inluding inputs and outputs), and ``split.csv``, where differents subsets can be provided (like the usual train, test, validation for instance). In the ``samples`` folder, each sample is itself stored in numbered folders (``0`` to ``yyyyyyyyy``). A sample is constituted of possibly numerous ``cgns`` files (numbered ``0`` to ``xxxxxxxxx``), containing a physics configuration at a given time step, and ``scalars.csv``, where constant scalars associated to the corresponding sample are listed.

The datamodel heavily relies on CGNS, see [Seven keys for practical understanding and use of CGNS](https://ntrs.nasa.gov/api/citations/20180006202/downloads/20180006202.pdf), where a very large number of possible physics configurations have already been formalized and standardized (like multiblock configuration, time evolution, etc...). The format is human-readable: the ``yaml`` and ``csv`` files can be opened with any text editor, and the physics configurations contained in ``cgns`` files can be explored using [``paraview``](https://www.paraview.org/), for instance.



## 2. How to use it ?


PLAID proposes high-level functions to construct and handling datasets.
In practice, the user should only use the classes [`Dataset`](plaid.containers.dataset.Dataset) and [`Sample`](plaid.containers.sample.Sample) when handling a database of physical solutions.

Example usage of each class are available and documented in the [`notebooks`](notebooks).
