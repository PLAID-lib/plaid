# Quickstart

Everything you need to start using PLAID and contributing effectively.

---

- [1 Using the library](#1-using-the-library)
- [2 Core concepts](#2-core-concepts)
- [3 Going further](#3-going-further)

---


## 1 Using the library

To use the library, the simplest way is to install it from the packages available:

- on conda-forge for Windows, macOS and Linux:
  ```bash
  conda install -c conda-forge plaid
  ```

- or on PyPi for Linux:
  ```bash
  pip install pyplaid
  ```

**Note**

- Only the conda-forge packages (all operating systems) and the Linux PyPI package include a bundled pyCGNS dependency. In other situations, which we have not tested, pyCGNS must be installed separately beforehand.
- On Apple Silicon, users can force an osx-64 conda environment using CONDA_SUBDIR=osx-64, allowing installation of the existing macOS-64 builds under Rosetta.


## 2 Core concepts

- [Sample](core_concepts/sample.md) → API: [`plaid.containers.sample.Sample`](api_reference.md#api-sample)
- [Dataset](core_concepts/dataset.md) → API: [`plaid.containers.dataset.Dataset`](api_reference.md#api-dataset)
- [Problem definition](core_concepts/problem_definition.md) → API: [`plaid.problem_definition.ProblemDefinition`](api_reference.md#api-problemdefinition)
- [Defaults](core_concepts/defaults.md)
- [On-disk format](core_concepts/disk_format.md)
- [Interoperability](core_concepts/interoperability.md)

## 3 Going further

Explore [Examples and Tutorials](examples_tutorials.md) for practical use cases and advanced techniques.

The [API documentation](api_reference.md) provides detailed information on all available classes and methods.

Two companion libraries extend the `plaid` standard to support machine-learning workflows in physics:

- [plaid-bridges](https://github.com/PLAID-lib/plaid-bridges): integrations with popular ML frameworks such as PyTorch Geometric.
- [plaid-ops](https://github.com/PLAID-lib/plaid-ops): standardized operations on PLAID samples and datasets, including advanced mesh processing (some requiring a finite-element engine) powered by [muscat](https://gitlab.com/drti/muscat).