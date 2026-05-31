# Quickstart

Everything you need to start using PLAID and contributing effectively.

---

- [1 Using the library](#1-using-the-library)
- [2 Core concepts](#2-core-concepts)
- [3 Going further](#3-going-further)

---


## 1 Using the library

To use the library, the simplest way is to install it from the packages available:

- on conda-forge for Linux, macOS, and Windows:
  ```bash
  conda install -c conda-forge plaid
  ```

- or on PyPI for Linux:
  ```bash
  pip install pyplaid
  ```

**Note**

- Conda-forge packages for all supported operating systems (Linux, macOS, and Windows) and the Linux PyPI package include a bundled pyCGNS dependency. In other situations, which we have not tested, pyCGNS must be installed separately beforehand.
- On Apple Silicon, users can force an osx-64 conda environment using CONDA_SUBDIR=osx-64, allowing installation of the existing macOS-64 builds under Rosetta.


## 2 Core concepts

- {doc}`core_concepts/sample` → API: {py:class}`plaid.containers.sample.Sample`
- {doc}`core_concepts/dataset` → API: {mod}`plaid.storage` backend datasets and converters
- {doc}`core_concepts/problem_definition` → API: {py:class}`plaid.problem_definition.ProblemDefinition`
- {doc}`core_concepts/defaults`
- {doc}`core_concepts/disk_format`
- {doc}`core_concepts/interoperability`

The main public imports are:

```python
from plaid import Sample, ProblemDefinition
from plaid.storage import (
    save_to_disk,
    init_from_disk,
    download_from_hub,
    init_streaming_from_hub,
    push_to_hub,
)
```

The package also ships two command-line tools:

```bash
plaid-check /path/to/plaid_dataset
plaid-viewer --datasets-root /path/to/datasets
```

## 3 Going further

Explore {doc}`example examples_tutorials <examples_tutorials>` for practical use cases and advanced techniques.

The {doc}`API documentation <../autoapi/plaid/index>` provides detailed information on all available classes and methods.

In-repository core capabilities are provided by `plaid` itself (containers, storage backends,
and converters under `plaid.storage`).

Two companion repositories extend the `plaid` ecosystem:

- [plaid-bridges](https://github.com/PLAID-lib/plaid-bridges): integrations with popular ML frameworks such as PyTorch Geometric.
- [plaid-ops](https://github.com/PLAID-lib/plaid-ops): standardized operations on PLAID samples and datasets, including advanced mesh processing (some requiring a finite-element engine) powered by [muscat](https://gitlab.com/drti/muscat).

These companion projects are external repositories (not modules shipped in this package).