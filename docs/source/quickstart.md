# :material-rocket-launch-outline:{ .lg .middle } Quickstart

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

There is no public top-level dataset container class: all operations are out-of-core, so full datasets are never loaded into RAM at once.

The package also ships two command-line tools:

```bash
plaid-check /path/to/plaid_dataset
plaid-viewer --datasets-root /path/to/datasets
```

Additional concepts are detailed in {doc}`Concepts <concepts>`.

## 3 Going further

Explore {doc}`Examples & Tutorials <examples_tutorials>` for practical use cases and advanced techniques.

The [API Reference](api/index.md) provides detailed information on all available classes and methods.
