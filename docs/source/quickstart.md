# :material-rocket-launch-outline:{ .lg .middle } Quickstart

## 1. Using the library

To use the library, the simplest way is to install it from the packages available:

- on conda-forge for Linux, macOS, and Windows:
  ```bash
  conda install -c conda-forge plaid
  ```

- or on PyPI for Linux:
  ```bash
  pip install pyplaid
  ```

!!! note "Installation notes"
    - Conda-forge packages for all supported operating systems (Linux, macOS, and Windows) and the Linux PyPI package include a bundled pyCGNS dependency. In other situations, which we have not tested, pyCGNS must be installed separately beforehand.
    - On Apple Silicon, users can force an osx-64 conda environment using CONDA_SUBDIR=osx-64, allowing installation of the existing macOS-64 builds under Rosetta.


## 2. Core concepts

The main classes of `plaid` are `Sample`, `ProblemDefinition` and `Infos`. There is no public top-level dataset container class. Datasets are accessed through backend-specific collections and `Converter` objects, so users typically materialize individual `Sample` objects only when needed instead of loading a whole dataset as one in-memory PLAID object.

The main use case is efficient, backend-agnostic access to heterogeneous PLAID samples in ML pipelines. The reading API is therefore centered on backend datasets and converters:

```python
from plaid.storage import init_from_disk

datasetdict, converterdict = init_from_disk(local_folder)

dataset = datasetdict[split]
converter = converterdict[split]

for i in range(len(dataset)):
    plaid_sample = converter.to_plaid(dataset, i)
```

Sample features can then be retrieved as follows:

```python
from plaid.storage import load_problem_definitions_from_disk

pb_defs = load_problem_definitions_from_disk(local_folder)
pb_def = pb_defs["my_pb_def"]

plaid_sample = ... # see above to instantiate a plaid sample

for t in plaid_sample.get_all_time_values():
    for path in pb_def.input_features:
        feature = plaid_sample.get_feature_by_path(path=path, time=t)
        ...
    for path in pb_def.output_features:
        feature = plaid_sample.get_feature_by_path(path=path, time=t)
        ...
```

These instructions are valid regardless of the storage backend or the heterogeneity of the data.

The package also ships three command-line tools:

```bash
plaid-check /path/to/plaid_dataset
plaid-serve --dataset /path/to/plaid_dataset
plaid-viewer --datasets-root /path/to/datasets
```

These concepts are described in more detail in [Concepts](concepts.md), including efficient dataset generation and writing.

## 3. Going further

Explore [Examples & Tutorials](examples_tutorials.md) for practical use cases and advanced techniques.

The [API Reference](api/index.md) provides detailed information on all available classes and methods.
