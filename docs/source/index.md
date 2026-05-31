<br>

<p align="center">
  <img src="https://plaid-lib.github.io/assets/images/plaid_logo2.png" alt="PLAID logo" width="150">
</p>

# PLAID

| | |
| --- | --- |
| **Testing** | [![CI Status](https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml/badge.svg)](https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml) [![Docs](https://readthedocs.org/projects/plaid-lib/badge/?version=latest)](https://plaid-lib.readthedocs.io/en/latest/?badge=latest) [![Coverage](https://codecov.io/gh/plaid-lib/plaid/branch/main/graph/badge.svg)](https://app.codecov.io/gh/plaid-lib/plaid/tree/main?search=&displayType=list) [![Last Commit](https://img.shields.io/github/last-commit/PLAID-lib/plaid/main)](https://github.com/PLAID-lib/plaid/commits/main) |
| **Package** | [![Conda Version](https://anaconda.org/conda-forge/plaid/badges/version.svg)](https://anaconda.org/conda-forge/plaid) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/plaid.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/plaid) [![PyPI Version](https://img.shields.io/pypi/v/pyplaid.svg)](https://pypi.org/project/pyplaid/) [![PyPI Downloads](https://static.pepy.tech/badge/pyplaid)](https://pepy.tech/projects/pyplaid) [![Platform](https://img.shields.io/badge/platform-POSIX-blue)](https://github.com/PLAID-lib/plaid) [![Python Version](https://img.shields.io/pypi/pyversions/pyplaid)](https://github.com/PLAID-lib/plaid) |
| **Other** | [![License](https://anaconda.org/conda-forge/plaid/badges/license.svg)](https://github.com/PLAID-lib/plaid/blob/main/LICENSE.txt) [![GitHub Stars](https://img.shields.io/github/stars/PLAID-lib/plaid?style=social)](https://github.com/PLAID-lib/plaid) [![JOSS paper](https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e/status.svg)](https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e) |


PLAID (Physics Learning AI Data Model) turns complex physics into AI-ready data: it keeps
meshes, fields, tags, time, multiphysics structure, and metadata explicit, while making massive
heterogeneous datasets scalable to store, stream, visualize, and learn from.

Key features include:

* **Fidelity** — keep all the complexity of your simulation data and exploit it in ML pipelines.
* **Efficient out-of-core datasets** — datasets are accessed sample by sample, so full
  datasets do not need to be loaded into memory.
* **Parallel dataset generation and writing** — `save_to_disk` can shard sample ids
  across multiple processes.
* **Multiple storage backends** — depending of you needs, you can use CGNS, Hugging Face Datasets,
  or Zarr backends through a unified API for local disk, Hub download, and streaming workflows.
* **Selective reading** — request only selected features, and, where supported, selected
  indices inside large variable arrays.
* **Interactive dataset viewer** — launch `plaid-viewer` to browse local or streamed
  datasets, inspect samples in 3D, select features and visualize fields.
* **Learning-problem metadata** — store dataset information and one or more
  `ProblemDefinition` objects alongside the data to make ML tasks explicit and
  reproducible.

PLAID has been developed at SafranTech, the research center of the
[Safran group](https://www.safran-group.com/). The code is hosted on
[GitHub](https://github.com/PLAID-lib/plaid).
