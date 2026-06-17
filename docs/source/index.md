<!-- ---
hide:
  - navigation
  - toc
--- -->

<p align="center">
  <img src="https://plaid-lib.github.io/assets/images/plaid_logo2.png" alt="PLAID logo" width="170">
</p>

<h1 align="center">PLAID</h1>

<p align="center"><em>Physics Learning AI Data Model — turning complex physics simulations into AI-ready data.</em></p>

<p align="center">
  <a href="https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml"><img alt="CI Status" src="https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml/badge.svg"></a>
  <a href="https://plaid-lib.readthedocs.io/en/latest/?badge=latest"><img alt="Docs" src="https://readthedocs.org/projects/plaid-lib/badge/?version=latest"></a>
  <a href="https://app.codecov.io/gh/plaid-lib/plaid/tree/main?search=&displayType=list"><img alt="Coverage" src="https://codecov.io/gh/plaid-lib/plaid/branch/main/graph/badge.svg"></a>
  <a href="https://github.com/PLAID-lib/plaid/commits/main"><img alt="Last Commit" src="https://img.shields.io/github/last-commit/PLAID-lib/plaid/main"></a>
  <br>
  <a href="https://anaconda.org/conda-forge/plaid"><img alt="Conda Version" src="https://anaconda.org/conda-forge/plaid/badges/version.svg"></a>
  <a href="https://pypi.org/project/pyplaid/"><img alt="PyPI Version" src="https://img.shields.io/pypi/v/pyplaid.svg"></a>
  <a href="https://packages.spack.io/package.html?name=py-plaid"><img alt="Spack Version" src="https://img.shields.io/spack/v/py-plaid"></a>
  <a href="https://github.com/PLAID-lib/plaid"><img alt="Platform" src="https://img.shields.io/badge/platform-POSIX-blue"></a>
  <a href="https://github.com/PLAID-lib/plaid"><img alt="Python Version" src="https://img.shields.io/pypi/pyversions/pyplaid"></a>
  <br>
  <a href="https://github.com/PLAID-lib/plaid/blob/main/LICENSE.txt"><img alt="License" src="https://anaconda.org/conda-forge/plaid/badges/license.svg"></a>
  <a href="https://github.com/PLAID-lib/plaid"><img alt="GitHub Stars" src="https://img.shields.io/github/stars/PLAID-lib/plaid?style=social"></a>
  <a href="https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e"><img alt="JOSS paper" src="https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e/status.svg"></a>
  <a href="https://arxiv.org/abs/2505.02974"><img alt="JOSS paper" src="https://img.shields.io/badge/arXiv-2505.02974-b31b1b.svg"></a>
</p>

---

## Overview

**PLAID** (Physics Learning AI Data Model) turns complex physics simulations into
AI-ready datasets. It preserves the full structure of each simulation — meshes
(including remeshing between time steps), physical fields, mesh tags, temporal
evolution, multiphysics couplings, and associated metadata — and exposes it through
a unified API designed for storing, streaming, visualizing, and learning from
massive, heterogeneous datasets.

## Key features

<div class="grid cards" markdown>

- :material-database-check:{ .lg .middle } **Fidelity**

    ---

    Keep all the complexity of your simulation data — meshes, fields, tags, time, and
    multiphysics structure — and exploit it directly in ML pipelines.

- :material-memory:{ .lg .middle } **Out-of-core datasets**

    ---

    Datasets are accessed sample by sample, so full datasets do not need to be loaded
    into memory.

- :material-cogs:{ .lg .middle } **Parallel I/O**

    ---

    `save_to_disk` can shard sample IDs across multiple processes for fast dataset
    generation and writing.

- :material-server-network:{ .lg .middle } **Multiple storage backends**

    ---

    Use **CGNS**, **Hugging Face Datasets**, or **Zarr** through a unified API for
    local disk, Hub download, and streaming workflows.

- :material-filter-variant:{ .lg .middle } **Selective reading**

    ---

    Request only the features you need and, when necessary, only selected indices within large variable arrays.

- :material-eye-outline:{ .lg .middle } **Interactive viewer**

    ---

    Launch `plaid-viewer` to browse local or streamed datasets, inspect samples in 3D,
    select features, and visualize fields.

</div>

## Get started

<div class="grid cards" markdown>

- :material-rocket-launch-outline:{ .lg .middle } **Quickstart**

    ---

    Install PLAID and run your first example in minutes.

    [:octicons-arrow-right-24: Quickstart](quickstart.md)

- :material-book-open-page-variant-outline:{ .lg .middle } **Concepts**

    ---

    Learn the core abstractions: samples, datasets, and problem definitions.

    [:octicons-arrow-right-24: Concepts](concepts.md)

- :material-school-outline:{ .lg .middle } **Examples & tutorials**

    ---

    Walk through end-to-end examples and notebooks.

    [:octicons-arrow-right-24: Examples & tutorials](examples_tutorials.md)

- :material-api:{ .lg .middle } **API reference**

    ---

    Browse the complete Python API.

    [:octicons-arrow-right-24: API reference](api/index.md)

</div>

## About

PLAID has been developed at **SafranTech**, the research center of the
[Safran group](https://www.safran-group.com/). The source code is hosted on
[GitHub](https://github.com/PLAID-lib/plaid) — contributions, issues, and feedback
are welcome.
