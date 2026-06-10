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




<!-- <a href="https://anaconda.org/conda-forge/plaid"><img alt="Conda Downloads" src="https://img.shields.io/conda/dn/conda-forge/plaid.svg?label=Conda%20downloads"></a>
<a href="https://pepy.tech/projects/pyplaid"><img alt="PyPI Downloads" src="https://static.pepy.tech/badge/pyplaid"></a>
   -->

<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/pyplaid.svg?label=PyPI%20downloads)](https://pypi.org/project/pyplaid/) -->



# Physics Learning AI Data Model (PLAID)

- [1. Description](#1-description)
- [2. Getting started](#2-getting-started)
- [3. Call for Contributions](#3-call-for-contributions)
- [4. Documentation](#4-documentation)


## 1. Description

This library proposes an implementation for a data model tailored for AI and ML learning of physics problems.
It has been developed at SafranTech, the research center of [Safran group](https://www.safran-group.com/).

- **Documentation:** https://plaid-lib.readthedocs.io/
- **Source code:** https://github.com/PLAID-lib/plaid
- **Contributing:** https://github.com/PLAID-lib/plaid/blob/main/CONTRIBUTING.md
- **License:** https://github.com/PLAID-lib/plaid/blob/main/LICENSE.txt
- **Bug reports:** https://github.com/PLAID-lib/plaid/issues
- **Report a security vulnerability:** https://github.com/PLAID-lib/plaid/security/advisories/new


## 2. Getting started


### 2.1 Using the library

To use the library, the simplest way is to install it from the packages available:

- on conda-forge for Linux, macOS, and Windows:
  ```bash
  conda install -c conda-forge plaid
  ```

- on PyPI for Linux:
  ```bash
  pip install pyplaid
  ```

- on Spack for Linux, macOS, and Windows:
  ```bash
  spack install py-plaid
  ```

**Note**

- Conda-forge packages for Linux, macOS, and Windows, as well as the Linux PyPI package, include a bundled pyCGNS dependency. Non-Linux PyPI installations require a separate pyCGNS installation and are untested.
- A Spack package recipe is available for Linux, macOS, and Windows, but has only been tested on Linux.
- On Apple Silicon, users can force an `osx-64` conda environment with `CONDA_SUBDIR=osx-64` to install the existing macOS-64 builds under Rosetta.

### 2.2 Contributing to the library

To contribute to the library, you need to clone the repo using git:

```bash
git clone https://github.com/PLAID-lib/plaid.git
```

#### 2.2.1 Development dependencies

To configure an environment:

- using conda (Windows, macOS and Linux):
  ```bash
  conda env create -n plaid-dev python=3.12 -f environment.yml
  pip install -e . --no-deps
  ```

- using uv (Linux):
  ```bash
  uv sync --dev
  ```

#### 2.2.2 Tests and examples

To check the installation, you can run the unit test suite:

```bash
pytest tests
```

To test further and learn about simple use cases, you can run and explore the examples:

```bash
cd examples
bash run_examples.sh  # [unix]
run_examples.bat      # [win]
```

#### 2.2.3 Documentation

The documentation is built with [Zensical](https://zensical.org/) and
[mkdocstrings](https://mkdocstrings.github.io/). To compile it locally, run:

```bash
cd docs
bash generate_doc.sh
```

Various notebooks are executed during compilation. The documentation can then be explored in ``docs/_build/html``.

#### 2.2.4 Formatting and linting with Ruff

We use [**Ruff**](https://docs.astral.sh/ruff/) for linting and formatting.

The configuration is defined in `ruff.toml`, and some folders like `docs/` and `examples/` are excluded from checks.

You can run Ruff manually as follows:

```bash
ruff --config ruff.toml check . --fix      # auto-fix linting issues
ruff --config ruff.toml format .           # auto-format code
```

#### 2.2.5 Setting up pre-commit

Pre-commit is configured to run the following hooks:

* Ruff check
* Ruff format
* Pytest

The selected hooks are defined in the `.pre-commit-config.yaml` file.

To run all hooks manually on the full codebase:

```bash
pre-commit run --all-files
```

You can also run (once):

```bash
pre-commit install
```

This ensures that every time you commit, all the hooks are executed automatically on the staged files.

## 3. Call for Contributions

The PLAID project welcomes your expertise and enthusiasm!

Small improvements or fixes are always appreciated.

Writing code isn’t the only way to contribute to PLAID. You can also:
- review pull requests
- help us stay on top of new and old issues
- develop tutorials, presentations, and other educational materials
- maintain and improve [our documentation](https://plaid-lib.readthedocs.io/)
- help with outreach and onboard new contributors

If you are new to contributing to open source, [this guide](https://opensource.guide/how-to-contribute/) helps explain why, what,
and how to successfully get involved.

## 4. Documentation

The documentation is deployed on [readthedocs](https://plaid-lib.readthedocs.io/).
