<div align="center">
<img src="https://plaid-lib.github.io/assets/images/PLAID-large-logo.png" width="300">
</div>

| | |
| --- | --- |
| Testing | [![CI Status](https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml/badge.svg)](https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml) [![Documentation Status](https://readthedocs.org/projects/plaid-lib/badge/?version=latest)](https://plaid-lib.readthedocs.io/en/latest/?badge=latest) [![Coverage](https://codecov.io/gh/plaid-lib/plaid/branch/main/graph/badge.svg)](https://app.codecov.io/gh/plaid-lib/plaid/tree/main?search=&displayType=list) ![Last Commit](https://img.shields.io/github/last-commit/PLAID-lib/plaid/main) |
| Package | [![Conda Latest Release](https://anaconda.org/conda-forge/plaid/badges/version.svg)](https://anaconda.org/conda-forge/plaid) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/plaid.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/plaid) [![PyPI Latest Release](https://img.shields.io/pypi/v/pyplaid.svg)](https://pypi.org/project/pyplaid/) [![PyPI Downloads](https://static.pepy.tech/badge/pyplaid)](https://pepy.tech/projects/pyplaid) ![Platform](https://img.shields.io/badge/platform-any-blue) ![Python Version](https://img.shields.io/pypi/pyversions/pyplaid)  |
| Other | [![License - BSD 3-Clause](https://anaconda.org/conda-forge/plaid/badges/license.svg)](https://github.com/PLAID-lib/plaid/blob/main/LICENSE.txt) ![GitHub stars](https://img.shields.io/github/stars/PLAID-lib/plaid?style=social) [![JOSS status](https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e/status.svg)](https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e) |


<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/pyplaid.svg?label=PyPI%20downloads)](https://pypi.org/project/pyplaid/) -->


# Physics Learning AI Datamodel (PLAID)

</div>


- [Physics Learning AI Datamodel (PLAID)](#physics-learning-ai-datamodel-plaid)
  - [1. Description](#1-description)
  - [2. Getting started](#2-getting-started)
  - [3. Call for Contributions](#3-call-for-contributions)
  - [4. Documentation](#4-documentation)


## 1. Description

This library proposes an implementation for a datamodel tailored for AI and ML learning of physics problems.
It has been developped at SafranTech, the research center of [Safran group](https://www.safran-group.com/).

- **Documentation:** https://plaid-lib.readthedocs.io/
- **Source code:** https://github.com/PLAID-lib/plaid
- **Contributing:** https://github.com/PLAID-lib/plaid/blob/main/CONTRIBUTING.md
- **License:** https://github.com/PLAID-lib/plaid/blob/main/LICENSE.txt
- **Bug reports:** https://github.com/PLAID-lib/plaid/issues
- **Report a security vulnerability:** https://github.com/PLAID-lib/plaid/issues


## 2. Getting started


### 2.1 Using the library

To use the library, the simplest way is to install it as follows:

```bash
conda install -c conda-forge plaid
```

or

```bash
sudo apt-get install -y libhdf5-dev
pip install pyplaid
```

**Note**

Only the conda-forge package comes with a bundled HDF5 dependency.


### 2.2 Contributing to the library

To contribute to the library, you need to clone the repo using git:

```bash
git clone https://github.com/PLAID-lib/plaid.git
```

#### 2.2.1 Development dependencies

To configure an environment manually, you can follow the dependencies listed in ``environment.yml``, or generate it using conda:

```bash
conda env create -f environment.yml
```

Then, to install the library:

```bash
pip install -e .
```

**Note**

The development dependency [**Muscat**](https://muscat.readthedocs.io/) is available on [``conda-forge``](https://anaconda.org/conda-forge/muscat) but not on [``PyPi``](https://pypi.org/project/muscat). A a consequence, using a conda environment is the only way to run tests and examples, and compile the documentation.


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

To compile locally the documentation, you can run:

```bash
cd docs
make html
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
