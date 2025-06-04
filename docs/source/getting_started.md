# Getting Started

Everything you need to know to start using the PLAID.

---

- [Getting Started](#getting-started)
  - [1. Using the library](#1-using-the-library)
  - [2. Contributing to the library](#2-contributing-to-the-library)
  - [3. Formatting and linting with Ruff](#3-formatting-and-linting-with-ruff)
  - [4. Settting up pre-commit](#4-setting-up-pre-commit)

---

## 1 Using the library

To use the library, the simplest way is to install as follows:

```bash
conda install -c conda-forge plaid
```

or

```bash
pip install pyplaid
```

## 2 Contributing to the library

To contribute to the library, you need to clone the repo using git:

```bash
git clone https://github.com/PLAID-lib/plaid.git
```

Configure an environment manually following the dependencies listed in ``conda_dev_env.yml``, or generate it using conda:

```bash
conda env create -f conda_dev_env.yml
```

Then, to install the library:

```bash
pip install -e .
```

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

## 3 Formatting and linting with ruff

We use [**ruff**](https://docs.astral.sh/ruff/) for linting and formatting.

The configuration is defined in `ruff.toml`, and some folders like `docs/` and `examples/` are excluded from checks.

You can run ruff manually as follows:

```bash
ruff --config ruff.toml check . --fix      # auto-fix linting issues
ruff --config ruff.toml format .           # auto-format code
```

## 4 Setting up pre-commit

If you’re contributing code, it is recommended to use pre-commit, which is configured to run the following hooks:

* Ruff check
* Ruff format
* Pytest

The selected hooks are defined in the .pre-commit-config.yaml file.

First, install pre-commit:

```bash
pip install pre-commit
```

To run all hooks manually on the full codebase:

```bash
pre-commit run --all-files
```

You can also run (once):

```bash
pre-commit install
```

This ensures that every time you commit, all the hooks are executed automatically on the staged files.


**Note**

[**ruff**](https://docs.astral.sh/ruff/), [**pytest**](https://anaconda.org/conda-forge/pytest) and [**Muscat**](https://anaconda.org/conda-forge/muscat) are development dependencies not included in the packages ``plaid`` on conda-forge or ``pyplaid`` on PyPi, but required to run the tests and examples. 

To install [**ruff**](https://docs.astral.sh/ruff/) and [**pytest**](https://anaconda.org/conda-forge/pytest):

```bash
pip install ruff pytest
``` 

[**Muscat**](https://pypi.org/project/muscat) is only available on conda-forge and can be installed as follows:

```bash
conda install -c -conda-forge muscat
```