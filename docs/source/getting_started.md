# Getting Started

Everything you need to know to start using the PLAID.

---

- [Getting Started](#getting-started)
  - [1. Using the library](#1-using-the-library)
  - [2. Contributing to the library](#2-contributing-to-the-library)

---


## 1 Using the library

To use the library, the simplest way is to install it as follows:

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

### 2.1 Development dependencies

Conda (or mamba) is needed to configure an environment development:

```bash
conda create -n plaid-dev python=3.11 muscat=2.4.1 poetry
conda activate plaid-dev
poetry install
```

**Note**

The development dependency [**Muscat**](https://muscat.readthedocs.io/) is available on [``conda-forge``](https://anaconda.org/conda-forge/muscat) but not on [``PyPi``](https://pypi.org/project/muscat).

### 2.2 Tests and examples

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

### 2.3 Documentation

To compile locally the documentation, you can run:

```bash
cd docs
make html
```

Various notebooks are executed during compilation. The documentation can then be explored in ``docs/_build/html``.

### 2.4 Formatting and linting with Ruff

We use [**Ruff**](https://docs.astral.sh/ruff/) for linting and formatting.

The configuration is defined in `ruff.toml`, and some folders like `docs/` and `examples/` are excluded from checks.

You can run Ruff manually as follows:

```bash
ruff --config ruff.toml check . --fix      # auto-fix linting issues
ruff --config ruff.toml format .           # auto-format code
```

### 2.5 Setting up pre-commit

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