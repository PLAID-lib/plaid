# Quickstart

Everything you need to start using PLAID and contributing effectively.

---

- [1. Using the library](#1-using-the-library)
- [2. Core concepts](#2-core-concepts)
- [3. Citation](#3-citation)
- [4. Contributing](#4-contributing)

---


## 1 Using the library

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

## 2 {doc}`core_concepts`

- {doc}`core_concepts/sample` → API: {py:class}`plaid.containers.sample.Sample`
- {doc}`core_concepts/dataset` → API: {py:class}`plaid.containers.dataset.Dataset`
- {doc}`core_concepts/problem` → API: {py:class}`plaid.problem_definition.ProblemDefinition`
- {doc}`core_concepts/feature_identifiers` → API: {py:class}`plaid.types.feature_types.FeatureIdentifier`
- {doc}`core_concepts/default` (flowchart)
- {doc}`core_concepts/disk_format`
- {doc}`core_concepts/interoperability`

## 3 Citation

If you use PLAID in your work, please cite the following.

JOSS paper (under review):

[![JOSS status](https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e/status.svg)](https://joss.theoj.org/papers/26b2e13a9fc8e012cc997ca28a7b565e)

PLAID datasets:

```bibtex
@misc{casenave2025plaid,
      title={{Physics-Learning AI Datamodel (PLAID) datasets: a collection of physics simulations for machine learning}},
      author={Casenave, F. and Roynard, X. and Staber, B. and Piat, W. and Bucci, M. A. and Akkari, N. and Kabalan, A. and Nguyen, X. M. V. and Saverio, L. and Carpintero Perez, R. and Kalaydjian, A. and Fouch\'{e}, S. and Gonon, T. and Najjar, G. and Menier, E. and Nastorg, M. and Catalani, G. and Rey, C.},
      year={2025},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.02974},
}
```

## 4 Contributing

To contribute to the library, you need to clone the repo using git:

```bash
git clone https://github.com/PLAID-lib/plaid.git
```

### 4.1 Development dependencies

To configure an environment manually, you can follow the dependencies listed in ``environment.yml``, or generate it using conda:

```bash
conda env create -f environment.yml
```

Then, to install the library:

```bash
pip install -e .
```

**Note**

The development dependency [**Muscat=2.5.0**](https://muscat.readthedocs.io/) is available on [``conda-forge``](https://anaconda.org/conda-forge/muscat) but not on [``PyPi``](https://pypi.org/project/muscat). A a consequence, using a conda environment is the only way to run tests and examples, and compile the documentation.


### 4.2 Tests and examples

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

### 4.3 Documentation

To compile locally the documentation, you can run:

```bash
cd docs
make html
```

Various notebooks are executed during compilation. The documentation can then be explored in ``docs/_build/html``.

### 4.4 Formatting and linting with Ruff

We use [**Ruff**](https://docs.astral.sh/ruff/) for linting and formatting.

The configuration is defined in `ruff.toml`, and some folders like `docs/` and `examples/` are excluded from checks.

You can run Ruff manually as follows:

```bash
ruff --config ruff.toml check . --fix      # auto-fix linting issues
ruff --config ruff.toml format .           # auto-format code
```

### 4.5 Setting up pre-commit

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

### 4.6 How to contribute

Before contributing, please review our guidelines and backward-compatibility policy.

- Coding standards: PEP 8, PEP 257; snake_case, 80-char lines; one class per file preferred.
- Rules: simple/flexible API, local/unit tests, examples preferred, 100% coverage.
- Reporting issues: include steps, expected behavior, PLAID version, logs.
- Feature requests: describe the feature and use cases.
- Contributor License Agreement (CLA): see details below.

Backward compatibility (API and disk format): see policy in this page and details in issues `#97` and `#14`.

CGNS resources: <http://cgns.github.io/>

#### Contributor License Agreement (CLA)

By contributing, you agree to the CLA terms summarized below (full text previously in Contributing page):
- Perpetual, worldwide, non-exclusive licenses for copyright and patent.
- You represent rights to contribute, originality, and disclose third-party restrictions.
- Contributions are provided “AS IS” without warranties.

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
