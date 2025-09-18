# Contributing

To contribute to the library, you need to clone the repo using git:

```bash
git clone https://github.com/PLAID-lib/plaid.git
```

## 1. Development dependencies

To configure an environment manually, you can follow the dependencies listed in ``environment.yml``, or generate it using conda:

```bash
conda env create -f environment.yml
```

Then, to install the library:

```bash
pip install -e .
```

**Note**

The development dependency [**Muscat=2.5.0**](https://muscat.readthedocs.io/) is available on [``conda-forge``](https://anaconda.org/conda-forge/muscat) but not on [``PyPi``](https://pypi.org/project/muscat). As a consequence, using a conda environment is the only way to run tests and examples, and compile the documentation.

## 2. Tests and examples

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

## 3. Documentation

To compile locally the documentation, you can run:

```bash
cd docs
make html
```

Various notebooks are executed during compilation. The documentation can then be explored in ``docs/_build/html``.

## 4. Formatting and linting with Ruff

We use [**Ruff**](https://docs.astral.sh/ruff/) for linting and formatting.

The configuration is defined in `ruff.toml`, and some folders like `docs/` and `examples/` are excluded from checks.

You can run Ruff manually as follows:

```bash
ruff --config ruff.toml check . --fix      # auto-fix linting issues
ruff --config ruff.toml format .           # auto-format code
```

## 5. Setting up pre-commit

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

## 6. How to contribute

Before contributing, please review our guidelines and backward-compatibility policy.

- Coding standards: PEP 8, PEP 257; snake_case, 80-char lines; one class per file preferred.
- Rules: simple/flexible API, local/unit tests, examples preferred, 100% coverage.
- Reporting issues: include steps, expected behavior, PLAID version, logs.
- Feature requests: describe the feature and use cases.
- Contributor License Agreement (CLA): see details below.

Backward compatibility (API and disk format): see policy in this page and details in issues `#97` and `#14`.

CGNS resources: <http://cgns.github.io/>

### Contributor License Agreement (CLA)

By contributing, you agree to the CLA terms summarized below (full text previously in Contributing page):
- Perpetual, worldwide, non-exclusive licenses for copyright and patent.
- You represent rights to contribute, originality, and disclose third-party restrictions.
- Contributions are provided “AS IS” without warranties.

