<div align="center">
<img src="https://plaid-lib.github.io/assets/images/PLAID-large-logo.png" width="300">
</div>

| | |
| --- | --- |
| Testing | [![CI Status](https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml/badge.svg)](https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml) [![Documentation Status](https://readthedocs.org/projects/plaid-lib/badge/?version=latest)](https://plaid-lib.readthedocs.io/en/latest/?badge=latest) [![Coverage](https://codecov.io/gh/plaid-lib/plaid/branch/main/graph/badge.svg)](https://app.codecov.io/gh/plaid-lib/plaid/tree/main?search=&displayType=list) ![Last Commit](https://img.shields.io/github/last-commit/PLAID-lib/plaid/main) |
| Package | [![PyPI Latest Release](https://img.shields.io/pypi/v/pyplaid.svg)](https://pypi.org/project/pyplaid/) [![Conda Latest Release](https://anaconda.org/conda-forge/plaid/badges/version.svg)](https://anaconda.org/conda-forge/plaid) ![Platform](https://img.shields.io/badge/platform-any-blue) ![Python Version](https://img.shields.io/pypi/pyversions/pyplaid) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/plaid.svg?label=Conda%20downloads)](https://anaconda.org/conda-forge/plaid) |
| Meta | [![License - BSD 3-Clause](https://anaconda.org/conda-forge/plaid/badges/license.svg)](https://github.com/PLAID-lib/plaid/blob/main/LICENSE.txt) ![GitHub stars](https://img.shields.io/github/stars/PLAID-lib/plaid?style=social) |

<!-- [![PyPI Downloads](https://img.shields.io/pypi/dm/pyplaid.svg?label=PyPI%20downloads)](https://pypi.org/project/pyplaid/) -->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3509134.svg)](https://doi.org/10.5281/zenodo.3509134)  -->


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

To use the library, the simplest way is to install the conda-forge or PyPi package:

```bash
conda install -c conda-forge plaid
pip install pyplaid
```

### 2.2 Contributing to the library

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

**Note**

[pytest](https://anaconda.org/conda-forge/pytest) and [Muscat](https://anaconda.org/conda-forge/muscat) are two dependencies not included in the packages ``plaid`` on conda-forge or ``pyplaid`` on PyPi, but required to run the tests and examples. [pytest](https://pypi.org/project/pytest) is available on PyPi, but not [Muscat](https://pypi.org/project/pytest).


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

A documentation is available in [readthedocs](https://plaid-lib.readthedocs.io/).
