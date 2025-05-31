![GitHub stars](https://img.shields.io/github/stars/PLAID-lib/plaid?style=social)
[![CI Status](https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml/badge.svg)](https://github.com/PLAID-lib/plaid/actions/workflows/testing.yml)
[![codecov](https://codecov.io/gh/plaid-lib/plaid/branch/main/graph/badge.svg)](https://app.codecov.io/gh/plaid-lib/plaid/tree/main?search=&displayType=list)
[![License](https://anaconda.org/conda-forge/plaid/badges/license.svg)](https://github.com/PLAID-lib/plaid/blob/main/LICENSE.txt)
[![Documentation Status](https://readthedocs.org/projects/plaid-lib/badge/?version=latest)](https://plaid-lib.readthedocs.io/en/latest/?badge=latest)
[![Conda Version](https://anaconda.org/conda-forge/plaid/badges/version.svg)](https://anaconda.org/conda-forge/plaid)
[![Conda Release Date](https://anaconda.org/conda-forge/plaid/badges/latest_release_date.svg)](https://anaconda.org/conda-forge/plaid)
[![Conda Platforms](https://anaconda.org/conda-forge/plaid/badges/platforms.svg)](https://anaconda.org/conda-forge/plaid)
[![Conda Downloads](https://anaconda.org/conda-forge/plaid/badges/downloads.svg)](https://anaconda.org/conda-forge/plaid)
<!-- ![Python Version](https://img.shields.io/pypi/pyversions/plaid-lib)
[![PyPI Version](https://img.shields.io/pypi/v/plaid-lib)](https://pypi.org/project/plaid-lib/)
[![codecov](https://codecov.io/gh/PLAID-lib/plaid/branch/main/graph/badge.svg)](https://codecov.io/gh/PLAID-lib/plaid) -->



<div align="center">
<img src="docs/source/images/plaid.jpg" width="70">

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

To use the library, the simplest way is to install the conda package:

```bash
conda install -c conda-forge plaid
```

### 2.2 Contributing to the library

To contribute to the library, you need to clone the repo using git:

```bash
git clone https://github.com/PLAID-lib/plaid.git
```

Configure an environment manually following the dependencies listed in ``conda_dev_env.yml``, or generate it using conda:

```bash
conda env create -f conda_dev_env'
```

Then, to install the library:

```bash
pip install -e .
```

To check the installation, you can run the unit test suite:

```bash
pytest tests
```

To test further and learn about simple use cases, you can run the examples:

```bash
cd examples
bash run_examples.sh  # [unix]
run_examples.bat      # [win]
```

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
