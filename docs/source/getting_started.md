# Getting Started

Everything you need to know to start using the PLAID.

---

- [Getting Started](#getting-started)
  - [1. Install guide](#1-install-guide)
    - [1.1 Using the library](#11-using-the-library)
    - [1.2 Contributing to the library](#12-contributing-to-the-library)
  - [2. Test installation](#2-test-installation)

---

## 1. Install guide

### 1.1 Using the library

To use the library, the simplest way is to install the conda package:

```bash
conda install -c conda-forge plaid
```

### 1.2 Contributing to the library

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

---
**Note**

[pytest](https://anaconda.org/conda-forge/pytest) and [Muscat](https://anaconda.org/conda-forge/muscat) are two dependencies not distributed in the plaid conda-forge package, but can be found on conda-forge as well.