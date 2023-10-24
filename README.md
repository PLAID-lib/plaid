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
- **Source code:** https://gitlab.com/drti/plaid/-/tree/main/src/plaid
- **Contributing:** https://gitlab.com/drti/plaid/-/blob/main/CONTRIBUTING.md
- **License:** https://gitlab.com/drti/plaid/-/blob/main/LICENSE.txt
- **Bug reports:** https://gitlab.com/drti/plaid/-/issues
- **Report a security vulnerability:** https://gitlab.com/drti/plaid/-/issues


## 2. Getting started

To use the library, the simplest way is to install the conda package:

```bash
conda install -c conda-forge plaid
```

Consider using setup.py for install, and exploring the examples folder.

```bash
pip setup.py install
```

To configure a compatible environment, consider using the provided conda yaml files.

```bash
conda env create -f '<environment_you_need>.yml'
```

To check the installation, you can run:
```bash
cd tests
python -m pytest
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
