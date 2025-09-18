# Quickstart

Everything you need to start using PLAID and contributing effectively.

---

- [1 Using the library](#1-using-the-library)
- [2 Core concepts](#2-core-concepts)
- [3 Going further](#3-going-further)

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

## 2 Core concepts

- {doc}`core_concepts/sample` → API: {py:class}`plaid.containers.sample.Sample`
- {doc}`core_concepts/dataset` → API: {py:class}`plaid.containers.dataset.Dataset`
- {doc}`core_concepts/problem_definition` → API: {py:class}`plaid.problem_definition.ProblemDefinition`
- {doc}`core_concepts/feature_identifiers` → API: {py:class}`plaid.types.feature_types.FeatureIdentifier`
- {doc}`core_concepts/defaults`
- {doc}`core_concepts/disk_format`
- {doc}`core_concepts/interoperability`

## 3 Going further

Explore {doc}`example notebooks <notebooks>` for practical use cases and advanced techniques.

The {doc}`API documentation <../autoapi/plaid/index>` provides detailed information on all available classes and methods.
