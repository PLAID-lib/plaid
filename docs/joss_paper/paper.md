---
title: "PLAID: Physics Learning AI Datamodel"
tags:
- python
- scientific machine learning
- data model
- physics simulation
date: "30 May 2025"

authors:
- name: Fabien Casenave
  orcid: 0000-0002-8810-9128
  affiliation: 1
- name: Xavier Roynard
  orcid: 0000-0001-7840-2120
  affiliation: 1
- name: Alexandre Devaux-Rivière
  affiliation: 1,2
affiliations:
- name: SafranTech, Safran Tech, Digital Sciences & Technologies, 78114 Magny-Les-Hameaux, France
  index: 1
- name: EPITA, 14-16 Rue Voltaire, 94270 Le Kremlin-Bicêtre, France
  index: 2
bibliography: paper.bib
---


# Summary

PLAID (Physics-Learning AI Datamodel) is a Python library and data format for representing, storing, and sharing physics simulation datasets for machine learning. It defines a unified, extensible schema (building on the CGNS standard [@poinot2018seven]) that can capture complex simulation data across domains. Unlike domain-specific formats, PLAID accommodates **time-dependent**, **multi-resolution** simulations and **heterogeneous meshes**. For example, a PLAID dataset directory has a `dataset` folder with subfolders for each sample; each sample contains one or more CGNS files (meshes and fields for each time step) and a `scalars.csv` for any global variables. Dataset metadata (descriptions, problem definitions, train/val/test splits) are stored in human-readable YAML and CSV files, making the format transparent and self-describing.

The library provides a high-level **API**: the core `Dataset` and `Sample` classes let users easily load, inspect, and save data. For instance, one can load an entire dataset with `dataset = Dataset("path_to_plaid_dataset")`, which automatically parses the PLAID folder or archive and reports the number of samples, fields, and scalars. The `Sample` class offers methods like `get_mesh(time, apply_links=True)` to retrieve a CGNS tree for a specific timestep, or `get_field(name, ...)` to access a particular field array.  PLAID also supports efficient I/O: datasets can be packaged into a single TAR file or directory; methods like `Dataset.load_from_file(...)` and `load_from_dir(...)` handle parallel loading of samples (with multi-processing).

Beyond basic I/O, PLAID includes utilities for machine-learning workflows. It provides converters (e.g. `init_with_tabular`) to build PLAID datasets from generic tabular data, and a Hugging Face “bridge” to push/pull datasets via the Hugging Face hub. It also supplies dataset splitting and augmentation tools: for example, `plaid.utils.split_dataset` can partition a dataset into train/val/test subsets. Analysis tools include an `OnlineStatistics` class to compute streaming stats (min, mean, variance, etc.) on large arrays, and post-processing functions for ML evaluation. For example, `compute_rRMSE_RMSE` and `compute_R2` in `plaid.post.metrics` compute regression errors (RMSE, R²) on scalar outputs.  These features (along with plotting helpers like bisect plots) facilitate benchmarking surrogate models on physics data. In short, PLAID couples a **flexible on-disk standard** with a rich software toolkit to manipulate physics data, addressing the needs of ML researchers in fluid dynamics, structural mechanics, and related fields.

# Statement of Need

Machine learning for physical systems often suffers from **inconsistent data representations** across different domains and simulators.  Existing initiatives typically target narrow problems: e.g., separate formats for CFD (CGNS/HDF5) or for finite-element data, and bespoke scripts to process each new dataset. As Casenave *et al.* observe, there is a "lack of large-scale, diverse, and standardized datasets" for simulation-based ML, and many prior efforts are "limited in scope … relying on fragmented tooling, or adhering to overly simplistic datamodels". This fragmentation hinders reproducibility and reuse of high-fidelity data.

PLAID addresses this gap by providing a **generic, unified datamodel** that can describe virtually any physics simulation data.  It leverages the CGNS (CFD General Notation System) standard to capture complex geometry and time evolution: for example, CGNS supports multi-block topologies and evolving meshes, with a data model that separates abstract topology (element families, etc.) from concrete mesh coordinates.  On top of CGNS, PLAID layers a lightweight organizational structure (folder layout with YAML/CSV metadata) suitable for ML tasks. Because the format is human-readable (YAML/CSV) and uses open CGNS/HDF5 for heavy data, it enables easier inspection and sharing of datasets.

By promoting a common standard, PLAID makes physics data **interoperable** across projects. It has already been used to package and publish multiple datasets covering structural mechanics and computational fluid dynamics. These PLAID-formatted datasets (hosted on Zenodo and Hugging Face) have supported ML benchmarks and competitions, democratizing access to simulation data. Additionally, several recent research efforts in surrogate modeling cite or build upon PLAID-formatted data (e.g. mesh-morphing Gaussian process regression), demonstrating its role in the community. In summary, PLAID fills an important need for a **flexible, extensible, and tool-supported** data standard that unifies diverse simulation data under a single framework.

# Functionality

The PLAID library implements the full datamodel as a Python package with modular components:

* **Data Model and Formats:** A PLAID dataset consists of a root folder (or archive) with a prescribed structure. Inside, a `dataset/` directory contains numbered sample subfolders (`sample_000...`), each holding one or more `.cgns` files under `meshes/` and a `scalars.csv`. The `dataset/infos.yaml` file contains human-readable descriptions and metadata.  A `problem_definition/` folder includes `problem_infos.yaml` (specifying the ML task inputs/outputs) and an optional `split.csv` (train/test splits).  This design supports **time series** (multiple CGNS per sample for multiple timesteps) and **multi-block/multi-geometry** problems out of the box. All metadata is in YAML/CSV, and fields/meshes are in CGNS/HDF5, so users can easily inspect or extend the data with standard tools (e.g. ParaView for CGNS).

* **Supported Data Types:** PLAID handles scalar outputs (from `scalars.csv`), vector/tensor field data on meshes (stored in CGNS fields), and sample-specific metadata. The CGNS helper routines (`plaid.utils.cgns_helper`) allow users to query available fields and retrieve data arrays. For example, `sample.get_field_names(base_name, zone_name, location, time)` returns all field names matching the query, and `sample.get_field(name, ...)` returns the numpy array for that field. The `get_mesh(time)` method reconstructs the full CGNS tree for a given timestep, with links resolved if requested (so the entire mesh connectivity is returned). Thus PLAID naturally supports **mesh-based simulation outputs** with arbitrary element types and refinements.

* **High-Level API:** The top-level `Dataset` class manages multiple `Sample` objects. Users can create an empty `Dataset()` and programmatically add samples via `add_sample()`, or load an existing PLAID data archive by calling `Dataset("path_to_plaid_dataset")`. The `Dataset` object summarizes itself (e.g. printing “Dataset(3 samples, 2 scalars, 5 fields)”) and provides access to samples by ID. Batch operations are supported: one can `dataset.add_samples(...)` to append many samples, or use the classmethods `Dataset.load_from_dir()` and `load_from_file()` to load data from disk, with optional parallel workers. Writing back to disk (saving the PLAID structure) is similarly easy. This high-level interface abstracts away low-level I/O, letting users focus on ML pipelines.

* **Extensibility:** The PLAID design allows custom physical configurations. By relying on CGNS, users can incorporate **user-defined families** and **CPEX extensions** (CGNS’s formal process for new data) without breaking PLAID. The YAML schema is open: any additional information can be added under the `infos.yaml` or sample CSV files without changing the library. Moreover, a “Hugging Face bridge” (`plaid.bridges.huggingface_bridge`) enables converting PLAID datasets to/from Hugging Face Dataset objects. For example, one can call a converter to upload a PLAID dataset to the Hugging Face Hub or instantiate a HF `Dataset` that yields PLAID samples. This integration has been used to publish PLAID benchmarks on Hugging Face.

* **Utilities:** PLAID includes helper modules for common tasks in data science workflows. The `plaid.utils.split` module provides a `split_dataset` function to partition data into training/validation/testing subsets according to user-defined ratios. The `plaid.utils.interpolation` module implements piecewise linear interpolation routines (and fast vectorized search) to resample time series fields or align datasets with different timesteps. The `plaid.utils.stats` module offers an `OnlineStatistics` class to compute running statistics (min, mean, variance, etc.) on arrays, which can be used to analyze dataset distributions. After ML model training, the `plaid.post` suite helps evaluate results: e.g. `plaid.post.metrics.compute_R2` and `compute_rRMSE_RMSE` compute standard regression error metrics, and `plaid.post.bisect` can generate bisect plots comparing predictions to true values for all samples. Together, these tools streamline dataset preparation, analysis, and benchmarking.

# Usage and Applications

PLAID is designed for AI/ML researchers and practitioners working with simulation data. Its broad feature set has already enabled various applications. The original PLAID paper released six datasets (2D/3D fluid and structural simulations) under this format and demonstrated baseline learning methods on them. These datasets are publicly available (e.g. on Zenodo and Hugging Face), and PLAID is used as the data backend in ongoing benchmarks like the NeurIPS ML4CFD competition. Beyond datasets, recent research has directly incorporated PLAID-based workflows. For example, [@casenave2024mmgp] used PLAID to store data for a Gaussian-process regression with mesh morphing, while [@kabalan2025elasticity, @kabalan2025ommgp] applied PLAID datasets in elasticity-based shape modeling. Likewise, [@perez2025learningsignalsdefinedgraphs; @perez2024gaussianprocessregressionsliced] leveraged PLAID data in graph-kernel regression studies of fluid/solid mechanics. These uses illustrate the library’s flexibility. In practice, users find that PLAID significantly reduces the overhead of data wrangling: once a simulation run is converted into PLAID format, standard ML libraries (PyTorch, TensorFlow, Scikit-learn) can consume the data via simple Python loaders, and all preprocessing (splits, normalization) can be managed within the PLAID ecosystem. This promotes **reproducible pipelines**: all geometry, fields, and metadata are captured in one place.

In summary, PLAID provides a **comprehensive framework** for physics-based ML data. By combining a unified schema, support for advanced mesh features, and helpful utilities, it addresses the longstanding need for interoperable, high-fidelity simulation datasets. We anticipate that PLAID will continue to accelerate ML research in engineering and the physical sciences by making complex simulation data more accessible and reusable.

# References