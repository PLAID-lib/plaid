# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- (storage/reader) Converter class: to_dict now takes features argument to prevent reader complete samples

### Changed

### Fixes

### Removed

## [0.1.12] - 2025-12-20

### Added

### Changed

- (tests) improve temp folder handling

### Fixes

- (storage/hf_datasets/reader) correct folder existence check in download_datasetdict_from_hub

### Removed

## [0.1.11] - 2025-12-19

### Added

- (storage) refactore the bridge to rely on a generic storage mechanism, with 3 backends developped: cgns, hf_datasets and zarr.
- (docs) improve contribution guidelines.

### Changed

- (huggingface bridge) full parallel support in `from_generator`, with optimization of constant leaf detection (no large data communicated between processes).
- (samples/features) add string support to globals.
- (huggingface bridge) correct split_constant tree derivation, add heuristic for number of shards usage in push_to_dict, robustify infer_hf_features_from_value with respect to numpy arrays of strings, modernize update_dataset_card.
- (dependencies) migrate to muscat=2.5.1
- (types/containers) move `FeatureIdentifier` class from `plaid.types` to `plaid.containers`.
- (features) rename `get_all_mesh_times()` to `get_all_time_values()`.

### Fixes

- (dataset/huggingface_bridge) add optional `warn` parameter to `Dataset.set_infos()` to allow silent replacement of infos; `huggingface_dataset_to_plaid` now uses `warn=False` to prevent unnecessary warnings.
- (types/containers) move `FeatureIdentifier` class from `plaid.types` to `plaid.containers`.
- (features) rename `get_all_mesh_times()` to `get_all_time_values()`.

### Removed

- (python) remove python 3.10 support for zarr compatibility

## [0.1.10] - 2025-10-29

### Added

- (problem_definition) add train_ and test_split mechanisms and constant_feature support.
- (dataset/problem_definition) add plaid version to infos.
- (sample) add `memory_safe` to save to run pyCGNS save in a subprocess and prevent memory leaks.
- (sample) add `del_feature` method and rename `_add_feature` without leading underscore.
- (docs) explain release process in Contributing page.

### Changed

- (dataset/sample/problem_definition) save and load methods starting with `_` are deprecated.
- (huggingface bridge) restructuration: do not rely on binary blobs anymore, but exploit native arrow types by flattening cgns trees into constant and variable parts (for stationnary and temporal datasets).
- (sample) Restructuring of the Sample class to store a global (tensor of arbitrary order) at a given time step: replaces scalar and time_series. All Sample data are now stored in CGNS trees.

### Fixes

- (meshes) fix `get_field_name`, could overwrite arguments during iteration over times, bases, zones and locations.
- (docs) explain release process in Contributing page.

### Removed

- (sample/features) cgns links and paths, as well as args `mesh_base_name` and `mesh_zone_name`
- (sample) time_series support, now handle directly globals at time steps.

## [0.1.9] - 2025-09-24

### Added

- (problem_definition) add extract_problem_definition_from_identifiers method.
- (dataset) `get_tabular_from_stacked_identifiers` now returns stacked tabular and the cumulated feature dims, to be able to split columns to match features and then to revert the operation.
- (dataset) add option `keep_cgns` to `extract_dataset_from_identifier` to keep CGNS tree structure even if no identifiers need it.
- (`huggingface_bridge.load_hf_dataset_from_hub`) new utility to load datasets from the Hugging Face Hub, supporting both proxy and non-proxy environments.

### Changed

- (`huggingface_bridge.to_plaid_sample`) now accepts hf_dataset[id] directly as input (with pickle loading handled internally).

### Fixes

- (pipelines) fix `ColumnTransformer.inverse_transform` to use appropriate input identifier, and to keep CGNS tree structure when extracting sub-dataset.
- (examples) corrected and improved downloadable examples and the associated documentation notebook.
  - Improved integration with huggingface_bridge.to_plaid_sample.
  - Significantly faster on first retrieval when the dataset is already cached locally.

## [0.1.8] - 2025-09-18

### Added

- (docs) configure `make clean`.
- (problem_definition) add methods using feature identifiers instead of names.
- (imports) add imports of `Sample`, `Dataset` from `plaid` and `plaid.containers` and `ProblemDefinition` from `plaid`.
- (dataset.py) add optional `ids` argument to `from_list_of_samples`.
- (sample) add summarize and check_completeness functions.
- (dataset) add summarize_features and check_feature_completeness functions.
- (Hugging Face bridge) add datasetdict conversion, and simple function for plaid sample init from hf sample.
- (pipelines/plaid_blocks.py) add column transformer inverse_transform.

### Changed

- (examples/docs) update use of deprecated functions.
- Reorder arguments in methods working on fields in `Sample` and `Dataset`, always use keyword arguments when using `add_field`, `get_field` or `del_field`.
- Refactor the `containers/sample.py` module by introducting `SampleScalars` and `SampleMeshes` in `containers/features.py` that handle the scalars and meshes mechanics. Some methods are removed from `Sample`.
- Move to jupytext for notebooks and examples handling (unique source for both).
- Move to Muscat=2.5.0 (for tests and examples support).
- Update repo configuration (actions: rely more on pypi dependencies, action versions).
- Rename types to remove `Type` from name of types: https://github.com/PLAID-lib/plaid/pull/164.
- Refactored method names for improved clarity:
  - `Dataset.from_tabular` → `Dataset.add_features_from_tabular`
  - `Dataset.from_features_identifier` → `Dataset.extract_dataset_from_identifier`
  - `Sample.from_features_identifier` → `Sample.extract_sample_from_identifier`

### Fixes

- (dataset) fix get tabular from dataset of samples containing multidimensional scalars.
- (plaid/examples) fix circular imports.
- (sample/dataset/problem_definition) fix incoherent path argument names in save/load methods -> `path` is now used everywhere.

### Removed

- (envs/packaging) drop python3.9 support and packaging.

## [0.1.7] - 2025-08-14

### Added

- (pipelines/*) add plaid_blocks.py and sklearn_block_wrappers.py: mechanisms to define ML pipeline on plaid datasets, that satisfy the sklearn conventions.
- (split.py) add mmd_subsample_fn to subsample datasets based on tabular input data.
- (CHANGELOG.md) initiale CHANGELOG.
- PLAID benchmarks and source code.
- (cgns_helper.py) add summarize_cgns_tree function.
- add python 3.13 support.
- (constants.py) locate additional constants to this file for clarity.
- (dataset.py, sample.py) initiate get/set feature_identifiers mechanisms.
- (dataset.py) add method `add_to_dir` to iteratively save `Sample` objects to a directory.

### Changed

- Update repo configuration (actions).
- Update README.
- Update documentation (including configuration and replacing data challenges page with PLAID benchmark one).
- (types/*) improve typing factorization.
- (stats.py) improve OnlineStatistics and Stats classes.

## [0.1.6] - 2025-06-19

### Added

- Add code of conduct and security policy.
- Initiate JOSS paper.

### Changed

- Update repo configuration (actions, PiPy packaging, doc generation, pre_commit_hooks).
- Update readme and licence.
- Update documentation.
- Update typing for compatibility down to python 3.9.

## [0.1.5] - 2025-06-05

### Changed

- Update repo configuration (actions, pre_commit_hooks).
- Update readme and contributing.
- Update documentation, including its configuration.
- Enforce ruff formatting.
- Improve coverage to 100%.

## [0.1.4] - 2025-06-01

### Changed

- Update PiPy packaging.

## [0.1.3] - 2025-06-01

### Changed

- Update PiPy packaging.

## [0.1.2] - 2025-06-01

### Added

- Initiated PiPy packaging.

### Changed

- Configure repo (PR templates, actions, readme).
- Update documentation.
- Update environment.
- (dataset.py): replace `Dataset._load_number_of_samples_` with `plaid.get_number_of_samples`.
- (sample.py): enforce snake_case name convention.

## [0.1.1] - 2025-05-29

### Added

- Migration from [GitLab](https://gitlab.com/drti/plaid).

[unreleased]: https://github.com/PLAID-lib/plaid/compare/0.1.12...HEAD
[0.1.12]: https://github.com/PLAID-lib/plaid/compare/0.1.11...0.1.12
[0.1.11]: https://github.com/PLAID-lib/plaid/compare/0.1.10...0.1.11
[0.1.10]: https://github.com/PLAID-lib/plaid/compare/0.1.9...0.1.10
[0.1.9]: https://github.com/PLAID-lib/plaid/compare/0.1.8...0.1.9
[0.1.8]: https://github.com/PLAID-lib/plaid/compare/0.1.7...0.1.8
[0.1.7]: https://github.com/PLAID-lib/plaid/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/PLAID-lib/plaid/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/PLAID-lib/plaid/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/PLAID-lib/plaid/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/PLAID-lib/plaid/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/PLAID-lib/plaid/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/PLAID-lib/plaid/releases/tag/0.1.1
