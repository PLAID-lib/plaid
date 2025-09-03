# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

### Changed

- (sample/dataset/problem_definition) fix incoherent path argument names in save/load methods -> `path` is now used everywhere
- (plaid/examples) fix circular imports
- (dataset.py) add optional `ids` argument to `from_list_of_samples`
- (sample) add summarize and check_completeness functions
- (dataset) add summarize_features and check_feature_completeness functions
- (Hugging Face bridge) add datasetdict conversion, and simple function for plaid sample init from hf sample
- (pipelines/plaid_blocks.py) add column transformer inverse_transform
- Move to Muscat=2.5.0 (for tests and examples support)
- Update repo configuration (actions: rely more on pypi dependencies, action versions)
- Rename types to remove `Type` from name of types: https://github.com/PLAID-lib/plaid/pull/164

### Removed

## [0.1.7] - 2025-08-14

### Added

- (pipelines/*) add plaid_blocks.py and sklearn_block_wrappers.py: mechanisms to define ML pipeline on plaid datasets, that staisfy the sklearn conventions
- (split.py) add mmd_subsample_fn to subsample datasets based on tabular input data
- (CHANGELOG.md) initiale CHANGELOG
- PLAID benchmarks and source code
- (cgns_helper.py) add summarize_cgns_tree function
- add python 3.13 support
- (constants.py) locate additional constants to this file for clarity
- (dataset.py, sample.py) initiate get/set feature_identifiers mechanisms
- (dataset.py) add method `add_to_dir` to iteratively save `Sample` objects to a directory

### Changed

- Update repo configuration (actions)
- Update README
- Update documentation (including configuration and replacing data challenges page with PLAID benchmark one)
- (types/*) improve typing factorization
- (stats.py) improve OnlineStatistics and Stats classes

## [0.1.6] - 2025-06-19

### Added

- Add code of conduct and security policy
- Initiate JOSS paper

### Changed

- Update repo configuration (actions, PiPy packaging, doc generation, pre_commit_hooks)
- Update readme and licence
- Update documentation
- Update typing for compatibility down to python 3.9

## [0.1.5] - 2025-06-05

### Changed

- Update repo configuration (actions, pre_commit_hooks)
- Update readme and contributing
- Update documentation, including its configuration
- Enforce ruff formatting
- Improve coverage to 100%

## [0.1.4] - 2025-06-01

### Changed

- Update PiPy packaging

## [0.1.3] - 2025-06-01

### Changed

- Update PiPy packaging

## [0.1.2] - 2025-06-01

### Added

- Initiated PiPy packaging

### Changed

- Configure repo (PR templates, actions, readme)
- Update documentation
- Update environment
- (dataset.py): replace `Dataset._load_number_of_samples_` with `plaid.get_number_of_samples`
- (sample.py): enforce snake_case name convention

## [0.1.1] - 2025-05-29

### Added

- Migration from [GitLab](https://gitlab.com/drti/plaid).

[unreleased]: https://github.com/PLAID-lib/plaid/compare/0.1.7...HEAD
[0.1.7]: https://github.com/PLAID-lib/plaid/compare/0.1.6...0.1.7
[0.1.6]: https://github.com/PLAID-lib/plaid/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/PLAID-lib/plaid/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/PLAID-lib/plaid/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/PLAID-lib/plaid/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/PLAID-lib/plaid/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/PLAID-lib/plaid/releases/tag/0.1.1
