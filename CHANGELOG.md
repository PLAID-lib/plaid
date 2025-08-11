# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- (CHANGELOG.md) initiale CHANGELOG
- PLAID benchmarks and source code
- (cgns_helper.py) add summarize_cgns_tree function
- add python 3.13 support
- (constants.py) locate additional constants to this file for clarity
- (dataset.py, sample.py) initiate get/set feature_identifiers mechanisms

### Changed

- Update repo configuration (actions)
- Update readme
- Update documentation (including replacing data challenges page with PLAID benchmark one)
- (types/*) improve typing factorization
- (stats.py) improve OnlineStatistics and Stats classes

### Removed

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

[unreleased]: https://github.com/PLAID-lib/plaid/compare/0.1.6...HEAD
[0.1.6]: https://github.com/PLAID-lib/plaid/compare/0.1.5...0.1.6
[0.1.5]: https://github.com/PLAID-lib/plaid/compare/0.1.4...0.1.5
[0.1.4]: https://github.com/PLAID-lib/plaid/compare/0.1.3...0.1.4
[0.1.3]: https://github.com/PLAID-lib/plaid/compare/0.1.2...0.1.3
[0.1.2]: https://github.com/PLAID-lib/plaid/compare/0.1.1...0.1.2
[0.1.1]: https://github.com/PLAID-lib/plaid/releases/tag/0.1.1

