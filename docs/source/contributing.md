# Contributing

- [1. Contributor License Agreement (CLA)](#1-contributor-license-agreement-cla)
- [2. How to contribute](#2-how-to-contribute)
  - [2.1. Coding Standards](#21-coding-standards)
  - [2.2. Contributing Process](#22-contributing-process)
  - [2.3. Issue Guidelines](#23-issue-guidelines)
- [3. Development setup](#3-development-setup)
  - [3.1. Prerequisites](#31-prerequisites)
  - [3.2. Installation Steps](#32-installation-steps)
  - [3.3. Important Note](#33-important-note)
- [4. Tests and examples](#4-tests-and-examples)
- [5. Documentation](#5-documentation)
- [6. Formatting and linting with Ruff](#6-formatting-and-linting-with-ruff)
- [7. Setting up pre-commit](#7-setting-up-pre-commit)
- [8. Backward-compatibility guidelines](#8-backward-compatibility-guidelines)
- [9. Release process](#9-release-process)
  - [9.1. Github](#91-github)
  - [9.2. conda-forge](#92-conda-forge)
  - [9.3. Readthedocs](#93-readthedocs)

## 1. Contributor License Agreement (CLA)

When you make your first contribution to this project, you will be automatically prompted to sign our Contributor License Agreement (CLA) through GitHub. This is a one-time process that you'll need to complete before your contributions can be accepted.

By signing the CLA, you agree to:

- Grant perpetual, worldwide, non-exclusive licenses for copyright and patent
- Represent that you have the rights to contribute, originality, and will disclose third-party restrictions
- Provide contributions "AS IS" without warranties

The agreement will be presented automatically through GitHub's interface when you make your first pull request.

## 2. How to contribute

### 2.1. Coding Standards
- Follow PEP 8 and PEP 257 guidelines
- Use snake_case naming convention
- Keep lines under 80 characters
- Prefer one class per file
- Aim for simple and flexible API design
- Maintain 100% test coverage

### 2.2. Contributing Process
1. Review the backward-compatibility guidelines
2. Write local/unit tests for your changes
3. Include examples when adding new features
4. Submit a pull request with your changes

### 2.3. Issue Guidelines
When reporting issues:
- Provide clear reproduction steps
- Describe expected behavior
- Include PLAID version information
- Attach relevant logs

For feature requests:
- Describe the feature in detail
- Explain the use cases
- Provide examples if possible

## 3. Development setup

### 3.1. Prerequisites

- Git
- Python (`>=3.10, <3.14`)
- Conda package manager

### 3.2. Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/PLAID-lib/plaid.git
    ```

2. Configure the development environment:
   - Using conda (recommended):

    ```bash
    conda env create -f environment.yml
    ```

   - Manual setup: Install dependencies listed in `environment.yml`

3. Install the library in editable mode:

    ```bash
    pip install -e .
    ```

### 3.3. Important Note

The development dependency [**Muscat=2.5**](https://muscat.readthedocs.io/) is only available on [``conda-forge``](https://anaconda.org/conda-forge/muscat) and not on [``PyPi``](https://pypi.org/project/muscat). Therefore, using a conda environment is required to:

- Run tests
- Execute examples
- Compile documentation


## 4. Tests and examples

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

## 5. Documentation

To compile locally the documentation, you can run:

```bash
cd docs
make html
```

Various notebooks are executed during compilation. The documentation can then be explored in ``docs/_build/html``.

## 6. Formatting and linting with Ruff

We use [**Ruff**](https://docs.astral.sh/ruff/) for linting and formatting.

The configuration is defined in `ruff.toml`, and some folders like `docs/` and `examples/` are excluded from checks.

You can run Ruff manually as follows:

```bash
ruff --config ruff.toml check . --fix      # auto-fix linting issues
ruff --config ruff.toml format .           # auto-format code
```

## 7. Setting up pre-commit

Pre-commit is configured to run the following hooks:

- Ruff check
- Ruff format
- Pytest

The selected hooks are defined in the `.pre-commit-config.yaml` file.

To run all hooks manually on the full codebase:

```bash
pre-commit run --all-files
```

You can also run (once):

```bash
pre-commit install
```

## 8. Backward-compatibility guidelines

Follow these concrete rules to preserve retro-compatibility for both the public API and on-disk formats.

- Public API
  - Prefer additive changes: add new functions, methods or keyword-only arguments instead of changing existing call signatures.
  - Mark any removed or renamed public function/method/argument with the `@deprecated` utilities and document the replacement and planned removal version.
  - When changing return types, keep a migration path (helper converting old format to new) and emit a deprecation warning for callers of the old format.
  - Add unit tests that exercise the old behaviour and the migration path.
  - Update the changelog and the release notes with clear migration steps and examples.

- On-disk formats (YAML/JSON/CSV, etc.)
  - Version any written problem/dataset metadata with a `version` field. When loading, accept older versions and migrate them in-memory.
  - Keep readers tolerant: accept both previous keys and new keys (using deprecation warnings when reading old keys). Don't break on extra unknown keys.
  - Write automated migration code for disk formats when the format changes. Prefer writing newer-format files while still supporting read of older files.
  - Add round-trip tests: save a structure with the current writer, then read it back; and also read legacy files and compare expected migrated structure.

- General process
  - Open an issue describing the compatibility impact for any breaking change and link it to the PR.
  - Add deprecation warnings at least one minor release before removal.
  - Keep `examples/` and `docs/` updated with migration examples and sample commands.

These guidelines help keep PLAID users' code and datasets working across releases. See `docs/` and issues `#97`, `#14` for past discussions and examples of migrations.

## 9. Release process

### 9.1. Github

- Create a new release branch from `main` called e.g. `release_0.1.10`
- Update the CHANGELOG
  - Rename the section [Unreleased] to the new version number (e.g., [0.1.10]) followed by the release date (YYYY-MM-DD)
  - Update links at the end of the file
- Create a pull request and request reviews
- Once approved, merge the pull request
- Tag the release (e.g., `git tag 0.1.10`)
- Create a new release on GitHub
  - Click `Generate release notes`
  - Include a link to the CHANGELOG file at the release tag, e.g.: `https://github.com/PLAID-lib/plaid/blob/0.1.10/CHANGELOG.md`
- Create a new pull request to add a new `Unreleased` section to the CHANGELOG, with sub-sections `Added`, `Changed`, `Fixes`, `Removed`

### 9.2. conda-forge

- Create a fork of https://github.com/conda-forge/plaid-feedstock or sync with the latest changes if you already have a fork
- Update the conda-forge recipe `plaid-feedstock/recipe/meta.yaml`
  - Update the `version` field to the new version number (line 1)
  - Update the SHA256 checksum for the new version (line 9), you can find it in the GitHub action: https://github.com/PLAID-lib/plaid/actions/workflows/checksum_release.yml, don’t take the one in `Digest` section of `Artifacts`, but take the one in the download button
  - Update requirements section if they change in the new version
- Submit a pull request to the conda-forge feedstock repository
- Follow the instructions provided in the PR message

### 9.3. Readthedocs

Just check a version was created for the release tag, it should have automatically triggered a new build on Read the Docs.
