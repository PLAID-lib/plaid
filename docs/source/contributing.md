# Contributing

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
- Prefer one class per file
- Aim for simple and flexible API design
- Maintain 100% test coverage

### 2.2. Contributing Process
1. Write local/unit tests for your changes
2. Include examples when adding new features
3. Submit a pull request with your changes

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
- Conda or uv package manager

### 3.2. Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/PLAID-lib/plaid.git
    ```

2. Configure the development environment:

   - Using conda (Windows, macOS and Linux):

    ```bash
    conda env create -n plaid-dev python=3.12 -f environment.yml
    pip install -e . --no-deps
    ```

   - Using uv (Linux):

    ```bash
    uv sync --dev --extra viewer
    ```


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

## 8. Release process

### 8.1. Github

- Create a new release branch from `main` called e.g. `release_0.1.10`
- Update the CHANGELOG
  - Rename the section [Unreleased] to the new version number (e.g., [0.1.10]) followed by the release date (YYYY-MM-DD)
  - Update links at the end of the file
- Create a pull request and request reviews
- Once approved, merge the pull request
- Tag the release (e.g., `git tag 0.1.10`). Important: pull the recently updated `main` branch before creating the tag.
- Create a new release on GitHub
  - Click `Generate release notes`
  - Include a link to the CHANGELOG file at the release tag, e.g.: `https://github.com/PLAID-lib/plaid/blob/0.1.10/CHANGELOG.md`
- Create a new pull request to add a new `Unreleased` section to the CHANGELOG, with sub-sections `Added`, `Changed`, `Fixes`, `Removed`

### 8.2. conda-forge

- Create a fork of https://github.com/conda-forge/plaid-feedstock or sync with the latest changes if you already have a fork
- Update the conda-forge recipe `plaid-feedstock/recipe/meta.yaml`
  - Update the `version` field to the new version number (line 1)
  - Update the SHA256 checksum for the new version (line 9), you can find it in the GitHub action: https://github.com/PLAID-lib/plaid/actions/workflows/checksum_release.yml, don’t take the one in `Digest` section of `Artifacts`, but take the one in the download button
  - Update requirements section if they change in the new version
- Submit a pull request to the conda-forge feedstock repository
- Follow the instructions provided in the PR message

### 8.3. Readthedocs

Just check a version was created for the release tag, it should have automatically triggered a new build on Read the Docs.

## 9. Documentation consistency checks

Before opening a PR that modifies docs or public APIs, run a quick consistency pass:

1. **Build docs locally**

   ```bash
   cd docs
   make html
   ```

2. **Validate API references**
   - Ensure documented methods/classes exist in current source.
   - Ensure renamed/removed APIs are either removed from docs or explicitly marked as legacy.

3. **Validate code snippets/imports**
   - Verify documented imports execute against the current package.
   - Prefer snippets that reflect maintained modules under `src/plaid`.

4. **Validate storage-layout claims**
   - Ensure on-disk format docs match current `plaid.storage` behavior.
   - Cross-check against tests/fixtures when updating storage documentation.
