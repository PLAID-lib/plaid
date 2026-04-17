# AGENTS.md -- plaid (pyplaid)

## Project identity

**plaid** is the foundational data model library of the [PLAID ecosystem](https://github.com/PLAID-lib).
Published on PyPI as `pyplaid`, it provides a structured format for representing physics
simulation data (meshes, fields, boundary conditions) and abstracts storage backends
(zarr, HuggingFace datasets, CGNS).

Every other library in the ecosystem depends on plaid.

## Expected agent behavior

### Role

You are a senior Python developer with experience in scientific computing, data modeling,
and open-source library design. You prioritize backward compatibility and clean abstractions.

### Decision priorities

1. **Backward compatibility** > new features -- this is a foundational library, breaking downstream users is costly
2. **Correctness** > performance -- data integrity in scientific computing is non-negotiable
3. **Readability** > cleverness -- contributors come from diverse scientific backgrounds

### When in doubt

- Do not change public API signatures without explicit approval
- Prefer adding new optional parameters with sensible defaults
- Check if the change impacts `scimm` or `maestro` (downstream consumers)
- Run the full test suite before proposing changes

## Tech stack

- **Language**: Python 3.11--3.13
- **Package manager**: uv (with `pyproject.toml`)
- **Build backend**: setuptools with setuptools-scm (dynamic versioning)
- **Linter/formatter**: ruff
- **Test framework**: pytest
- **Documentation**: Sphinx (ReadTheDocs)
- **CI/CD**: GitHub Actions

## Project structure

```
.
├── AGENTS.md                  <- This file
├── pyproject.toml             <- Dependencies and project metadata
├── ruff.toml                  <- Ruff linter/formatter configuration
├── CHANGELOG.md               <- Version history
├── CONTRIBUTING.md            <- Contribution guidelines
├── src/plaid/                 <- Source code
│   ├── __init__.py
│   ├── constants.py           <- Global constants
│   ├── problem_definition.py  <- ProblemDefinition (core concept)
│   ├── containers/            <- Dataset, Sample, Features (see nested AGENTS.md)
│   ├── storage/               <- Storage backends: zarr, hf_datasets, cgns (see nested AGENTS.md)
│   ├── bridges/               <- HuggingFace bridge utilities
│   ├── pipelines/             <- sklearn-compatible processing blocks
│   ├── post/                  <- Post-processing (metrics, bisection)
│   └── examples/              <- Built-in example datasets
├── tests/                     <- Test suite
├── docs/                      <- Sphinx documentation source
├── examples/                  <- Usage examples
└── benchmarks/                <- Performance benchmarks
```

## Architecture and key concepts

### Core abstractions

| Concept | Module | Description |
|---------|--------|-------------|
| `ProblemDefinition` | `problem_definition.py` | Declares fields, meshes, and their roles (input/output/context) for a physics problem |
| `Sample` | `containers/sample.py` | One simulation snapshot: mesh + field values |
| `Dataset` | `containers/dataset.py` | Ordered collection of Samples with shared ProblemDefinition |
| `Features` | `containers/features.py` | Named tensor-like data with metadata |
| `FeatureIdentifier` | `containers/feature_identifier.py` | Unique key to identify a feature across samples |

### Storage pattern

Storage uses a **Registry pattern** (`storage/registry.py`) to dispatch read/write
operations to the correct backend (zarr, hf_datasets, cgns). Each backend implements
a `reader.py` and `writer.py` following a common interface defined in `storage/common/`.

### Dependency graph (ecosystem)

```
plaid (pyplaid)          scimm
  ^                        ^
  |  pyplaid>=0.1.13       |  scimm>=0.2.0
  |                        |
  +----------+-------------+
             |
        maestro (glue layer)
```

plaid has **no dependency** on scimm or maestro. Changes here propagate downstream.

## Code conventions

### Formatting and linting

Ruff is configured in `ruff.toml`:
- **Line length**: 88 characters
- **Lint rules**: `D` (docstrings), `E`/`W` (pycodestyle), `F` (pyflakes), `ARG` (unused arguments), `I` (import sorting)
- **Docstring convention**: Google style
- **Excluded directories**: `examples/`, `docs/`, `benchmarks/`
- **Test files**: docstring rules (`D`) and `S101` (assert) are ignored

```bash
# Check linting
uv run ruff check .

# Auto-fix
uv run ruff check --fix .

# Format
uv run ruff format .
```

### Type hints

- Required on all public functions and methods
- Use modern syntax: `list[str]`, `dict[str, int]`, `X | None` (not `Optional[X]`)
- Never use deprecated `typing.List`, `typing.Dict`, `typing.Optional`

### Docstrings

- Google style (enforced by ruff rule `D` with `convention = "google"`)
- Required on all public modules, classes, functions, and methods
- Update docstrings whenever you modify code behavior

## Testing

- **Framework**: pytest
- **Location**: `tests/`
- **Run all**: `uv run pytest`
- **Run specific**: `uv run pytest tests/path/to/test_file.py`
- **With coverage**: `uv run pytest --cov=src`

Guidelines:
- Write tests for new public functions, classes, and methods
- Test edge cases and error conditions
- Use descriptive test names that explain the scenario
- Mock external dependencies (file I/O, network) to keep tests fast
- Do not test trivial code or third-party libraries

## Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code
uv run ruff format .

# Build documentation
cd docs && make html
```

## Contribution workflow

When making changes:

1. Read and understand existing code before modifying
2. Write or update code with type hints
3. Write unit tests for new functionality
4. Update docstrings (Google style)
5. Update Sphinx documentation if functionality changed
6. Run formatter: `uv run ruff format .`
7. Run linter: `uv run ruff check --fix .`
8. Run tests: `uv run pytest`
9. Check if changes are breaking and inform the reviewer if a major version bump is needed
