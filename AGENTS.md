# AGENTS.md -- plaid (pyplaid)

## Project identity

**plaid** is the foundational data model library of the [PLAID ecosystem](https://github.com/PLAID-lib).
Published on PyPI as `pyplaid`, it provides a structured format for representing physics
simulation data (meshes, fields, boundary conditions) and abstracts storage backends
(zarr, HuggingFace datasets, CGNS).

Other libraries in the ecosystem depend on plaid.

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
- Check if the change impacts downstream consumers of plaid
- Run the full test suite before proposing changes

### Confidentiality

plaid is a **public** repository. Some downstream libraries in the PLAID ecosystem are not public.
Never mention non-public repository names, internal project names, or confidential details
in any public-facing content (code comments, docstrings, commit messages, PR descriptions,
issues, or documentation).

### Communication rules

- All interactions on this repository (issues, PRs, reviews, comments) must be in **English**.
- Be direct and concise. Avoid compliments, flattery, or filler sentences.

### Efficiency and minimalism

These rules cut token usage, latency, and the volume of generated code without
sacrificing correctness.

**Write the least code that works.** Climb this ladder, stop at the first rung that holds:

1. Does this need to exist at all? Speculative need = skip it, say so in one line (YAGNI).
2. Does the standard library do it? Use it.
3. Does an already-installed dependency solve it? Use it -- never add a new dependency for what a few lines cover.
4. Can it be one line? Make it one line.
5. Only then: the minimum code that works.

- No unrequested abstractions: no interface with a single implementation, no factory for one product, no config for a value that never changes.
- Deletion over addition. The shortest working diff wins. Fewest files possible.
- Mark a deliberate shortcut with a `# NOTE(shortcut):` comment naming its ceiling and upgrade trigger, e.g. `# NOTE(shortcut): O(n) scan, index it if the sample count grows`.
- Non-trivial logic leaves one runnable check behind (an `assert`-based self-check or a small `test_*.py`) -- no heavy fixtures unless asked.

**Communicate tersely.** Why over what; the diff already says what.

- Drop filler, pleasantries, and hedging. Code first, then at most a few short lines: what was skipped and when to add it.
- No tool-call narration, no decorative tables or emoji in explanations, no dumping long raw logs -- quote the shortest decisive line.
- If the explanation is longer than the code, delete the explanation.

**Never simplify away** (these override the ladder): backward compatibility, public API
stability, input validation at trust boundaries, error handling that prevents data loss
or corruption, security, or anything explicitly requested. In scientific computing, data
integrity is non-negotiable -- see Decision priorities above.

## Tech stack

- **Language**: Python 3.11--3.14 (`requires-python = ">=3.11, <3.15"`, all four versions tested in CI)
- **Package manager**: uv (with `pyproject.toml`; dev tools in the `dev` dependency group, viewer dependencies in the `viewer` extra)
- **Build backend**: setuptools with setuptools-scm (dynamic versioning)
- **Linter/formatter**: ruff
- **Test framework**: pytest
- **Documentation**: Zensical (with mkdocstrings for the API reference), published on ReadTheDocs
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
│   ├── __init__.py            <- Public API: Sample, Infos, ProblemDefinition + storage helpers
│   ├── version.py             <- Package version (setuptools-scm)
│   ├── constants.py           <- Global constants
│   ├── problem_definition.py  <- ProblemDefinition (core concept)
│   ├── infos.py               <- Infos (dataset metadata)
│   ├── containers/            <- Sample container + helpers (see nested AGENTS.md)
│   ├── storage/               <- Storage backends: zarr, hf_datasets, cgns (see nested AGENTS.md)
│   ├── types/                 <- Shared type aliases and definitions
│   ├── utils/                 <- Internal helpers (CGNS trees, JSON codecs, worker processes)
│   ├── cli/                   <- plaid-check and plaid-serve entry points, ParaView plugin
│   ├── viewer/                <- Dataset viewer (trame) and its plaid-viewer entry point
│   └── downloadable_examples/ <- Built-in downloadable example datasets
├── tests/                     <- Test suite
├── docs/                      <- Zensical (Markdown) documentation source
└── examples/                  <- Usage examples
```

> Note: the v1.0.0 reorganization removed the top-level `Dataset` re-export and the
> `bridges/`, `pipelines/` and `post/` source packages under `src/plaid/`. Data is now
> handled through `Sample` objects and the `storage` layer. (The top-level `examples/`
> directory still exists — it holds usage scripts, not an importable package.)
> See `docs/source/upgrade_guide.md`.

## Architecture and key concepts

### Core abstractions

| Concept | Module | Description |
|---------|--------|-------------|
| `ProblemDefinition` | `problem_definition.py` | Declares the input and output features and the train/test splits of a learning problem |
| `Infos` | `infos.py` | Dataset metadata (owner, license, data production, number of samples, storage backend, etc.) |
| `Sample` | `containers/sample.py` | One simulation snapshot: mesh + field values (a pydantic `BaseModel`) |

`Sample`, `Infos` and `ProblemDefinition` are re-exported at the top level of the
`plaid` package, together with the helpers `get_number_of_samples` and `get_sample_ids`
from `containers/utils.py` and the main storage entry points (`init_from_disk`,
`download_from_hub`, `load_infos_from_disk`, ...); see `__all__` in `src/plaid/__init__.py`.

### Storage pattern

Storage uses a **Registry pattern** (`storage/registry.py`) to dispatch read/write
operations to the correct backend (zarr, hf_datasets, cgns). Each backend is a class
conforming to the `BackendModule` Protocol defined in `storage/backend_api.py`, with
its `reader.py` and `writer.py`; backend-independent helpers live in `storage/common/`.
Reading/writing a collection of samples is done through this storage layer rather
than through a dedicated `Dataset` class.

## Code conventions

### Formatting and linting

Ruff is configured in `ruff.toml`:
- **Line length**: 88 characters -- but `E501` (line-too-long) is in the `ignore` list, so this limit is enforced by the **formatter** (`ruff format`), not checked by the linter
- **Lint rules**: `D` (docstrings), `E`/`W` (pycodestyle), `F` (pyflakes), `ARG` (unused arguments), `I` (import sorting)
- **Ignored rules**: `E501` (line too long) and `D107` (missing docstring in `__init__`)
- **Docstring convention**: Google style
- **Excluded directories**: `examples/`, `docs/`, `benchmarks/`
- **Per-file ignores**: test files ignore docstring rules (`D`) and `S101` (assert); `__init__.py` files ignore `F401` (unused imports, to allow re-exports)

CI (`lint-format-check.yaml`) runs both `ruff check .` and `ruff format --check .`;
the commands are listed under [Commands](#commands).

### Type hints

- Required on all public functions and methods
- In new code, use modern syntax: `list[str]`, `dict[str, int]`, `X | None`
- Do not introduce deprecated `typing.List` / `typing.Dict`; existing modules still use
  `Optional[X]` / `Union[...]` -- do not rewrite them in unrelated changes

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

## Pull request rules

PR titles **must start with one of the following emojis** to indicate the type of change
(checked by `.github/workflows/pr-title-checker.yml`):

| Emoji | Type |
|-------|------|
| 🐛 | Bug fix |
| 📄 | Documentation |
| 🎉 | New feature or initial commit |
| 🚀 | Performance or deployment |
| ♻️ | Refactor or cleanup |
| 📦 | Packaging or dependency management |

PR checklist (from `.github/pull_request_template.md`):
- Typing enforced
- Documentation updated
- Changelog updated
- Tests and example updates
- Coverage should be 100% (a review goal: CI reports coverage but sets no `fail_under` threshold)

## Commands

```bash
# Install dependencies (dev group + viewer extra, as in CI; tests/viewer needs the extra)
uv sync --extra viewer

# Run tests
uv run pytest

# Check linting
uv run ruff check .

# Auto-fix linting issues
uv run ruff check --fix .

# Format code / check formatting (as in CI)
uv run ruff format .
uv run ruff format --check .

# Build documentation (executes the example notebooks, then runs Zensical;
# system packages needed by the viewer screenshots: see .github/workflows/doc.yml)
uv run bash docs/generate_doc.sh
```

## Contribution workflow

When making changes:

1. Read and understand existing code before modifying
2. Write or update code with type hints
3. Write unit tests for new functionality
4. Update docstrings (Google style)
5. Update the Zensical documentation if functionality changed
6. Run formatter: `uv run ruff format .`
7. Run linter: `uv run ruff check --fix .`
8. Run tests: `uv run pytest`
9. Check if changes are breaking and inform the reviewer if a major version bump is needed
