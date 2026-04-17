# AGENTS.md -- plaid/containers

This module defines the core data containers of the PLAID data model.

## Key classes

| Class | File | Description |
|-------|------|-------------|
| `Dataset` | `dataset.py` | Ordered collection of `Sample` objects sharing a common `ProblemDefinition`. Main entry point for loading and manipulating simulation data. |
| `Sample` | `sample.py` | Single simulation snapshot containing mesh coordinates and field values as `Features`. |
| `Features` | `features.py` | Named tensor-like container with shape and dtype metadata. Wraps numpy arrays. |
| `FeatureIdentifier` | `feature_identifier.py` | Immutable key (name + location) used to uniquely identify a feature across samples. |
| `DefaultManager` | `managers/default_manager.py` | Manages default values and missing data for features within a dataset. |

## Design constraints

- `Dataset` is a **large class** (~1800 lines). Avoid adding new responsibilities to it. Prefer extracting logic into helper functions or dedicated modules.
- `Sample` and `Features` are **value objects** -- they should remain simple, with minimal business logic.
- `FeatureIdentifier` is **immutable and hashable** -- it is used as dictionary keys throughout the codebase. Do not add mutable state.
- All containers must support **serialization** through the storage backends (zarr, hf_datasets, cgns).

## Downstream impact

These classes are the public API surface consumed by `maestro` and end users. Any signature change is a **breaking change** that requires a major version bump.

## Testing

Tests are in `tests/`. When modifying a container class, verify that storage round-trips (write then read) still produce identical data.
