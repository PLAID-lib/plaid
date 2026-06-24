# AGENTS.md -- plaid/containers

This module defines the core data container of the PLAID data model.

## Key classes

| Class | File | Description |
|-------|------|-------------|
| `Sample` | `sample.py` | Single simulation snapshot containing mesh coordinates and field values. Implemented as a pydantic `BaseModel`. This is the main data container exposed by plaid. |
| `DefaultManager` | `managers/default_manager.py` | Manages default values and missing data for features within a sample. |

Helper functions live in `utils.py` (e.g. `get_number_of_samples`, `get_sample_ids`)
and are re-exported at the top level of the `plaid` package.

> Note: the v1.0.0 reorganization removed the `Dataset`, `Features` and
> `FeatureIdentifier` classes. A collection of samples is now read/written through the
> `storage` layer rather than a dedicated `Dataset` class. See `docs/source/upgrade_guide.md`.

## Design constraints

- `Sample` is a **value object** built on pydantic -- keep it focused on holding mesh
  and field data, with minimal business logic. Prefer extracting heavy logic into
  helper functions or dedicated modules.
- `DefaultManager` centralizes default/missing-data handling -- do not duplicate this
  logic inside `Sample`.
- All containers must support **serialization** through the storage backends
  (zarr, hf_datasets, cgns).

## Downstream impact

`Sample` is part of the public API surface consumed by downstream libraries and end
users. Any signature change is a **breaking change** that requires a major version bump.

## Testing

Tests are in `tests/`. When modifying a container class, verify that storage round-trips
(write then read) still produce identical data.
