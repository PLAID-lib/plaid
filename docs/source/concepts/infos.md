---
title: Infos
---

# Infos

`Infos` defines the structured metadata stored in the `infos.yaml` file at the
root of a PLAID dataset.

In the current API, infos stores:

- `owner` and `license`, required string entries describing the dataset
  ownership and licensing
- `data_production`, for optional production context such as simulator,
  hardware, contact, or location
- `data_description`, for optional dataset description entries such as the
  number of samples, DOE, inputs, and outputs
- `num_samples`, as a dictionary keyed by split name, populated by storage writers
- `storage_backend`, as a storage backend identifier, populated by storage writers

## Basic usage

```python
from plaid.infos import DataProduction, Infos

infos = Infos(
    owner="Safran",
    license="proprietary",
    data_production=DataProduction(
        type="simulation",
        physics="fluid dynamics",
        simulator="ExampleSolver",
    ),
    data_description="ExampleDescription",
)
```

Infos can also be built from a plain mapping, for instance after reading YAML:

```python
infos = Infos.model_validate(
    {
        "owner": "Safran",
        "license": "proprietary",
    }
)
```

`num_samples` and `storage_backend` are derived from the chosen storage backend
and the saved split contents. They can be omitted when creating an `Infos`
object that will later be passed to `save_to_disk(...)`; PLAID fills them before
writing `infos.yaml`.

## Loading from disk

Load infos from a complete dataset path or directly from an `infos.yaml` file:

```python
infos = Infos.from_path("/path/to/plaid_dataset")
```

When a directory is provided, `Infos.from_path(...)` looks for `infos.yaml`
inside that directory. By default, loading from disk requires the persisted
storage metadata (`num_samples` and `storage_backend`) to be present. To load a
draft infos file that has not been produced by `save_to_disk(...)`, use:

```python
infos = Infos.from_path("/path/to/draft/infos.yaml", require_persisted=False)
```

## Saving

Save to YAML:

```python
infos.save_to_file("/path/to/plaid_dataset/infos.yaml")
```

If a directory path is provided, the file is saved as `infos.yaml` inside that
directory.

## Typed access and serialization

`Infos` is a Pydantic model. Access metadata through typed attributes and use
Pydantic serialization when a plain mapping is needed:

```python
owner = infos.owner
backend = infos.storage_backend
payload = infos.model_dump(exclude_none=True)
```

## Notes

- `owner` and `license` are required when creating infos.
- `num_samples` and `storage_backend` are required when loading persisted dataset infos.
- `num_samples` and `storage_backend` are overwritten with the actual saved dataset values when `save_to_disk(..., infos=...)` is called before writing `infos.yaml`.
- Unknown keys are rejected during validation.
- `save_to_file(...)` writes YAML using the standard infos key order.
