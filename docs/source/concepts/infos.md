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
  hardware, computation duration, script, or contact
- `data_description`, for an optional free-form dataset description string
- `num_samples`, as a dictionary keyed by split name, populated by storage writers
- `storage_backend`, as a storage backend identifier, populated by storage writers

## Basic usage

```python
from plaid import Infos

infos = Infos(
    owner="Safran",
    license="proprietary",
    data_production={
        "type": "simulation",
        "physics": "fluid dynamics",
        "simulator": "ExampleSolver",
    },
    data_description="ExampleDescription",
)
```

To inspect the public constructor fields accepted by `Infos`, use:

```python
Infos.print_available_fields()
```

`num_samples` and `storage_backend` are derived from the chosen storage backend
and the saved split contents. They can be omitted when creating an `Infos`
object that will later be passed to `save_to_disk(...)`; PLAID fills them before
writing `infos.yaml`.

## Loading from disk

Load infos directly from an `infos.yaml` file:

```python
infos = Infos.from_path("/path/to/plaid_dataset/infos.yaml")
```

Use `Infos.from_path(...)` when you have the YAML file path. Use
`plaid.storage.load_infos_from_disk("/path/to/plaid_dataset")` when you have the
dataset root directory.

When the path has no suffix, `Infos.from_path(...)` appends `.yaml`. For
example, `Infos.from_path("/path/to/plaid_dataset/infos")` reads
`/path/to/plaid_dataset/infos.yaml`. Existing directories are rejected.

## Saving

Save to YAML:

```python
infos.save_to_file("/path/to/plaid_dataset/infos.yaml")
```

When the path has no suffix, `save_to_file(...)` appends `.yaml`; if a non-YAML
suffix is provided, it is replaced with `.yaml`. Existing directories are
rejected. Direct YAML writing requires complete persisted metadata: `owner`,
`license`, `num_samples`, and `storage_backend`. When using `save_to_disk(...,
infos=...)`, PLAID fills `num_samples` and `storage_backend` automatically before
writing `infos.yaml`.

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
