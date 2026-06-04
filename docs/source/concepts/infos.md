---
title: Infos
---

# Infos

`Infos` defines the structured metadata stored in the `infos.yaml` file at the
root of a PLAID dataset.

In the current API, infos stores:

- `legal`, with required `owner` and `license` entries
- `data_production`, for optional production context such as simulator,
  hardware, contact, or location
- `data_description`, for optional dataset description entries such as the
  number of samples, DOE, inputs, and outputs
- `num_samples`, as a required dictionary keyed by split name
- `storage_backend`, as a required storage backend identifier

## Basic usage

```python
from plaid.infos import DataProduction, Infos, Legal

infos = Infos(
    legal=Legal(owner="Safran", license="proprietary"),
    num_samples={"train": 10, "test": 5},
    storage_backend="zarr",
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
        "legal": {
            "owner": "Safran",
            "license": "proprietary",
        },
        "num_samples": {"train": 10, "test": 5},
        "storage_backend": "zarr",
    }
)
```

## Loading from disk

Load infos from a dataset path or directly from an `infos.yaml` file:

```python
infos = Infos.from_path("/path/to/plaid_dataset")
```

When a directory is provided, `Infos.from_path(...)` looks for `infos.yaml`
inside that directory.

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
owner = infos.legal.owner
backend = infos.storage_backend
payload = infos.model_dump(exclude_none=True)
```

## Notes

- `legal.owner`, `legal.license`, `num_samples`, and `storage_backend` are required when validating complete infos.
- `num_samples` and `storage_backend` are overwritten with the actual saved dataset values when `save_to_disk(..., infos=...)` is called before writing `infos.yaml`.
- Unknown keys are rejected during validation.
- `save_to_file(...)` writes YAML using the standard infos key order.
