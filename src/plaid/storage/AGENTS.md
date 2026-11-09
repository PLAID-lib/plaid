# AGENTS.md -- plaid/storage

This module implements the multi-backend storage layer for reading and writing PLAID datasets.

## Architecture

Storage follows a **Registry pattern**:

```
storage/
├── registry.py        <- Dispatches to the correct backend based on its name
├── backend_api.py     <- Backend contract (BackendModule Protocol)
├── callbacks.py       <- Callback contracts (SampleCallbackContext, post-write hooks)
├── reader.py          <- Public read API (init_from_disk, Converter, ...)
├── writer.py          <- Public write API (save_to_disk, push_to_hub, ...)
├── common/            <- Backend-independent helpers
│   ├── reader.py      <- Read infos, problem definitions and metadata (disk and Hub)
│   ├── writer.py      <- Write infos, problem definitions and metadata (disk and Hub)
│   ├── bridge.py      <- Sample <-> flat dict conversion helpers
│   └── preprocessor.py <- Sample flattening, dtype inference, sharded preprocessing
├── zarr/              <- Zarr backend (__init__.py, reader.py, writer.py, bridge.py)
├── hf_datasets/       <- HuggingFace datasets backend (__init__.py, reader.py, writer.py, bridge.py)
└── cgns/              <- CGNS backend (__init__.py, reader.py, writer.py)
```

## How it works

1. The **registry** (`registry.py`) holds a `BACKENDS` dict mapping each backend name
   (`"cgns"`, `"hf_datasets"`, `"zarr"`) to its backend class, exposed through
   `get_backend(name)` and `available_backends()`.
2. The public API delegates to the selected backend: `writer.save_to_disk` takes a
   `backend` argument (default `"hf_datasets"`) and records it in the dataset infos;
   `reader.init_from_disk` reads it back from `Infos.storage_backend`.
3. Each backend exposes, in its `__init__.py`, a backend class (`ZarrBackend`, `HFBackend`,
   `CgnsBackend`) that conforms to the `BackendModule` Protocol in `backend_api.py`, and
   implements the read/write logic in its `reader.py` and `writer.py`.

## Adding a new backend

1. Create a new subdirectory under `storage/` (e.g., `storage/my_format/`).
2. Implement a backend class conforming to the `BackendModule` Protocol
   (`backend_api.py`), with its `reader.py` and `writer.py`, reusing the helpers in `common/`.
3. Register the new backend by adding it to the `BACKENDS` dict in `registry.py`.
4. Add round-trip tests (write then read) to verify data integrity.

## Design constraints

- Backends must be **stateless** -- all configuration is passed through function parameters.
- Read/write operations must preserve **data integrity** exactly (no lossy conversions without explicit user consent).
- The `BackendModule` Protocol is the **contract** -- do not add backend-specific parameters to the public API without updating the contract first.
- Partial reads (`features=[...]`, `indexers={...}`) are supported by the `hf_datasets` and `zarr` backends; the `cgns` backend does not provide them.

## Testing

Each backend should have round-trip tests that write a dataset and read it back, asserting equality. Tests are in `tests/storage/`.
