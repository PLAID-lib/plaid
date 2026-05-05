# AGENTS.md -- plaid/storage

This module implements the multi-backend storage layer for reading and writing PLAID datasets.

## Architecture

Storage follows a **Registry pattern**:

```
storage/
├── registry.py        <- Dispatches to the correct backend based on format
├── reader.py          <- Public read API (delegates to backend readers)
├── writer.py          <- Public write API (delegates to backend writers)
├── common/            <- Abstract interfaces and shared utilities
│   ├── reader.py      <- Base reader interface
│   ├── writer.py      <- Base writer interface
│   ├── bridge.py      <- Format conversion helpers
│   └── preprocessor.py
├── zarr/              <- Zarr backend (reader.py, writer.py, bridge.py)
├── hf_datasets/       <- HuggingFace datasets backend (reader.py, writer.py, bridge.py)
└── cgns/              <- CGNS backend (reader.py, writer.py)
```

## How it works

1. The **registry** (`registry.py`) maps format identifiers to backend modules.
2. The public `reader.py` and `writer.py` at the top level accept a format parameter and delegate to the appropriate backend.
3. Each backend implements the interfaces defined in `common/reader.py` and `common/writer.py`.

## Adding a new backend

1. Create a new subdirectory under `storage/` (e.g., `storage/my_format/`).
2. Implement `reader.py` and `writer.py` following the interfaces in `common/`.
3. Register the new backend in `registry.py`.
4. Add round-trip tests (write then read) to verify data integrity.

## Design constraints

- Backends must be **stateless** -- all configuration is passed through function parameters.
- Read/write operations must preserve **data integrity** exactly (no lossy conversions without explicit user consent).
- The `common/` interfaces are the **contract** -- do not add backend-specific parameters to the public API without updating the contract first.
- `zarr` is the primary backend and the most feature-complete. Use it as the reference when implementing others.

## Testing

Each backend should have round-trip tests that write a dataset and read it back, asserting equality. Tests are in `tests/`.
