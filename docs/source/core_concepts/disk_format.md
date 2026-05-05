---
title: On-disk format
---

# On-disk format

PLAID storage is backend-oriented. A saved dataset folder contains shared metadata
at the root and split data written by the selected backend.

Typical structure:

```
dataset_root/
├── infos.yaml
├── variable_schema.yaml
├── cgns_types.yaml
├── constants/
│   ├── train/
│   └── test/
├── data/
│   ├── train/
│   └── test/
└── problem_definitions/
```

## Root metadata

- `infos.yaml`: global dataset metadata (including storage backend and number of
  samples per split).
- `variable_schema.yaml`: schema of variable (per-sample) features.
- `cgns_types.yaml`: CGNS typing metadata used by converters.

## Split content

- `constants/<split>/`: split-level constant feature payloads.
- `data/<split>/`: backend-specific variable sample payloads.

The exact files under `data/<split>/` depend on the backend (`cgns`,
`hf_datasets`, `zarr`).

## Problem definitions

- `problem_definitions/`: serialized
  {py:class}`~plaid.problem_definition.ProblemDefinition` files (YAML).

## Notes

- The current layout is produced by the storage writer API
  ({py:meth}`plaid.storage.save_to_disk`).
- Historical layouts from older PLAID versions may differ.
