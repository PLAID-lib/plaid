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

The CGNS backend only contains `infos.yaml`, `data/` and `problem_definitions/`, since all samples are complete.

## Root metadata

- `infos.yaml`: global dataset metadata (including storage backend and number of
  samples per split).
- `variable_schema.yaml`: schema of variable (per-sample) features.
- `cgns_types.yaml`: CGNS typing metadata used by converters.

## Split content

- `constants/<split>/`: split-level constant feature payloads.
- `data/<split>/`: backend-specific variable sample payloads.

Each `constants/<split>/` directory contains:

- `data.mmap`: concatenated bytes for constant numeric/string payloads;
- `layout.json`: byte offsets, shapes and dtypes inside `data.mmap`;
- `constant_schema.yaml`: schema for constant features in that split.

The exact files under `data/<split>/` depend on the backend (`cgns`,
`hf_datasets`, `zarr`).

## Problem definitions

- `problem_definitions/`: optional serialized `ProblemDefinition` files (YAML).

## Loading policy for constants

When metadata are loaded from local disk, numeric constants are kept as
`np.memmap` arrays for memory efficiency.  When metadata are loaded from the
Hugging Face Hub, numeric constants are materialized into in-memory arrays so
that they remain valid after temporary download folders are cleaned up.

## Validation

The `plaid-check` command validates the required on-disk layout and performs
integrity checks on metadata, sample conversion, numeric values, duplicates and
problem definitions:

```bash
plaid-check /path/to/plaid_dataset
plaid-check /path/to/plaid_dataset --split train --json
plaid-check /path/to/plaid_dataset --strict
```

The minimal required layout checked by the CLI is:

- `infos.yaml`
- `variable_schema.yaml`
- `cgns_types.yaml`
- `constants/`
- `data/`
