---
title: Dataset check
---

# Dataset check

`plaid-check` validates the integrity of a local PLAID dataset.

It checks:

- required on-disk files and directories;
- `infos.yaml`, metadata and split sample counts;
- sample conversion through the declared storage backend;
- invalid numeric values such as `None`, empty arrays, `NaN` and `Inf`;
- duplicated samples;
- optional `problem_definitions/` feature names, splits and indices.

## Basic usage

```bash
plaid-check /path/to/plaid_dataset
```

A valid dataset prints an `[OK]` line and returns exit code `0`.

## Options

Check only selected splits:

```bash
plaid-check /path/to/plaid_dataset --split train --split test
```

Check only selected problem definitions:

```bash
plaid-check /path/to/plaid_dataset --problem-definition regression_500
```

Emit a machine-readable report:

```bash
plaid-check /path/to/plaid_dataset --json
```

Make warnings fail the command:

```bash
plaid-check /path/to/plaid_dataset --strict
```

## Report format

Messages are reported with a severity, a stable code, a location and a short
description.  Errors return exit code `1`; warnings return exit code `2` only in
strict mode.

## Notes

- For CGNS datasets, only `infos.yaml` and `data/` are required at the root.
- For other backends, metadata files and `constants/` are checked as well.
- Without `--problem-definition`, all discovered problem definitions are checked.
- In JSON mode, progress bars are disabled to keep the output parseable.