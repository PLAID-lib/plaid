---
title: Upgrade guide
---

# Upgrade guide

This page explains how to upgrade an existing code base to **PLAID v1.0.0**.

PLAID follows [Semantic Versioning](https://semver.org/). The `v1.0.0` release is
the first major release: it consolidates the data model, removes deprecated and
out-of-scope modules, and simplifies several public APIs. As a major release, it
contains **breaking changes** compared to the `0.1.x` series.

The guide is organized by **version jump**. Read the section that matches the
version you are upgrading *from*. For the exhaustive, change-by-change history,
see the [`CHANGELOG.md`](https://github.com/PLAID-lib/plaid/blob/main/CHANGELOG.md).

!!! tip "Upgrade incrementally"
    If you are several versions behind, pin `pyplaid`, upgrade one step at a
    time, and run your test suite between steps. The last release of the `0.x`
    series is **`0.1.15`**; the sections below describe the jump from `0.1.x` to
    `1.0.0`.

---

## Upgrade to v1.0.0 from 0.1.x

`v1.0.0` reorganizes the package. The changes most likely to affect your code are
listed below, with before/after examples.

### Top-level imports

The `Dataset` class is no longer re-exported from the top-level `plaid` package,
a new `Infos` object is exported, and the version string moved module.

```python
# Before (0.1.x)
from plaid import Dataset, Sample, ProblemDefinition
from plaid import __version__            # backed by plaid._version

# After (1.0.0)
from plaid import Sample, ProblemDefinition, Infos
from plaid import __version__            # backed by plaid.version
from plaid.containers.dataset import Dataset   # import Dataset from its module
```

The helpers `get_number_of_samples` and `get_sample_ids` are still exported from
the top-level package.

### `plaid.types`

Several type aliases were removed and a few were renamed.

| Removed in 1.0.0 | Notes / replacement |
| --- | --- |
| `Scalar`, `Field`, `TimeSequence`, `Feature` | feature-level aliases removed |
| `SklearnBlock` | tied to the removed `pipelines` module |
| `FeatureIdentifier` | import from `plaid.containers.feature_identifier` if needed |
| `ArrayDType` | replaced by `ScalarDType` |
| `IndexType` | replaced by `IndexArrayType` |

The public aliases now exported from `plaid.types` are:

```python
from plaid.types import (
    Array,
    ScalarDType,
    IndexArrayType,
    ScalarOrArray,
    ScalarOrArrayOrStr,
    CGNSNode,
    CGNSTree,
)
```

### Removed modules

The following modules were removed from the `plaid` package in `1.0.0`. They were
either out of the scope of the data model or superseded:

| Removed module | What to do instead |
| --- | --- |
| `plaid.pipelines` (`plaid_blocks`, `sklearn_block_wrappers`) | build ML pipelines outside PLAID, on top of the data model |
| `plaid.post` (`bisect`, `metrics`) | compute post-processing / metrics in your own code |
| `plaid.utils.split` | manage dataset splits via `ProblemDefinition` train/test splits |
| `plaid.utils.stats` | compute statistics in your own code |
| `plaid.utils.interpolation` | use an external interpolation routine |
| `plaid.utils.init_with_tabular` | construct samples explicitly |
| `plaid.utils.deprecation`, `plaid.utils.base` | internal helpers, no public replacement |
| `plaid.examples` | see the examples and tutorials in the documentation |

If you imported any of these, remove the import and move the corresponding logic
into your own project, or rely on the supported data-model APIs.

### `ProblemDefinition`

`ProblemDefinition` was rewritten as a compact [pydantic](https://docs.pydantic.dev/)
model. The many `*_features_identifiers` accessors were replaced by two methods,
and YAML key order is now enforced.

```python
# Before (0.1.x)
pb.add_in_features_identifiers([...])
pb.add_out_features_identifiers([...])
pb.set_in_features_identifiers([...])
pb.set_out_features_identifiers([...])
pb.get_in_features_identifiers()
pb.get_out_features_identifiers()

# After (1.0.0)
pb.add_input_features([...])
pb.add_output_features([...])
```

The public surface of `ProblemDefinition` in `1.0.0` is intentionally small:
`from_path`, `add_input_features`, `add_output_features`, `save_to_file`, and the
train/test split accessors (`get_train_split_name`, `get_train_split_indices`,
`get_test_split_name`, `get_test_split_indices`). The previous
`constant_features_identifiers` accessors were removed together with the
in/out identifier accessors.

### Storage / CGNS backend

The constant/variable mechanism used in the CGNS backend reading and writing
paths was removed. If you relied on that distinction at the storage level, review
your read/write code against the current
[storage API](api/storage/backend_api.md).

### New in v1.0.0

`v1.0.0` also introduces new building blocks you can adopt:

- **`plaid.infos`** — a dedicated pydantic `Infos` class, now living at the same
  level as `ProblemDefinition` (see [Infos](concepts/infos.md)).
- **`plaid.viewer`** — an interactive [trame](https://kitware.github.io/trame/)
  application for visual dataset exploration (see [Viewer](concepts/viewer.md)).

---

## Upgrading from an older 0.1.x release

If you are upgrading from a release earlier than `0.1.15`, first move up to
`0.1.15` and account for the intermediate breaking changes documented in the
[`CHANGELOG.md`](https://github.com/PLAID-lib/plaid/blob/main/CHANGELOG.md), in
particular:

- **0.1.15** — `save_to_disk` API simplified: `generators` replaced by
  `sample_constructor` and `ids`.
- **0.1.13** — `get_mesh` renamed to `get_tree`; `get_<x>_assignment` renamed to
  `resolve_<x>` (e.g. `get_time_assignment` → `resolve_time`).
- **0.1.11** — `get_all_mesh_times()` renamed to `get_all_time_values()`;
  `FeatureIdentifier` moved from `plaid.types` to `plaid.containers`; Python 3.10
  support dropped.
- **0.1.10** — `Sample` restructured to store globals at time steps (scalars and
  time series unified into CGNS trees).
- **0.1.8** — `Dataset.from_tabular` → `Dataset.add_features_from_tabular`;
  `Dataset.from_features_identifier` → `Dataset.extract_dataset_from_identifier`;
  `Sample.from_features_identifier` → `Sample.extract_sample_from_identifier`.

Once on `0.1.15`, follow the [Upgrade to v1.0.0 from 0.1.x](#upgrade-to-v100-from-01x)
section above.
