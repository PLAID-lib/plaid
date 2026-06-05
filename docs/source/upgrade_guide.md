---
title: Upgrade guide
---

# Upgrade guide

This page explains how to upgrade an existing code base to **PLAID v1.0.0**.

PLAID follows [Semantic Versioning](https://semver.org/). The `v1.0.0` release is
the first major release: it consolidates the data model, removes deprecated and
out-of-scope modules, and simplifies several public APIs. As a major release, it
contains **breaking changes**.

The guide is organized by **version jump**. Read the section that matches the
version you are upgrading *from*. For the exhaustive, change-by-change history,
see the [`CHANGELOG.md`](https://github.com/PLAID-lib/plaid/blob/main/CHANGELOG.md).

!!! tip "Upgrade incrementally"
    If you are several versions behind, pin `pyplaid`, upgrade one step at a
    time, and run your test suite between steps. The last release of the `0.x`
    series is **`0.1.15`**; the sections below describe the jump from `0.1.x` to
    `1.0.0`.

!!! info "Related documentation"
    This guide focuses on *what changed* and *how to migrate*. For *how the new
    API works*, see:

    - [Quickstart](quickstart.md) — the new read/write pattern in a nutshell.
    - [Concepts](concepts.md) — [Sample](concepts/sample.md),
      [Problem definition](concepts/problem_definition.md),
      [Infos](concepts/infos.md), [Dataset](concepts/dataset.md),
      [Disk format](concepts/disk_format.md).
    - [Conversion tutorial](tutorials/storage.md) — end-to-end storage workflow
      (save, load, backends, Hub, parallel I/O).
    - [API reference](api/index.md) — in particular
      [`plaid.storage`](api/storage/backend_api.md).

---

## Upgrade to v1.0.0 from 0.1.x

`v1.0.0` reorganizes the package. The changes most likely to affect your code are
listed below, with before/after examples.

### Top-level imports

The single biggest change is that **the `Dataset` class has been removed**: it is
no longer exported from the top-level `plaid` package *and no longer exists as a
module either*. A new `Infos` object is exported, and the version string moved
module. See [Removing the `Dataset` class](#removing-the-dataset-class-use-plaidstorage)
below for the full migration.

```python
# Before (0.1.x)
from plaid import Dataset, Sample, ProblemDefinition
from plaid import __version__            # backed by plaid._version

# After (1.0.0)
from plaid import Sample, ProblemDefinition, Infos
from plaid import __version__            # backed by plaid.version
# `Dataset` no longer exists — there is no `plaid.containers.dataset` module.
# Use the storage helpers instead:
from plaid.storage import save_to_disk, init_from_disk
```

The helpers `get_number_of_samples` and `get_sample_ids` are still exported from
the top-level package.

### Removing the `Dataset` class: use `plaid.storage`

In the `0.1.x` series, `plaid.Dataset` was a **monolithic, in-memory container**:
you built one `Dataset` object, appended every `Sample` to it, kept the whole
collection in RAM, and called `save_to_dir` / `load` on that object.

In `v1.0.0` this class is **removed entirely** — there is no public high-level
dataset container class anymore, and there is no `plaid.containers.dataset`
module to import from. The data model is now centered on three objects —
[`Sample`](concepts/sample.md), [`ProblemDefinition`](concepts/problem_definition.md)
and [`Infos`](concepts/infos.md) — plus the storage helpers in
[`plaid.storage`](api/storage/backend_api.md). A dataset on disk is a shared
metadata layout plus backend-specific sample payloads; loading it back gives you,
**per split**, a backend dataset object and a `Converter` that materializes
individual `Sample` objects lazily.

This is a deliberate shift away from "load the whole dataset into one in-memory
object" toward **backend-agnostic, lazy, per-sample access**, so that large
datasets that do not fit in memory can be streamed sample by sample into ML
pipelines. The concepts are introduced in [Quickstart](quickstart.md) and the
[Dataset concept page](concepts/dataset.md); the end-to-end workflow is in the
[Conversion tutorial](tutorials/storage.md).

#### Writing: build-then-append → `save_to_disk(sample_constructor, ids)`

Instead of building a `Dataset` and appending samples, you provide a
`sample_constructor(id) -> Sample` callable plus an `ids` mapping of split names
to sliceable id sequences. PLAID handles iteration, generator creation and
parallel sharding internally, and writes directly to the chosen backend.

```python
# Before (0.1.x) — everything in memory, then dumped
from plaid import Dataset, Sample

dataset = Dataset()
for raw in raw_items:
    sample = Sample()
    # fill the sample: add_tree, add_field, ...
    dataset.add_sample(sample)
dataset.save_to_dir("my_plaid_dataset")

# After (1.0.0) — lazy, per-sample, backend-aware
from plaid import Sample
from plaid.storage import save_to_disk

def sample_constructor(sample_id):
    sample = Sample()
    # fill the sample: add_tree, add_field, ...
    return sample

save_to_disk(
    "my_plaid_dataset",
    sample_constructor=sample_constructor,
    ids={"train": [0, 1, 2], "test": [3, 4]},
    backend="zarr",   # one of "hf_datasets", "cgns", "zarr"
)
```

See the [Conversion tutorial](tutorials/storage.md) for a complete example
(including `num_proc` parallel writing and `push_to_hub`) and the
[writer API](api/storage/writer.md).

#### Reading: `Dataset.load(...)` → `init_from_disk(...)` + converter

Loading no longer returns a single object you index into. It returns a
dictionary of backend datasets and a dictionary of converters, one per split.
You materialize a `Sample` on demand with `converter.to_plaid(dataset, idx)`.

```python
# Before (0.1.x)
from plaid import Dataset

dataset = Dataset()
dataset.load("my_plaid_dataset")
sample = dataset[0]
n = len(dataset)

# After (1.0.0)
from plaid.storage import init_from_disk

datasetdict, converterdict = init_from_disk("my_plaid_dataset")
dataset = datasetdict["train"]
converter = converterdict["train"]

sample = converter.to_plaid(dataset, 0)   # materialize one Sample lazily
n = len(dataset)
```

The same shape is used for the Hub (`download_from_hub`,
`init_streaming_from_hub`). See the [Dataset concept page](concepts/dataset.md),
the [reader API](api/storage/reader.md), and the [backend API](api/storage/backend_api.md).

#### Operation-by-operation map

| `0.1.x` — `Dataset` method | `1.0.0` — replacement |
| --- | --- |
| `Dataset()` + `add_sample` / `add_samples` / `from_list_of_samples` | `save_to_disk(sample_constructor=..., ids=...)` |
| `Dataset.save_to_dir(path)` / `add_to_dir` | `save_to_disk(path, sample_constructor=..., ids=...)` |
| `Dataset.load(path)` | `init_from_disk(path)` → `(datasetdict, converterdict)` |
| `dataset[i]` / `get_samples()` | `converter.to_plaid(dataset, i)` |
| `len(dataset)` / `get_number_of_samples()` | `len(dataset)` (per-split backend object) |
| `dataset.set_infos(...)` / `get_infos()` | pass [`Infos`](concepts/infos.md) to `save_to_disk(infos=...)`; read back with `Infos.from_path(path)` |
| persisting a `ProblemDefinition` with the dataset | `save_to_disk(..., pb_defs=...)`; read back with `load_problem_definitions_from_disk(path)` |
| `Dataset.add_features_from_tabular` (ex-`from_tabular`) | build the corresponding `Sample` objects in `sample_constructor` |
| `Dataset.extract_dataset_from_identifier` | request features at read time: `converter.to_plaid(dataset, i, features=[...])` |
| `Dataset.get_tabular_from_stacked_identifiers` | gather features yourself from the materialized `Sample` objects |
| `plaid.examples` | `plaid.downloadable_examples` |
| change backend (e.g. CGNS → HF) | `init_from_disk` then `save_to_disk` with the new `backend` (see the [Conversion tutorial](tutorials/storage.md)) |

If you only need a subset of features or spatial indices, the converter supports
`features=[...]` and `indexers={...}` for partial reads on the `hf_datasets` and
`zarr` backends — see the [Conversion tutorial](tutorials/storage.md#indexed-extraction-with-indexers).

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

If you imported any of these, remove the import and move the corresponding logic
into your own project, or rely on the supported data-model APIs.

### `ProblemDefinition`

`ProblemDefinition` was rewritten as a compact [pydantic](https://docs.pydantic.dev/)
model with four required fields — `input_features`, `output_features`,
`train_split` and `test_split`. The many `*_features_identifiers` accessors were
collapsed into two methods, splits became plain model attributes, and YAML key
order is now enforced on save.

```python
# Before (0.1.x)
pb.add_in_features_identifiers([...])
pb.add_out_features_identifiers([...])
pb.set_in_features_identifiers([...])
pb.set_out_features_identifiers([...])
pb.get_in_features_identifiers()
pb.get_out_features_identifiers()
pb.get_split("train")            # split accessors

# After (1.0.0)
from plaid import ProblemDefinition

pb = ProblemDefinition(
    input_features=["Base/Zone/GridCoordinates/CoordinateX"],
    output_features=["Base/Zone/VertexFields/pressure"],
    train_split={"train": [0, 1, 2]},
    test_split={"test": [3, 4]},
)
pb.add_input_features([...])
pb.add_output_features([...])
pb.train_split                   # direct attribute access
pb.test_split
```

The public surface of `ProblemDefinition` in `1.0.0` is intentionally small:
`from_path`, `model_validate`, `add_input_features`, `add_output_features`,
`save_to_file`, and the four model fields (`input_features`, `output_features`,
`train_split`, `test_split`). The previous `constant_features_identifiers`
accessors and the `get_*_split_*` / `set_*_split_*` helpers were removed
together with the in/out identifier accessors; splits are now read and assigned
directly via the `train_split` / `test_split` attributes, and feature lists are
normalized (stringified, sorted, deduplicated, non-empty) by pydantic
validators. The problem name is no longer stored in the model — on disk it is
the YAML filename stem, in memory it is the dictionary key returned by
`load_problem_definitions_from_disk`. See the
[Problem definition concept page](concepts/problem_definition.md) and the
[`problem_definition` API](api/problem_definition.md).

### Storage / CGNS backend

The constant/variable mechanism used in the CGNS backend reading and writing
paths was removed. If you relied on that distinction at the storage level, review
your read/write code against the current
[backend API](api/storage/backend_api.md) and the
[CGNS backend API](api/storage/cgns/index.md). The on-disk layout written by
`save_to_disk` (shared metadata + per-backend payloads) is described in the
[Disk format concept page](concepts/disk_format.md), and the three backends
(`hf_datasets`, `cgns`, `zarr`) are compared in the
[Conversion tutorial](tutorials/storage.md#choosing-a-backend).

### New in v1.0.0

`v1.0.0` also introduces new building blocks you can adopt:

- **`plaid.infos`** — a dedicated pydantic `Infos` class, now living at the same
  level as `ProblemDefinition` (see [Infos](concepts/infos.md)).
- **`plaid.viewer`** — an interactive [trame](https://kitware.github.io/trame/)
  application for visual dataset exploration (see [Viewer](concepts/viewer.md)).
- **`plaid-check`** — a CLI tool that validates the integrity of a local PLAID
  dataset (on-disk layout, `infos.yaml`, splits, sample conversion, invalid
  numeric values, duplicated samples, and optional problem definitions); see
  [Dataset check](concepts/check.md).

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
