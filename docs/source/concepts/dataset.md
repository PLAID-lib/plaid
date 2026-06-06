---
title: Dataset
---

# Dataset

In the current PLAID API, there is no public high-level dataset container class.
A PLAID dataset is represented on disk by a shared metadata layout plus backend-specific sample payloads.  Loading a dataset returns:

- a dictionary of backend datasets, one per split;
- a dictionary of `Converter` objects, one per split.

This keeps storage concerns explicit while providing a common way to materialize
backend-native samples as PLAID `Sample`
objects.


## Storage backends

Persistent backends currently used for disk and Hub workflows are:

- `hf_datasets`, with efficient zero-copy instatiation,
- `cgns`, the human-readable backend (samples can by opened with Paraview),
- `zarr`, with powerful large-scale capabilities.


## Save a dataset

Datasets are written with `save_to_disk`.  The user
provides:

- `sample_constructor(id) -> Sample`, a callable returning one `Sample`;
- `ids`, a dictionary mapping split names to sliceable sequences of identifiers;
- a persistent backend: `"hf_datasets"`, `"cgns"`, or `"zarr"`.

```python
from plaid import Sample
from plaid.storage import save_to_disk

def sample_constructor(sample_id):
    sample = Sample()
    # Fill the sample: add_tree, add_global, add_field, ...
    return sample

save_to_disk(
    "my_plaid_dataset",
    sample_constructor=sample_constructor,
    ids={"train": [0, 1, 2], "test": [3, 4]},
    backend="zarr",
)
```

## Load from disk

Use `init_from_disk`:

```python
from plaid.storage import init_from_disk

datasetdict, converterdict = init_from_disk("my_plaid_dataset")

dataset = datasetdict["train"]
converter = converterdict["train"]
```

Backend dataset objects expose backend-specific behavior:

- `hf_datasets` returns Hugging Face `datasets.Dataset` objects;
- `cgns` returns local `CGNSDataset` objects containing `Sample` directories;
- `zarr` returns `ZarrDataset` objects;
- streaming from the Hub returns iterable datasets when supported.

## Converter API

The converter normalizes backend-specific data access:

```python
plaid_sample = converter.to_plaid(dataset, idx=0)
sample_dict = converter.to_dict(dataset, idx=0)
```

For non-CGNS backends, `to_plaid(...)` reconstructs a `Sample` from dictionaries and shared
metadata.  For the CGNS backend, samples are already PLAID `Sample` objects.

Selected features can be requested when supported by the backend:

```python
sample = converter.to_plaid(
    dataset,
    idx=0,
    features=["Base/Zone/VertexFields/pressure"],
)
```

Spatial index extraction is possible with `indexers`:

```python
sample = converter.to_plaid(
    dataset,
    idx=0,
    features=["Base/Zone/VertexFields/pressure"],
    indexers={"Base/Zone/VertexFields/pressure": [0, 10, 20]},
)
```


## Metadata and problem definitions

`save_to_disk(...)` writes `infos.yaml` and can also persist one or more
`ProblemDefinition` objects. For non-CGNS backends, it also writes shared
schemas, CGNS types, and constants. The CGNS backend stores self-contained
samples and therefore does not write these derived metadata files.

The dataset-level `infos.yaml` payload is represented by
[`Infos`](infos.md). It stores metadata such as legal ownership, licensing,
data production context, data description, split sample counts, and the storage
backend. The `infos` argument accepts an `Infos` instance:

```python
from plaid import ProblemDefinition
from plaid.infos import Infos
from plaid.storage import save_to_disk

pb_def = ProblemDefinition(
    input_features=["Global/input"],
    output_features=["Base/Zone/VertexFields/pressure"],
    train_split={"train": [0, 1, 2]},
    test_split={"test": "all"},
)
infos = Infos(
    owner="CompanyX",
    license="proprietary",
    data_description="Example dataset with three training samples.",
)

save_to_disk(
    "my_plaid_dataset",
    sample_constructor=sample_constructor,
    ids={"train": [0, 1, 2]},
    infos=infos,
    pb_defs={"regression_1": pb_def},
)
```

The metadata and problem definitions can be loaded later with:

```python
from plaid.infos import Infos
from plaid.storage import load_problem_definitions_from_disk

infos = Infos.from_path("my_plaid_dataset/infos.yaml")
pb_defs = load_problem_definitions_from_disk("my_plaid_dataset")
pb_def = pb_defs["regression_1"]
```

See also [Infos](infos.md) and [Problem definition](problem_definition.md) for
details on each metadata object.