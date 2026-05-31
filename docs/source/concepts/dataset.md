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

- `hf_datasets`, with power zero-copy instatiation,
- `cgns`, the human-readable backend (samples can by open with Paraview),
- `zarr`, with powerfull large-scale capabilities.


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

`save_to_disk(...)` writes shared metadata (`infos.yaml`, schemas, CGNS types,
constants) and can also persist one or more `ProblemDefinition` objects:

```python
from plaid import ProblemDefinition
from plaid.storage import save_to_disk

pb_def = ProblemDefinition(name="regression_1")

save_to_disk(
    "my_plaid_dataset",
    sample_constructor=sample_constructor,
    ids={"train": [0, 1, 2]},
    infos={"legal": {"owner": "CompanyX", "license": "proprietary"}},
    pb_defs=pb_def,
)
```

Problem definitions can be loaded later with:

```python
from plaid.storage import load_problem_definitions_from_disk

pb_defs = load_problem_definitions_from_disk("my_plaid_dataset")
pb_def = pb_defs["regression_1"]
```