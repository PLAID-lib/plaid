---
title: Dataset
---

# Dataset

A PLAID {py:class}`~plaid.containers.dataset.Dataset` is a lightweight container/view
that accesses samples through a backend.

In the current API, `Dataset` focuses on:

- loading data from disk,
- exposing a split/view (`indices`),
- delegating sample access to its backend,
- saving a dataset view back to disk.

## Create and load

### Empty dataset

```python
from plaid.containers.dataset import Dataset

dataset = Dataset()
```

### Load from a local directory or archive

Use the factory method:

```python
dataset = Dataset.from_path("/path/to/plaid_dataset", split="train")
```

Or instantiate and load later:

```python
dataset = Dataset(path="/path/to/plaid_dataset", split="train")
dataset.load()
```

`Dataset.load(...)` accepts either:

- a directory path,
- or a tar archive path.

When no split is provided, the default split is `"train"`.

See also: {doc}`../notebooks/containers/dataset_example`.

## Main attributes

- `path`: source dataset path.
- `split`: loaded split name.
- `stage`: optional label (`"training"` or `"evaluating"`).
- `problem_definition`: attached
  {py:class}`~plaid.problem_definition.ProblemDefinition`.
- `indices`: either `"all"` or an explicit integer index array.
- `infos`: normalized metadata dictionary.

## Access samples

```python
len(dataset)        # number of exposed samples
sample0 = dataset[0]
samples = dataset.get_samples()
ids = dataset.get_sample_ids()
```

### About indexing/slicing

`Dataset.__getitem__` delegates directly to the backend. Depending on backend behavior,
a slice can return a backend-native object (for example a `list[Sample]`), not
necessarily another `Dataset` instance.

## Backends

Access the backend with:

```python
backend = dataset.get_backend()
```

The default backend is in-memory, and backend-specific operations (such as adding
samples in memory) are performed on the backend object itself.

## Metadata (`infos`)

Set normalized metadata with:

```python
dataset.set_infos(
    {
        "legal": {"owner": "CompanyX"},
    }
)
```

## Save to disk

Save the current dataset view:

```python
dataset.save_to_dir("/path/to/output_dir")
```