---
title: Dataset
---

# Dataset

A PLAID [`plaid.containers.dataset.Dataset`](../api_reference.md#api-dataset) is a collection of physics configurations, organized into [`plaid.containers.sample.Sample`](../api_reference.md#api-sample). Each [`plaid.containers.sample.Sample`](../api_reference.md#api-sample) contains all the necessary features to define a specific configuration, including mesh and scalar data.

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

See also: `notebooks/containers/dataset_example`.

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
dataset.add_info("legal", "owner", "CompanyX")
dataset.add_infos("data_production", {"type": "simulation", "simulator": "Z-Set"})
infos = dataset.get_infos()
dataset.print_infos()
```

## Quality checks and summaries

```python
print(dataset.summarize_features())         # coverage of feature names
print(dataset.check_feature_completeness()) # detect missing features per sample
```

## Best practices

- Prefer FeatureIdentifiers for unambiguous selection and stable keys.
- Keep sample IDs contiguous when possible (simplifies slicing and joins).
- For large datasets, consider using `processes_number` when loading from disk to parallelize I/O.
- When building learning tasks, pair `Dataset` with [`plaid.problem_definition.ProblemDefinition`](../api_reference.md#api-problemdefinition) and rely on identifiers for inputs/outputs.
