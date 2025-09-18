---
title: Dataset
---

# Dataset

A PLAID {py:class}`~plaid.containers.dataset.Dataset` is a collection of physics configurations, organized into {py:class}`~plaid.containers.sample.Sample`. Each {py:class}`~plaid.containers.sample.Sample` contains all the necessary features to define a specific configuration, including mesh and scalar data.

## Create and load

- Empty dataset:

```python
from plaid.containers.dataset import Dataset

dataset = Dataset()
print(dataset)  # Dataset(0 samples, 0 scalars, 0 fields)
```

- From directory in PLAID format:

```python
dataset = Dataset.load_from_dir("/path/to/dataset_dir")
```

- From .plaid archive (TAR produced by `Dataset.save`):

```python
dataset = Dataset.load_from_file("/path/to/dataset.plaid")
```

- From a list of Samples (IDs optional):

```python
from plaid.containers.sample import Sample

samples = [Sample(...), Sample(...)]
dataset = Dataset.from_list_of_samples(samples)
```

See also: {doc}`notebooks/containers/dataset_example`.

## Basic usage

- Length, iteration, indexing:

```python
len(dataset)                # number of samples
for sample in dataset: ...  # iterate samples
sample_3 = dataset[3]       # get sample by id
subset = dataset[0:10]      # returns a Dataset with selected ids
```

- Manage samples and IDs:

```python
sid = dataset.add_sample(Sample(...))
dataset.del_sample(sid)
ids = dataset.get_sample_ids()
```

## Discover features across the dataset

```python
dataset.get_scalar_names(ids=None)
dataset.get_time_series_names(ids=None)
dataset.get_field_names(ids=None, zone_name=None, base_name=None)

# Structured, hashable descriptors of features (recommended)
feat_ids = dataset.get_all_features_identifiers(ids=None)
node_ids = dataset.get_all_features_identifiers_by_type("nodes")
```

Learn more about identifiers: {doc}`core_concepts/feature_identifiers`.

## Retrieve features by identifier(s)

```python
from plaid.types import FeatureIdentifier

fid_scalar = FeatureIdentifier({"type": "scalar", "name": "Re"})
fid_field  = FeatureIdentifier({
    "type": "field", "name": "pressure", "base_name": "Base",
    "zone_name": "Zone", "location": "Vertex", "time": 0.0,
})

# One feature for all samples (dict: sample_id -> feature)
scalar_by_sample = dataset.get_feature_from_identifier(fid_scalar)

# Several features per sample (dict: sample_id -> list[feature])
features_by_sample = dataset.get_features_from_identifiers([fid_scalar, fid_field])
```

## Convert to/from tabular data

Extract homogeneous features (same sizes) to a 3D array `(n_samples, n_features, dim_feature)`:

```python
tab = dataset.get_tabular_from_homogeneous_identifiers([fid_scalar, fid_field])
```

Extract and stack features to a 2D array `(n_samples, dim_stacked)`:

```python
tab = dataset.get_tabular_from_stacked_identifiers([fid_scalar, fid_field])
```

Update/add features from tabular data (optionally restricting the output dataset to only those features):

```python
updated = dataset.add_features_from_tabular(
    tabular=tab,
    feature_identifiers=[fid_scalar, fid_field],
    restrict_to_features=True,
)
```

## Merge and extract

- Extract a dataset containing only selected features:

```python
slim = dataset.extract_dataset_from_identifier([fid_field])
```

- Merge entire datasets (append samples) or merge only features:

```python
ids_added = dataset.merge_dataset(other_dataset)          # append samples
merged    = dataset.merge_features(other_dataset)         # union of features
```

## Save to disk

```python
# Save to directory (PLAID format)
dataset._save_to_dir_("/path/to/output_dir")

# Save to .plaid archive (TAR)
dataset.save("/path/to/output.plaid")
```

## Dataset metadata (infos)

Datasets can carry metadata grouped by categories (e.g., legal, data_production):

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
- When building learning tasks, pair `Dataset` with {py:class}`~plaid.problem_definition.ProblemDefinition` and rely on identifiers for inputs/outputs.
