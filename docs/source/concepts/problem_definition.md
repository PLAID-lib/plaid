---
title: Problem definition
---

# Problem definition

`ProblemDefinition` defines the feature selection and split configuration used on top of a PLAID dataset.

In the current API, a problem definition stores:

- `name`
- `input_features` (`list[str]`)
- `output_features` (`list[str]`)
- `train_split` and `test_split`

## Basic usage

```python
from plaid import ProblemDefinition

pb = ProblemDefinition(name="regression_1")

pb.add_input_features([
    "Base/Zone/GridCoordinates/CoordinateX",
    "Base/Zone/GridCoordinates/CoordinateY",
])
pb.add_output_features([
    "Base/Zone/VertexFields/pressure",
])

pb.train_split = {"train": [0, 1, 2]}
pb.test_split = {"test": [3, 4]}
```

Feature lists are normalized by the model: entries are converted to strings,
sorted, and checked for duplicates.

## Loading from disk

Load a definition from a dataset path:

```python
pb = ProblemDefinition.from_path("/path/to/plaid_dataset", name="regression_1")
```

At storage level, problem definitions are loaded as a dictionary keyed by name:

```python
from plaid.storage import load_problem_definitions_from_disk

pb_defs = load_problem_definitions_from_disk("/path/to/plaid_dataset")
pb = pb_defs["regression_1"]
```

## Saving

Save to YAML:

```python
pb.save_to_file("problem_definitions/regression_1.yaml")
```

## Notes

- Input/output features are plain strings correspond to CGNS paths.
- Splits are represented by `train_split` and `test_split` dictionaries.
- Split values can be explicit index sequences or the string `"all"`.
- `add_input_features(...)` and `add_output_features(...)` accept either a
  single string or a sequence of strings.