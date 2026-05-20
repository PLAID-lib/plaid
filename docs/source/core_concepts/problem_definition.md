---
title: Problem definition
---

# Problem definition

{py:class}`~plaid.problem_definition.ProblemDefinition` defines the feature
selection and split configuration used on top of a PLAID dataset.

In the current API, a problem definition stores:

- `name`
- `input_features` (`list[str]`)
- `output_features` (`list[str]`)
- `train_split` and `test_split`

## Basic usage

```python
from plaid.problem_definition import ProblemDefinition

pb = ProblemDefinition(name="regression_1")

pb.add_in_features_identifiers([
    "Base/Zone/GridCoordinates/CoordinateX",
    "Base/Zone/GridCoordinates/CoordinateY",
])
pb.add_out_features_identifiers([
    "Base/Zone/Solution/pressure",
])

pb.train_split = {"train": [0, 1, 2]}
pb.test_split = {"test": [3, 4]}
```

## Loading from disk

Load a definition from a dataset path:

```python
pb = ProblemDefinition.from_path("/path/to/plaid_dataset", name="regression_1")
```

## Saving

Save to YAML:

```python
pb.save_to_file("problem_definitions/regression_1.yaml")
```

## Notes

- Input/output features are plain strings in the current implementation.
- Splits are represented by `train_split` and `test_split` dictionaries.
- Each split dictionary is expected to contain a single split entry.