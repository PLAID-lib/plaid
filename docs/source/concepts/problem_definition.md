---
title: Problem definition
---

# Problem definition

`ProblemDefinition` defines the feature selection and split configuration used on top of a PLAID dataset.

In the current API, a problem definition stores:

- `name` (`str`, required)
- `input_features` (`list[str]`, required and non-empty)
- `output_features` (`list[str]`, required and non-empty)
- `train_split` and `test_split` (required)

## Basic usage

```python
from plaid import ProblemDefinition

pb = ProblemDefinition(
    name="regression_1",
    input_features=[
        "Base/Zone/GridCoordinates/CoordinateX",
        "Base/Zone/GridCoordinates/CoordinateY",
    ],
    output_features=[
        "Base/Zone/VertexFields/pressure",
    ],
    train_split={"train": [0, 1, 2]},
    test_split={"test": [3, 4]},
)
```

Feature lists are normalized by the model: entries are converted to strings,
sorted, checked for duplicates, and rejected if empty.

Problem definitions can also be validated from a plain mapping, for instance
after reading YAML:

```python
pb = ProblemDefinition.model_validate(
    {
        "name": "regression_1",
        "input_features": ["Base/Zone/GridCoordinates/CoordinateX"],
        "output_features": ["Base/Zone/VertexFields/pressure"],
        "train_split": {"train": [0, 1, 2]},
        "test_split": {"test": [3, 4]},
    }
)
```

## Loading from disk

Load a definition from a dataset path:

```python
pb = ProblemDefinition.from_path(
    "/path/to/plaid_dataset/problem_definitions/regression_1.yaml"
)
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

- Input/output features are plain strings corresponding to CGNS paths.
- Splits are represented by `train_split` and `test_split` dictionaries and are accessed directly as model attributes.
- Split values can be explicit index sequences or the string `"all"`.
- `add_input_features(...)` and `add_output_features(...)` accept either a
  single string or a sequence of strings after initialization.