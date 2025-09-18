---
title: Problem definition
---

# Problem definition

{py:class}`~plaid.problem_definition.ProblemDefinition` gathers all the information defining a learning problem:
- task: e.g., regression or classification
- inputs: list of FeatureIdentifiers
- outputs: list of FeatureIdentifiers
- split: arbitrary named splits (train/val/test, etc.) stored as JSON

Typical usage:

```python
from plaid.problem_definition import ProblemDefinition
from plaid.types import FeatureIdentifier

pb = ProblemDefinition()
pb.set_task("regression")

pb.add_in_feature_identifier(FeatureIdentifier({"type": "scalar", "name": "Re"}))
pb.add_out_feature_identifier(FeatureIdentifier({
    "type": "field", "name": "pressure", "base_name": "Base", "zone_name": "Zone", "location": "Vertex", "time": 0.0
}))

splits = {"train": [0, 1, 2], "test": [3, 4]}
pb.set_split(splits)

pb._save_to_dir_("problem_definition")
```

{py:class}`~plaid.problem_definition.ProblemDefinition` supports filtering helpers to intersect existing inputs/outputs with a candidate list of identifiers.
