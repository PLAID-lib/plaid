---
title: Feature identifiers
---

# Feature identifiers

Feature identifiers are a concise, unambiguous way to point to any feature in PLAID. They replace legacy name-only APIs (now deprecated) and make it possible to uniquely address features across time steps, bases, zones and locations.

- A feature is one of: scalar, time_series, field, nodes.
- A FeatureIdentifier is a small dictionary that encodes the feature type and, when relevant, its context (e.g., base, zone, location, time).

Why this matters:
- Names alone can be ambiguous (e.g., a field called "pressure" may exist at several locations, times, or zones). Identifiers remove ambiguity and make operations deterministic and hashable.
- Identifiers are stable keys, so they can be used in sets, dicts, and sorting. See the underlying implementation in {py:class}`~plaid.types.feature_types.FeatureIdentifier`.
- Discussion and design notes are available in the project discussion: [Feature identifier concept](https://github.com/orgs/PLAID-lib/discussions/107).

## Structure

FeatureIdentifier is a `dict[str, str | float]` with a mandatory `type` key. Depending on the feature type, other keys are required or optional:

- scalar: `{"type": "scalar", "name": <str>}`
- time_series: `{"type": "time_series", "name": <str>}`
- field: `{"type": "field", "name": <str>, "base_name": <str>, "zone_name": <str>, "location": <str>, "time": <float>}`
  - `location` must be one of: `Vertex`, `EdgeCenter`, `FaceCenter`, `CellCenter`.
  - `base_name`, `zone_name`, `location`, `time` are optional if default value mechanics apply (see {doc}`defaults`).
- nodes: `{"type": "nodes", "base_name": <str>, "zone_name": <str>, "time": <float>}`

Notes:
- Time must be a float when present.
- FeatureIdentifier is hashable and orderable (internally sorted), enabling deduplication and stable sorting.

## Examples

Minimal identifiers:

```python
from plaid.types import FeatureIdentifier

fid_scalar = FeatureIdentifier({"type": "scalar", "name": "Re"})
fid_ts     = FeatureIdentifier({"type": "time_series", "name": "load_curve"})

fid_field = FeatureIdentifier({
    "type": "field",
    "name": "pressure",
    "base_name": "Base",
    "zone_name": "Zone",
    "location": "Vertex",
    "time": 0.0,
})

fid_nodes = FeatureIdentifier({
    "type": "nodes",
    "base_name": "Base",
    "zone_name": "Zone",
    "time": 0.0,
})
```

## Using identifiers with {py:class}`~plaid.containers.sample.Sample`

The {py:class}`~plaid.containers.sample.Sample` container exposes helpers to retrieve, update, extract and merge features via identifiers.

```python
from plaid.containers.sample import Sample

sample = Sample(path)

# 1) Retrieve one feature
u = sample.get_feature_from_identifier(fid_field)

# 2) Retrieve several features
features = sample.get_features_from_identifiers([fid_scalar, fid_field])

# 3) Update one or several features
updated = sample.update_features_from_identifier(fid_scalar, 0.5)            # scalar
updated = sample.update_features_from_identifier([fid_field], [u_new])       # field

# 4) Extract a sub-sample containing only selected features
sub = sample.extract_sample_from_identifier([fid_field, fid_nodes])

# 5) Merge all features from another sample
merged = sample.merge_features(sub)
```

`Sample` also offers string identifiers for convenience:

```python
u = sample.get_feature_from_string_identifier(
    "field::pressure/Base/Zone/Vertex/0.0"
)

Re = sample.get_feature_from_string_identifier("scalar::Re")
```

String format is: `<type>::<detail1>/<detail2>/...`. The order is fixed per type. If `time` is provided, it is parsed as float.

## Using identifiers with {py:class}`~plaid.problem_definition.ProblemDefinition`

{py:class}`~plaid.problem_definition.ProblemDefinition` stores learning inputs/outputs as lists of FeatureIdentifiers and offers utilities to add and filter them.

```python
from plaid.problem_definition import ProblemDefinition

problem = ProblemDefinition()

problem.add_in_feature_identifier(fid_scalar)
problem.add_in_features_identifiers([fid_field])

problem.add_out_feature_identifier(fid_nodes)

ins  = problem.get_in_features_identifiers()
outs = problem.get_out_features_identifiers()

# Filtering among registered identifiers
subset_in  = problem.filter_in_features_identifiers([fid_scalar, fid_nodes])
subset_out = problem.filter_out_features_identifiers([fid_nodes])
```

Legacy name-based methods (e.g., `add_input_scalars_names`) are deprecated; prefer the identifier-based ones.

## Best practices

- Always include enough context to disambiguate a feature. For fields/nodes on multiple bases/zones/times, set all relevant keys.
- Use {py:meth}`~plaid.containers.sample.Sample.get_all_features_identifiers()` to introspect what identifiers exist in a sample.
- Use sets to deduplicate identifiers safely: `set(list_of_identifiers)`.
- When authoring problem definitions on disk, {py:meth}`~plaid.problem_definition.ProblemDefinition._save_to_dir_` persists identifiers under `problem_definition/problem_infos.yaml` (keys `input_features` and `output_features`).

## See also

- {doc}`../core_concepts`
- {doc}`defaults`
- {doc}`../notebooks`
