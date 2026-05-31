---
title: Sample
---

# Sample

`Sample` represents one observation. It contains:

- globals: named scalar, string or array values stored in a dedicated CGNS base
- meshes containing:
    - nodes: mesh node coordinates, that can be located:
        - in different bases
        - in different zones in each base
    - fields, arrays that can be located:
        - in different bases
        - in different zones in each base
        - in different locations in each base/zone among: `Vertex`, `EdgeCenter`, `FaceCenter`, or `CellCenter`

Key APIs include:
- Feature accessors: `get_global`, `get_field`, `get_nodes`
- Feature updates: `add_global`, `add_field`, `set_nodes`, and path-based updates with `add_feature`
- Discovery: `get_global_names`, `get_field_names`, and `get_all_features_identifiers_by_type`

## Time, base and zone defaults

Most mesh/field methods accept optional `time`, `base` and `zone` arguments.
When they are omitted, PLAID resolves them through the sample's default manager:

- `set_default_time`
- `set_default_base`
- `set_default_zone_base`
- `resolve_time`, `resolve_base`, `resolve_zone`

This is useful for concise code on simple single-time, single-base, single-zone
samples, while still allowing explicit selection for more complex CGNS trees.

## Mesh tree operations

Samples store their mesh and field data as CGNS trees indexed by time.  Important
tree-level methods include:

- `init_tree`, `add_tree`, `set_trees`, `get_tree`, `del_tree`
- `init_base`, `get_base`, `get_base_names`, `del_base`
- `init_zone`, `get_zone`, `get_zone_names`, `get_zone_type`, `del_zone`
- `show_tree`

`get_tree(only_mesh=True)` returns a tree where global values and fields are
removed, keeping only the mesh support.

## Geometry, fields and globals

Geometry accessors:

- `get_nodes`, `set_nodes`
- `get_elements`
- `get_nodal_tags`, `get_element_tags`

Field accessors:

- `add_field`, `get_field`, `get_field_names`, `del_field`

`add_field` currently expects a one-dimensional NumPy array.  It checks that the
array length matches the geometrical support for `Vertex` and `CellCenter`
locations.  Integer arrays may be converted to floating-point arrays for CGNS
compatibility.

Global feature accessors:

- `add_global`, `get_global`, `get_global_names`, `del_global`

## Path-based access

For generic workflows, features can be addressed by CGNS-like paths:

```python
value = sample.get_feature_by_path("Base/Zone/VertexFields/pressure")
sample = sample.update_features_by_path(
    "Base/Zone/VertexFields/pressure",
    new_pressure,
)
```

Path-based updates dispatch internally to globals, fields or coordinates based on
the path structure.

## Save, load and diagnostics

Samples can be saved as sample directories containing CGNS mesh files:

```python
sample.save_to_dir("sample_000000000", overwrite=True)
sample = Sample.load_from_dir("sample_000000000")
```

For large samples, `save_to_dir(..., memory_safe=True)` writes CGNS files through
a subprocess to reduce pyCGNS memory-leak risks.

Diagnostic helpers include:

- `summarize()` for a detailed textual overview;
- `check_completeness()` for a compact feature-completeness report.

See also: {doc}`../examples_tutorials` for hands-on examples.
