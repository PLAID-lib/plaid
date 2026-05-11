---
title: Sample
---

# Sample

{py:class}`~plaid.containers.sample.Sample` represents one observation. It contains:
- globals: name → value
- meshes containing:
  - nodes: mesh node coordinates, that can be located:
    - in different bases
    - in different zones in each base
  - fields, arrays that can be located:
    - in different bases
    - in different zones in each base
    - in different locations in each base/zone among: `Vertex`, `EdgeCenter`, `FaceCenter`, or `CellCenter`

Key APIs include:
- Feature accessors: {py:meth}`~plaid.containers.sample.Sample.get_global`, {py:meth}`~plaid.containers.sample.Sample.get_field`, {py:meth}`~plaid.containers.sample.Sample.get_nodes`
- Feature updates: {py:meth}`~plaid.containers.sample.Sample.add_global`, {py:meth}`~plaid.containers.sample.Sample.add_field`, {py:meth}`~plaid.containers.sample.Sample.set_nodes`, and path-based updates with {py:meth}`~plaid.containers.sample.Sample.add_feature`
- Discovery: {py:meth}`~plaid.containers.sample.Sample.get_global_names`, {py:meth}`~plaid.containers.sample.Sample.get_field_names`, and {py:meth}`~plaid.containers.sample.Sample.get_all_features_identifiers_by_type`

See also: {doc}`../examples_tutorials` for hands-on examples.
