---
title: Sample
---

# Sample

{py:class}`~plaid.containers.sample.Sample` represents one observation. It contains {doc}`feature_identifiers` among (all optional):
- scalars: name → value
- meshes containing:
  - nodes: mesh node coordinates, that can be located:
    - in different bases
    - in different zones in each base
  - fields, arrays that can be located:
    - in different bases
    - in different zones in each base
    - in different locations in each base/zone among: `Vertex`, `EdgeCenter`, `FaceCenter`, or `CellCenter`

Key APIs include:
- Feature accessors: {py:meth}`~plaid.containers.sample.Sample.get_scalar`, {py:meth}`~plaid.containers.sample.Sample.get_field`, {py:meth}`~plaid.containers.sample.Sample.get_nodes`
- Feature updates: {py:meth}`~plaid.containers.sample.Sample.add_scalar`, {py:meth}`~plaid.containers.sample.Sample.add_field`, {py:meth}`~plaid.containers.sample.Sample.set_nodes`, and high-level identifier-based updates
- Discovery: {py:meth}`~plaid.containers.sample.Sample.get_all_features_identifiers()` or {py:meth}`~plaid.containers.sample.Sample.get_all_features_identifiers_by_type()` to list all available features with their context

See also: {doc}`../examples_tutorials` for hands-on examples.
