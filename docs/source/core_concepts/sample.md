---
title: Sample
---

# Sample

[`plaid.containers.sample.Sample`](../api_reference.md#api-sample) represents one observation. It contains:
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
- Feature accessors: `Sample.get_scalar`, `Sample.get_field`, `Sample.get_nodes`
- Feature updates: `Sample.add_scalar`, `Sample.add_field`, `Sample.set_nodes`, and high-level identifier-based updates
- Discovery: `Sample.get_all_features_identifiers()` or `Sample.get_all_features_identifiers_by_type()` to list all available features with their context

See also: [Examples and Tutorials](../examples_tutorials.md) for hands-on examples.
