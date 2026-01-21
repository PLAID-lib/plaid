# Pattern: Nodal vs Cell-Centered Fields

## Context

Scientific datasets may define physical quantities either:
- at mesh nodes (nodal fields), or
- over mesh elements or cells (cell-centered fields).

PLAID conversion scripts must respect this distinction explicitly.

---

## Nodal fields

### Typical characteristics
- One value per mesh node
- Common for displacements, velocities, phase fields
- Often stored as `VertexFields` or node-based arrays

### Code patterns
- Assignment to `mesh.nodeFields`
- Flattened arrays with length equal to number of nodes
- Fields added via `sample.add_field(...)`

---

## Cell-centered fields

### Typical characteristics
- One value per mesh element or cell
- Common for PDE coefficients, fluxes, or averaged quantities
- Often stored as `CellData` or element-based arrays

### Code patterns
- Assignment to `mesh.elemFields`
- Flattened arrays with length equal to number of elements
- Fields appear under `CellData` in CGNS trees

---

## PLAID semantics
- Field location (node vs cell) is part of the data semantics
- Mixing nodal and cell-centered quantities is allowed but must be explicit
- Downstream models may rely on this distinction

---

## Common pitfalls
- Treating cell-centered data as nodal data
- Reshaping arrays without respecting element ordering
- Assuming nodal location by default

---

## When to use this pattern
Use this pattern when:
- datasets originate from PDE solvers,
- coefficients or outputs are defined per cell rather than per node.
