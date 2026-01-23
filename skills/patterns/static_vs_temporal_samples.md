# Pattern: Static vs Temporal Samples

## Context

PLAID datasets may represent either:
- **static samples** (single geometry or state), or
- **temporal samples** (time-dependent evolution of a system).

Conversion scripts should make this distinction explicit, as it strongly
affects how `Sample` objects are constructed.

---

## Static samples

### Typical characteristics
- No explicit time axis
- One `Sample` corresponds to one geometry or configuration
- Fields are added once and do not vary in time
- Common in geometry datasets, shape repositories, or steady simulations

### Code patterns
- No loop over time
- A single call to `sample.add_tree(...)`
- Fields attached directly to the mesh or tree
- Generator yields one `Sample` per input geometry

### PLAID semantics
- `Sample` represents a single, time-independent state
- Temporal semantics are absent by design
- Fields are interpreted as static quantities

### Common pitfalls
- Introducing artificial time steps where none exist
- Treating multiple geometries as time steps of one `Sample`

---

## Temporal samples

### Typical characteristics
- Explicit time axis provided by the dataset
- One `Sample` corresponds to one physical trajectory
- Fields vary over time and are associated with specific time steps
- Common in transient simulations and dynamical systems

### Code patterns
- Loop over time steps
- Repeated calls to `sample.add_tree(time=...)`
- Fields added with an associated time
- Generator yields one `Sample` per trajectory

### PLAID semantics
- `Sample` represents a full temporal evolution
- Time ordering is explicit and meaningful
- Fields are interpreted as time-dependent quantities

### Common pitfalls
- Mixing multiple trajectories into one `Sample`
- Treating time indices as batch dimensions

---

## Choosing between static and temporal samples

When writing a conversion script:
- Use **static samples** if the dataset provides independent configurations
  with no temporal meaning.
- Use **temporal samples** if the dataset encodes evolution over time for
  a given system.

This choice should be reflected directly in the structure of the conversion
code and the construction of `Sample` objects.
