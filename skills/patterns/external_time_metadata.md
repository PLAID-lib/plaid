# Pattern: External Time Metadata

## Context

In some datasets, the temporal structure is not implicit in file ordering
but defined by external metadata files (e.g. PVD, XML, sidecar text files).

Conversion scripts must explicitly extract and use this metadata to define
PLAID time semantics.

---

## Typical characteristics
- Time values stored in separate metadata files
- Simulation outputs split across multiple files per time step
- File indices do not reliably encode physical time
- Temporal subsampling may be applied

---

## Code patterns
- Explicit parsing of metadata files (e.g. XML, PVD)
- Construction of a time array independent of file iteration
- Calls to `sample.add_tree(..., time=t)` using extracted values
- Assertions to ensure consistency between fields and time

---

## PLAID semantics
- Time is an explicit, meaningful coordinate
- Ordering and spacing of time steps matter
- Global and nodal quantities may both depend on time

---

## Common pitfalls
- Using file index as a proxy for time
- Ignoring metadata-defined time values
- Mixing subsampled and full-resolution time series
- Treating time-varying globals as constants

---

## When to use this pattern
Use this pattern when:
- simulation time is defined outside the main data files,
- or when multiple files correspond to one physical time step.
