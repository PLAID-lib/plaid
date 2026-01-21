# Pattern: Trajectory Datasets

## What you typically see in code
- Nested loops over files, trajectories, and time steps
- One `Sample` object created per trajectory
- Repeated calls to `add_tree(time=...)`

## PLAID semantics
- A `Sample` represents one physical trajectory
- Time steps are explicit and ordered
- Fields are attached to a given time

## Common pitfalls
- Mixing multiple trajectories into one `Sample`
- Treating the trajectory index as a batch dimension
