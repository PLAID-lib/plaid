# Dataset Conversion Examples

This folder contains **working examples** of converting external datasets
into the PLAID format.

Each script in this folder:
- was used to generate a real PLAID dataset,
- targets a **specific dataset and input format**,
- prioritizes correctness, clarity, and explicit semantics over generality.

These examples are meant to help users understand *how* to write PLAID
conversion code and to serve as reference material for tooling that assists
with PLAID usage.

---

## Scope and philosophy

The scripts in this folder are intentionally **dataset-specific**.

They:
- encode assumptions about the input data layout,
- rely on domain knowledge (e.g. mesh structure, time semantics),
- may use external scientific libraries to parse or construct data.

They are **not** intended to be:
- generic converters,
- reusable pipelines,
- or minimal tutorial examples.

For introductory or API-focused examples, see the other subfolders in
`examples/`.

---

## External dependencies

Some conversion scripts depend on libraries that are **not part of PLAID’s
core dependencies** (as declared in `pyproject.toml`).

These external dependencies are:
- required only to *convert* the corresponding dataset,
- not required to *load or use* the resulting PLAID dataset,
- intentionally not added to PLAID’s dependency list.

Users should inspect the imports at the top of each script to identify any
dataset-specific requirements.

---

## What to expect from a conversion script

A typical conversion script will:

- read data from one or more input files (e.g. HDF5, NetCDF),
- construct meshes or trees required by PLAID,
- assemble `Sample` objects with appropriate time and feature semantics,
- define a `ProblemDefinition`,
- write the resulting dataset to disk and optionally publish it.

Explicit control flow and dataset-specific logic are preferred over hidden
abstractions.

---

## Notes for contributors

When adding a new conversion example:

- add a single script per dataset,
- keep dataset assumptions explicit in the code,
- avoid introducing new PLAID APIs solely for one dataset,
- do not refactor existing scripts to fit a uniform style.

Optional but encouraged:
- add a short header docstring describing the dataset and its structure,
- document any non-obvious PLAID semantics (e.g. initial conditions, trajectories).

---

## Intended audience

This folder is intended for:
- users converting their own scientific datasets to PLAID,
- contributors wanting to understand real-world PLAID usage,
- automated tools (including LLM-based assistants) that analyze PLAID code.

These examples reflect **how PLAID is used in practice**, not idealized or
simplified workflows.
