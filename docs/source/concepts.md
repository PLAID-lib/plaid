# Concepts

PLAID is a data model and library for organizing samples of heterogeneous physics data and defining
learning problems on physics datasets.

PLAID relies on the CGNS standard for representing complex physics meshes and
uses human-readable metadata files such as YAML for dataset schemas, problem
definitions, and dataset information.

The current public API is centered on `plaid.containers.sample.Sample`,
`plaid.problem_definition.ProblemDefinition`, and the storage helpers
from `plaid.storage`. Datasets are loaded and saved through backend-specific
collections plus converters.

For practical examples, see the [examples and tutorials](examples_tutorials.md) pages.

## Concept pages

* [Sample](concepts/sample.md)
* [Dataset](concepts/dataset.md)
* [Problem definition](concepts/problem_definition.md)
* [Defaults](concepts/defaults.md)
* [Disk format](concepts/disk_format.md)
* [Viewer](concepts/viewer.md)
