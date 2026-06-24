# :material-book-open-page-variant-outline:{ .lg .middle } Concepts

PLAID relies on the CGNS standard for representing complex physics meshes and
uses human-readable metadata files such as YAML for dataset schemas, problem
definitions, and dataset information.

The current public API is centered on `Sample`,
`ProblemDefinition` and `Infos`, and the storage helpers
from `storage`. Datasets are loaded and saved through backend-specific
collections plus converters.

For practical examples, see the [Examples & Tutorials](examples_tutorials.md) pages.

## Concept pages

* [Sample](concepts/sample.md)
* [Dataset](concepts/dataset.md)
* [Problem definition](concepts/problem_definition.md)
* [Infos](concepts/infos.md)
* [Default values](concepts/defaults.md)
* [Disk format](concepts/disk_format.md)
* [Serve API](concepts/serve.md)
* [Viewer](concepts/viewer.md)
