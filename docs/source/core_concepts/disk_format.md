---
title: On-disk format
---

# On-disk format

A PLAID [`plaid.containers.dataset.Dataset`](../api_reference.md#api-dataset) is a collection of physics configurations, stored on disk in a human-readable, tool-friendly structure:

```
folder
├── dataset
│   ├── samples
│   │   ├── sample_000000000
│   │   │   ├── meshes/
│   │   │   │   ├── mesh_000000000.cgns
│   │   │   │   ├── mesh_000000001.cgns
│   │   │   └── scalars.csv
│   │   └── sample_yyyyyyyyy
│   └── infos.yaml
└── problem_definition
    ├── problem_infos.yaml
    └── split.json (or split.csv for <=0.1.7)
```

- `dataset/samples/`: one directory per [`plaid.containers.sample.Sample`](../api_reference.md#api-sample).
- `meshes/`: CGNS files for time steps; can be explored in ParaView.
- `scalars.csv`: constant scalars for [`plaid.containers.sample.Sample`](../api_reference.md#api-sample).
- `problem_definition/`: learning task definition and splits.
