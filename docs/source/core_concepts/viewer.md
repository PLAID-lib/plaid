# Dataset viewer

The dataset viewer is a small trame/VTK web application that lets
you browse PLAID datasets stored on disk and inspect their samples in 3D.
It ships as the `plaid-viewer` console script.

## Architecture

The viewer runs as a single trame server process:

- `plaid.viewer.services.PlaidDatasetService` discovers datasets and
  loads `plaid.Sample` instances. It uses
  `plaid.storage.init_from_disk` to obtain `(dataset_dict,
  converter_dict)` and materialises a sample on demand with
  `converter.to_plaid(dataset, index)`, so every PLAID backend
  (`hf_datasets`, `cgns`, `zarr`, ...) is supported uniformly.
  Hugging Face Hub datasets are also supported: when a dataset id is
  registered as a repo id, the service dispatches to
  `plaid.storage.init_streaming_from_hub` instead, so samples are
  streamed lazily without a full local copy.
- `plaid.viewer.services.ParaviewArtifactService` writes each selected
  sample to a CGNS file (or `.cgns.series` sidecar for time-dependent
  samples) in a per-process cache directory.
- `plaid.viewer.trame_app.server.build_server` assembles the UI
  (Vuetify side drawer with dataset/split/sample selectors and display
  options) and a VTK pipeline: `vtkCGNSReader` → optional cut plane →
  optional threshold → composite-data geometry → mapper/actor.

There is no separate FastAPI backend and no second port: dataset
discovery, CGNS export and the 3D view are all served by trame.

## Launching the viewer

```bash
uv run plaid-viewer --datasets-root /path/to/datasets
```

Useful options:

| Option            | Default     | Description                                                                                      |
| ----------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| `--datasets-root` | *required*  | Directory containing one sub-directory per PLAID dataset. A single-dataset directory also works. |
| `--cache-dir`     | `None`      | Persistent artifact cache. When omitted, an ephemeral temp dir is used and cleaned at shutdown.  |
| `--host`          | `127.0.0.1` | Bind address for the trame HTTP server.                                                          |
| `--port`          | `8080`      | Port exposed by the trame HTTP server.                                                           |
| `--backend-id`    | `disk`      | PLAID backend identifier embedded in sample references and the cache key.                        |
| `--hub-repo`      | `None`      | Hugging Face Hub repo id (`namespace/name`) streamed via `init_streaming_from_hub`. Repeat the flag to pre-register multiple repos. |

Open `http://<host>:<port>/` in your browser.

### Streaming from the Hugging Face Hub

Hub datasets can be added at launch time with `--hub-repo` or from the
running UI through the **Hub** tab in the side drawer (the drawer now
groups the local datasets root and the Hugging Face repo input under a
`Local / Hub` tab selector, hidden when `--disable-root-change` is set).
Each registered repo shows up as a removable chip and as a new entry in
the **Dataset** dropdown. Samples are loaded on demand through
`plaid.storage.init_streaming_from_hub`, so only the selected sample's
shards are fetched.

```bash
# Start with one or more hub datasets pre-registered.
uv run plaid-viewer --hub-repo PLAID-lib/VKI-LS59 --hub-repo PLAID-lib/Rotor37
```

Streaming splits returned by PLAID are forward-only
`datasets.IterableDataset` objects without `__len__`. The viewer adapts
accordingly:

- A `streaming` chip appears in the toolbar to advertise the mode.
- The **Sample** slider starts at a single reachable step and grows by
  one every time the user moves it to the right; each right-arrow press
  consumes the next element from the iterator.
- Revisiting an already-fetched index simply re-renders the cached
  sample; the slider cannot be rewound because the underlying iterator
  cannot.
- Switching split or dataset rebuilds a fresh iterator from the Hub.
- When the stream is exhausted the slider caps at the last consumed
  index and the counter label shows `(end of stream)`.


## Using the UI

The side drawer provides, from top to bottom:

1. **Dataset / Split** - two `VSelect` controls that pick the active
   dataset and split.
2. **Sample** - a `VSlider` over the integer sample index of the current
   split; the selected `sample_id` (and the total count) is shown under
   the slider.
3. **Base** - a `VBtnToggle` with exclusive, mandatory selection: exactly
   one renderable CGNS base exposed by `vtkCGNSReader.GetBaseSelection()`
   is active at any time. Bases that contain
   no `Zone_t` children (for example, a `Global` base storing only
   reference scalars or free-standing tensors) are not rendered but are
   summarised in the **Non-visual bases** accordion further down the
   drawer: each `DataArray_t` is listed with its name, dtype, shape and a
   short value preview.
4. **Field / Colormap / Show edges** - colour the geometry by any point
   or cell array (all point and cell arrays are enabled on the reader
   by default so every field shows up in the dropdown), pick from a set
   of built-in colormaps and optionally overlay wireframe edges.
5. **Cut plane** - toggle a `vtkCutter` and interactively adjust its
   normal and signed offset along that normal (the plane origin is the
   current dataset's bounding-box centre).
6. **Threshold** - toggle a `vtkThreshold` filter on the currently
   selected field and set the `[min, max]` range. Defaults are populated
   from the field's data range.
7. **Select features** - an expandable panel listing the field paths
   available for the current dataset (retrieved from the PLAID metadata
   schema). Toggling checkboxes and clicking **Apply** filters the loaded
   samples down to the selected fields:
   - For disk-backed datasets the selection is forwarded to
     `converter.to_plaid(dataset, index, features=...)`. PLAID expands
     the list internally with
     `plaid.utils.cgns_helper.update_features_for_CGNS_compatibility`
     to preserve the CGNS conventions (coordinates, zones, grid
     locations, etc. that make the kept fields renderable). The
     user-facing selection is first intersected with the active split's
     own feature catalogue, so paths that only live in another split
     (for example a field present in `train` but not in `test`) do not
     trigger a `Missing features` error.
   - For streaming (Hugging Face Hub) datasets the expansion must be
     done ahead of `init_streaming_from_hub`. The viewer calls
     `update_features_for_CGNS_compatibility` itself and hands the
     expanded list to the streaming loader, then invalidates the
     current iterator so the next sample is materialised with the new
     filter.
   The **Clear** / **Select all** buttons in the panel header provide
   shortcuts; an empty selection loads only the geometric support
   (mesh + zones + metadata).
8. **Reset camera** - re-frames the current actor.

The 3D view is a server-side `VtkRemoteView` (images are rendered on the
server and streamed to the browser). Camera manipulation uses the
ParaView-like trackball style:

- Left mouse button: rotate.
- Middle mouse button (or Shift + left): pan.
- Mouse wheel (or right button drag): zoom.

A status line at the bottom of the drawer reports the last action or
error.

## Cache layout

Artifacts are written under:

```
<cache_root>/datasets/<dataset_id>/<split>/<sample_id>/<key_prefix>/
    meshes/                   # one CGNS per timestep (time-dependent)
    meshes.cgns.series        # ParaView file-series sidecar (time-dependent)
    mesh.cgns                 # single static mesh
    metadata.json             # cache key, sample ref, export version, ...
```

The cache key is a SHA-256 of the sample reference, backend id, PLAID
version and `ViewerConfig.export_version`. Re-running the viewer with
the same inputs reuses existing artifacts; bumping `export_version`
invalidates them.

## Programmatic usage

```python
from pathlib import Path
from plaid.viewer.cache import CacheRoot
from plaid.viewer.config import ViewerConfig
from plaid.viewer.services import ParaviewArtifactService, PlaidDatasetService
from plaid.viewer.trame_app.server import build_server

config = ViewerConfig(datasets_root=Path("/path/to/datasets"))
with CacheRoot(persistent_dir=config.cache_dir) as cache:
    datasets = PlaidDatasetService(config)
    artifacts = ParaviewArtifactService(datasets, cache.path)
    server = build_server(datasets, artifacts)
    server.start(host="127.0.0.1", port=8080, open_browser=False)
```
