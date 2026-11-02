# Dataset viewer

`plaid-viewer` is a small web application for opening a PLAID dataset and
looking at its samples in 3D. It is meant for exploration: choose where the
data comes from, choose what part of the sample you want to see, then inspect
the mesh and fields interactively.

## Start the viewer

From a terminal, run:

```bash
plaid-viewer
```

The viewer needs the optional `viewer` dependencies (`pip install
"pyplaid[viewer]"`). From a `uv`-managed development checkout, use
`uv run plaid-viewer` instead.

You can also start directly from a local datasets folder:

```bash
plaid-viewer --datasets-root /path/to/datasets
```

Then open the address shown by the command, usually:

```text
http://127.0.0.1:8080/
```

Useful launch options:

| Option | What it does |
| --- | --- |
| `--datasets-root /path/to/datasets` | Opens a local folder containing PLAID datasets. |
| `--hub-repo namespace/name` | Adds a Hugging Face Hub dataset to stream. Can be repeated. |
| `--dataset-id name` | Selects this dataset when the viewer starts. |
| `--browse-roots /path ...` | Limits the folders the root field and folder browser can open (home directory by default). |
| `--disable-root-change` | Hides the datasets root controls. |
| `--disable-dataset-change` | Hides the dataset selector, pinning the selected dataset. |
| `--host 0.0.0.0` | Makes the viewer reachable from another machine. |
| `--port 8080` | Changes the web server port. |

Without `--datasets-root`, the viewer reopens the last local root selected in
the interface.

## Run with Docker

The repository includes a Docker image definition for the browser-based
viewer. Build it locally from the repository root (no prebuilt image is
published by this setup):

```bash
cd /absolute/path/to/plaid
docker build -f docker/Dockerfile -t plaid-viewer:local .
```

The image installs the checkout, not the PyPI release, with the `viewer`
dependencies from `uv.lock`. It uses Python 3.12 and CPU-based Mesa/EGL
rendering, so no GPU, X server, or display forwarding is required. The
runtime runs as the non-root `viewer` user (UID 1000). The default package
version is `0.0.0+docker` because Git history is excluded from the build
context. For a release checkout, pass its Python package version explicitly,
for example `--build-arg PLAID_VERSION=1.0.0`.

Start the viewer with a read-only mount of your datasets directory:

```bash
docker run --rm --init \
  -p 127.0.0.1:8080:8080 \
  --mount type=bind,source=/absolute/path/to/datasets,target=/datasets,readonly \
  plaid-viewer:local
```

Open `http://127.0.0.1:8080/` in your host browser. The mounted directory must
exist and be readable/traversable by UID 1000. On WSL, use Linux paths
accessible to the Docker daemon. Paths entered in the viewer are container
paths, such as `/datasets`, not paths on the host. The folder browser is
restricted to `/datasets` by default.

Alternatively, the included Compose configuration provides the same
read-only mount and loopback-only port binding:

```bash
export PLAID_DATASETS_ROOT=/absolute/path/to/datasets
docker compose -f docker/compose.yaml up --build
# Stop and remove the service afterwards:
docker compose -f docker/compose.yaml down
```

Arguments after the image name replace its default arguments. For example,
to stream a Hub dataset without mounting local data:

```bash
docker run --rm --init -p 127.0.0.1:8080:8080 plaid-viewer:local \
  --port 8080 --hub-repo PhysArena/VKI-LS59
```

The entry point always supplies `--host 0.0.0.0` for container networking.
Keep the host port bound to `127.0.0.1` unless you intentionally expose the
viewer to a trusted network; this setup does not add authentication or TLS.
For local datasets, preserve `--datasets-root /datasets --browse-roots
/datasets` when replacing the default arguments. To change the host port,
use, for example, `-p 127.0.0.1:8090:8080` without changing the container port.

Preferences and Hub downloads are discarded when a container is removed.
Optionally persist them by adding a named home volume:
`--mount type=volume,source=plaid-viewer-home,target=/home/viewer`.
Rendering/export artifacts remain temporary under `/tmp`.

**ParaView limitation:** the image does not include the ParaView desktop
application. **Open in ParaView** cannot launch ParaView on the host from
this container. Use the browser viewer, download a CGNS export and open it
manually in host ParaView, or run `plaid-viewer` directly on the host when
you need that integration.

### Validate the image

After building, run this from the checkout with a running Docker daemon:

```bash
python3 docker/smoke_test_viewer_image.py plaid-viewer:local
```

The smoke test starts the image, checks its HTTP endpoint, non-root user,
and read-only dataset mount, then loads and renders a real CGNS sample
using the viewer's VTK pipeline. It removes the test container on success
or failure. The build-only GitHub Actions workflow runs the same test;
it does not publish an image.

## Follow the steps in the side panel

![Sample 3D View tab showing a sample colored by a field](../images/viewer.jpg)

The left panel groups dataset selection, sample loading, rendering options,
and the current sample's Globals into separate cards.

### Select a datasets root and dataset

Choose where the datasets come from.

- **Local**: enter the path to a folder on your computer. You can also click
  the folder button to browse, then validate with the check button.
- **Hub**: enter a Hugging Face Hub repository name such as
  `PhysArena/VKI-LS59`, then click the plus button. Added Hub datasets appear
  as chips and can be removed from the same area.

If you started the viewer with `--datasets-root` or `--hub-repo`, the source may
already be filled in. The root controls and dataset selector are hidden by
`--disable-root-change` and `--disable-dataset-change`.

Pick the **Dataset** you want to inspect.

For a **local** dataset, click **Open in ParaView** to launch ParaView with the
PLAID plugin connected to a loopback-only `plaid-serve` data API. The viewer
chooses the first available API port from **8081–8090**, keeping it close to the
default viewer port (8080), and keeps the server running until that ParaView
instance exits (or the viewer shuts down). If all ports in this range are in use
or denied, the launch reports an error instead of using a high-numbered port.
Your firewall must allow the selected API port. ParaView must be installed
and available as `paraview` on your `PATH`, or you must set `PARAVIEW_EXEC` to
the ParaView executable **before starting `plaid-viewer`**. For example, when
running from WSL with ParaView installed on Windows:

```bash
export PARAVIEW_EXEC=/mnt/c/Users/User/Apps/ParaView-6.2.0-Windows-Python3.12-msvc2017-AMD64/bin/paraview.exe
uv run plaid-viewer
```

Replace the example path with the location of your ParaView installation.
The launch script adds and displays a **Plaid Explorer** source in ParaView
automatically so you can inspect and compare samples.
Hub-streamed datasets are not supported by `plaid-serve` and cannot be opened
with this button.

The list depends on the selected source tab: local datasets are shown on the
**Local** tab, while streamed Hub datasets are shown on the **Hub** tab.

### Load a sample: split, Globals, base and fields

Choose a **Split**, for example `train`, `test`, or any other split provided by
the dataset.

Choose which high-level part of the sample to load:

- Click an **Available Bases** button to display its mesh. Click the active
  button again to deselect it.
- Enable **Sample Globals** to load global values associated with the sample.
  Loaded values appear in the **Sample Globals Output** card.

By default, the viewer starts light: it does not load every field immediately.
Pick a base and/or enable Sample Globals first, then choose fields.

When the dataset exposes selectable fields, including datasets streamed from
the Hub, **Pre-select browsable field features** chooses which fields are
loaded. This is useful for large datasets because you can avoid loading
everything at once.

- Click **Load all** to select and load every available field.
- Expand **Select features** to tick only the fields you need.
- Click **Apply** after changing the checked fields.
- Click **Clear** to uncheck the current selection, then **Apply** to load no
  fields (the mesh can still be shown).

For streamed Hub datasets, choosing a Base initially loads only the mesh.
Select fields here and click **Apply** before **Next** to include them in the
stream. Applying a new selection restarts the stream from the first sample;
click **Next** again to load it with the selected fields.

### Select a sample

- For local datasets, enter a zero-based sample index in **Select Sample**,
  use its small +/− buttons to step one sample at a time, or drag the slider.
- For streamed Hub datasets, click **Next** to fetch the next sample. Streaming
  is forward-only: the viewer loads samples one after another instead of
  knowing the full list in advance.

The message under this control reports the loaded sample, the streaming
step, or the latest error.

### Configure 3D rendering

The **3D Rendering options** card appears when a base is selected. Choose a
**Field to plot** to color the 3D view.

If no field is selected, the viewer still shows the mesh, but without scalar
coloring. When a field is selected, a color legend appears in the 3D view.

The same card also contains the colormap, edge visibility, iso contour, and
camera reset controls described in [Rendering options](#rendering-options).

The **Min** and **Max** controls below **Colormap** independently set the
color scale limits. Each displays the limit currently used by the 3D colors
and legend. Choose **Dynamic** (the default) to follow the current field,
**Fix** to enter a numeric limit, or **Grow only** to retain the lowest
minimum or highest maximum encountered while moving between samples and time
steps. Changing the plotted field starts a new Grow-only range. A fixed limit
must be finite and the minimum must stay below the maximum; an invalid entry
leaves the last valid color scale in use and shows an error.

## Explore Globals across splits

Switch from **Sample 3D View** to the **Dataset Globals** tab.

### Extract Globals

In **Data Extraction**, choose **First Time Step Globals** or **Last Time Step
Globals** (last by default) and click **Extract Globals**. This reads the
selected time step's `Global/` values in every sample of every local split;
timeless Globals remain available. It does not alter the feature selection
used by the sample view. Changing the time step discards the extracted
Globals: click **Extract Globals** again to reload them.

Numeric Global arrays with multiple values are expanded into scalar components
named with 1-based suffixes (for example, `Global/foo_1`, `Global/foo_2`).
Text Globals (for example, case names) are kept as table columns and point
labels, but are never plotted. Extracted values are retained even if they are
arrays or missing. Extraction may take time for large datasets.

**Streaming Hub datasets are not supported** in the Globals tab; the extract
button is disabled for these datasets.

### Select Globals

Nothing is selected after an extraction. The **Selected Globals** list drives
the parallel axes, the X/Y/Z choices, the selected-Globals CSV and the skrub
report. Enter a regular expression in **Filter Globals** to show only matching
Globals; an invalid expression displays an error and matches nothing.
**Check** and **Uncheck** affect only the visible Globals, so **Check** with an
empty filter selects them all. Hidden selections remain unchanged. Clear the
filter to show the full list again.

### Plot Globals

In **Plot**, tick one or more **Splits** and choose a **Plot type**: **1D**,
**2D**, **3D**, or **parallel**, then click **Draw plot**. Any change to the
configuration clears the plot until **Draw plot** is clicked again, so heavy
configurations are never drawn by accident.

- Scatter plots (1D, 2D, 3D) use one, two, or three numeric Globals chosen in
  **X**, **Y** and **Z** among the selected Globals (among all of them when none
  is selected). In 1D the horizontal axis is the sample id. Points use
  different colors and shapes for each split. Enable **Label points** and pick
  **Label from** to annotate points with their sample id or another Global;
  labelling more than about 1,000 points can be slow.
- Parallel plots use the selected Globals as axes. Each axis is scaled
  independently from 0 to 1 for comparison across different units; hover over
  a point to see the original value. **Parallel renderer** switches between
  partial-line plotting (`lines`), limited to 500 samples across the selected
  splits, and Plotly's interactive `parcoords`. `parcoords` needs a value for
  every selected Global, so samples with missing selected values are omitted
  in that mode.

The Dataset Globals view below shows extracted values plotted in 3D (the side
panel is collapsed with the menu button to widen the plot).

![Dataset Globals tab with a 3D scatter plot of extracted Globals](../images/viewerglobals.jpg)

Scatter plots skip samples with missing, non-finite, or non-scalar values on
the chosen axes. `lines` parallel plots show the available parts of each
sample's line, leaving gaps for missing axes and marking isolated values;
samples without any plottable values are skipped. Time-companion (`*_times`)
columns are not plotted. All plots use interactive Plotly views: pan, zoom,
reset, and export the plot with the plot toolbar; rotate the 3D scatter with
the mouse.

### Select samples

In **1D** and **2D**, click points to toggle their selection, or select
multiple points with Plotly's box or lasso tools. In `parcoords`, brush one or
more axes to select lines. **Selected rows** lists each matching split and
sample id, with a count, in the format of a problem-definition split: the copy
button next to the count copies the list, ready to paste under `train_split:`
or `test_split:`. **Download selected samples (CSV)** saves the selected
samples with every extracted Global. The button is disabled when no points are
selected or no plotted lines match the brush. Changing the plot or extracting
again clears the selection.

### Export and report

- **Download all Globals (CSV)**, in **Data Extraction**, saves every extracted
  sample across all splits, with `split` and `sample_id` columns and vector
  Globals expanded into numbered components.
- **Download selected Globals (CSV)**, in **Export & Report**, saves the same
  rows with only the selected Globals (and text Globals).
- **Generate skrub report** builds a standalone data-profiling report of the
  selected Globals, then **Open skrub report** opens it in a new browser tab.
  Like the CSV exports, the report covers every extracted sample across all
  splits, regardless of which splits are selected for plotting. It is limited
  to 2,000 columns. Report generation may take time for large tables; if it
  fails, the CSV exports and plots remain available.

The report link is cleared when Globals are extracted again, the selected
Globals change, the dataset or time step changes, or the report mode is
switched.

The report below is generated by the viewer from a small synthetic dataset,
where the `test` split has no `Global/Efficiency` values. It is interactive:
switch tabs (Table, Stats, Distributions, Associations) or click a column to
see its statistics.

<iframe src="viewer_skrub_report.html" title="Interactive skrub report of extracted Globals" style="width: 100%; height: 720px; border: 1px solid #ddd;" loading="lazy"></iframe>

To compare splits, enable **Compare individual splits** before clicking
**Generate skrub report** (available when the dataset has at least two splits).
It builds one skrub report per extracted split and stacks them on a single
page, in extraction order. Switching tab in any report switches every report
to the same tab, so the distributions of, for example, `train` and `test` can
be read one below the other. This is a side-by-side view, not a statistical
test between splits. Turn the switch off to generate a global report again.

## Time-dependent samples

If the selected sample contains several time steps, a **Time** slider appears
under the sample selector.

You can:

- move through time with the slider;
- press play/pause to animate the sample;
- press stop to go back to the first time step;
- toggle loop playback;
- adjust the playback speed with **Animation speed (fps)**.

## Rendering options

The controls in **3D Rendering options** only change how the current sample is
drawn. They do not change which data is loaded.

- **Colormap** changes the colors used for the selected field.
- **Show edges** overlays mesh edges.
- **Iso contour** limits the colormap to 9 color bands, to visualize the iso
  contours of the field.
- **Reset camera** recenters the view after you rotate, pan, or zoom.

## Interacting with the 3D view

Use the mouse directly in the 3D area:

- left mouse button: rotate;
- middle mouse button, or `Shift` + left mouse button: pan;
- mouse wheel, or right mouse button drag: zoom.

The small axes marker in the corner helps you keep track of the orientation.

## Local datasets and Hub datasets

The viewer can open data from two places:

- **Local datasets** are read from disk and can be accessed in any order. The
  sample control is therefore an index field and a slider.
- **Hub datasets** are streamed from Hugging Face Hub. This avoids downloading
  the full dataset first, but samples are consumed forward with the **Next**
  button.

When streaming is active, a `streaming` chip appears in the toolbar.

## Troubleshooting

!!! tip "Viewer tips"
    - If the 3D view is empty, first select a button under **Available Bases**.
      The message "Pick a Base or enable Globals to load the sample" means that
      neither a base nor **Sample Globals** is selected.
    - If the field dropdown is empty, load fields with **Load all**, or tick them
      under **Select features** and click **Apply**. For Hub streams, click
      **Next** after applying the selection to fetch a sample with those fields.
    - If the viewer feels slow on a large dataset, load only one base and a small
      number of fields.
    - The message under the sample selector reports the latest action or error;
      on the **Dataset Globals** tab, the message under **Extract Globals** does.
    - **Draw plot** and **Download all Globals (CSV)** stay disabled until
      Globals are extracted. **Download selected Globals (CSV)**, **Generate
      skrub report** and parallel plots also need selected Globals: check them
      with **Filter Globals** and **Check** first.
    - Hover over **Selected Globals** for a reminder when nothing is selected.
