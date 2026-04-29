"""Trame server for the dataset viewer.

This module builds a self-contained trame application that lets users
browse PLAID datasets and visualize their samples. All UI is exposed as
trame/Vuetify widgets in a side drawer; the 3D view is a VTK *remote*
view (server-side rendering, streamed as images) driven by a lightweight
VTK pipeline (reader -> geometry -> mapper). Remote rendering avoids the
rare vtk.js rendering artefacts observed when geometry with several
disjoint 1D connected components (e.g. VKI-LS59 ``Base_1_2`` with two
airfoil profiles) is streamed to the browser.



Architecture:

- A :class:`PlaidDatasetService` is used to discover datasets and load
  samples.
- A :class:`ParaviewArtifactService` converts a sample to a single CGNS
  file (or ``.cgns.series`` sidecar for time-dependent samples).
- ``vtkCGNSReader`` (optionally wrapped in ``vtkCGNSFileSeriesReader``) feeds
  the VTK pipeline.
- The user can colour the geometry by any point or cell field and
  choose a colormap preset.


The server is started by :mod:`plaid.viewer.cli` but can also be used
as a library.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from pathlib import Path

from plaid.viewer.models import SampleRef
from plaid.viewer.services import ParaviewArtifactService, PlaidDatasetService
from plaid.viewer.services.plaid_dataset_service import STREAM_CURSOR_ID

logger = logging.getLogger(__name__)

_COLORMAPS = ["viridis", "plasma", "inferno", "magma", "coolwarm", "turbo", "jet"]

_VTK_LOG_ROUTER_INSTALLED = False
_C_STDERR_REROUTED = False


def _reroute_c_stderr() -> None:
    """Permanently redirect the process's stderr file descriptor to /dev/null.

    VTK's CGNS reader and the underlying HDF5 library emit informational
    messages such as ``Mismatch in number of children and child IDs read``
    directly via ``fprintf(stderr, ...)``. Those are not routed through
    ``vtkOutputWindow`` and cannot be captured by a Python logger without
    hijacking file descriptor 2.

    To keep Python's ``sys.stderr`` functional (pytest, tracebacks, etc.) we
    save the current fd 2, reopen ``sys.stderr`` on top of the saved fd, and
    only *then* redirect fd 2 itself to ``/dev/null``. C libraries that
    write directly to ``stderr`` are silenced while Python ``print(...,
    file=sys.stderr)`` and logging handlers keep working.

    Installed once per process.
    """
    global _C_STDERR_REROUTED
    if _C_STDERR_REROUTED:
        return
    import sys  # noqa: PLC0415

    try:
        saved_fd = os.dup(2)
    except OSError:  # pragma: no cover - no fd 2
        return
    try:
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass
    try:
        sys.stderr = os.fdopen(saved_fd, "w", buffering=1)
    except OSError:  # pragma: no cover - defensive
        os.close(saved_fd)
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 2)
    os.close(devnull_fd)
    _C_STDERR_REROUTED = True


def _install_vtk_log_router() -> None:
    """Route VTK / HDF5 warnings to the Python ``logger`` at DEBUG level.

    ``vtkCGNSReader`` (through HDF5) emits chatty but harmless warnings such
    as ``Mismatch in number of children and child IDs read`` when opening
    CGNS files that contain bases without zones (e.g. ``Global``). By default
    VTK writes those to ``stderr`` through a ``vtkOutputWindow``, which
    pollutes the trame server console. We redirect all VTK messages to the
    Python logger so users can opt in with ``PLAID_VIEWER_LOG=DEBUG``
    without any noise at INFO level.

    Installed once per process.
    """
    global _VTK_LOG_ROUTER_INSTALLED
    if _VTK_LOG_ROUTER_INSTALLED:
        return
    try:
        import vtk  # noqa: PLC0415
    except ImportError:  # pragma: no cover - VTK is required in practice
        return

    # ``vtkPythonStdStreamCaptureHelper`` is not available in every VTK wheel,
    # so we subclass ``vtkOutputWindow`` in Python and forward all messages.
    class _LoggingOutputWindow(vtk.vtkOutputWindow):  # type: ignore[misc]
        def DisplayText(self, text: str) -> None:  # noqa: N802 - VTK API
            logger.debug("vtk: %s", text.rstrip())

        def DisplayErrorText(self, text: str) -> None:  # noqa: N802 - VTK API
            logger.debug("vtk error: %s", text.rstrip())

        def DisplayWarningText(self, text: str) -> None:  # noqa: N802 - VTK API
            logger.debug("vtk warning: %s", text.rstrip())

        def DisplayGenericWarningText(  # noqa: N802 - VTK API
            self, text: str
        ) -> None:
            logger.debug("vtk warning: %s", text.rstrip())

        def DisplayDebugText(self, text: str) -> None:  # noqa: N802 - VTK API
            logger.debug("vtk debug: %s", text.rstrip())

    vtk.vtkOutputWindow.SetInstance(_LoggingOutputWindow())
    # Also silence VTK's own warning channel entirely; the logger now owns it.
    vtk.vtkObject.GlobalWarningDisplayOff()
    # VTK 9 routes most reader warnings (e.g. CGNS ``Mismatch in number of
    # children and child IDs read``) through loguru via ``vtkLogger``, which
    # writes to stderr independently from ``vtkOutputWindow``. Silence that
    # channel as well so the server console stays clean.
    if hasattr(vtk, "vtkLogger"):
        try:
            vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
        except AttributeError:  # pragma: no cover - very old VTK
            pass
    _VTK_LOG_ROUTER_INSTALLED = True


@contextlib.contextmanager
def _silence_stderr():
    """Temporarily redirect file descriptor 2 to ``/dev/null``.

    Needed around ``vtkCGNSReader`` updates because the CGNS C library
    writes messages such as ``Mismatch in number of children and child IDs
    read`` directly to ``stderr`` (via ``fprintf``), bypassing VTK's
    ``vtkOutputWindow`` and therefore our Python logger override.
    """
    try:
        saved = os.dup(2)
    except OSError:  # pragma: no cover - no fd 2 (unlikely)
        yield
        return
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, 2)
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull_fd)


# ---------------------------------------------------------------------------
# VTK helpers
# ---------------------------------------------------------------------------


def _enable_all_selections(cgns_reader) -> None:
    """Enable every base / point / cell array known to a ``vtkCGNSReader``.

    ``vtkCGNSReader`` selections are OFF by default for arrays (and for
    any base beyond the first one) so the VTK output would otherwise miss
    half of the data. We enable everything after ``UpdateInformation`` so
    the UI can expose it to the user.
    """
    cgns_reader.UpdateInformation()
    cgns_reader.EnableAllBases()
    cgns_reader.EnableAllPointArrays()
    cgns_reader.EnableAllCellArrays()


def _disable_bases_on_reader(reader, base_names: list[str]) -> None:
    """Disable the given bases on the reader's base selection.

    Keeps every other base enabled. Useful to hide zone-less CGNS bases
    from ``vtkCGNSReader`` which otherwise logs ``No zones in base ...``
    warnings on every update.
    """
    cgns = _cgns_reader_of(reader)
    selection = cgns.GetBaseSelection()
    for name in base_names:
        if selection.ArrayExists(name):
            selection.DisableArray(name)
    cgns.Modified()


def _load_reader(cgns_path: Path):
    """Return a ready-to-use VTK reader for ``cgns_path``.

    For a ``.cgns.series`` sidecar, the reader is wrapped in
    ``vtkCGNSFileSeriesReader`` so ParaView's time controls work out of the
    box. (Note: the generic ``vtkFileSeriesReader`` is not exposed by the
    ``vtk`` PyPI wheel, only the CGNS-specialised series reader is.)

    All bases, point arrays and cell arrays are enabled by default; the
    side drawer lets the user narrow the selection later.
    """
    import vtk  # noqa: PLC0415

    if cgns_path.suffix == ".series":
        payload = json.loads(cgns_path.read_text())
        entries = sorted(
            payload.get("files", []),
            key=lambda entry: float(entry.get("time", 0.0)),
        )
        base_dir = cgns_path.parent
        inner = vtk.vtkCGNSReader()
        series = vtk.vtkCGNSFileSeriesReader()
        series.SetReader(inner)
        for entry in entries:
            series.AddFileName(str((base_dir / entry["name"]).resolve()))
        # ``vtkCGNSFileSeriesReader`` does not expose per-entry time setters:
        # the timestep values are read from each CGNS file itself when the
        # series reader pulls information from the underlying reader.
        series.UpdateInformation()
        inner.EnableAllBases()
        inner.EnableAllPointArrays()
        inner.EnableAllCellArrays()
        # Do not call Update() here: the caller disables zone-less bases
        # first (see ``_refresh_sample_view``) to avoid ``vtkCGNSReader``
        # logging ``No zones in base ...`` warnings. The pipeline's
        # ``_apply_base_selection`` triggers the first Update().
        return series

    reader = vtk.vtkCGNSReader()
    reader.SetFileName(str(cgns_path))
    _enable_all_selections(reader)
    return reader


def _cgns_reader_of(reader):
    """Return the underlying ``vtkCGNSReader`` for a plain or series reader."""
    if hasattr(reader, "GetReader"):
        return reader.GetReader()
    return reader


def _selection_names(selection) -> list[str]:
    """Return the array names exposed by a ``vtkDataArraySelection``."""
    return [selection.GetArrayName(i) for i in range(selection.GetNumberOfArrays())]


def _reader_bases_and_fields(reader) -> tuple[list[str], list[str], list[str]]:
    """Return ``(bases, point_fields, cell_fields)`` exposed by the reader."""
    cgns = _cgns_reader_of(reader)
    bases = _selection_names(cgns.GetBaseSelection())
    point_fields = _selection_names(cgns.GetPointDataArraySelection())
    cell_fields = _selection_names(cgns.GetCellDataArraySelection())
    return bases, point_fields, cell_fields


def _advance_reader_time(reader, time_value: float) -> None:
    """Ask a VTK reader to update to the given time value.

    Works both on a plain ``vtkCGNSReader`` (static sample, no-op on the
    reader itself) and on a ``vtkCGNSFileSeriesReader`` wrapping it. We call
    ``UpdateTimeStep`` when available and otherwise fall back to the
    executive's ``SetUpdateTimeStep`` API. Any failure is logged but does
    not propagate to the UI.
    """
    try:
        with _silence_stderr():
            update_time_step = getattr(reader, "UpdateTimeStep", None)
            if callable(update_time_step):
                update_time_step(time_value)
            else:
                executive = reader.GetExecutive()
                executive.SetUpdateTimeStep(0, time_value)
            reader.Update()
    except Exception as exc:  # noqa: BLE001 - defensive, VTK may be strict
        logger.warning("Failed to advance reader to time %s: %s", time_value, exc)


def _apply_base_selection(reader, active_bases: list[str]) -> None:
    """Enable exactly ``active_bases`` on the reader's base selection."""
    cgns = _cgns_reader_of(reader)
    selection = cgns.GetBaseSelection()
    selection.DisableAllArrays()
    for name in active_bases:
        selection.EnableArray(name)
    cgns.Modified()
    with _silence_stderr():
        reader.Update()


def _list_point_and_cell_fields(dataset) -> tuple[list[str], list[str]]:
    """Return the point and cell field names available on ``dataset``."""
    point_fields: set[str] = set()
    cell_fields: set[str] = set()

    def _visit(obj):
        if obj is None:
            return
        if hasattr(obj, "GetNumberOfBlocks"):
            for i in range(obj.GetNumberOfBlocks()):
                _visit(obj.GetBlock(i))
            return
        pd = obj.GetPointData() if hasattr(obj, "GetPointData") else None
        cd = obj.GetCellData() if hasattr(obj, "GetCellData") else None
        if pd is not None:
            for i in range(pd.GetNumberOfArrays()):
                point_fields.add(pd.GetArrayName(i))
        if cd is not None:
            for i in range(cd.GetNumberOfArrays()):
                cell_fields.add(cd.GetArrayName(i))

    _visit(dataset)
    return sorted(point_fields), sorted(cell_fields)


def _compute_field_range(
    dataset, field_name: str, association: str
) -> tuple[float, float]:
    """Return the (min, max) range of ``field_name`` across ``dataset``."""
    lo = float("inf")
    hi = float("-inf")

    def _visit(obj):
        nonlocal lo, hi
        if obj is None:
            return
        if hasattr(obj, "GetNumberOfBlocks"):
            for i in range(obj.GetNumberOfBlocks()):
                _visit(obj.GetBlock(i))
            return
        data = obj.GetPointData() if association == "point" else obj.GetCellData()
        if data is None:
            return
        arr = data.GetArray(field_name)
        if arr is None:
            return
        r = arr.GetRange(-1)
        lo = min(lo, r[0])
        hi = max(hi, r[1])

    _visit(dataset)
    if lo == float("inf"):
        return 0.0, 1.0
    return lo, hi


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class _VtkPipeline:
    """Minimal reader -> (cut) -> (threshold) -> geometry -> actor pipeline."""

    def __init__(self) -> None:
        import vtk  # noqa: PLC0415

        self.render_window = vtk.vtkRenderWindow()
        # Off-screen rendering is required on headless servers (no X
        # display). It does not prevent the interactor from receiving
        # events forwarded from the browser by ``VtkRemoteView``: the
        # events are dispatched to the interactor style, which mutates
        # the server-side camera before the next frame is streamed.
        self.render_window.OffScreenRenderingOn()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.12, 0.12, 0.14)
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        # Without an explicit interactor style, ``vtkRenderWindowInteractor``
        # does not translate mouse events into camera manipulation, so the
        # remote view appears frozen in the browser even though events are
        # correctly forwarded. ``vtkInteractorStyleTrackballCamera`` is the
        # standard ParaView-like style (LMB rotate, MMB pan, wheel zoom).
        interactor_style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(interactor_style)
        self.interactor.Initialize()
        self._interactor_style = interactor_style  # keep a reference alive

        self.reader = None
        self.actor = vtk.vtkActor()
        # Gouraud shading (per-vertex normals interpolated across the
        # triangle) looks noticeably smoother than flat shading on curved
        # surfaces. Combined with a ``vtkPolyDataNormals`` step below, it
        # gives a nice continuous lighting on CFD meshes without changing
        # the geometry.
        self.actor.GetProperty().SetInterpolationToGouraud()
        self.mapper = vtk.vtkCompositePolyDataMapper()
        self.actor.SetMapper(self.mapper)
        self.renderer.AddActor(self.actor)

        self.lut = vtk.vtkLookupTable()
        self.lut.SetHueRange(0.667, 0.0)  # blue -> red
        self.lut.Build()

        self._current_dataset = None

    def load(self, cgns_path: Path) -> None:
        """Load a new CGNS/series file and reset the pipeline."""
        self.reader = _load_reader(cgns_path)
        self._rebuild()

    def update(
        self,
        *,
        field: str | None,
        association: str,
        cmap: str,
        show_edges: bool,
    ) -> None:
        """Rebuild the downstream pipeline with the current options."""
        if self.reader is None:
            return
        import vtk  # noqa: PLC0415

        pipeline_output = self.reader.GetOutputPort()

        geom = vtk.vtkCompositeDataGeometryFilter()
        geom.SetInputConnection(pipeline_output)
        geom.Update()
        self._current_dataset = geom.GetOutput()
        self.mapper.SetInputConnection(geom.GetOutputPort())

        if field is not None:
            self.mapper.SelectColorArray(field)

            if association == "point":
                self.mapper.SetScalarModeToUsePointFieldData()
            else:
                self.mapper.SetScalarModeToUseCellFieldData()
            self.mapper.SetColorModeToMapScalars()
            self.mapper.ScalarVisibilityOn()
            lo, hi = _compute_field_range(self.reader.GetOutput(), field, association)
            self.lut = _build_lut(cmap, lo, hi)
            self.mapper.SetLookupTable(self.lut)
            self.mapper.SetScalarRange(lo, hi)
        else:
            self.mapper.ScalarVisibilityOff()

        self.actor.GetProperty().SetEdgeVisibility(bool(show_edges))
        self.actor.GetProperty().SetLineWidth(1.0)

    def reset_camera(self) -> None:
        """Reset the camera to the default view orientation and framing.

        ``vtkRenderer.ResetCamera()`` only adjusts the camera *distance*
        so the current actor fits in the viewport; it leaves the camera
        orientation (position direction, view up) untouched. To match the
        first-load behaviour after the user has rotated the scene, we
        also reset the orientation to the VTK defaults (looking down
        ``-Z`` with ``+Y`` up) before reframing.
        """
        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0.0, 0.0, 1.0)
        camera.SetFocalPoint(0.0, 0.0, 0.0)
        camera.SetViewUp(0.0, 1.0, 0.0)
        camera.SetViewAngle(30.0)
        self.renderer.ResetCamera()

    def _rebuild(self) -> None:
        self.renderer.ResetCamera()


def _build_lut(cmap: str, lo: float, hi: float):
    """Build a simple ``vtkLookupTable`` approximating a matplotlib colormap."""
    import vtk  # noqa: PLC0415

    # Minimal built-in approximations - use HueRange for the common cases.
    lut = vtk.vtkLookupTable()
    lut.SetTableRange(lo, hi)
    lut.SetNumberOfColors(256)
    presets = {
        "viridis": (0.75, 0.0),
        "plasma": (0.8, 0.05),
        "inferno": (0.0, 0.15),
        "magma": (0.85, 0.0),
        "coolwarm": (0.667, 0.0),
        "turbo": (0.7, 0.0),
        "jet": (0.667, 0.0),
    }
    h0, h1 = presets.get(cmap, (0.667, 0.0))
    lut.SetHueRange(h0, h1)
    lut.SetSaturationRange(1.0, 1.0)
    lut.SetValueRange(1.0, 1.0)
    lut.Build()
    return lut


# ---------------------------------------------------------------------------
# Trame server
# ---------------------------------------------------------------------------


def build_server(
    dataset_service: PlaidDatasetService,
    artifact_service: ParaviewArtifactService,
):
    """Create a configured trame :class:`Server` instance.

    Args:
        dataset_service: Discovers datasets and loads PLAID samples.
        artifact_service: Converts a :class:`SampleRef` to a ParaView-readable
            artifact on disk.

    Returns:
        The configured ``trame.app.Server``. Call ``.start(host=..., port=...)``
        to run it.
    """
    from trame.app import (
        asynchronous,  # noqa: PLC0415
        get_server,  # noqa: PLC0415
    )
    from trame.ui.vuetify3 import SinglePageWithDrawerLayout  # noqa: PLC0415
    from trame.widgets import html  # noqa: PLC0415
    from trame.widgets import vtk as vtk_widgets  # noqa: PLC0415
    from trame.widgets import vuetify3 as v3  # noqa: PLC0415

    _install_vtk_log_router()

    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller

    pipeline = _VtkPipeline()
    # Background task handle for the time-series playback loop (see
    # ``_on_playing`` below). Kept here so successive toggles cancel the
    # previous task instead of spawning duplicates.
    play_task: dict[str, object] = {"task": None}
    # One-shot flag raised by ``_apply_features`` so the next
    # ``_refresh_sample_view_impl`` call rebuilds the ParaView artifact
    # from scratch (its on-disk cache key does not include the feature
    # filter, so without this force-refresh the renderer would keep
    # showing the pre-filter CGNS file).
    force_artifact_refresh: dict[str, bool] = {"pending": False}

    with _silence_stderr():
        datasets = dataset_service.list_datasets()
    # Dataset ids are kept in two disjoint lists driven by the
    # Local / Hub tabs so the dropdown always matches the active source
    # (``init_from_disk`` vs ``init_streaming_from_hub``). The UI reads
    # the right list via a ternary expression on ``source_tab``.
    hub_ids_set = set(dataset_service.hub_repos)
    local_dataset_ids = [
        d.dataset_id for d in datasets if d.dataset_id not in hub_ids_set
    ]
    hub_dataset_ids = [d.dataset_id for d in datasets if d.dataset_id in hub_ids_set]
    dataset_ids = local_dataset_ids + hub_dataset_ids

    # --- Default state ----------------------------------------------------
    # Datasets root panel. ``allow_root_change`` gates the UI on the
    # client: when False, the panel is hidden so a public deployment can
    # pin the root from the CLI (``--datasets-root /data
    # --disable-root-change``).
    state.setdefault(
        "datasets_root_text",
        str(dataset_service.datasets_root) if dataset_service.datasets_root else "",
    )
    state.setdefault("allow_root_change", dataset_service._config.allow_root_change)
    state.setdefault("browse_dialog", False)
    state.setdefault("browse_cwd", "")
    state.setdefault("browse_parent", None)
    state.setdefault("browse_entries", [])

    # Hugging Face Hub streaming. ``hub_repos`` mirrors the service state
    # and ``hub_repo_input`` is the text field bound to the "Add hub
    # dataset" panel. Hub datasets are exposed alongside local ones in
    # ``dataset_ids``; the service dispatches to
    # ``plaid.storage.init_streaming_from_hub`` when the selected dataset
    # is a registered repo id.
    state.setdefault("hub_repos", list(dataset_service.hub_repos))
    state.setdefault("hub_repo_input", "")
    # Active side-panel tab: "local" drives ``datasets_root_text`` and
    # directory browsing, "hub" drives the Hugging Face repo input. The
    # selection only gates which form is rendered; registered datasets
    # from either source always land in ``dataset_ids`` together.
    state.setdefault("source_tab", "local")

    # Initial ``dataset_id`` follows the default ``source_tab`` ("local"):
    # pick the first local dataset when any is available, otherwise fall
    # back to the first hub dataset (so a viewer launched with only
    # ``--hub-repo`` still has something selected).
    initial_dataset_id = (
        local_dataset_ids[0]
        if local_dataset_ids
        else (hub_dataset_ids[0] if hub_dataset_ids else None)
    )
    state.setdefault("dataset_id", initial_dataset_id)
    # Separate lists per source so the dropdown only shows datasets that
    # match the active tab. ``dataset_ids`` is kept for backwards
    # compatibility (e.g. tests that inspect the full list) but the UI
    # reads from ``local_dataset_ids`` / ``hub_dataset_ids`` directly.
    state.setdefault("local_dataset_ids", local_dataset_ids)
    state.setdefault("hub_dataset_ids", hub_dataset_ids)
    state.setdefault("dataset_ids", dataset_ids)

    state.setdefault("splits", [])
    state.setdefault("split", None)
    state.setdefault("sample_ids", [])
    state.setdefault("sample_id", None)
    state.setdefault("sample_index", 0)
    state.setdefault("sample_count", 0)
    # Streaming (Hugging Face Hub) navigation. Hub datasets expose
    # ``IterableDataset`` splits without a ``__len__``, so the slider is
    # driven by a forward-only cursor rather than a random-access index
    # list. ``stream_position`` mirrors the service cursor (-1 before any
    # fetch), ``stream_exhausted`` is set when the iterator raises
    # ``StopIteration`` so the slider caps at the last consumed index.
    state.setdefault("is_streaming", False)
    state.setdefault("stream_position", -1)
    state.setdefault("stream_exhausted", False)

    # Feature filtering state. ``available_features`` is the full list of
    # feature paths declared in the dataset metadata (populated whenever
    # ``dataset_id`` changes), ``selected_features`` is the subset the
    # user kept through the checkbox panel. An empty ``selected_features``
    # means "no filter": every feature is loaded (default behaviour).
    state.setdefault("available_features", [])
    state.setdefault("selected_features", [])

    state.setdefault("base_options", [])
    # Single active base (exclusive selection). Kept as a list internally
    # so `_apply_base_selection` has a uniform interface, but the UI
    # exposes it as a ``VBtnToggle`` with ``multiple=False``.
    state.setdefault("active_base", None)
    # PLAID globals (``sample.get_global_names`` / ``sample.get_global``)

    # for the current sample, minus the ``IterationValues`` / ``TimeValues``
    # bookkeeping arrays which describe time steps rather than physical
    # scalars.
    state.setdefault("sample_globals", [])
    # Time axis. ``time_values`` mirrors ``sample.features.get_all_time_values()``
    # and ``time_index`` is the index of the currently displayed step.
    state.setdefault("time_values", [])
    state.setdefault("time_index", 0)
    state.setdefault("time_count", 0)
    state.setdefault("current_time", None)
    state.setdefault("field_options", [])
    state.setdefault("field", None)  # "point:name" or "cell:name"
    state.setdefault("cmap", "viridis")
    state.setdefault("cmaps", _COLORMAPS)
    state.setdefault("show_edges", False)
    state.setdefault("field_range", [0.0, 1.0])
    state.setdefault("status", "Select a dataset to start.")
    # Loading indicator: True while the VTK reader is opening a new sample
    # or advancing to a new time step. Consumed by a ``VProgressLinear`` in
    # the header and an overlay on top of the 3D view.
    state.setdefault("loading", False)
    # Time-series playback controls.
    state.setdefault("playing", False)
    state.setdefault("play_fps", 5)
    state.setdefault("play_loop", True)

    # --- Helpers ----------------------------------------------------------

    def _refresh_splits() -> None:
        if not state.dataset_id:
            state.splits = []
            state.split = None
            # Propagate "no dataset" to sample list + 3D scene so the
            # view does not linger on the last local sample when the
            # user switches to the Hub tab without any registered repo.
            _refresh_samples()
            return

        try:
            with _silence_stderr():
                detail = dataset_service.get_dataset(state.dataset_id)
            splits = list(detail.splits.keys())
        except Exception as exc:  # noqa: BLE001
            state.status = f"Failed to load dataset: {exc}"
            splits = []
        state.splits = splits
        new_split = splits[0] if splits else None
        # When the new dataset exposes the same first split name as the
        # previous one (e.g. both default to ``train``), ``state.split``
        # does not change and the ``@state.change("split")`` listener is
        # skipped: the sample list would keep pointing at the old dataset.
        # Force a refresh in that case.
        same_split = state.split == new_split
        state.split = new_split
        if same_split:
            _refresh_samples()

    def _clear_scene(status: str | None = None) -> None:
        """Empty the VTK view and all sample-related panels.

        Used whenever no sample should be displayed (no dataset
        selected, streaming dataset waiting for the first ``Next``
        click, ...). Keeping this in a single place ensures the 3D
        view never lingers on a stale frame from a previous selection.
        """
        pipeline.reader = None
        pipeline.mapper.RemoveAllInputConnections(0)
        pipeline.mapper.ScalarVisibilityOff()
        state.base_options = []
        state.active_base = None
        state.field_options = []
        state.field = None
        state.sample_globals = []
        state.time_values = []
        state.time_count = 0
        state.time_index = 0
        state.current_time = None
        state.sample_ids = []
        state.sample_id = None
        state.sample_count = 0
        state.sample_index = 0
        if status is not None:
            state.status = status
        ctrl.view_update()

    def _refresh_samples() -> None:
        if not state.dataset_id:
            # No dataset selected: clear everything, including the 3D
            # scene. This matters when the user switches to the Hub tab
            # without any registered repo - otherwise the view would
            # keep showing the last local sample.
            state.is_streaming = False
            _clear_scene(status="Select a dataset to start.")
            return

        split_key = state.split
        if split_key == "__default__":
            split_key = None
        # Streaming datasets (HF Hub) are not random-access. The service
        # returns a single synthetic ``SampleRef`` with the
        # ``STREAM_CURSOR_ID`` sentinel per split, and we advance the
        # cursor forward through ``advance_stream_cursor`` as the user
        # moves the slider to the right. The slider exposes indices
        # ``[0 .. cursor_position + 1]`` so the user can still revisit
        # already-fetched samples via the converter cache but never
        # rewind the underlying iterator (which is by construction
        # forward-only).
        try:
            streaming = dataset_service.is_streaming(state.dataset_id)
        except Exception:  # noqa: BLE001
            streaming = False
        state.is_streaming = streaming
        if streaming:
            # Reset the cursor so each (dataset, split) selection starts
            # at the first available sample regardless of previous state.
            try:
                dataset_service.reset_stream_cursor(state.dataset_id, split_key)
            except Exception as exc:  # noqa: BLE001
                state.status = f"Failed to reset stream cursor: {exc}"
                return
            state.stream_position = -1
            state.stream_exhausted = False
            state.sample_ids = []
            state.sample_count = 0
            state.sample_index = 0
            # No sample has been fetched yet: the status bar invites the
            # user to click "Next" to consume the first element of the
            # stream. ``sample_id`` stays ``None`` so ``_refresh_sample_view``
            # short-circuits until the cursor has actually advanced.
            state.sample_id = None
            # Clear the VTK scene so the 3D view is empty while waiting
            # for the first ``Next`` click. Without this, switching back
            # to the Hub tab would still show the mesh of the previously
            # loaded local dataset (or the previous streaming sample),
            # which is confusing since no hub sample has been fetched yet.
            pipeline.reader = None
            pipeline.mapper.RemoveAllInputConnections(0)
            pipeline.mapper.ScalarVisibilityOff()
            state.base_options = []
            state.active_base = None
            state.field_options = []
            state.field = None
            state.sample_globals = []
            state.time_values = []
            state.time_count = 0
            state.time_index = 0
            state.current_time = None
            ctrl.view_update()
            state.status = "Streaming: click Next to fetch the first sample."
            return

        try:
            with _silence_stderr():
                refs = dataset_service.list_samples(state.dataset_id)
        except Exception as exc:  # noqa: BLE001
            state.status = f"Failed to list samples: {exc}"
            refs = []
        ids = [r.sample_id for r in refs if r.split == split_key]
        state.sample_ids = ids
        state.sample_count = len(ids)
        state.sample_index = 0
        new_sample_id = ids[0] if ids else None
        # Switching dataset/split may leave ``state.sample_id`` unchanged
        # (e.g. both new and old first sample are "0"); in that case the
        # ``@state.change("sample_id")`` hook would not fire and the 3D
        # view would keep the previous sample. Force a refresh whenever
        # the sample id is the same but the dataset/split context changed.
        same_id = state.sample_id == new_sample_id
        state.sample_id = new_sample_id
        if same_id and new_sample_id is not None:
            _refresh_sample_view()

    def _refresh_field_options() -> None:
        """Restrict the field dropdown to arrays present in the active base.

        ``_list_point_and_cell_fields`` walks the reader's current output,
        which reflects the currently enabled base selection, so fields
        belonging to unselected bases are hidden.
        """
        if pipeline.reader is None:
            state.field_options = []
            state.field = None
            return
        points, cells = _list_point_and_cell_fields(pipeline.reader.GetOutput())
        options = [f"point:{n}" for n in points] + [f"cell:{n}" for n in cells]
        state.field_options = options
        # Preserve the previously selected field if it is still available.
        if state.field not in options:
            state.field = options[0] if options else None

    def _refresh_sample_view() -> None:
        """Reload the current sample and refresh the full UI state.

        The call is intentionally synchronous: trame schedules state
        broadcasts after the callback returns, so we rely on the
        ``VProgressLinear`` shown while ``state.loading`` is True to
        indicate activity. A previous async variant that ran the VTK work
        in an executor caused the viewer to appear frozen, so we keep the
        simple blocking flow and just expose ``state.loading`` for visual
        feedback.
        """
        if not (state.dataset_id and state.sample_id is not None):
            return
        state.loading = True
        try:
            _refresh_sample_view_impl()
        finally:
            state.loading = False

    def _refresh_sample_view_impl() -> None:
        split = state.split if state.split != "__default__" else None
        # Streaming datasets expose a "hub" backend regardless of the
        # CLI-default backend id, so ``SampleRef`` carries the correct
        # loader hint and the paraview artifact cache remains coherent
        # across local/streaming switches.
        backend_id = "hub" if state.is_streaming else dataset_service._config.backend_id
        ref = SampleRef(
            backend_id=backend_id,
            dataset_id=state.dataset_id,
            split=split,
            sample_id=str(state.sample_id),
        )

        # Refresh time axis + globals panel (independent of VTK rendering).
        # PLAID's CGNS loading (pyCGNS / CHLone) writes low-level HDF5
        # warnings such as "Mismatch in number of children and child IDs
        # read" directly to stderr. Wrap every call that can trigger a
        # CGNS read with ``_silence_stderr`` so the server console stays
        # clean.
        try:
            with _silence_stderr():
                times = dataset_service.list_time_values(ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list time values: %s", exc)
            times = []
        state.time_values = times
        state.time_count = len(times)
        state.time_index = 0
        state.current_time = times[0] if times else None
        try:
            with _silence_stderr():
                state.sample_globals = dataset_service.describe_globals(
                    ref, time=state.current_time
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to describe globals: %s", exc)
            state.sample_globals = []
        try:
            # Streaming samples all share the same ``SampleRef`` (the
            # ``STREAM_CURSOR_ID`` sentinel) and would therefore hit the
            # paraview artifact cache on every Next click, returning the
            # first consumed sample forever. ``force=True`` tells
            # ``ensure_artifact`` to rebuild the on-disk CGNS from the
            # freshly advanced stream cursor instead.
            #
            # Disk datasets additionally set ``force_artifact_refresh``
            # after the user applies a new feature filter: the artifact
            # cache key is derived from ``SampleRef`` alone (no feature
            # list), so without forcing a rebuild the renderer would
            # keep displaying the pre-filter CGNS file.
            force = state.is_streaming or force_artifact_refresh["pending"]
            force_artifact_refresh["pending"] = False
            with _silence_stderr():
                artifact = artifact_service.ensure_artifact(ref, force=force)
            pipeline.load(artifact.cgns_path)
            # Disable zone-less bases *before* the reader's first Update()
            # so ``vtkCGNSReader`` does not log ``No zones in base ...``
            # warnings for auxiliary bases like ``Global``.
            try:
                with _silence_stderr():
                    non_visual_names = list(
                        dataset_service.describe_non_visual_bases(ref).keys()
                    )
            except Exception:  # noqa: BLE001
                non_visual_names = []
            if non_visual_names:
                _disable_bases_on_reader(pipeline.reader, non_visual_names)
            with _silence_stderr():
                pipeline.reader.Update()
            bases, _points, _cells = _reader_bases_and_fields(pipeline.reader)
            non_visual_set = set(non_visual_names)
            # The ``Global`` CGNS base is a PLAID bookkeeping base used to
            # store sample-level metadata (scalar inputs/outputs, time
            # values, ...). It is surfaced separately in the "Globals"
            # panel of the drawer and should never appear alongside the
            # ``Base_<topo_dim>_<geom_dim>`` rendering bases in the base
            # toggle: selecting it would hide every ``Base_x_y`` base and
            # leave the 3D view empty.
            visual_bases = [
                name
                for name in bases
                if name not in non_visual_set and name != "Global"
            ]
            state.base_options = visual_bases

            # Preserve the user's base selection across samples when the
            # same base still exists; otherwise fall back to the first
            # renderable base.
            previous = state.active_base
            if previous in visual_bases:
                state.active_base = previous
            else:
                state.active_base = visual_bases[0] if visual_bases else None
            if state.active_base is not None:
                _apply_base_selection(pipeline.reader, [state.active_base])
            _refresh_field_options()
            # For streaming datasets the sentinel ``cursor`` sample id
            # would look like ``hub:repo:split:cursor``; replace it with
            # a 0-based step counter that is meaningful to the user.
            if state.is_streaming:
                state.status = (
                    f"Loaded streaming sample #{state.stream_position} "
                    f"from {state.dataset_id}"
                    + (f" / {state.split}" if state.split else "")
                )
            else:
                state.status = f"Loaded sample {ref.encode()}"
            _apply_pipeline(reset_camera=True)
        except Exception as exc:  # noqa: BLE001
            # "Missing features" errors bubble up from the PLAID converter
            # when a feature path selected by the user does not exist in
            # the current split's schema (constant/variable features are
            # declared per-split). The raw exception dumps the full list
            # of missing paths, which is both noisy and unactionable in
            # the viewer. We shorten it to a hint that the user should
            # check the split-specific availability of the filter.
            message = str(exc)
            if "Missing features" in message:
                state.status = (
                    "Failed to load sample: Missing features in dataset, check split"
                )
            else:
                state.status = f"Failed to load sample: {exc}"

    def _apply_pipeline(*, reset_camera: bool = False) -> None:
        """Rebuild the VTK pipeline and push the result to the client.

        With ``VtkRemoteView`` the VTK camera lives on the server, so
        resetting it server-side and calling ``ctrl.view_update`` is
        sufficient: the next rendered frame sent to the browser already
        reflects the default orientation and reframed bounds.
        """
        if pipeline.reader is None:
            return
        association = "point"
        name: str | None = None
        if state.field:
            association, name = state.field.split(":", 1)
        if name is not None:
            lo, hi = _compute_field_range(
                pipeline.reader.GetOutput(), name, association
            )
            state.field_range = [float(lo), float(hi)]
        pipeline.update(
            field=name,
            association=association,
            cmap=state.cmap,
            show_edges=bool(state.show_edges),
        )
        if reset_camera:
            pipeline.reset_camera()
        ctrl.view_update()

    # --- State change handlers -------------------------------------------

    def _refresh_available_features() -> None:
        """Populate ``available_features`` and ``selected_features`` from PLAID.

        Called whenever the active ``dataset_id`` changes so the feature
        checkbox panel in the drawer reflects what the current dataset
        actually exposes. Errors during metadata loading (missing
        ``variable_schema.yaml`` on non-PLAID directories, network
        failures for Hub datasets, ...) are caught and logged: the panel
        is simply emptied in that case.
        """
        if not state.dataset_id:
            state.available_features = []
            state.selected_features = []
            return
        try:
            with _silence_stderr():
                available = dataset_service.list_available_features(state.dataset_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to list features: %s", exc)
            state.available_features = []
            state.selected_features = []
            return
        state.available_features = available
        current = dataset_service.get_features(state.dataset_id)
        state.selected_features = list(current) if current else []

    @ctrl.set("apply_features")
    def _apply_features() -> None:
        """Push ``selected_features`` to the service and reload the sample.

        The selection is forwarded verbatim to
        :meth:`PlaidDatasetService.set_features`. In particular an
        empty list is kept empty (not converted to ``None``): the user
        then sees a sample that only contains the auto-injected non-field
        paths (globals, mesh coordinates, ...), which removes every
        coloured array from the 3D view. To restore the full dataset
        the user can click the "Load all" shortcut or re-check every
        feature manually.
        """
        if not state.dataset_id:
            return
        features = list(state.selected_features or [])
        try:
            with _silence_stderr():
                # Pass the list unconditionally: ``None`` means "no
                # filter at all" and is reserved for the initial state /
                # explicit reset via :meth:`PlaidDatasetService.set_features`.
                dataset_service.set_features(state.dataset_id, features)
        except Exception as exc:  # noqa: BLE001
            state.status = f"Failed to set features: {exc}"
            return
        # Changing the feature filter invalidates the in-memory store
        # cache (for streaming datasets, the iterator is rebuilt) and
        # any cached paraview artifact for this dataset. The simplest
        # way to propagate the change to the view is to run the full
        # split/sample refresh cascade.
        state.status = (
            f"Applied feature filter ({len(features)} selected)."
            if features
            else "Feature filter cleared (no field loaded)."
        )
        # Force the next ``ensure_artifact`` call to rebuild the CGNS
        # file; otherwise the cache would still return the pre-filter
        # artifact and the renderer's field list would not change.
        force_artifact_refresh["pending"] = True
        _refresh_samples()

    @ctrl.set("clear_features")
    def _clear_features() -> None:
        """Clear the feature selection.

        After calling this, the sample contains only the auto-injected
        non-field paths (globals, coordinates, connectivities) so the
        3D view shows the mesh with no coloured field. Use the
        top-level "Load all" shortcut to restore every feature.
        """
        state.selected_features = []
        _apply_features()

    @ctrl.set("select_all_features")
    def _select_all_features() -> None:
        """Select every available feature and apply the filter.

        Used by the top-level "Load all" shortcut button so the user
        can restore the full-dataset view in a single click without
        having to open the checkbox panel. Internally this is
        equivalent to clearing the filter (an empty / full selection
        both load every feature once non-field paths are re-injected
        by :meth:`PlaidDatasetService.set_features`), but reflecting
        the selection in the checkboxes gives clearer visual feedback.
        """
        state.selected_features = list(state.available_features or [])
        _apply_features()

    @state.change("dataset_id")
    def _on_dataset(**_: object) -> None:
        _refresh_available_features()
        _refresh_splits()

    @state.change("source_tab")
    def _on_source_tab(**_: object) -> None:
        """Switch ``dataset_id`` to the first entry of the active source.

        The dropdown's ``items`` binding filters by ``source_tab`` on the
        client, but the currently selected ``dataset_id`` may belong to
        the other source and would then display as a stale selection. We
        proactively pick the first id from the active list (or ``None``
        when empty) so the dropdown always reflects the active tab.
        """
        active_ids = (
            list(state.hub_dataset_ids or [])
            if state.source_tab == "hub"
            else list(state.local_dataset_ids or [])
        )
        new_id = active_ids[0] if active_ids else None
        if state.dataset_id == new_id:
            # ``@state.change('dataset_id')`` would not fire; refresh
            # splits explicitly so the split dropdown and sample list
            # stay coherent with the active tab.
            _refresh_splits()
        else:
            state.dataset_id = new_id

    @state.change("split")
    def _on_split(**_: object) -> None:
        # Clear the active feature selection on every split switch so
        # the user starts from a predictable, lightweight state: only
        # the geometric supports (mesh coordinates, connectivities,
        # globals, ...) associated with the split's available features
        # are loaded, and no field is coloured in the 3D view. This
        # avoids "Missing features in dataset, check split" errors when
        # the previously-selected fields do not exist in the new split,
        # and lets the user opt-in to specific fields through the
        # checkbox panel. ``_apply_features`` triggers ``_refresh_samples``
        # under the hood, so we do not need to call it again here.
        #
        # Streaming (Hugging Face Hub) datasets are handled differently:
        # they typically expose a single default split, so the multi-
        # split "Missing features" issue does not apply. Pushing an
        # empty feature filter through ``set_features`` would invalidate
        # the store cache and force :meth:`_open` to re-instantiate the
        # streaming iterator with an ``update_features_for_CGNS_compatibility``
        # expansion derived from the dataset-wide metadata union, which
        # may not match the hub split's actual schema and ends up
        # loading the wrong feature catalogue. We therefore skip the
        # auto-clear for streaming datasets and let the user apply
        # filters explicitly through the checkbox panel.
        if not state.dataset_id:
            _refresh_samples()
            return
        try:
            streaming = dataset_service.is_streaming(state.dataset_id)
        except Exception:  # noqa: BLE001
            streaming = False
        if streaming:
            _refresh_samples()
            return
        state.selected_features = []
        _apply_features()

    @state.change("sample_index")
    def _on_sample_index(**_: object) -> None:
        try:
            idx = int(state.sample_index)
        except (TypeError, ValueError):
            idx = 0
        # Streaming datasets: drive the forward-only cursor. The slider's
        # maximum (``sample_count - 1``) always matches the most recent
        # position the user has reached, so a right-arrow press grows the
        # cursor by exactly one step; when the stream is exhausted the
        # index is clamped back to the last valid position.
        if state.is_streaming:
            if state.dataset_id is None:
                return
            split = state.split if state.split != "__default__" else None
            position = int(state.stream_position)
            if idx <= position:
                # Already-visited step: a streaming iterator cannot be
                # rewound, so the view keeps the most recently fetched
                # sample. We simply update the slider label.
                state.sample_index = max(0, position)
                return
            # Advance the cursor step-by-step until it matches ``idx``
            # (the slider can only advance by one in normal use, but we
            # stay robust to multi-step jumps).
            while int(state.stream_position) < idx:
                try:
                    dataset_service.advance_stream_cursor(state.dataset_id, split)
                except StopIteration:
                    state.stream_exhausted = True
                    # Clamp back to the last fetched position.
                    state.sample_index = max(0, int(state.stream_position))
                    state.status = "Stream exhausted."
                    return
                state.stream_position = int(state.stream_position) + 1
            # Grow the slider's reachable range by one so the user can
            # fetch the next sample on the next right-arrow press.
            state.sample_count = int(state.stream_position) + 2
            state.sample_id = "cursor"
            # ``sample_id`` did not actually change ("cursor" both times),
            # so the ``@state.change("sample_id")`` listener is skipped.
            # Force a refresh explicitly.
            _refresh_sample_view()
            return
        ids = list(state.sample_ids or [])
        if not ids:
            state.sample_id = None
            return
        idx = max(0, min(idx, len(ids) - 1))
        state.sample_id = ids[idx]

    @state.change("sample_id")
    def _on_sample(**_: object) -> None:
        _refresh_sample_view()

    def _apply_time_step_impl() -> None:
        """Synchronous work behind a time-axis update.

        Pushes the selected time step into the VTK pipeline and refreshes
        the globals panel for the new time. Both are safe to call at
        playback rates now that ``_on_time_index`` short-circuits during
        playback, so the loop only performs one VTK update and one
        globals read per frame.
        """
        if pipeline.reader is not None and state.current_time is not None:
            _advance_reader_time(pipeline.reader, float(state.current_time))
            _apply_pipeline()
        if state.dataset_id and state.sample_id is not None:
            split = state.split if state.split != "__default__" else None
            ref = SampleRef(
                backend_id=dataset_service._config.backend_id,
                dataset_id=state.dataset_id,
                split=split,
                sample_id=str(state.sample_id),
            )
            try:
                with _silence_stderr():
                    state.sample_globals = dataset_service.describe_globals(
                        ref, time=state.current_time
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to describe globals: %s", exc)

    @state.change("time_index")
    def _on_time_index(**_: object) -> None:
        times = list(state.time_values or [])
        if not times:
            state.current_time = None
            return
        try:
            idx = int(state.time_index)
        except (TypeError, ValueError):
            idx = 0
        idx = max(0, min(idx, len(times) - 1))
        state.current_time = times[idx]
        # During playback the loop (``_play_loop``) already advances the
        # time step itself; without this short-circuit the listener
        # would run a second ``_apply_time_step_impl`` per frame (double
        # VTK update + double PLAID read), which saturates the trame
        # WebSocket and stalls playback.
        if state.playing:
            return
        state.loading = True
        try:
            _apply_time_step_impl()
        finally:
            state.loading = False

    async def _play_loop() -> None:
        """Advance ``time_index`` at ``play_fps`` while ``playing`` is True.

        The loop directly updates ``time_index``, ``current_time`` and
        runs the VTK time-step update synchronously (the VTK calls are
        fast enough for typical CFD meshes). Relying on the
        ``@state.change("time_index")`` listener was unreliable because
        trame dispatches it asynchronously, so the playback could end
        before the last frame was actually rendered.

        When the end of the time axis is reached, the loop either wraps
        around (``play_loop=True``) or stops playback
        (``play_loop=False``). The loop exits cleanly on
        :class:`asyncio.CancelledError` so the Stop button can cancel the
        task immediately.
        """
        try:
            while state.playing:
                count = int(state.time_count or 0)
                if count <= 1:
                    with state:
                        state.playing = False
                    break
                nxt = int(state.time_index or 0) + 1
                if nxt >= count:
                    if state.play_loop:
                        nxt = 0
                    else:
                        with state:
                            state.playing = False
                        break
                times = list(state.time_values or [])
                # Trame state mutations inside an asyncio task must be
                # wrapped in ``with state:`` for the ``@state.change``
                # handlers to actually fire and for the client to receive
                # the broadcast. Without this block, the slider / time
                # label on the client do not update during playback.
                with state:
                    state.time_index = nxt
                    state.current_time = times[nxt] if nxt < len(times) else None
                _apply_time_step_impl()
                fps = max(1, int(state.play_fps or 1))
                await asyncio.sleep(1.0 / fps)
        except asyncio.CancelledError:  # pragma: no cover - cooperative cancel
            pass

    @state.change("playing")
    def _on_playing(**_: object) -> None:
        existing = play_task.get("task")
        if existing is not None and not existing.done():  # type: ignore[union-attr]
            existing.cancel()  # type: ignore[union-attr]
            play_task["task"] = None
        if state.playing and int(state.time_count or 0) > 1:
            play_task["task"] = asynchronous.create_task(_play_loop())

    @ctrl.set("toggle_play")
    def _toggle_play() -> None:
        state.playing = not bool(state.playing)

    @ctrl.set("stop_playback")
    def _stop_playback() -> None:
        """Stop playback and reset the time axis back to the first step.

        Using a controller callback is more robust than the inline
        ``click="playing = false; time_index = 0"`` expression: if the
        slider is already at index 0 the client-side assignment is a
        no-op and no ``@state.change("time_index")`` listener runs, so
        the VTK view would keep showing the last-played frame. Here we
        always force a refresh by calling ``_apply_time_step_impl``.
        """
        state.playing = False
        times = list(state.time_values or [])
        state.time_index = 0
        state.current_time = times[0] if times else None
        state.loading = True
        try:
            _apply_time_step_impl()
        finally:
            state.loading = False

    @state.change("active_base")
    def _on_base(**_: object) -> None:
        if pipeline.reader is None:
            return
        active = [state.active_base] if state.active_base else []
        try:
            _apply_base_selection(pipeline.reader, active)
        except Exception as exc:  # noqa: BLE001
            state.status = f"Failed to update base: {exc}"
            return
        # Narrow the field dropdown to arrays that actually exist on the
        # newly-selected base.
        _refresh_field_options()
        _apply_pipeline(reset_camera=True)

    @state.change("field", "cmap", "show_edges")
    def _on_view_params(**_: object) -> None:
        _apply_pipeline()

    # --- Datasets root management ----------------------------------------

    def _reload_dataset_list() -> None:
        """Re-discover datasets under the (possibly new) datasets root."""
        try:
            with _silence_stderr():
                new_datasets = dataset_service.list_datasets()
        except Exception as exc:  # noqa: BLE001
            state.status = f"Failed to list datasets: {exc}"
            new_datasets = []
        hub_set = set(dataset_service.hub_repos)
        local_ids = [d.dataset_id for d in new_datasets if d.dataset_id not in hub_set]
        hub_ids = [d.dataset_id for d in new_datasets if d.dataset_id in hub_set]
        new_ids = local_ids + hub_ids
        state.local_dataset_ids = local_ids
        state.hub_dataset_ids = hub_ids
        state.dataset_ids = new_ids
        # Force ``dataset_id`` to change so ``@state.change('dataset_id')``
        # fires and cascades through splits / samples / view refresh.
        # Pick from the list that matches the active source tab.
        active_ids = hub_ids if state.source_tab == "hub" else local_ids
        state.dataset_id = active_ids[0] if active_ids else None

        if not new_ids:
            state.splits = []
            state.split = None
            state.sample_ids = []
            state.sample_id = None
            state.sample_count = 0
            state.base_options = []
            state.active_base = None
            state.field_options = []
            state.field = None
            state.sample_globals = []
            state.status = "No dataset found under the configured root."

    @ctrl.set("apply_datasets_root")
    def _apply_datasets_root() -> None:
        """Change the datasets root from the text field."""
        if not state.allow_root_change:
            return
        raw = (state.datasets_root_text or "").strip()
        if not raw:
            try:
                dataset_service.set_datasets_root(None)
            except Exception as exc:  # noqa: BLE001
                state.status = f"Failed to clear datasets root: {exc}"
                return
            _reload_dataset_list()
            state.status = "Datasets root cleared."
            return
        try:
            resolved = dataset_service.set_datasets_root(raw)
        except Exception as exc:  # noqa: BLE001
            state.status = f"Invalid datasets root: {exc}"
            return
        state.datasets_root_text = str(resolved) if resolved else ""
        _reload_dataset_list()
        state.status = f"Datasets root set to {resolved}"

    def _load_browse_view(path: str | None) -> None:
        try:
            listing = dataset_service.list_subdirs(path)
        except Exception as exc:  # noqa: BLE001
            state.status = f"Cannot browse: {exc}"
            return
        state.browse_cwd = listing["path"]
        state.browse_parent = listing["parent"]
        state.browse_entries = listing["entries"]

    @ctrl.set("open_browse_dialog")
    def _open_browse_dialog() -> None:
        if not state.allow_root_change:
            return
        start = (state.datasets_root_text or "").strip() or None
        try:
            _load_browse_view(start)
        except Exception:  # noqa: BLE001
            _load_browse_view(None)
        state.browse_dialog = True

    @ctrl.set("browse_cd")
    def _browse_cd(path: str) -> None:
        _load_browse_view(path)

    @ctrl.set("browse_up")
    def _browse_up() -> None:
        if state.browse_parent:
            _load_browse_view(state.browse_parent)

    @ctrl.set("browse_select")
    def _browse_select() -> None:
        """Use ``browse_cwd`` as the new datasets root."""
        state.datasets_root_text = state.browse_cwd
        state.browse_dialog = False
        _apply_datasets_root()

    @ctrl.set("add_hub_repo")
    def _add_hub_repo() -> None:
        """Register the repo id from the text field for streaming.

        Calls :meth:`PlaidDatasetService.add_hub_dataset`, then rebuilds
        the dataset list so the new entry is immediately selectable from
        the dropdown.
        """
        if not state.allow_root_change:
            return
        raw = (state.hub_repo_input or "").strip()
        if not raw:
            state.status = "Enter a Hugging Face repo id (e.g. namespace/name)."
            return
        try:
            normalised = dataset_service.add_hub_dataset(raw)
        except Exception as exc:  # noqa: BLE001
            state.status = f"Invalid repo id: {exc}"
            return
        state.hub_repos = list(dataset_service.hub_repos)
        state.hub_repo_input = ""
        _reload_dataset_list()
        # Select the newly added hub dataset to give immediate feedback.
        if normalised in (state.dataset_ids or []):
            state.dataset_id = normalised
        state.status = f"Streaming from {normalised}"

    @ctrl.set("remove_hub_repo")
    def _remove_hub_repo(repo_id: str) -> None:
        """Unregister a previously added hub repo."""
        if not state.allow_root_change:
            return
        dataset_service.remove_hub_dataset(repo_id)
        state.hub_repos = list(dataset_service.hub_repos)
        _reload_dataset_list()
        state.status = f"Removed hub dataset {repo_id}"

    @ctrl.set("stream_next")
    def _stream_next() -> None:
        """Advance the streaming cursor and load the next sample.

        Handler behind the "Next" button shown (instead of the sample
        slider) when the active dataset is a Hugging Face Hub stream.
        The cursor is advanced one step on the service-side
        ``_StreamCursor``; ``sample_id`` is then set to the new 0-based
        step number so the existing ``@state.change("sample_id")``
        plumbing fires and pushes the fresh sample through the VTK
        pipeline.
        """
        if not state.is_streaming or state.dataset_id is None:
            return
        if state.stream_exhausted:
            return
        split = state.split if state.split != "__default__" else None
        try:
            dataset_service.advance_stream_cursor(state.dataset_id, split)
        except StopIteration:
            state.stream_exhausted = True
            state.status = "Stream exhausted."
            return
        # Advance the UI counters. ``sample_id`` stays at the
        # ``STREAM_CURSOR_ID`` sentinel ("cursor") because
        # :meth:`PlaidDatasetService.load_sample` needs that sentinel to
        # route through ``converter.sample_to_plaid`` (IterableDatasets
        # have no ``to_plaid(dataset, index)`` random-access path).
        # Instead of mutating ``sample_id`` we refresh the view
        # directly; the service-side cursor has already moved one step
        # forward so ``load_sample`` will pick up the new record.
        new_position = int(state.stream_position) + 1
        state.stream_position = new_position
        state.sample_count = new_position + 1
        state.sample_index = new_position
        state.sample_id = STREAM_CURSOR_ID
        # ``sample_id`` did not actually change (both times the sentinel
        # ``STREAM_CURSOR_ID``), so the ``@state.change("sample_id")``
        # listener is skipped. Refresh the view directly instead. The
        # status bar text is set inside ``_refresh_sample_view_impl`` as
        # a 0-based step label for streaming mode.
        _refresh_sample_view()

    @ctrl.set("reset_camera")
    def _reset_camera() -> None:

        # With VtkRemoteView the camera lives on the server, so resetting
        # it server-side in ``pipeline.reset_camera`` and pushing a new
        # frame via ``ctrl.view_update`` is enough: the browser only
        # renders the images we send it.
        _apply_pipeline(reset_camera=True)

    # --- UI ---------------------------------------------------------------

    with SinglePageWithDrawerLayout(server) as layout:
        layout.title.set_text("Dataset Viewer")

        with layout.drawer as drawer:
            # Wider drawer to accommodate long CGNS feature paths such as
            # ``Base_2_2/Zone/FlowSolution/Pressure`` without wrapping.
            drawer.width = 460
            with v3.VContainer(classes="pa-2"):
                # Source-selection tabs: pick between a local datasets
                # root (``init_from_disk``) and Hugging Face Hub streaming
                # (``init_streaming_from_hub``). The tabs only drive which
                # form is rendered; registered datasets from either
                # source always land in ``dataset_ids`` together. Hidden
                # when ``--disable-root-change`` was passed on the CLI so
                # a public deployment can pin the root for good.
                with html.Div(v_if=("allow_root_change",), classes="mb-2"):
                    with v3.VTabs(
                        v_model=("source_tab",),
                        density="compact",
                        grow=True,
                        classes="mb-2",
                    ):
                        v3.VTab("Local", value="local")
                        v3.VTab("Hub", value="hub")
                    # Local datasets root form.
                    with html.Div(v_if=("source_tab === 'local'",)):
                        html.Div("Datasets root", classes="text-caption")
                        with html.Div(classes="d-flex align-center"):
                            v3.VTextField(
                                v_model=("datasets_root_text",),
                                density="compact",
                                hide_details=True,
                                placeholder="/absolute/path/to/datasets",
                                classes="mr-2",
                                clearable=True,
                                __events=[("keyup_enter", "keyup.enter")],
                                keyup_enter=ctrl.apply_datasets_root,
                            )
                            v3.VBtn(
                                icon="mdi-folder-open",
                                click=ctrl.open_browse_dialog,
                                density="compact",
                                variant="tonal",
                                classes="mr-1",
                            )
                            v3.VBtn(
                                icon="mdi-check",
                                click=ctrl.apply_datasets_root,
                                density="compact",
                                variant="tonal",
                                color="primary",
                            )
                    # Hugging Face Hub streaming form.
                    with html.Div(v_if=("source_tab === 'hub'",)):
                        html.Div(
                            "Hugging Face Hub dataset",
                            classes="text-caption",
                        )
                        with html.Div(classes="d-flex align-center"):
                            v3.VTextField(
                                v_model=("hub_repo_input",),
                                density="compact",
                                hide_details=True,
                                placeholder="namespace/name",
                                prepend_inner_icon="mdi-cloud-download",
                                classes="mr-2",
                                clearable=True,
                                __events=[("keyup_enter", "keyup.enter")],
                                keyup_enter=ctrl.add_hub_repo,
                            )
                            v3.VBtn(
                                icon="mdi-plus",
                                click=ctrl.add_hub_repo,
                                density="compact",
                                variant="tonal",
                                color="primary",
                            )
                        # Chip list of registered repos with a remove button.
                        with html.Div(
                            v_if=("(hub_repos || []).length > 0",),
                            classes="mt-1 d-flex flex-wrap",
                        ):
                            v3.VChip(
                                "{{ repo }}",
                                v_for="repo in hub_repos",
                                key="repo",
                                closable=True,
                                size="small",
                                classes="mr-1 mb-1",
                                click_close=(ctrl.remove_hub_repo, "[repo]"),
                            )
                    v3.VDivider(classes="my-2")

                # The dropdown ``items`` are filtered by ``source_tab``:
                # Local tab -> ``local_dataset_ids`` (``init_from_disk``
                # datasets), Hub tab -> ``hub_dataset_ids``
                # (``init_streaming_from_hub`` datasets). The user never
                # sees ids from the inactive source in the same menu.
                v3.VSelect(
                    label="Dataset",
                    v_model=("dataset_id",),
                    items=(
                        "source_tab === 'hub' ? hub_dataset_ids : local_dataset_ids",
                    ),
                    density="compact",
                )

                v3.VSelect(
                    label="Split",
                    v_model=("split",),
                    items=("splits",),
                    density="compact",
                )
                # Sample picker. Two mutually-exclusive widgets:
                # - Local datasets expose a random-access slider over
                #   the integer sample indices.
                # - Hub streaming datasets have no ``__len__`` and can
                #   only be consumed forward, so we expose a "Next"
                #   button that advances the ``_StreamCursor`` by one
                #   step via ``ctrl.stream_next``.
                html.Div("Sample", classes="text-caption mt-2")
                v3.VSlider(
                    v_if=("!is_streaming",),
                    v_model_number=("sample_index",),
                    min=0,
                    max=("sample_count > 0 ? sample_count - 1 : 0",),
                    step=1,
                    thumb_label=True,
                    hide_details=True,
                    disabled=("sample_count === 0",),
                )
                with html.Div(
                    v_if=("is_streaming",),
                    classes="d-flex align-center mb-1",
                ):
                    v3.VBtn(
                        "Next",
                        prepend_icon="mdi-arrow-right",
                        click=ctrl.stream_next,
                        disabled=("stream_exhausted",),
                        color="primary",
                        variant="tonal",
                        density="compact",
                        classes="mr-2",
                    )
                # Sample counter: for local datasets the slider exposes
                # all ids up-front; for streaming datasets we report the
                # step number (the total is unknown until the iterator
                # is exhausted, at which point "end of stream" appears).
                html.Div(
                    "{{ is_streaming"
                    " ? ('step ' + (stream_position + 1) + (stream_exhausted"
                    " ? ' (end of stream)' : ' (streaming)'))"
                    " : ((sample_id ?? '-') + ' / ' + sample_count + ' samples') }}",
                    classes="text-caption text-medium-emphasis mb-2",
                )

                # Time axis slider, only shown when the sample actually
                # exposes a time axis (time-dependent samples).
                with html.Div(v_if=("time_count > 1",), classes="mb-2"):
                    html.Div("Time", classes="text-caption mt-2")
                    v3.VSlider(
                        v_model_number=("time_index",),
                        min=0,
                        max=("time_count > 0 ? time_count - 1 : 0",),
                        step=1,
                        thumb_label=True,
                        hide_details=True,
                    )
                    html.Div(
                        "t = {{ current_time }} "
                        "<span class='text-medium-emphasis'>"
                        "({{ time_index + 1 }} / {{ time_count }})</span>",
                        classes="text-caption text-medium-emphasis",
                    )
                    # Playback controls: Play/Pause + FPS slider + loop.
                    with html.Div(classes="d-flex align-center mt-2"):
                        v3.VBtn(
                            icon=("playing ? 'mdi-pause' : 'mdi-play'",),
                            click="playing = !playing",
                            density="compact",
                            variant="tonal",
                            classes="mr-2",
                        )
                        v3.VBtn(
                            icon="mdi-stop",
                            click=ctrl.stop_playback,
                            density="compact",
                            variant="tonal",
                            classes="mr-2",
                        )
                        v3.VBtn(
                            icon=("play_loop ? 'mdi-repeat' : 'mdi-repeat-off'",),
                            click="play_loop = !play_loop",
                            density="compact",
                            variant="tonal",
                        )
                    html.Div("FPS: {{ play_fps }}", classes="text-caption mt-1")
                    v3.VSlider(
                        v_model_number=("play_fps",),
                        min=1,
                        max=30,
                        step=1,
                        hide_details=True,
                        density="compact",
                    )
                v3.VDivider(classes="my-2")
                html.Div("Base", classes="text-caption")

                with v3.VBtnToggle(
                    v_model=("active_base",),
                    mandatory=True,
                    density="compact",
                    divided=True,
                    classes="flex-wrap mb-2",
                ):
                    v3.VBtn(
                        "{{ base }}",
                        v_for="base in base_options",
                        key="base",
                        value=("base",),
                        size="small",
                    )
                v3.VSelect(
                    label="Field",
                    v_model=("field",),
                    items=("field_options",),
                    density="compact",
                )
                v3.VSelect(
                    label="Colormap",
                    v_model=("cmap",),
                    items=("cmaps",),
                    density="compact",
                )
                v3.VSwitch(
                    label="Show edges",
                    v_model=("show_edges",),
                    density="compact",
                    hide_details=True,
                )
                v3.VDivider(classes="my-2")
                v3.VBtn("Reset camera", click=ctrl.reset_camera, block=True)

                # Feature filter panel. Only rendered when the active
                # dataset exposes any feature path (otherwise the panel
                # would be empty and misleading). Driven by the
                # ``available_features`` / ``selected_features`` state
                # vectors populated by ``_refresh_available_features``;
                # the Apply button forwards the selection to
                # :meth:`PlaidDatasetService.set_features`, which in turn
                # invalidates the store cache and (for streaming
                # datasets) rebuilds the iterator with an
                # ``update_features_for_CGNS_compatibility`` expansion of
                # the user selection.
                # Feature filter panel. The expansion panel starts
                # collapsed: most users only need the "Load all" shortcut
                # button exposed above it, and the full checkbox list is
                # only expanded when they actually want to subset the
                # dataset. The top-level "Load all" button clears the
                # current selection and forces a reload without the user
                # having to open the panel at all.
                # Hidden for streaming (Hugging Face Hub) datasets:
                # feature filtering goes through ``init_streaming_from_hub``
                # which rebuilds the iterator from the dataset-wide
                # metadata union, a workflow that does not fit the
                # per-split viewer model and led to confusing "Missing
                # features" errors. Streaming users therefore always see
                # the full feature payload; local disk datasets keep the
                # complete feature selection UI unchanged.
                with html.Div(
                    v_if=("!is_streaming && (available_features || []).length > 0",),
                    classes="mt-3",
                ):
                    v3.VDivider(classes="my-2")
                    with html.Div(classes="d-flex align-center mb-1"):
                        html.Div("Features", classes="text-subtitle-2 flex-grow-1")
                        v3.VBtn(
                            "Load all",
                            click=ctrl.select_all_features,
                            size="x-small",
                            color="primary",
                            variant="tonal",
                        )
                    with v3.VExpansionPanels(variant="accordion", multiple=True):
                        with v3.VExpansionPanel():
                            v3.VExpansionPanelTitle(
                                "Select features ({{ (selected_features"
                                " || []).length }} / {{ (available_features"
                                " || []).length }})"
                            )
                            with v3.VExpansionPanelText():
                                html.Div(
                                    "Empty selection loads every feature.",
                                    classes="text-caption text-medium-emphasis mb-1",
                                )
                                with html.Div(classes="d-flex mb-1"):
                                    v3.VBtn(
                                        "Clear",
                                        click="selected_features = []",
                                        size="x-small",
                                        variant="text",
                                        classes="mr-1",
                                    )
                                    v3.VBtn(
                                        "Apply",
                                        click=ctrl.apply_features,
                                        size="x-small",
                                        color="primary",
                                        variant="tonal",
                                    )
                                with html.Div(
                                    style="max-height: 240px; overflow: auto;",
                                    classes="pa-1",
                                ):
                                    v3.VCheckbox(
                                        v_for="feat in available_features",
                                        key="feat",
                                        v_model=("selected_features",),
                                        value=("feat",),
                                        label=("feat",),
                                        density="compact",
                                        hide_details=True,
                                        multiple=True,
                                    )

                html.Div("{{ status }}", classes="text-caption mt-2")

                # PLAID globals for the current sample (filtered out of
                # ``IterationValues`` / ``TimeValues`` bookkeeping arrays).
                with html.Div(
                    v_if=("(sample_globals || []).length > 0",),
                    classes="mt-3",
                ):
                    html.Div("Globals", classes="text-subtitle-2 mb-1")
                    with v3.VList(density="compact"):
                        with v3.VListItem(v_for="g in sample_globals", key="g.name"):
                            v3.VListItemTitle(
                                "{{ g.name }} "
                                "<span class='text-caption text-medium-emphasis'>"
                                "({{ g.dtype }}, shape={{ g.shape }})"
                                "</span>"
                            )
                            v3.VListItemSubtitle(
                                "{{ g.preview }}", classes="text-caption"
                            )

        # File-system browser dialog for the datasets root. Scoped to the
        # server's ``browse_roots`` sandbox so the user can only reach
        # directories explicitly allowed by the operator.
        with v3.VDialog(v_model=("browse_dialog",), max_width="640"):
            with v3.VCard():
                v3.VCardTitle("Select datasets root")
                v3.VCardSubtitle(
                    "{{ browse_cwd }}", classes="text-caption text-medium-emphasis"
                )
                with v3.VCardText(style="max-height: 50vh; overflow: auto;"):
                    with v3.VList(density="compact"):
                        v3.VListItem(
                            prepend_icon="mdi-arrow-up",
                            title="..",
                            click=ctrl.browse_up,
                            v_if=("browse_parent",),
                        )
                        with v3.VListItem(
                            v_for="e in browse_entries",
                            key="e.path",
                            click=(ctrl.browse_cd, "[e.path]"),
                        ):
                            v3.VListItemTitle("{{ e.name }}")
                            v3.VListItemSubtitle(
                                "PLAID dataset",
                                v_if=("e.is_plaid_candidate",),
                                classes="text-success",
                            )
                with v3.VCardActions():
                    v3.VSpacer()
                    v3.VBtn(
                        "Cancel",
                        click="browse_dialog = false",
                        variant="text",
                    )
                    v3.VBtn(
                        "Use this directory",
                        click=ctrl.browse_select,
                        color="primary",
                        variant="tonal",
                    )

        # Indeterminate progress bar shown under the app bar while a sample
        # or time step is being loaded on the server.
        with layout.toolbar:
            # Small chip in the toolbar that advertises whether the
            # current dataset is streamed from the Hugging Face Hub (the
            # sample slider is then forward-only) or browsed from a
            # local PLAID directory (random access).
            v3.VChip(
                "streaming",
                v_if=("is_streaming",),
                size="small",
                color="secondary",
                prepend_icon="mdi-cloud-download",
                classes="mr-2",
            )
            v3.VProgressLinear(
                indeterminate=True,
                absolute=True,
                location="bottom",
                color="primary",
                v_if=("loading",),
            )

        with layout.content:
            with v3.VContainer(fluid=True, classes="fill-height pa-0 ma-0"):
                view = vtk_widgets.VtkRemoteView(pipeline.render_window, ref="view")

                ctrl.view_update = view.update
                ctrl.view_reset_camera = view.reset_camera

    # Trigger initial population.
    _refresh_splits()

    return server
