"""Smoke tests for the trame dataset viewer server."""

from __future__ import annotations

import json
import sys
import types
from html.parser import HTMLParser
from pathlib import Path

import pytest


@pytest.fixture
def headless_trame_view(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid creating a native VTK interactor in UI/state-only tests.

    Args:
        monkeypatch: Pytest fixture for replacing rendering components.
    """
    from trame.widgets import html, vtk  # noqa: PLC0415

    from plaid.viewer.trame_app import server as server_module  # noqa: PLC0415

    class HeadlessPipeline:
        """Provide only the empty-scene operations used by state callbacks."""

        def __init__(self) -> None:
            self.render_window = None
            self.reader = None
            self.mapper = types.SimpleNamespace(
                RemoveAllInputConnections=lambda _port: None,
                ScalarVisibilityOff=lambda: None,
            )
            self.scalar_bar = types.SimpleNamespace(SetVisibility=lambda _value: None)

    class HeadlessRemoteView(html.Div):
        """Keep the trame UI tree without registering a VTK render window."""

        def __init__(self, _render_window: object, **kwargs: object) -> None:
            super().__init__(**kwargs)

        def update(self) -> None:
            """No frame is needed for state-only tests."""

        def reset_camera(self) -> None:
            """No camera is needed for state-only tests."""

    monkeypatch.setattr(server_module, "_VtkPipeline", HeadlessPipeline)
    monkeypatch.setattr(vtk, "VtkRemoteView", HeadlessRemoteView)


class _FakeCGNSReader:
    def __init__(self) -> None:
        self.file_name: str | None = None
        self.enable_calls: list[str] = []

    def SetFileName(self, name: str) -> None:  # noqa: N802 - VTK API
        self.file_name = name

    def UpdateInformation(self) -> None:  # noqa: N802 - VTK API
        self.enable_calls.append("UpdateInformation")

    def EnableAllBases(self) -> None:  # noqa: N802 - VTK API
        self.enable_calls.append("EnableAllBases")

    def EnableAllPointArrays(self) -> None:  # noqa: N802 - VTK API
        self.enable_calls.append("EnableAllPointArrays")

    def EnableAllCellArrays(self) -> None:  # noqa: N802 - VTK API
        self.enable_calls.append("EnableAllCellArrays")


class _FakeCGNSFileSeriesReader:
    def __init__(self) -> None:
        self.inner: _FakeCGNSReader | None = None
        self.file_names: list[str] = []
        self.update_information_calls = 0

    def SetReader(self, inner) -> None:  # noqa: N802 - VTK API
        self.inner = inner

    def AddFileName(self, name: str) -> None:  # noqa: N802 - VTK API
        self.file_names.append(name)

    def UpdateInformation(self) -> None:  # noqa: N802 - VTK API
        self.update_information_calls += 1


def test_pipeline_load_keeps_camera_until_explicit_reset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replacing a reader must not move the server-side camera."""
    import vtk  # noqa: PLC0415

    from plaid.viewer.trame_app import server as server_module  # noqa: PLC0415

    pipeline = server_module._VtkPipeline.__new__(server_module._VtkPipeline)
    pipeline.renderer = vtk.vtkRenderer()
    pipeline.camera_initialized = True
    camera = pipeline.renderer.GetActiveCamera()
    camera.SetPosition(4, 5, 6)
    camera.SetFocalPoint(1, 2, 3)
    camera.SetViewUp(0, 0, 1)
    camera.SetParallelScale(7)
    monkeypatch.setattr(server_module, "_load_reader", lambda _path: object())

    pipeline.load(tmp_path / "first.cgns")
    pipeline.load(tmp_path / "second.cgns")
    assert camera.GetPosition() == (4, 5, 6)
    assert camera.GetFocalPoint() == (1, 2, 3)
    assert camera.GetViewUp() == (0, 0, 1)
    assert camera.GetParallelScale() == 7

    pipeline.reset_camera()
    assert camera.GetPosition() != (4, 5, 6)


@pytest.mark.usefixtures("headless_trame_view")
@pytest.mark.parametrize(
    ("next_times", "expected_index", "expected_time"),
    [
        ([0.0, 2.0, 4.0], 1, 2.0),
        ([0.0, 2.5, 5.0], 1, 2.5),
        ([0.0, 1.0], 1, 1.0),
        ([], 0, None),
    ],
)
def test_sample_change_preserves_closest_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    next_times: list[float],
    expected_index: int,
    expected_time: float | None,
) -> None:
    """Changing samples keeps the physical time when possible."""
    from trame import app  # noqa: PLC0415

    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services.paraview_artifact_service import (  # noqa: PLC0415
        ParaviewArtifactService,
    )
    from plaid.viewer.services.plaid_dataset_service import (  # noqa: PLC0415
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app import server as server_module  # noqa: PLC0415
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    get_server = app.get_server
    monkeypatch.setattr(
        app,
        "get_server",
        lambda **kwargs: get_server(name=f"time-{next_times}", **kwargs),
    )

    class Reader:
        """Represent a loaded mesh in the headless test."""

        def Update(self) -> None:  # noqa: N802 - VTK API
            """Accept an update to the current step."""

        def GetOutput(self) -> None:  # noqa: N802 - VTK API
            """Return an empty mesh."""

    class Pipeline:
        """Keep the camera's initialization state across sample loads."""

        def __init__(self) -> None:
            self.render_window = None
            self.reader = None
            self.camera_initialized = False
            self.mapper = types.SimpleNamespace(
                RemoveAllInputConnections=lambda _port: None,
                ScalarVisibilityOff=lambda: None,
            )
            self.scalar_bar = types.SimpleNamespace(SetVisibility=lambda _value: None)
            self.reset_count = 0

        def load(self, _path: Path) -> None:
            """Replace the reader but retain camera state."""
            self.reader = Reader()

        def update(self, **_kwargs: object) -> None:
            """Accept a pipeline update."""

        def reset_camera(self) -> None:
            """Record explicit or initial camera resets."""
            self.reset_count += 1
            self.camera_initialized = True

    pipeline = Pipeline()
    reader_times: list[float] = []
    monkeypatch.setattr(server_module, "_VtkPipeline", lambda: pipeline)
    monkeypatch.setattr(
        server_module, "_reader_bases_and_fields", lambda _reader: ([], [], [])
    )
    monkeypatch.setattr(server_module, "_apply_base_selection", lambda *_args: None)
    monkeypatch.setattr(
        server_module,
        "_advance_reader_time",
        lambda _reader, time: reader_times.append(time),
    )

    class TimedService(PlaidDatasetService):
        """Expose distinct time axes for local samples."""

        def get_features(self, dataset_id: str) -> list[str]:  # noqa: ARG002
            """Keep a mesh feature enabled while navigating samples."""
            return ["Base_2_3"]

        def list_time_values(self, ref) -> list[float]:
            """Return the time axis for the selected sample."""
            return [0.0, 2.0, 4.0] if ref.sample_id == "0" else next_times

    root = Path(__file__).resolve().parents[1] / "containers"
    service = TimedService(ViewerConfig(datasets_root=root))
    server = build_server(service, ParaviewArtifactService(service, tmp_path))
    server.state.ready()
    server.state.active_base = "Base_2_3"
    server.state.flush()
    assert server.state.status.startswith("Loaded sample")
    server.state.time_index = 1
    server.state.flush()
    assert server.state.current_time == 2.0

    server.state.sample_index = 1
    server.state.flush()
    assert server.state.sample_id == "1"
    assert pipeline.reset_count == 1
    assert server.state.time_values == next_times
    assert server.state.time_index == expected_index
    assert server.state.current_time == expected_time
    if expected_time is not None:
        assert reader_times[-1] == expected_time


@pytest.mark.parametrize("allow_root_change", [True, False])
@pytest.mark.usefixtures("headless_trame_view")
def test_dataset_selector_is_independent_of_root_controls(
    tmp_path: Path,
    allow_root_change: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep dataset selection available when the datasets root is pinned."""
    from trame import app  # noqa: PLC0415

    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services.paraview_artifact_service import (  # noqa: PLC0415
        ParaviewArtifactService,
    )
    from plaid.viewer.services.plaid_dataset_service import (  # noqa: PLC0415
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    get_server = app.get_server
    monkeypatch.setattr(
        app,
        "get_server",
        lambda **kwargs: get_server(
            name=f"dataset-selector-{allow_root_change}", **kwargs
        ),
    )

    class DatasetSelectorParser(HTMLParser):
        """Record visibility conditions surrounding the dataset selector."""

        def __init__(self) -> None:
            super().__init__()
            self.conditions: list[str] = []
            self.selector_conditions: list[str] | None = None

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            """Track open div conditions and inspect the dataset selector."""
            attributes = dict(attrs)
            if tag == "div":
                self.conditions.append(attributes.get("v-if", ""))
            if tag == "vselect" and attributes.get("label") == "Dataset":
                self.selector_conditions = list(self.conditions)

        def handle_endtag(self, tag: str) -> None:
            """Remove the visibility condition for a closed div."""
            if tag == "div":
                self.conditions.pop()

    service = PlaidDatasetService(
        ViewerConfig(datasets_root=tmp_path, allow_root_change=allow_root_change)
    )
    server = build_server(service, ParaviewArtifactService(service, tmp_path))
    server.state.ready()

    parser = DatasetSelectorParser()
    parser.feed(server.state["trame__template_main"])
    assert server.state.allow_root_change is allow_root_change
    assert parser.selector_conditions is not None
    assert "allow_dataset_change" in parser.selector_conditions
    assert "allow_root_change" not in parser.selector_conditions


@pytest.mark.usefixtures("headless_trame_view")
def test_open_in_paraview_uses_current_local_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Launch only the selected local dataset, not another root or Hub repo."""
    from trame import app  # noqa: PLC0415

    from plaid.viewer import paraview_session  # noqa: PLC0415
    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services import (  # noqa: PLC0415
        ParaviewArtifactService,
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    for name in ("first", "second"):
        (tmp_path / name / "data").mkdir(parents=True)
    get_server = app.get_server
    monkeypatch.setattr(
        app,
        "get_server",
        lambda **kwargs: get_server(name="open-paraview-test", **kwargs),
    )
    launched: list[Path] = []

    def launch(path: Path) -> types.SimpleNamespace:
        """Record the selected path without starting a desktop application."""
        launched.append(path)
        return types.SimpleNamespace(port=8123)

    monkeypatch.setattr(paraview_session, "open_dataset_in_paraview", launch)
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset("org/repo")
    server = build_server(service, ParaviewArtifactService(service, tmp_path))
    server.state.ready()
    assert "Open in ParaView" in server.state["trame__template_main"]
    assert "Add paraview to your PATH" in server.state["trame__template_main"]
    assert "PARAVIEW_EXEC" in server.state["trame__template_main"]
    assert "Hub datasets are not supported" in server.state["trame__template_main"]

    server.state.dataset_id = "second"
    server.state.flush()
    server.controller.open_in_paraview()
    assert launched == [tmp_path / "second"]
    assert "8123" in server.state.paraview_status

    server.state.dataset_id = "org/repo"
    server.state.flush()
    server.controller.open_in_paraview()
    assert launched == [tmp_path / "second"]
    assert "Select a local dataset" in server.state.paraview_status

    server.state.dataset_id = "first"
    server.state.flush()
    monkeypatch.setattr(
        paraview_session,
        "open_dataset_in_paraview",
        lambda _path: (_ for _ in ()).throw(OSError("not installed")),
    )
    server.controller.open_in_paraview()
    assert "not installed" in server.state.paraview_status

    def missing_executable(_path: Path) -> None:
        """Model the launcher failing to find the ParaView executable."""
        raise FileNotFoundError(2, "No such file or directory", "paraview")

    monkeypatch.setattr(
        paraview_session, "open_dataset_in_paraview", missing_executable
    )
    server.controller.open_in_paraview()
    assert "paraview to your PATH" in server.state.paraview_status
    assert "PARAVIEW_EXEC" in server.state.paraview_status

    monkeypatch.setenv("PARAVIEW_EXEC", "/opt/paraview/bin/paraview")

    def missing_custom_executable(_path: Path) -> None:
        """Model a missing custom ParaView executable."""
        raise FileNotFoundError(
            2, "No such file or directory", "/opt/paraview/bin/paraview"
        )

    monkeypatch.setattr(
        paraview_session, "open_dataset_in_paraview", missing_custom_executable
    )
    server.controller.open_in_paraview()
    assert "PARAVIEW_EXEC" in server.state.paraview_status

    monkeypatch.setattr(
        paraview_session,
        "open_dataset_in_paraview",
        lambda _path: (_ for _ in ()).throw(FileNotFoundError("dataset missing")),
    )
    server.controller.open_in_paraview()
    assert "dataset missing" in server.state.paraview_status
    assert "Add paraview" not in server.state.paraview_status


@pytest.mark.usefixtures("headless_trame_view")
def test_local_sample_picker_has_number_input_before_slider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Use native steppers and a numeric maximum for local sample navigation."""
    from trame import app  # noqa: PLC0415

    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services.paraview_artifact_service import (  # noqa: PLC0415
        ParaviewArtifactService,
    )
    from plaid.viewer.services.plaid_dataset_service import (  # noqa: PLC0415
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    class SamplePickerParser(HTMLParser):
        """Collect the controls inside the local-only sample picker."""

        def __init__(self) -> None:
            super().__init__()
            self.depth = 0
            self.controls: list[tuple[str, dict[str, str | None]]] = []

        def handle_starttag(
            self, tag: str, attrs: list[tuple[str, str | None]]
        ) -> None:
            """Record controls nested inside the local picker.

            Args:
                tag: HTML element name.
                attrs: Element attributes.
            """
            attributes = dict(attrs)
            if tag == "div":
                if self.depth:
                    self.depth += 1
                elif attributes.get("v-if") == "!is_streaming":
                    self.depth = 1
            elif self.depth:
                self.controls.append((tag, attributes))

        def handle_endtag(self, tag: str) -> None:
            """Leave the picker when its outer div closes.

            Args:
                tag: HTML element name.
            """
            if tag == "div" and self.depth:
                self.depth -= 1

    get_server = app.get_server
    monkeypatch.setattr(
        app,
        "get_server",
        lambda **kwargs: get_server(name="sample-picker-layout", **kwargs),
    )
    dataset_root = Path(__file__).resolve().parents[1] / "containers"
    service = PlaidDatasetService(ViewerConfig(datasets_root=dataset_root))
    server = build_server(service, ParaviewArtifactService(service, tmp_path))
    server.state.ready()
    parser = SamplePickerParser()
    parser.feed(server.state["trame__template_main"])

    assert [tag for tag, _ in parser.controls] == ["vnumberinput", "vslider"]
    number_input, slider = (attrs for _, attrs in parser.controls)
    # Vuetify parses the number input's text internally. Vue's .number
    # modifier on it turns its intermediate text into a number too early,
    # triggering a `t.replace is not a function` JS error while typing.
    assert number_input["v-model"] == slider["v-model.number"] == "sample_index"
    assert "v-model.number" not in number_input
    # Native increment computes model + step; static HTML step="1" can
    # reach Vuetify as a string and produce "01" instead of 1.
    assert number_input[":min"] == slider["min"] == "0"
    assert number_input[":max"] == slider[":max"] == "sample_max_index"
    assert number_input[":step"] == slider["step"] == "1"
    assert number_input["controlvariant"] == "stacked"
    assert number_input[":precision"] == "0"
    assert number_input[":disabled"] == slider[":disabled"]
    assert server.state.sample_count == 10
    assert server.state.sample_index == 0
    assert server.state.sample_max_index == 9
    assert len(server.state.sample_ids) == 10

    server.state.sample_index = 1
    server.state.flush()
    assert server.state.sample_id == "1"
    assert server.state.sample_max_index == 9
    assert server.state.sample_count == 10

    server.state.sample_index = "5"
    server.state.flush()
    assert server.state.sample_index == 5
    assert server.state.sample_id == "5"

    server.state.sample_index = 20
    server.state.flush()
    assert server.state.sample_index == 9
    assert server.state.sample_id == "9"

    server.state.sample_index = -1
    server.state.flush()
    assert server.state.sample_index == 0
    assert server.state.sample_id == "0"


def test_load_reader_series_uses_vtk_cgns_file_series_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A ``.cgns.series`` sidecar must drive a ``vtkCGNSFileSeriesReader``.

    This guards against regressing to ``vtkFileSeriesReader``, which is not
    available in the ``vtk`` PyPI wheel and would silently break time-series
    rendering.
    """
    series_path = tmp_path / "meshes.cgns.series"
    sidecar = {
        "file-series-version": "1.0",
        "files": [
            {"name": "meshes/mesh_000000001.cgns", "time": 1.5},
            {"name": "meshes/mesh_000000000.cgns", "time": 0.0},
        ],
    }
    series_path.write_text(json.dumps(sidecar))

    fake_vtk = types.SimpleNamespace(
        vtkCGNSReader=_FakeCGNSReader,
        vtkCGNSFileSeriesReader=_FakeCGNSFileSeriesReader,
    )
    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)

    from plaid.viewer.trame_app.server import _load_reader  # noqa: PLC0415

    reader = _load_reader(series_path)

    assert isinstance(reader, _FakeCGNSFileSeriesReader)
    assert isinstance(reader.inner, _FakeCGNSReader)
    # File names are added in ascending time order, not sidecar order.
    expected_order = [
        str((tmp_path / "meshes/mesh_000000000.cgns").resolve()),
        str((tmp_path / "meshes/mesh_000000001.cgns").resolve()),
    ]
    assert reader.file_names == expected_order
    assert reader.update_information_calls == 1
    # Inner reader must have had its selections enabled so the pipeline
    # produces non-empty output.
    assert reader.inner.enable_calls == [
        "EnableAllBases",
        "EnableAllPointArrays",
        "EnableAllCellArrays",
    ]


def test_select_initial_dataset_id_prefers_configured_dataset() -> None:
    from plaid.viewer.trame_app.server import (
        _select_initial_dataset_id,  # noqa: PLC0415
    )

    assert _select_initial_dataset_id("b", ["a", "b"], ["org/repo"]) == "b"
    assert (
        _select_initial_dataset_id("org/repo", ["a", "b"], ["org/repo"]) == "org/repo"
    )


def test_select_initial_dataset_id_falls_back_to_existing_dataset() -> None:
    from plaid.viewer.trame_app.server import (
        _select_initial_dataset_id,  # noqa: PLC0415
    )

    assert _select_initial_dataset_id("missing", ["a", "b"], ["org/repo"]) == "a"
    assert _select_initial_dataset_id(None, [], ["org/repo"]) == "org/repo"
    assert _select_initial_dataset_id(None, [], []) is None


def test_update_view_skips_push_before_protocol_is_ready() -> None:
    from plaid.viewer.trame_app.server import _update_view  # noqa: PLC0415

    calls: list[None] = []
    server = types.SimpleNamespace(protocol=None)
    ctrl = types.SimpleNamespace(view_update=lambda: calls.append(None))

    _update_view(server, ctrl)

    assert calls == []


def test_update_view_pushes_frame_when_protocol_is_ready() -> None:
    from plaid.viewer.trame_app.server import _update_view  # noqa: PLC0415

    calls: list[None] = []
    server = types.SimpleNamespace(protocol=object())
    ctrl = types.SimpleNamespace(view_update=lambda: calls.append(None))

    _update_view(server, ctrl)

    assert calls == [None]


@pytest.mark.usefixtures("headless_trame_view")
def test_color_range_controls_follow_field_and_reject_invalid_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the actual trame callbacks with successive field ranges."""
    from trame import app  # noqa: PLC0415

    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services import (  # noqa: PLC0415
        ParaviewArtifactService,
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app import server as module  # noqa: PLC0415

    get_server = app.get_server
    monkeypatch.setattr(
        app,
        "get_server",
        lambda **kwargs: get_server(name="color-range-controls", **kwargs),
    )
    field_range = [1.0, 5.0]
    applied: list[tuple[float, float]] = []

    class Reader:
        """Return a distinct field range on successive redraws."""

        def GetOutput(self) -> None:  # noqa: N802
            """Represent the current reader output."""

    class Pipeline:
        """Record the color range passed to the actual VTK pipeline."""

        def __init__(self) -> None:
            self.reader = None
            self.render_window = None
            self.mapper = types.SimpleNamespace(
                RemoveAllInputConnections=lambda _port: None,
                ScalarVisibilityOff=lambda: None,
            )
            self.scalar_bar = types.SimpleNamespace(SetVisibility=lambda _value: None)

        def update(self, **kwargs: object) -> None:
            """Record the scalar limits selected by the server."""
            applied.append(kwargs["color_range"])

    pipeline = Pipeline()
    monkeypatch.setattr(module, "_VtkPipeline", lambda: pipeline)
    monkeypatch.setattr(module, "_compute_field_range", lambda *_: tuple(field_range))
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    server = module.build_server(service, ParaviewArtifactService(service, tmp_path))
    state = server.state
    state.ready()
    pipeline.reader = Reader()
    state.field = "point:pressure"
    state.flush()
    assert applied[-1] == (1.0, 5.0)
    field_range[:] = [2.0, 4.0]
    state.color_min_mode = "Grow only"
    state.color_max_mode = "Grow only"
    state.flush()
    assert applied[-1] == (1.0, 5.0)
    field_range[:] = [-3.0, 7.0]
    state.cmap = "jet"
    state.flush()
    assert applied[-1] == (-3.0, 7.0)
    state.color_max_mode = "Fix"
    state.color_max = "10"
    state.flush()
    assert applied[-1] == (-3.0, 10.0)
    state.color_min = "oops"  # Disabled in the UI while Grow only.
    state.color_min_mode = "Fix"
    state.flush()
    assert applied[-1] == (-3.0, 10.0)
    assert state.color_range_error
    assert state.color_min == "oops"
    state.color_min = "-5"
    state.flush()
    assert applied[-1] == (-5.0, 10.0)
    assert not state.color_range_error
    state.color_min = "11"
    state.flush()
    assert applied[-1] == (-5.0, 10.0)
    assert "less than" in state.color_range_error
    state.color_min = "-5"
    state.flush()
    state.color_min_mode = "Dynamic"
    field_range[:] = [0.0, 3.0]
    state.flush()
    assert applied[-1] == (0.0, 10.0)
    state.field = "cell:temperature"
    field_range[:] = [1.0, 2.0]
    state.flush()
    assert applied[-1] == (1.0, 10.0)


@pytest.mark.usefixtures("headless_trame_view")
def test_color_range_ui_has_independent_limits_and_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose both editable values and independent mode selectors."""
    from trame import app  # noqa: PLC0415

    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services import (  # noqa: PLC0415
        ParaviewArtifactService,
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    get_server = app.get_server
    monkeypatch.setattr(
        app,
        "get_server",
        lambda **kwargs: get_server(name="color-range-ui", **kwargs),
    )
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    server = build_server(service, ParaviewArtifactService(service, tmp_path))
    server.state.ready()
    template = server.state["trame__template_main"]
    for name in ("min", "max"):
        assert f'label="{name.title()}"' in template
        assert f'v-model="color_{name}"' in template
        assert f'v-model="color_{name}_mode"' in template
    assert server.state.color_range_modes == ["Dynamic", "Fix", "Grow only"]


@pytest.mark.usefixtures("headless_trame_view")
def test_iso_contour_switch_reaches_the_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The Iso contour switch is shown and redraws the view with 9 bands."""
    from trame import app  # noqa: PLC0415

    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services import (  # noqa: PLC0415
        ParaviewArtifactService,
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app import server as module  # noqa: PLC0415

    get_server = app.get_server
    monkeypatch.setattr(
        app,
        "get_server",
        lambda **kwargs: get_server(name="iso-contour", **kwargs),
    )
    applied: list[object] = []

    class Reader:
        def GetOutput(self) -> None:  # noqa: N802
            """Represent the current reader output."""

    class Pipeline:
        """Record the iso-contour flag passed to the VTK pipeline."""

        def __init__(self) -> None:
            self.reader = None
            self.render_window = None
            self.mapper = types.SimpleNamespace(
                RemoveAllInputConnections=lambda _port: None,
                ScalarVisibilityOff=lambda: None,
            )
            self.scalar_bar = types.SimpleNamespace(SetVisibility=lambda _value: None)

        def update(self, **kwargs: object) -> None:
            applied.append(kwargs["iso_contour"])

    pipeline = Pipeline()
    monkeypatch.setattr(module, "_VtkPipeline", lambda: pipeline)
    monkeypatch.setattr(module, "_compute_field_range", lambda *_: (0.0, 1.0))
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    server = module.build_server(service, ParaviewArtifactService(service, tmp_path))
    state = server.state
    state.ready()

    template = state["trame__template_main"]
    assert 'label="Iso contour"' in template
    assert 'v-model="iso_contour"' in template
    assert state.iso_contour is False

    pipeline.reader = Reader()
    state.field = "point:pressure"
    state.flush()
    assert applied[-1] is False
    state.iso_contour = True
    state.flush()
    assert applied[-1] is True
    state.iso_contour = False
    state.flush()
    assert applied[-1] is False


@pytest.mark.usefixtures("headless_trame_view")
def test_streaming_base_options_survive_initialisation_and_split_switch(
    tmp_path: Path,
) -> None:
    """A streamed dataset's metadata bases remain selectable before Next."""
    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.models import DatasetDetail  # noqa: PLC0415
    from plaid.viewer.services.paraview_artifact_service import (  # noqa: PLC0415
        ParaviewArtifactService,
    )
    from plaid.viewer.services.plaid_dataset_service import (  # noqa: PLC0415
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    repo_id = "test/force_asr"
    mesh_path = "Base_2_2/Zone/GridCoordinates/CoordinateX"
    field_path = "Base_2_2/Zone/VertexFields/pressure"

    class StreamingService(PlaidDatasetService):
        """Use real metadata filtering without contacting the Hub."""

        def is_streaming(self, dataset_id: str) -> bool:  # noqa: ARG002
            """Mark the registered repository as a stream."""
            return True

        def _open(self, dataset_id: str) -> tuple[dict, dict]:  # noqa: ARG002
            """Return empty streams without connecting to the Hub."""
            return {"res_SENS": iter(()), "other": iter(())}, {}

        def get_dataset(self, dataset_id: str) -> DatasetDetail:
            """Expose two splits without opening a remote dataset."""
            return DatasetDetail(
                dataset_id=dataset_id,
                is_streaming=True,
                path=f"hf://{dataset_id}",
                splits={"res_SENS": None, "other": None},
            )

        def reset_stream_cursor(self, dataset_id: str, split: str | None) -> None:
            """Avoid fetching Parquet data while resetting an empty stream."""

    service = StreamingService(ViewerConfig())
    service.add_hub_dataset(repo_id)
    service._feature_metadata[repo_id] = (["Base_2_2", mesh_path], [field_path])
    artifact_service = ParaviewArtifactService(service, tmp_path)
    server = build_server(service, artifact_service)

    # State callbacks are dispatched only when the trame state becomes ready.
    server.state.ready()
    assert server.state.base_options == ["Base_2_2"]
    assert server.state.sample_id is None
    assert server.state.stream_position == -1

    server.state.split = "other"
    server.state.flush()
    assert server.state.base_options == ["Base_2_2"]
    assert server.state.sample_id is None

    server.state.active_base = "Base_2_2"
    server.state.flush()
    assert server.state.base_options == ["Base_2_2"]
    assert mesh_path in service.get_features(repo_id)
    assert field_path in server.state.available_features
    assert field_path not in service.get_features(repo_id)

    # Applying a field after a mesh-only sample was displayed must
    # invalidate the previous step as well as update the feature filter.
    server.state.stream_position = 0
    server.state.sample_id = "cursor"
    server.state.field_options = ["point:old_field"]
    server.state.selected_features = [field_path]
    server.controller.apply_features()
    server.state.flush()

    assert mesh_path in service.get_features(repo_id)
    assert field_path in service.get_features(repo_id)
    assert server.state.stream_position == -1
    assert server.state.sample_id is None
    assert server.state.field_options == []


@pytest.mark.usefixtures("headless_trame_view")
def test_time_step_change_discards_extracted_globals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Switching time step drops stale Globals without re-extracting them."""
    from trame import app  # noqa: PLC0415

    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services.paraview_artifact_service import (  # noqa: PLC0415
        ParaviewArtifactService,
    )
    from plaid.viewer.services.plaid_dataset_service import (  # noqa: PLC0415
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    get_server = app.get_server
    monkeypatch.setattr(
        app, "get_server", lambda **kwargs: get_server(name="time-step", **kwargs)
    )
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    monkeypatch.setattr(
        service,
        "extract_globals",
        lambda *_a, **_k: pytest.fail("extraction must not be automatic"),
    )
    server = build_server(service, ParaviewArtifactService(service, tmp_path))
    server.state.ready()

    server.state.explore_names = ["Global/a"]
    server.state.explore_selected_globals = ["Global/a"]
    server.state.explore_csv_url = "csv/all-token"
    server.state.explore_report_url = "report/old.html"
    server.state.explore_compare_splits = True
    server.state.flush()

    server.state.explore_time_step = "first"
    server.state.flush()

    assert server.state.explore_names == []
    assert server.state.explore_selected_globals == []
    assert server.state.explore_csv_url == ""
    assert server.state.explore_report_url == ""
    assert server.state.explore_compare_splits is False
    assert server.state.explore_status == "Click Extract Globals to load all splits."


@pytest.mark.usefixtures("headless_trame_view")
def test_skrub_split_switch_clears_report_and_keeps_global_default(
    tmp_path: Path,
) -> None:
    """Switching report modes invalidates the previous link in either direction.

    Args:
        tmp_path: Temporary directory for the viewer artifact cache.
    """
    from plaid.viewer.config import ViewerConfig  # noqa: PLC0415
    from plaid.viewer.services.paraview_artifact_service import (  # noqa: PLC0415
        ParaviewArtifactService,
    )
    from plaid.viewer.services.plaid_dataset_service import (  # noqa: PLC0415
        PlaidDatasetService,
    )
    from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    server = build_server(service, ParaviewArtifactService(service, tmp_path))
    state = server.state
    state.ready()

    markup = state["trame__template_main"]
    assert state.explore_compare_splits is False
    assert 'label="Compare individual splits"' in markup
    assert 'v-model="explore_compare_splits"' in markup
    assert 'disabled="splits.length < 2"' in markup
    assert 'label="Compare splits"' not in markup
    assert markup.count("Generate skrub report") == 1

    for enabled in (True, False):
        state.explore_report_url = "previous-report.html"
        state.explore_compare_splits = enabled
        state.flush()
        assert state.explore_report_url == ""
        assert state.explore_compare_splits is enabled
