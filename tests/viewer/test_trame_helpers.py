"""Headless tests for trame server helper functions using fakes."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import yaml

from plaid.viewer.trame_app import server as srv


@pytest.mark.parametrize(
    ("mode", "is_min", "previous", "expected"),
    [
        ("Dynamic", True, -10.0, 2.0),
        ("Dynamic", False, 20.0, 2.0),
        ("Grow only", True, -10.0, -10.0),
        ("Grow only", True, 5.0, 2.0),
        ("Grow only", False, 20.0, 20.0),
        ("Grow only", False, 1.0, 2.0),
        ("Grow only", True, None, 2.0),
        ("Fix", True, None, 7.5),
    ],
)
def test_resolve_color_limit(
    mode: str, is_min: bool, previous: float | None, expected: float
) -> None:
    """Resolve each end independently, using input only for fixed limits."""
    assert (
        srv._resolve_color_limit(mode, 2.0, "7.5", previous, is_min=is_min) == expected
    )


@pytest.mark.parametrize("value", ["", "abc", "nan", "inf", None])
def test_fixed_color_limit_requires_finite_number(value: object) -> None:
    """Reject missing, invalid and non-finite fixed values."""
    with pytest.raises(ValueError, match="finite numeric"):
        srv._resolve_color_limit("Fix", 2.0, value, None, is_min=True)


def test_filter_global_names_regex_and_invalid_pattern() -> None:
    names = ["Global/pressure", "Global/temperature", "Global/pressure_1"]

    assert srv._filter_global_names(names, r"pressure(_\d+)?$") == (
        ["Global/pressure", "Global/pressure_1"],
        "",
    )
    assert srv._filter_global_names(names, "") == (names, "")
    assert srv._filter_global_names(names, "missing") == ([], "")
    matches, error = srv._filter_global_names(names, "[")
    assert matches == []
    assert error.startswith("Invalid regular expression:")


def test_label_options_follow_selected_globals_and_keep_label_only_fields() -> None:
    """Keep sample IDs and text labels while narrowing numeric labels."""
    labels = ["sample_id", "Global/a", "Global/case", "Global/b"]
    names = ["Global/a", "Global/b"]

    assert srv._label_options(labels, names, []) == labels
    assert srv._label_options(labels, names, ["Global/b"]) == [
        "sample_id",
        "Global/case",
        "Global/b",
    ]
    assert srv._label_options(labels, names, ["Global/a"]) == [
        "sample_id",
        "Global/a",
        "Global/case",
    ]


def test_select_filtered_globals_preserves_hidden_selection() -> None:
    names = ["Global/a", "Global/b", "Global/c", "Global/d"]
    selected = ["Global/b", "Global/d"]
    filtered = ["Global/a", "Global/b"]

    assert srv._select_filtered_globals(names, selected, filtered, True) == [
        "Global/a",
        "Global/b",
        "Global/d",
    ]
    assert srv._select_filtered_globals(names, selected, filtered, False) == [
        "Global/d"
    ]
    assert srv._select_filtered_globals(names, selected, [], False) == selected
    assert srv._select_filtered_globals(names, selected, [], True) == selected


class _Selection:
    def __init__(self, names: list[str]) -> None:
        self.names = names
        self.enabled: list[str] = []
        self.disabled: list[str] = []

    def GetNumberOfArrays(self) -> int:  # noqa: N802
        return len(self.names)

    def GetArrayName(self, i: int) -> str:  # noqa: N802
        return self.names[i]

    def ArrayExists(self, name: str) -> bool:  # noqa: N802
        return name in self.names

    def DisableArray(self, name: str) -> None:  # noqa: N802
        self.disabled.append(name)

    def DisableAllArrays(self) -> None:  # noqa: N802
        self.disabled.extend(self.names)

    def EnableArray(self, name: str) -> None:  # noqa: N802
        self.enabled.append(name)


class _Reader:
    def __init__(self) -> None:
        self.base = _Selection(["Base", "Global"])
        self.point = _Selection(["p"])
        self.cell = _Selection(["c"])
        self.modified = False
        self.updated = False

    def GetBaseSelection(self):  # noqa: N802
        return self.base

    def GetPointDataArraySelection(self):  # noqa: N802
        return self.point

    def GetCellDataArraySelection(self):  # noqa: N802
        return self.cell

    def Modified(self) -> None:  # noqa: N802
        self.modified = True

    def Update(self) -> None:  # noqa: N802
        self.updated = True


def test_reader_selection_helpers() -> None:
    reader = _Reader()
    wrapper = types.SimpleNamespace(GetReader=lambda: reader)
    srv._disable_bases_on_reader(wrapper, ["Global", "Missing"])
    assert reader.base.disabled == ["Global"]
    assert reader.modified is True
    assert srv._reader_bases_and_fields(wrapper) == (["Base", "Global"], ["p"], ["c"])
    srv._apply_base_selection(reader, ["Base"])
    assert reader.base.enabled == ["Base"]
    assert reader.updated is True


def test_advance_reader_time_update_and_fallback() -> None:
    calls: list[object] = []

    class WithUpdate:
        def UpdateTimeStep(self, value: float) -> None:  # noqa: N802
            calls.append(("time", value))

        def Update(self) -> None:  # noqa: N802
            calls.append("update")

    srv._advance_reader_time(WithUpdate(), 2.5)
    assert calls == [("time", 2.5), "update"]

    class Exec:
        def SetUpdateTimeStep(self, port: int, value: float) -> None:  # noqa: N802
            calls.append(("exec", port, value))

    class WithoutUpdate:
        def GetExecutive(self):  # noqa: N802
            return Exec()

        def Update(self) -> None:  # noqa: N802
            calls.append("fallback-update")

    srv._advance_reader_time(WithoutUpdate(), 3.0)
    assert ("exec", 0, 3.0) in calls


def test_advance_reader_time_swallows_reader_errors() -> None:
    class Broken:
        def UpdateTimeStep(self, _value: float) -> None:  # noqa: N802
            raise RuntimeError("boom")

    srv._advance_reader_time(Broken(), 1.0)


class _Data:
    def __init__(self, arrays: dict[str, tuple[float, float]]) -> None:
        self.arrays = arrays

    def GetNumberOfArrays(self) -> int:  # noqa: N802
        return len(self.arrays)

    def GetArrayName(self, i: int) -> str:  # noqa: N802
        return list(self.arrays)[i]

    def GetArray(self, name: str):  # noqa: N802
        rng = self.arrays.get(name)
        return None if rng is None else types.SimpleNamespace(GetRange=lambda _idx: rng)


class _Leaf:
    def __init__(
        self,
        point: dict[str, tuple[float, float]],
        cell: dict[str, tuple[float, float]],
    ) -> None:
        self.point = _Data(point)
        self.cell = _Data(cell)

    def GetPointData(self):  # noqa: N802
        return self.point

    def GetCellData(self):  # noqa: N802
        return self.cell


class _Blocks:
    def __init__(self, blocks: list[object | None]) -> None:
        self.blocks = blocks

    def GetNumberOfBlocks(self) -> int:  # noqa: N802
        return len(self.blocks)

    def GetBlock(self, i: int):  # noqa: N802
        return self.blocks[i]


def test_dataset_field_helpers() -> None:
    dataset = _Blocks(
        [
            None,
            _Leaf({"p": (1.0, 2.0)}, {}),
            _Leaf({"p": (-1.0, 4.0)}, {"c": (5.0, 6.0)}),
        ]
    )
    assert srv._list_point_and_cell_fields(dataset) == (["p"], ["c"])
    assert srv._compute_field_range(dataset, "p", "point") == (-1.0, 4.0)
    assert srv._compute_field_range(dataset, "missing", "point") == (0.0, 1.0)

    class NoData:
        def GetPointData(self):  # noqa: N802
            return None

        def GetCellData(self):  # noqa: N802
            return _Data({})

    assert srv._compute_field_range(_Blocks([NoData()]), "p", "point") == (0.0, 1.0)


def test_scalar_bar_helpers() -> None:
    class ScalarBar:
        def __init__(self) -> None:
            self.lut = None
            self.title = None
            self.visible = None

        def SetLookupTable(self, lut) -> None:  # noqa: N802, ANN001
            self.lut = lut

        def SetTitle(self, title: str) -> None:  # noqa: N802
            self.title = title

        def SetVisibility(self, visible: bool) -> None:  # noqa: N802, FBT001
            self.visible = visible

    scalar_bar = ScalarBar()
    lut = object()

    srv._show_scalar_bar_for_field(scalar_bar, lut, "pressure", "point")
    assert scalar_bar.lut is lut
    assert scalar_bar.title == "pressure (point)"
    assert scalar_bar.visible is True

    srv._hide_scalar_bar(scalar_bar)
    assert scalar_bar.visible is False


def test_load_reader_plain_and_build_lut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeCGNSReader:
        def __init__(self) -> None:
            self.file_name = None
            self.calls: list[str] = []

        def SetFileName(self, name: str) -> None:  # noqa: N802
            self.file_name = name

        def UpdateInformation(self) -> None:  # noqa: N802
            self.calls.append("info")

        def EnableAllBases(self) -> None:  # noqa: N802
            self.calls.append("bases")

        def EnableAllPointArrays(self) -> None:  # noqa: N802
            self.calls.append("points")

        def EnableAllCellArrays(self) -> None:  # noqa: N802
            self.calls.append("cells")

    class FakeLookupTable:
        def __init__(self) -> None:
            self.hue = None
            self.table: dict[int, tuple[float, ...]] = {}

        def SetTableRange(self, *_args):
            pass  # noqa: ANN002

        def SetNumberOfColors(self, *_args):
            pass  # noqa: ANN002

        def SetHueRange(self, *args):
            self.hue = args  # noqa: ANN002

        def SetTableValue(self, index, *rgba):
            self.table[index] = rgba  # noqa: ANN001, ANN002

        def Build(self):
            pass

    fake_vtk = types.SimpleNamespace(
        vtkCGNSReader=FakeCGNSReader, vtkLookupTable=FakeLookupTable
    )
    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)
    path = tmp_path / "mesh.cgns"
    reader = srv._load_reader(path)
    assert reader.file_name == str(path)
    assert reader.calls == ["info", "bases", "points", "cells"]
    assert srv._build_lut("unknown", 0.0, 1.0).hue == (0.667, 0.0)
    # Known colormaps sample matplotlib instead of a hue ramp.
    import matplotlib  # noqa: PLC0415

    viridis = srv._build_lut("viridis", 0.0, 1.0)
    assert viridis.hue is None
    assert len(viridis.table) == 256
    assert viridis.table[0] == pytest.approx(matplotlib.colormaps["viridis"](0.0))
    assert viridis.table[255] == pytest.approx(matplotlib.colormaps["viridis"](1.0))


def test_install_vtk_log_router_with_fake_vtk(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    captured: dict[str, object] = {}

    class FakeOutputWindow:
        @staticmethod
        def SetInstance(instance) -> None:  # noqa: N802
            captured["instance"] = instance
            calls.append(("output", instance.__class__.__name__))

    class FakeObject:
        @staticmethod
        def GlobalWarningDisplayOff() -> None:  # noqa: N802
            calls.append("warnings-off")

    class FakeLogger:
        VERBOSITY_OFF = 0

        @staticmethod
        def SetStderrVerbosity(value: int) -> None:  # noqa: N802
            calls.append(("verbosity", value))

    fake_vtk = types.SimpleNamespace(
        vtkOutputWindow=FakeOutputWindow,
        vtkObject=FakeObject,
        vtkLogger=FakeLogger,
    )
    monkeypatch.setitem(sys.modules, "vtk", fake_vtk)
    monkeypatch.setattr(srv, "_VTK_LOG_ROUTER_INSTALLED", False)

    srv._install_vtk_log_router()
    srv._install_vtk_log_router()

    output = captured["instance"]
    output.DisplayText("text")
    output.DisplayErrorText("error")
    output.DisplayWarningText("warning")
    output.DisplayGenericWarningText("generic")
    output.DisplayDebugText("debug")
    assert calls == [
        ("output", "_LoggingOutputWindow"),
        "warnings-off",
        ("verbosity", 0),
    ]


def test_install_vtk_log_router_ignores_missing_and_old_vtk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = __import__

    def missing_vtk(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002, ANN001, ANN002
        if name == "vtk":
            raise ImportError("no vtk")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.delitem(sys.modules, "vtk", raising=False)
    monkeypatch.setattr("builtins.__import__", missing_vtk)
    monkeypatch.setattr(srv, "_VTK_LOG_ROUTER_INSTALLED", False)
    srv._install_vtk_log_router()
    assert srv._VTK_LOG_ROUTER_INSTALLED is False

    class FakeOutputWindow:
        @staticmethod
        def SetInstance(_instance) -> None:  # noqa: N802
            pass

    class FakeObject:
        @staticmethod
        def GlobalWarningDisplayOff() -> None:  # noqa: N802
            pass

    class OldLogger:
        @staticmethod
        def SetStderrVerbosity(_value: int) -> None:  # noqa: N802
            raise AttributeError("old")

    monkeypatch.setattr("builtins.__import__", real_import)
    monkeypatch.setitem(
        sys.modules,
        "vtk",
        types.SimpleNamespace(
            vtkOutputWindow=FakeOutputWindow,
            vtkObject=FakeObject,
            vtkLogger=OldLogger,
        ),
    )
    srv._install_vtk_log_router()
    assert srv._VTK_LOG_ROUTER_INSTALLED is True


def test_parcoords_brush_exports_only_matching_plotted_samples() -> None:
    from plaid.viewer.global_plots import build_globals_figure

    records = {
        "train": [
            {"sample_id": "0", "values": {"Global/a": 1, "Global/b": 10}},
            {"sample_id": "1", "values": {"Global/a": 2, "Global/b": 20}},
        ],
        "test": [
            {"sample_id": "0", "values": {"Global/a": 3, "Global/b": 30}},
            {"sample_id": "1", "values": {"Global/a": 4}},
        ],
    }
    figure = build_globals_figure(
        records,
        ["train", "test"],
        "parallel",
        [],
        parallel_fields=["Global/a", "Global/b"],
        parallel_mode="parcoords",
    )
    assert srv._parcoords_selected_rows(figure, {}) == {
        ("train", "0"),
        ("train", "1"),
        ("test", "0"),
    }
    assert srv._parcoords_selected_rows(figure, {0: [1.5, 3.5]}) == {
        ("train", "1"),
        ("test", "0"),
    }
    assert srv._parcoords_selected_rows(
        figure, {0: [[0.5, 1.5], [2.5, 3.5]], 1: [25, 35]}
    ) == {("test", "0")}
    assert srv._parcoords_selected_rows(
        figure, {0: [[[0.5, 1.5], [2.5, 3.5]]], 1: [[25, 35]]}
    ) == {("test", "0")}
    assert srv._parcoords_selected_rows(figure, {0: [5, 6]}) == set()
    assert srv._parcoords_selected_rows(None, {0: [1, 2]}) == set()


def test_stream_csv_writes_every_chunk_as_attachment() -> None:
    """CSV exports are streamed block by block over HTTP."""
    import asyncio

    from aiohttp import web
    from aiohttp.test_utils import make_mocked_request

    written: list[bytes] = []

    class _Writer:
        buffer_size = 0
        output_size = 0

        async def write(self, chunk: bytes) -> None:
            written.append(bytes(chunk))

        async def write_headers(self, *_args: object) -> None:
            return None

        async def write_eof(self, *_args: object) -> None:
            return None

        async def drain(self) -> None:
            return None

        def enable_compression(self, *_args: object, **_kwargs: object) -> None:
            return None

        def enable_chunking(self) -> None:
            return None

        def send_headers(self, *_args: object) -> None:
            return None

    request = make_mocked_request("GET", "/csv", writer=_Writer())
    response = asyncio.run(
        srv._stream_csv(request, iter([b"a,b\n", b"1,2\n"]), "out.csv")
    )
    assert isinstance(response, web.StreamResponse)
    assert response.headers["Content-Disposition"] == 'attachment; filename="out.csv"'
    assert response.headers["Content-Type"].startswith("text/csv")
    assert b"".join(written).endswith(b"a,b\n1,2\n")


def test_constraint_ranges_cleared_brush() -> None:
    """A cleared or malformed brush has no active ranges."""
    assert srv._constraint_ranges(None) == []
    assert srv._constraint_ranges([]) == []
    assert srv._constraint_ranges(()) == []


def test_scatter_selection_resolves_plot_indices_across_splits() -> None:
    from plaid.viewer.global_plots import build_globals_figure, globals_dataframe

    records = {
        "train": [
            {"sample_id": "0", "values": {"Global/a": 1, "Global/b": 2}},
            {"sample_id": "1", "values": {"Global/a": None, "Global/b": 4}},
            {"sample_id": "2", "values": {"Global/a": 5, "Global/b": 6}},
        ],
        "test": [
            {"sample_id": "0", "values": {"Global/a": 7, "Global/b": 8}},
        ],
    }
    points = [
        None,  # Ignore non-dictionary selection events.
        {"curveNumber": 0, "pointNumber": 1},
        {"curveNumber": 1, "pointNumber": 0},
        {"curveNumber": 1, "pointNumber": 0},
        {"curveNumber": 0, "pointNumber": 2},  # Beyond plotted points.
        {"curveNumber": 2, "pointNumber": 0},
        {"curveNumber": -1, "pointNumber": 0},
        {"curveNumber": 0, "pointNumber": -1},
        {"curveNumber": 0, "pointNumber": True},
        {"curveNumber": "0", "pointNumber": 0},
        {},
    ]
    for mode in ("1D", "2D"):
        figure = build_globals_figure(
            records, ["train", "test"], mode, ["Global/a", "Global/b"]
        )
        selected = srv._scatter_selected_rows(figure, points)
        assert selected == {("train", "2"), ("test", "0")}
        frame = globals_dataframe(records, selected)
        assert list(zip(frame["split"], frame["sample_id"])) == [
            ("train", "2"),
            ("test", "0"),
        ]
        assert srv._scatter_selected_rows(figure, []) == set()
        assert srv._scatter_selected_rows(figure, None) == set()
    assert srv._scatter_selected_rows(None, points) == set()


def test_selected_rows_yaml_match_export_order() -> None:
    """The YAML block lists identities in CSV order, grouped by split."""
    from plaid.viewer.global_plots import globals_dataframe, prepare_globals

    records = {
        "train": [
            {"sample_id": "0", "values": {"Global/a": 1}},
            {"sample_id": "1", "values": {"Global/a": 2}},
        ],
        "test": [{"sample_id": "0", "values": {"Global/a": 3}}],
    }
    prepared = prepare_globals(records)
    selected = {("train", "1"), ("test", "0")}
    frame = globals_dataframe(prepared, selected)
    assert frame[["split", "sample_id"]].to_dict("records") == [
        {"split": "train", "sample_id": "1"},
        {"split": "test", "sample_id": "0"},
    ]
    text = srv._selected_rows_yaml(prepared, selected)
    assert text == "    train:\n    - 1\n    test:\n    - 0"
    assert yaml.safe_load(f"train_split:\n{text}") == {
        "train_split": {"train": [1], "test": [0]}
    }
    assert srv._selected_rows_yaml(prepared, set()) == ""


def test_plot_blocker_guards_heavy_or_empty_parallel_plots() -> None:
    """Parallel plots need axes, and per-sample lines are capped."""
    limit = srv.PARALLEL_LINES_MAX_SAMPLES
    assert srv._plot_blocker("2D", "lines", [], 10**6) == ""
    assert "Select the Globals" in srv._plot_blocker("parallel", "parcoords", [], 1)
    assert srv._plot_blocker("parallel", "lines", ["Global/a"], limit) == ""
    message = srv._plot_blocker("parallel", "lines", ["Global/a"], limit + 1)
    assert "parcoords" in message
    assert srv._plot_blocker("parallel", "parcoords", ["Global/a"], 10**6) == ""
