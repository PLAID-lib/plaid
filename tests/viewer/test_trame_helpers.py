"""Headless tests for trame server helper functions using fakes."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from plaid.viewer.trame_app import server as srv


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

        def SetTableRange(self, *_args):
            pass  # noqa: ANN002

        def SetNumberOfColors(self, *_args):
            pass  # noqa: ANN002

        def SetHueRange(self, *args):
            self.hue = args  # noqa: ANN002

        def SetSaturationRange(self, *_args):
            pass  # noqa: ANN002

        def SetValueRange(self, *_args):
            pass  # noqa: ANN002

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
