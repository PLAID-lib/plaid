"""Smoke tests for the trame dataset viewer server."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def empty_datasets_root(tmp_path: Path) -> Path:
    """Return an existing but empty datasets directory."""
    root = tmp_path / "datasets"
    root.mkdir()
    return root


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
