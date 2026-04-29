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


# TODO: Re-enable after fixing VTK segfault in CI environment
# def test_build_server_returns_trame_server(empty_datasets_root: Path) -> None:
#     """``build_server`` should return a configured trame server for empty roots."""
#     pytest.importorskip("vtk")
#     pytest.importorskip("trame")
#     from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

#     config = ViewerConfig(datasets_root=empty_datasets_root)
#     dataset_service = PlaidDatasetService(config)

#     with CacheRoot(install_signal_handlers=False, run_orphan_sweep=False) as cache:
#         artifact_service = ParaviewArtifactService(dataset_service, cache.path)
#         server = build_server(dataset_service, artifact_service)

#     # The server should expose state and controller attributes.
#     assert hasattr(server, "state")
#     assert hasattr(server, "controller")
#     assert server.state.dataset_ids == []
#     assert server.state.status.startswith("Select a dataset")

# TODO: Re-enable after fixing VTK segfault in CI environment
# def test_browse_cd_updates_browse_state(tmp_path: Path) -> None:
#     """``ctrl.browse_cd`` must load the given directory into the browser state.

#     Regression guard for a bug where the file-browser list items dispatched
#     through the client-side ``trigger(...)`` helper (which only resolves
#     names registered as server triggers), while ``browse_cd`` was only
#     registered as a controller method via ``@ctrl.set``. Clicking a folder
#     in the browser dialog was therefore a no-op.
#     """
#     pytest.importorskip("vtk")
#     pytest.importorskip("trame")
#     from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

#     datasets_root = tmp_path / "datasets"
#     datasets_root.mkdir()
#     child = datasets_root / "child"
#     child.mkdir()

#     config = ViewerConfig(
#         datasets_root=datasets_root,
#         browse_roots=(tmp_path,),
#     )
#     dataset_service = PlaidDatasetService(config)

#     with CacheRoot(install_signal_handlers=False, run_orphan_sweep=False) as cache:
#         artifact_service = ParaviewArtifactService(dataset_service, cache.path)
#         server = build_server(dataset_service, artifact_service)

#         assert hasattr(server.controller, "browse_cd")
#         server.controller.browse_cd(str(child))

#         assert Path(server.state.browse_cwd) == child.resolve()
#         assert server.state.browse_parent == str(datasets_root.resolve())


class _FakeSelection:
    def __init__(self) -> None:
        self.enabled: list[str] = []

    def DisableAllArrays(self) -> None:  # noqa: N802 - VTK API
        self.enabled = []

    def EnableArray(self, name: str) -> None:  # noqa: N802 - VTK API
        self.enabled.append(name)


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
