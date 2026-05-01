"""Tests for the viewer CLI parser that do not start the VTK/trame runtime."""

from __future__ import annotations

from pathlib import Path

import pytest

from plaid.viewer import cli as cli_mod
from plaid.viewer.cli import _build_parser


def test_build_parser_defaults() -> None:
    args = _build_parser().parse_args([])

    assert args.datasets_root is None
    assert args.browse_roots is None
    assert args.disable_root_change is False
    assert args.dataset_id is None
    assert args.disable_dataset_change is False
    assert args.cache_dir is None
    assert args.host == "127.0.0.1"
    assert args.port == 8080
    assert args.backend_id == "disk"
    assert args.hub_repo is None


def test_build_parser_accepts_all_options(tmp_path: Path) -> None:
    datasets_root = tmp_path / "datasets"
    cache_dir = tmp_path / "cache"
    browse_a = tmp_path / "a"
    browse_b = tmp_path / "b"

    args = _build_parser().parse_args(
        [
            "--datasets-root",
            str(datasets_root),
            "--browse-roots",
            str(browse_a),
            str(browse_b),
            "--disable-root-change",
            "--dataset-id",
            "dataset-b",
            "--disable-dataset-change",
            "--cache-dir",
            str(cache_dir),
            "--host",
            "0.0.0.0",
            "--port",
            "9000",
            "--backend-id",
            "zarr",
            "--hub-repo",
            "org/one",
            "--hub-repo",
            "org/two",
        ]
    )

    assert args.datasets_root == datasets_root
    assert args.browse_roots == [browse_a, browse_b]
    assert args.disable_root_change is True
    assert args.dataset_id == "dataset-b"
    assert args.disable_dataset_change is True
    assert args.cache_dir == cache_dir
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.backend_id == "zarr"
    assert args.hub_repo == ["org/one", "org/two"]


def test_main_wires_services_without_starting_real_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, object]] = []

    class FakeCache:
        def __init__(self, persistent_dir=None):
            calls.append(("cache", persistent_dir))
            self.path = tmp_path / "cache-root"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            calls.append(("cache-exit", exc_type))

    class FakeDatasetService:
        def __init__(self, config):
            self.config = config
            calls.append(("dataset-root", config.datasets_root))
            calls.append(("allow-root-change", config.allow_root_change))
            calls.append(("initial-dataset-id", config.initial_dataset_id))
            calls.append(("allow-dataset-change", config.allow_dataset_change))

        def add_hub_dataset(self, repo_id: str) -> str:
            calls.append(("hub", repo_id))
            if repo_id == "bad/repo":
                raise ValueError("bad")
            return repo_id

    class FakeArtifactService:
        def __init__(self, _dataset_service, cache_path):
            calls.append(("artifact-cache", cache_path))

    class FakeServer:
        def start(self, *, host: str, port: int, open_browser: bool) -> None:
            calls.append(("start", (host, port, open_browser)))

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002, ANN001, ANN002
        if name == "plaid.viewer.trame_app.server" and "_reroute_c_stderr" in fromlist:
            return type(
                "ServerModule",
                (),
                {"_reroute_c_stderr": lambda: calls.append(("stderr", None))},
            )
        if name == "plaid.viewer.trame_app.server" and "build_server" in fromlist:
            return type(
                "ServerModule", (), {"build_server": lambda _ds, _as: FakeServer()}
            )
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr(cli_mod, "CacheRoot", FakeCache)
    monkeypatch.setattr(cli_mod, "PlaidDatasetService", FakeDatasetService)
    monkeypatch.setattr(cli_mod, "ParaviewArtifactService", FakeArtifactService)
    monkeypatch.setattr(
        cli_mod, "get_last_datasets_root", lambda: tmp_path / "persisted"
    )
    monkeypatch.setattr("builtins.__import__", fake_import)

    assert (
        cli_mod.main(
            [
                "--cache-dir",
                str(tmp_path / "cache"),
                "--host",
                "0.0.0.0",
                "--port",
                "9001",
                "--disable-root-change",
                "--dataset-id",
                "dataset-b",
                "--disable-dataset-change",
                "--hub-repo",
                "org/repo",
                "--hub-repo",
                "bad/repo",
            ]
        )
        == 0
    )

    assert ("stderr", None) in calls
    assert ("dataset-root", tmp_path / "persisted") in calls
    assert ("allow-root-change", False) in calls
    assert ("initial-dataset-id", "dataset-b") in calls
    assert ("allow-dataset-change", False) in calls
    assert ("hub", "org/repo") in calls
    assert ("hub", "bad/repo") in calls
    assert ("artifact-cache", tmp_path / "cache-root") in calls
    assert ("start", ("0.0.0.0", 9001, False)) in calls
