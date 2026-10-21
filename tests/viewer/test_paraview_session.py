"""Tests for the viewer-owned local ParaView data API."""

from __future__ import annotations

import threading
from pathlib import Path
from urllib import request

import pytest

from plaid.viewer import paraview_session


def test_open_dataset_serves_selected_path_on_actual_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "selected"
    dataset.mkdir()
    launched: list[int] = []
    exited = threading.Event()

    class Process:
        """Model a running ParaView until the test releases it."""

        def wait(self) -> None:
            """Block until ParaView exits."""
            exited.wait(timeout=5)

    def launch(port: int) -> Process:
        """Capture the actual data API port passed to ParaView."""
        launched.append(port)
        with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            assert response.read() == b'{"status": "ok"}'
        return Process()

    monkeypatch.setattr(
        paraview_session, "run_paraview_with_plugin_client_server", launch
    )
    session = paraview_session.open_dataset_in_paraview(dataset)
    try:
        assert launched == [session.port]
        assert session.server.server_address[0] == "127.0.0.1"
        assert session.server.context.default_dataset_uri == str(dataset.resolve())
        with request.urlopen(
            f"http://127.0.0.1:{session.port}/health", timeout=2
        ) as response:
            assert response.read() == b'{"status": "ok"}'
    finally:
        exited.set()
        session.close()
        session.close()


def test_launch_failure_releases_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "selected"
    dataset.mkdir()

    def fail(port: int) -> None:
        """Raise when ParaView cannot be started."""
        raise OSError(f"Cannot launch on {port}")

    monkeypatch.setattr(
        paraview_session, "run_paraview_with_plugin_client_server", fail
    )
    with pytest.raises(OSError, match="Cannot launch on"):
        paraview_session.open_dataset_in_paraview(dataset)


def test_missing_dataset_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        paraview_session.open_dataset_in_paraview(tmp_path / "missing")

    file = tmp_path / "file"
    file.touch()
    with pytest.raises(FileNotFoundError, match="not a directory"):
        paraview_session.open_dataset_in_paraview(file)


def test_session_closes_after_paraview_exits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The process watcher releases the API even when waiting fails."""
    dataset = tmp_path / "selected"
    dataset.mkdir()
    finished = threading.Event()

    class Process:
        """Model a ParaView wait failure after its process exits."""

        def wait(self) -> None:
            """Signal the watcher, then raise a process error."""
            finished.wait(timeout=5)
            raise OSError("wait failed")

    monkeypatch.setattr(
        paraview_session,
        "run_paraview_with_plugin_client_server",
        lambda port: Process(),  # noqa: ARG005 - named launcher keyword
    )
    session = paraview_session.open_dataset_in_paraview(dataset)
    try:
        assert session in paraview_session._sessions
        finished.set()
        # Both the watcher and shutdown cleanup should be safe to call.
        paraview_session._close_sessions()
        assert session not in paraview_session._sessions
    finally:
        finished.set()
        session.close()
