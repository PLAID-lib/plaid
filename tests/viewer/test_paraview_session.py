"""Tests for the viewer-owned local ParaView data API."""

from __future__ import annotations

import errno
import socket
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
    launched: list[tuple[int, dict[str, str]]] = []
    exited = threading.Event()

    class Process:
        """Model a running ParaView until the test releases it."""

        def wait(self) -> None:
            """Block until ParaView exits."""
            exited.wait(timeout=5)

    def launch(port: int, options: dict[str, str]) -> Process:
        """Capture the API port and source options passed to ParaView.

        Args:
            port: Port of the viewer-owned data API.
            options: ParaView source settings, including the tree name.

        Returns:
            The simulated running ParaView process.
        """
        launched.append((port, options))
        with request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
            assert response.read() == b'{"status": "ok"}'
        return Process()

    monkeypatch.setattr(
        paraview_session, "run_paraview_with_plugin_client_server", launch
    )
    session = paraview_session.open_dataset_in_paraview(dataset)
    try:
        assert 8081 <= session.port <= 8090
        assert launched == [(session.port, {"tree_name": "selected"})]
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

    def fail(port: int, options: dict[str, str]) -> None:
        """Raise when ParaView cannot be started.

        Args:
            port: Port of the viewer-owned data API.
            options: ParaView source settings.
        """
        assert options == {"tree_name": "selected"}
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
        lambda port, options: Process(),  # noqa: ARG005 - launcher keywords
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


def test_server_skips_occupied_port(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the next candidate when a real socket occupies the first port."""
    with socket.socket() as occupied, socket.socket() as available:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        available.bind(("127.0.0.1", 0))
        occupied_port = occupied.getsockname()[1]
        available_port = available.getsockname()[1]
        available.close()
        monkeypatch.setattr(
            paraview_session, "_PARAVIEW_PORTS", (occupied_port, available_port)
        )
        server = paraview_session._create_server(paraview_session.ServeContext())
        try:
            assert server.server_port == available_port
        finally:
            server.server_close()


def test_server_reports_exhausted_ports(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report exhaustion rather than fall back to an OS-assigned port."""
    with socket.socket() as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        monkeypatch.setattr(
            paraview_session, "_PARAVIEW_PORTS", (occupied.getsockname()[1],)
        )
        with pytest.raises(OSError, match="no available port in 8081-8090"):
            paraview_session._create_server(paraview_session.ServeContext())


@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EINVAL])
def test_server_handles_bind_errors(
    monkeypatch: pytest.MonkeyPatch, error_number: int
) -> None:
    """Skip denied ports, but propagate unexpected server errors."""
    attempted: list[int] = []

    def fail(address: tuple[str, int], context: object) -> None:
        """Record the bind attempt and simulate an operating system error.

        Args:
            address: Loopback address and candidate API port.
            context: Dataset serving context.
        """
        assert isinstance(context, paraview_session.ServeContext)
        attempted.append(address[1])
        raise OSError(error_number, "bind failed")

    monkeypatch.setattr(paraview_session, "_ServeHTTPServer", fail)
    message = "no available port" if error_number == errno.EACCES else "bind failed"
    with pytest.raises(OSError, match=message):
        paraview_session._create_server(paraview_session.ServeContext())
    expected = list(range(8081, 8091)) if error_number == errno.EACCES else [8081]
    assert attempted == expected
