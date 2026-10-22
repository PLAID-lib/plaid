"""Manage a local PLAID data API for a ParaView launched from the viewer."""

from __future__ import annotations

import atexit
import errno
import logging
import threading
from pathlib import Path
from subprocess import Popen
from typing import Any

from plaid.cli.paraview_plugin import run_paraview_with_plugin_client_server
from plaid.cli.serve import ServeContext, _ServeHTTPServer

logger = logging.getLogger(__name__)
_PARAVIEW_PORTS = range(8081, 8091)
_sessions: set[ParaViewSession] = set()
_sessions_lock = threading.Lock()


class ParaViewSession:
    """Own the local HTTP server while its ParaView process is running."""

    def __init__(
        self,
        server: _ServeHTTPServer,
        process: Popen[Any],
        server_thread: threading.Thread,
    ) -> None:
        """Store the running HTTP server and ParaView process.

        Args:
            server: Local PLAID data API server.
            process: ParaView process using the API.
            server_thread: Thread already serving HTTP requests.
        """
        self.server = server
        self.process = process
        self.port = server.server_port
        self._server_thread = server_thread

    def start(self) -> None:
        """Serve requests and stop serving once ParaView exits."""
        with _sessions_lock:
            _sessions.add(self)
        threading.Thread(target=self._wait_for_paraview, daemon=True).start()

    def _wait_for_paraview(self) -> None:
        """Release the HTTP server when the associated ParaView exits."""
        try:
            self.process.wait()
        except Exception:  # noqa: BLE001
            logger.exception("Failed to wait for ParaView")
        finally:
            self.close()

    def close(self) -> None:
        """Stop the HTTP server, including on viewer shutdown."""
        with _sessions_lock:
            if self not in _sessions:
                return
            _sessions.remove(self)
        self.server.shutdown()
        self.server.server_close()
        self._server_thread.join(timeout=5)


def _close_sessions() -> None:
    """Close all data API servers when the viewer exits."""
    with _sessions_lock:
        sessions = list(_sessions)
    for session in sessions:
        session.close()


atexit.register(_close_sessions)


def _create_server(context: ServeContext) -> _ServeHTTPServer:
    """Bind the data API to the first available port near the viewer port.

    Args:
        context: Dataset context to expose through the data API.

    Returns:
        A loopback-only server bound to a port between 8081 and 8090.

    Raises:
        OSError: If no permitted port is available or server creation fails.
    """
    for port in _PARAVIEW_PORTS:
        try:
            return _ServeHTTPServer(("127.0.0.1", port), context)
        except OSError as error:
            if error.errno not in (errno.EADDRINUSE, errno.EACCES):
                raise
    raise OSError(
        "Cannot start the ParaView data API: no available port in 8081-8090. "
        "Free or allow a port in this range and try again."
    )


def open_dataset_in_paraview(dataset_path: Path) -> ParaViewSession:
    """Start a loopback-only data API and launch ParaView for a local dataset.

    Args:
        dataset_path: Directory of the selected local PLAID dataset.

    Returns:
        The session, with the selected API port between 8081 and 8090.

    Raises:
        FileNotFoundError: If the dataset directory no longer exists.
        OSError: If the server or ParaView cannot start.
    """
    path = dataset_path.resolve(strict=True)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset is not a directory: {path}")
    server = _create_server(ServeContext(default_dataset_uri=str(path)))
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    try:
        # Bind and serve before launching: the plugin can connect immediately.
        server_thread.start()
        process = run_paraview_with_plugin_client_server(
            port=server.server_port, options={"tree_name": path.name}
        )
        session = ParaViewSession(server, process, server_thread)
        session.start()
    except BaseException:
        if server_thread.is_alive():
            server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)
        raise
    return session
