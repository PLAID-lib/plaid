"""Tests for the PLAID HTTP server entry points."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.client import HTTPConnection

import pytest

from plaid.cli.serve import ServeContext, _ServeHTTPServer


@pytest.fixture()
def serve_url() -> Generator[str, None, None]:
    """Run a local PLAID server and yield its host/port endpoint.

    Yields:
        URL base for the temporary HTTP server.
    """
    server = _ServeHTTPServer(("127.0.0.1", 0), ServeContext())
    server_address = server.server_address
    host = str(server_address[0])
    port = int(server_address[1])
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield f"{host}:{port}"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


def _get_json(url_base: str, path: str) -> tuple[int, dict[str, object]]:
    """Perform a GET request and decode the JSON payload.

    Args:
        url_base: Host/port string.
        path: HTTP path.

    Returns:
        Tuple of status code and JSON-decoded object.
    """
    connection = HTTPConnection(url_base)
    connection.request("GET", path)
    response = connection.getresponse()
    payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, payload


def test_health_entry_point_returns_ok(serve_url: str) -> None:
    """Health route should return status OK."""
    status, payload = _get_json(serve_url, "/health")

    assert status == 200
    assert payload == {"status": "ok"}


def test_entry_points_route_returns_available_endpoints(serve_url: str) -> None:
    """Entry points route should expose supported server capabilities."""
    status, payload = _get_json(serve_url, "/entry_points")

    assert status == 200
    assert payload == {
        "samples_step": True,
        "predict": False,
        "splits": True,
        "timesteps": True,
        "samples": True,
    }


def test_unknown_route_returns_not_found(serve_url: str) -> None:
    """Unknown routes should return a 404 payload."""
    status, payload = _get_json(serve_url, "/unknown")

    assert status == 404
    assert payload == {"GET error": "Not Found"}
