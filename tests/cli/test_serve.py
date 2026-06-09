"""Tests for the PLAID HTTP server entry points."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.client import HTTPConnection
from pathlib import Path

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


def _post_json(
    url_base: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> tuple[int, dict[str, object]]:
    """Perform a POST request and decode the JSON payload.

    Args:
        url_base: Host/port string.
        path: HTTP path.
        payload: JSON payload sent in the request body.

    Returns:
        Tuple of status code and JSON-decoded object.
    """
    body = json.dumps(payload or {}).encode("utf-8")
    connection = HTTPConnection(url_base)
    connection.request(
        "POST",
        path,
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    response_payload = json.loads(response.read().decode("utf-8"))
    connection.close()
    return response.status, response_payload


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


def test_post_health_entry_point_returns_ok(serve_url: str) -> None:
    """POST health route should match the Maestro serve interface."""
    status, payload = _post_json(serve_url, "/health")

    assert status == 200
    assert payload == {"status": "ok"}


def test_post_predict_returns_not_implemented(serve_url: str) -> None:
    """POST predict route should be explicit but unsupported by PLAID serve."""
    status, payload = _post_json(serve_url, "/predict")

    assert status == 501
    assert payload == {"error": "Endpoint /predict is not supported by PLAID serve"}


def test_post_unknown_route_returns_not_found(serve_url: str) -> None:
    """Unknown POST routes should return a 404 payload."""
    status, payload = _post_json(serve_url, "/unknown")

    assert status == 404
    assert payload == {"POST error": "Not Found"}


def test_post_infos_returns_dataset_infos(serve_url: str) -> None:
    """POST infos route should load dataset infos from the provided dataset."""
    dataset = Path("datamain/PhysArena_Tensile2d")

    status, payload = _post_json(serve_url, "/infos", {"dataset": str(dataset)})

    assert status == 200
    assert payload["storage_backend"] == "hf_datasets"
    assert payload["num_samples"] == {"OOD": 2, "test": 200, "train": 500}


def test_post_problem_definition_returns_selected_definition(serve_url: str) -> None:
    """POST problem_definition route should load the requested definition."""
    dataset = Path("datamain/PhysArena_Tensile2d")

    status, payload = _post_json(
        serve_url,
        "/problem_definition",
        {"dataset": str(dataset), "problem_definition": "regression_8"},
    )

    assert status == 200
    assert payload["name"] == "regression_8"
    assert isinstance(payload["input_features"], list)
    assert isinstance(payload["output_features"], list)
