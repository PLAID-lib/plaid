"""Tests for the PLAID HTTP server entry points."""

from __future__ import annotations

import json
import threading
from collections.abc import Generator
from http.client import HTTPConnection
from types import SimpleNamespace

import pytest

from plaid.cli import serve
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


@pytest.fixture()
def sample_serve_url(
    monkeypatch,
) -> Generator[tuple[str, list[dict[str, object]]], None, None]:
    """Run a local PLAID server with fake sample serialization.

    Yields:
        Tuple containing URL base and calls captured by the fake context.
    """
    calls: list[dict[str, object]] = []

    class FakeContext:
        def resolve_dataset_uri(self, payload):
            calls.append({"method": "resolve_dataset_uri", "payload": payload})
            return str(payload["dataset"])

        def get_sample_objects(self, dataset_uri, split, sample_ids):
            calls.append(
                {
                    "method": "get_sample_objects",
                    "dataset_uri": dataset_uri,
                    "split": split,
                    "sample_ids": sample_ids,
                }
            )
            return [f"sample-{sample_id}" for sample_id in sample_ids]

    monkeypatch.setattr(
        serve,
        "sample_to_json_payload",
        lambda sample: {"serialized": sample},
    )
    server = _ServeHTTPServer(("127.0.0.1", 0), FakeContext())
    server_address = server.server_address
    host = str(server_address[0])
    port = int(server_address[1])
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        yield f"{host}:{port}", calls
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


def _post_raw(
    url_base: str,
    path: str,
    body: bytes,
) -> tuple[int, dict[str, object]]:
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
        "process": False,
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
    """POST health route should match the Plaid serve interface."""
    status, payload = _post_json(serve_url, "/health")

    assert status == 200
    assert payload == {"status": "ok"}


def test_post_predict_returns_not_implemented(serve_url: str) -> None:
    """POST process route should be explicit but unsupported by PLAID serve."""
    status, payload = _post_json(serve_url, "/process")

    assert status == 501
    assert payload == {"error": "Endpoint /process is not supported by PLAID serve"}


def test_post_unknown_route_returns_not_found(serve_url: str) -> None:
    """Unknown POST routes should return a 404 payload."""
    status, payload = _post_json(serve_url, "/unknown")

    assert status == 404
    assert payload == {"POST error": "Not Found"}


def test_post_infos_returns_dataset_infos(monkeypatch, serve_url: str) -> None:
    """POST infos route should load dataset infos from the provided dataset."""
    monkeypatch.setattr(
        ServeContext,
        "get_infos",
        staticmethod(lambda dataset_uri: {"dataset_uri": dataset_uri}),
    )

    status, payload = _post_json(serve_url, "/infos", {"dataset": "memory://dataset"})

    assert status == 200
    assert payload == {"dataset_uri": "memory://dataset"}


def test_post_problem_definition_returns_selected_definition(
    monkeypatch,
    serve_url: str,
) -> None:
    """POST problem_definition route should load the requested definition."""
    calls = []

    def fake_get_problem_definition(dataset_uri, name=None):
        calls.append({"dataset_uri": dataset_uri, "name": name})
        return {
            "name": name,
            "input_features": [],
            "output_features": [],
        }

    monkeypatch.setattr(
        ServeContext,
        "get_problem_definition",
        staticmethod(fake_get_problem_definition),
    )

    status, payload = _post_json(
        serve_url,
        "/problem_definition",
        {"dataset": "memory://dataset", "problem_definition": "regression_8"},
    )

    assert status == 200
    assert payload["name"] == "regression_8"
    assert isinstance(payload["input_features"], list)
    assert isinstance(payload["output_features"], list)
    assert calls == [{"dataset_uri": "memory://dataset", "name": "regression_8"}]


def test_post_samples_rejects_missing_sample_ids(serve_url: str) -> None:
    """Sample route should validate request payload fields."""
    status, payload = _post_json(
        serve_url,
        "/samples",
        {"dataset": "datamain/PhysArena_Tensile2d"},
    )

    assert status == 400
    assert "sample_ids" in str(payload["error"])


def test_post_dataset_route_rejects_invalid_json_body(serve_url: str) -> None:
    """Dataset routes should reject invalid JSON request bodies."""
    status, payload = _post_raw(serve_url, "/infos", b"[")

    assert status == 400
    assert "Expecting value" in str(payload["error"])


def test_post_problem_definition_rejects_non_string_name(serve_url: str) -> None:
    """Problem definition selector must be a string when provided."""
    status, payload = _post_json(
        serve_url,
        "/problem_definition",
        {"dataset": "datamain/PhysArena_Tensile2d", "problem_definition": 1},
    )

    assert status == 400
    assert payload == {"error": "problem_definition must be a string"}


def test_post_samples_returns_serialized_samples(sample_serve_url) -> None:
    """Sample route should resolve samples and serialize them to JSON payloads."""
    url_base, calls = sample_serve_url

    status, payload = _post_json(
        url_base,
        "/samples",
        {"dataset": "memory://dataset", "split": " train ", "sample_ids": [0, 2]},
    )

    assert status == 200
    assert payload == {
        "samples": [{"serialized": "sample-0"}, {"serialized": "sample-2"}]
    }
    assert calls == [
        {
            "method": "resolve_dataset_uri",
            "payload": {
                "dataset": "memory://dataset",
                "split": " train ",
                "sample_ids": [0, 2],
            },
        },
        {
            "method": "get_sample_objects",
            "dataset_uri": "memory://dataset",
            "split": "train",
            "sample_ids": [0, 2],
        },
    ]


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"dataset": " /tmp/data "}, "/tmp/data"),
        ({"uri": "s3://bucket"}, "s3://bucket"),
    ],
)
def test_parse_dataset_uri_accepts_dataset_or_uri(payload, expected):
    """Dataset URI parsing should accept both supported field names."""
    assert serve._parse_dataset_uri(payload) == expected


def test_resolve_dataset_uri_uses_default_when_request_has_no_dataset():
    """The serve context should fall back to its configured default dataset."""
    context = ServeContext(default_dataset_uri=" /data/default ")

    assert context.resolve_dataset_uri({}) == "/data/default"


def test_resolve_dataset_uri_requires_dataset_without_default():
    """A request needs a dataset URI when the server has no default dataset."""
    with pytest.raises(ValueError, match="dataset path/URI is required"):
        ServeContext().resolve_dataset_uri({})


@pytest.mark.parametrize(
    ("payload", "expected"),
    [({}, None), ({"split": " train "}, "train")],
)
def test_parse_split_accepts_optional_non_empty_string(payload, expected):
    """Split parsing should strip valid split names and allow omission."""
    assert serve._parse_split(payload) == expected


@pytest.mark.parametrize("split", ["", 1])
def test_parse_split_rejects_invalid_split_values(split):
    """Split parsing should reject non-string and empty split values."""
    with pytest.raises(ValueError, match="split must be a non-empty string"):
        serve._parse_split({"split": split})


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"sample_ids": []},
        {"sample_ids": [1, "2"]},
        {"sample_ids": [-1]},
    ],
)
def test_validate_sample_request_rejects_invalid_sample_ids(payload):
    """Sample requests require a non-empty list of non-negative integers."""
    with pytest.raises(ValueError):
        serve._validate_sample_request(payload)


def test_validate_sample_request_returns_sample_ids():
    """Valid sample IDs should be returned unchanged."""
    assert serve._validate_sample_request({"sample_ids": [0, 2]}) == [0, 2]


def test_serve_context_get_store_loads_and_caches_dataset(monkeypatch, tmp_path):
    """Dataset stores should be loaded once and reused by URI."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    calls = []

    def fake_init_from_disk(path):
        calls.append(path)
        return {"train": ["raw"]}, {
            "train": SimpleNamespace(to_plaid=lambda data, i: (data[i], i))
        }

    monkeypatch.setattr(serve, "init_from_disk", fake_init_from_disk)
    context = ServeContext()

    first = context._get_store(str(dataset))
    second = context._get_store(str(dataset))

    assert first is second
    assert calls == [str(dataset)]
    assert context.get_sample_objects(str(dataset), "train", [0]) == [("raw", 0)]


def test_serve_context_get_store_rejects_missing_dataset(tmp_path):
    """Dataset loading should fail before storage initialization for bad paths."""
    with pytest.raises(ValueError, match="does not exist"):
        ServeContext()._get_store(str(tmp_path / "missing"))


def test_serve_context_get_store_rejects_empty_dataset(monkeypatch, tmp_path):
    """Datasets without splits should be rejected."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    monkeypatch.setattr(serve, "init_from_disk", lambda _path: ({}, {}))

    with pytest.raises(ValueError, match="no splits"):
        ServeContext()._get_store(str(dataset))


def test_resolve_split_selects_only_split():
    """A single available split can be inferred."""
    store = serve._DatasetStore(datasetdict={"train": []}, converterdict={})

    assert ServeContext._resolve_split(store, None) == "train"


def test_resolve_split_rejects_unknown_or_ambiguous_split():
    """Unknown and ambiguous split requests should produce clear errors."""
    store = serve._DatasetStore(datasetdict={"train": [], "test": []}, converterdict={})

    with pytest.raises(ValueError, match="Unknown split"):
        ServeContext._resolve_split(store, "OOD")
    with pytest.raises(ValueError, match="split is required"):
        ServeContext._resolve_split(store, None)


def test_get_infos_serializes_loaded_infos(monkeypatch):
    """Dataset infos should be returned through model_dump."""
    monkeypatch.setattr(
        serve,
        "load_infos_from_disk",
        lambda dataset_uri: SimpleNamespace(model_dump=lambda: {"uri": dataset_uri}),
    )

    assert ServeContext.get_infos("dataset") == {"uri": "dataset"}


def test_get_problem_definition_selects_requested_definition(monkeypatch):
    """Requested problem definitions should be selected by name."""
    monkeypatch.setattr(
        serve,
        "load_problem_definitions_from_disk",
        lambda _dataset_uri: {
            "first": SimpleNamespace(model_dump=lambda: {"name": "first"}),
            "second": SimpleNamespace(model_dump=lambda: {"name": "second"}),
        },
    )

    assert ServeContext.get_problem_definition("dataset", "second") == {
        "name": "second"
    }
    with pytest.raises(ValueError, match="not found"):
        ServeContext.get_problem_definition("dataset", "missing")


def test_get_problem_definition_prefers_benchmark_then_sorted_first(monkeypatch):
    """Default problem definition selection should be deterministic."""
    monkeypatch.setattr(
        serve,
        "load_problem_definitions_from_disk",
        lambda _dataset_uri: {
            "zeta": SimpleNamespace(model_dump=lambda: {"name": "zeta"}),
            "PLAID_benchmark": SimpleNamespace(
                model_dump=lambda: {"name": "PLAID_benchmark"}
            ),
        },
    )
    assert ServeContext.get_problem_definition("dataset") == {"name": "PLAID_benchmark"}

    monkeypatch.setattr(
        serve,
        "load_problem_definitions_from_disk",
        lambda _dataset_uri: {
            "zeta": SimpleNamespace(model_dump=lambda: {"name": "zeta"}),
            "alpha": SimpleNamespace(model_dump=lambda: {"name": "alpha"}),
        },
    )
    assert ServeContext.get_problem_definition("dataset") == {"name": "alpha"}


def test_get_problem_definition_returns_only_available_definition(monkeypatch):
    """A single problem definition should be selected by default."""
    monkeypatch.setattr(
        serve,
        "load_problem_definitions_from_disk",
        lambda _dataset_uri: {
            "only": SimpleNamespace(model_dump=lambda: {"name": "only"})
        },
    )

    assert ServeContext.get_problem_definition("dataset") == {"name": "only"}


def test_run_server_until_process_exits_starts_and_stops_server():
    """The lifecycle helper should shut the server down after ParaView exits."""
    calls = []
    server = SimpleNamespace(
        serve_forever=lambda: calls.append("serve_forever"),
        shutdown=lambda: calls.append("shutdown"),
        server_close=lambda: calls.append("server_close"),
    )
    process = SimpleNamespace(wait=lambda: calls.append("wait"))

    serve._run_server_until_process_exits(server, process)

    assert "wait" in calls
    assert calls[-2:] == ["shutdown", "server_close"]


def test_run_server_until_process_exits_terminates_on_keyboard_interrupt():
    """KeyboardInterrupt should terminate ParaView and still close the server."""
    calls = []

    def wait(timeout=None):
        calls.append(("wait", timeout))
        if timeout is None:
            raise KeyboardInterrupt

    server = SimpleNamespace(
        serve_forever=lambda: None,
        shutdown=lambda: calls.append("shutdown"),
        server_close=lambda: calls.append("server_close"),
    )
    process = SimpleNamespace(
        wait=wait,
        terminate=lambda: calls.append("terminate"),
    )

    serve._run_server_until_process_exits(server, process)

    assert "terminate" in calls
    assert ("wait", 5) in calls
    assert calls[-2:] == ["shutdown", "server_close"]


def test_run_server_until_process_exits_kills_process_on_error():
    """Unexpected lifecycle errors should kill ParaView and close the server."""
    calls = []

    server = SimpleNamespace(
        serve_forever=lambda: None,
        shutdown=lambda: calls.append("shutdown"),
        server_close=lambda: calls.append("server_close"),
    )
    process = SimpleNamespace(
        wait=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        kill=lambda: calls.append("kill"),
    )

    with pytest.raises(RuntimeError, match="boom"):
        serve._run_server_until_process_exits(server, process)

    assert calls == ["kill", "shutdown", "server_close"]


def test_main_runs_server_forever(monkeypatch):
    """The CLI entry point should create a server and run it without ParaView."""
    calls = []

    class FakeServer:
        def __init__(self, address, context):
            calls.append(("init", address, context.default_dataset_uri))

        def serve_forever(self):
            calls.append("serve_forever")

    monkeypatch.setattr(serve, "_ServeHTTPServer", FakeServer)

    assert serve.main(["--host", "127.0.0.1", "--port", "0", "--dataset", "data"]) == 0
    assert calls == [("init", ("127.0.0.1", 0), "data"), "serve_forever"]


def test_main_runs_until_paraview_process_exits(monkeypatch):
    """ParaView mode should launch the plugin and use process-bound serving."""
    calls = []
    process = object()

    class FakeServer:
        def __init__(self, address, context):
            calls.append(("init", address, context.default_dataset_uri))

    monkeypatch.setattr(serve, "_ServeHTTPServer", FakeServer)
    monkeypatch.setattr(
        serve,
        "_run_server_until_process_exits",
        lambda server, pv_process: calls.append(("run_until_exit", server, pv_process)),
    )
    monkeypatch.setattr(
        "plaid.cli.paraview_plugin.run_paraview_with_plugin",
        lambda: process,
    )

    assert serve.main(["--port", "8123", "--ParaViewRun"]) == 0
    assert calls[0] == ("init", ("0.0.0.0", 8123), None)
    assert calls[1][0] == "run_until_exit"
    assert calls[1][2] is process
