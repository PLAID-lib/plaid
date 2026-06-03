"""Tests for the PLAID prediction HTTP client."""

import json
from types import SimpleNamespace

from plaid.containers.sample import Sample
from plaid.utils.cgns_helper import compare_cgns_trees
from plaid.utils.predict_client import PlaidClient
from plaid.utils.sample_json import sample_to_json_payload


class _FakeResponse:
    """Small context-manager response used to avoid real HTTP calls."""

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


def _assert_same_sample_content(reference: Sample, candidate: Sample) -> None:
    """Assert that two samples contain equivalent timestamped CGNS trees."""
    assert sorted(reference.data.keys()) == sorted(candidate.data.keys())
    for time in reference.data:
        assert compare_cgns_trees(reference.data[time], candidate.data[time])


def test_plaid_client_initializes_default_configuration():
    """Client construction stores host, port, endpoints, protocol and timeout."""
    client = PlaidClient("localhost", 8080)

    assert client.host == "localhost"
    assert client.port == 8080
    assert client.protocol == "http"
    assert client.timeout == 100
    assert client.endpoints == {
        "health": "/health",
        "predict": "/predict",
        "problem_definition": "/problem_definition",
        "samples": "/samples",
    }


def test_request_json_posts_payload_and_decodes_response(monkeypatch):
    """Low-level JSON requests are sent as POST and decoded from UTF-8 JSON."""
    calls = []
    response_payload = {"status": "ok", "answer": 42}

    def fake_urlopen(req, timeout):
        calls.append(SimpleNamespace(req=req, timeout=timeout))
        return _FakeResponse(response_payload)

    monkeypatch.setattr("plaid.utils.predict_client.request.urlopen", fake_urlopen)
    client = PlaidClient("example.test", 1234)

    result = client._request_json("health", {"ping": True})

    assert result == response_payload
    assert len(calls) == 1
    req = calls[0].req
    assert req.full_url == "http://example.test:1234/health"
    assert req.get_method() == "POST"
    assert json.loads(req.data.decode("utf-8")) == {"ping": True}
    assert req.headers["Content-type"] == "application/json"
    assert calls[0].timeout == 100


def test_check_connection_returns_true_for_ok_status(monkeypatch):
    """The health endpoint is considered connected only when status is ok."""
    client = PlaidClient("localhost", 8000)
    calls = []

    def fake_request_json(endpoint, payload):
        calls.append((endpoint, payload))
        return {"status": "ok"}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert client.check_connection() is True
    assert calls == [("health", {})]


def test_check_connection_returns_false_for_bad_status(monkeypatch, capsys):
    """A non-ok health response returns False and reports the bad status."""
    client = PlaidClient("localhost", 8000)
    monkeypatch.setattr(
        client, "_request_json", lambda _endpoint, _payload: {"status": "bad"}
    )

    assert client.check_connection() is False

    captured = capsys.readouterr()
    assert "Server health check failed" in captured.out
    assert "Received data: bad" in captured.out


def test_check_connection_returns_false_on_exception(monkeypatch, capsys):
    """Connection exceptions are caught and converted to False."""
    client = PlaidClient("localhost", 8000)

    def raise_error(_endpoint, _payload):
        raise RuntimeError("server unavailable")

    monkeypatch.setattr(client, "_request_json", raise_error)

    assert client.check_connection() is False

    assert "Connection check failed: server unavailable" in capsys.readouterr().out


def test_predict_sends_sample_payload_and_decodes_first_sample(
    monkeypatch, sample_with_tree
):
    """Prediction serializes one sample and returns the first sample in response."""
    client = PlaidClient("localhost", 8000)
    response_sample = sample_with_tree.copy()
    calls = []

    def fake_request_json(endpoint, payload):
        calls.append((endpoint, payload))
        return {"samples": [sample_to_json_payload(response_sample)]}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    predicted = client.predict(sample_with_tree)

    assert calls == [("predict", {"sample": sample_to_json_payload(sample_with_tree)})]
    _assert_same_sample_content(response_sample, predicted)


def test_problem_definition_requests_problem_definition_endpoint(monkeypatch):
    """Problem definition requests are delegated to their configured endpoint."""
    client = PlaidClient("localhost", 8000)
    expected = {"features": ["pressure"]}
    calls = []

    def fake_request_json(endpoint, payload):
        calls.append((endpoint, payload))
        return expected

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    assert client.problem_definition() == expected
    assert calls == [("problem_definition", {})]


def test_samples_sends_selection_payload_and_decodes_samples(
    monkeypatch, sample_with_tree
):
    """Sample retrieval sends ids/split and reconstructs all returned samples."""
    client = PlaidClient("localhost", 8000)
    first_sample = sample_with_tree.copy()
    second_sample = Sample(path=None)
    calls = []

    def fake_request_json(endpoint, payload):
        calls.append((endpoint, payload))
        return {
            "samples": [
                sample_to_json_payload(first_sample),
                sample_to_json_payload(second_sample),
            ]
        }

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    samples = client.samples(sample_ids=[3, 5], split="test")

    assert calls == [("samples", {"sample_ids": [3, 5], "split": "test"})]
    assert len(samples) == 2
    _assert_same_sample_content(first_sample, samples[0])
    _assert_same_sample_content(second_sample, samples[1])
