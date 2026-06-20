"""Class to use the /process capability of a server."""

import json
from typing import Any
from urllib import request

from plaid.containers.sample import Sample
from plaid.utils.sample_json import sample_from_json_payload, sample_to_json_payload


class PlaidClient:
    """Client for making requests to a PLAID process server."""

    def __init__(self, host: str, port: int):
        """Initialize a process server client.

        Args:
            host: Hostname or IP address of the process server.
            port: Port number used by the process server.

        """
        self.host = host
        self.port = port
        self.endpoints = {
            "health": "/health",
            "process": "/process",
            "problem_definition": "/problem_definition",
            "infos": "/infos",
            "samples": "/samples",
        }
        self.protocol = "http"
        self.timeout = 100  # timeout for the response

    def _request_json(
        self, endpoint: str, payload: dict[str, object]
    ) -> dict[str, Any]:
        """Send a JSON POST request to an endpoint and decode the response.

        Args:
            endpoint: Endpoint key configured in ``self.endpoints``.
            payload: JSON-serializable payload to send in the request body.

        Returns:
            Decoded JSON response payload.

        """
        req = request.Request(
            url=f"{self.protocol}://{self.host}:{self.port}{self.endpoints[endpoint]}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def check_connection(self) -> bool:
        """Check if the server is healthy by querying the health endpoint."""
        try:
            data = self._request_json("health", {}).get("status", "payload missing")
            if data != "ok":
                print("Server health check failed: status not ok")
                print(f"Received data: {data}")
                return False
            return True
        except Exception as e:
            print(f"Connection check failed: {e}")
            return False

    def process(self, sample: Sample | None = None, **fields: Any) -> Sample:
        """Send a single-sample request to the process endpoint.

        The optional ``sample`` and any ``Sample`` passed as a keyword field are
        JSON-encoded automatically. All other keyword fields are forwarded
        verbatim, so the server operation contract stays opaque to this client.
        The client operates on a single sample at a time.

        Args:
            sample: Optional inline sample, sent as the ``sample`` field.
            **fields: Additional request fields forwarded to the server. Any
                ``Sample`` value is JSON-encoded before being sent.

        Returns:
            The single ``Sample`` returned by the server.

        """
        payload: dict[str, Any] = {
            key: sample_to_json_payload(value) if isinstance(value, Sample) else value
            for key, value in fields.items()
        }
        if sample is not None:
            payload["sample"] = sample_to_json_payload(sample)

        response = self._request_json("process", payload)
        return sample_from_json_payload(response["samples"][0])

    def problem_definition(self) -> dict[str, Any]:
        """Get the problem definition from the server.

        Returns:
            A dictionary containing the problem definition as provided by the server.
        """
        return self._request_json("problem_definition", {})

    def infos(self) -> dict[str, Any]:
        """Get the infos from the server.

        Returns:
            A dictionary containing the infos as provided by the server.
        """
        return self._request_json("infos", {})

    def samples(self, sample_ids: list[int], split: str) -> list[Sample]:
        """Request samples from the server by sample IDs and split.

        Args:
            sample_ids: A list of integers representing the IDs of the samples to request.
            split: A string indicating the data split (e.g., "training", "validation", "test") from which to request the samples.

        Returns:
            A list of Sample objects corresponding to the requested sample IDs and split.
        """
        payload: dict[str, Any] = {"sample_ids": sample_ids, "split": split}
        response = self._request_json("samples", payload)
        return [
            sample_from_json_payload(sample_payload)
            for sample_payload in response["samples"]
        ]
