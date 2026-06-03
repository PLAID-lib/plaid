"""Class to use the /predict capability of a server."""

import json
from typing import Any
from urllib import request

from plaid.containers.sample import Sample
from plaid.utils.sample_json import sample_from_json_payload, sample_to_json_payload


class PlaidClient():
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.endpoints = {
            "health": "/health",
            "predict": "/predict",
            "problem_definition": "/problem_definition",
            "samples": "/samples",
        }
        self.protocol = "http"
        self.timeout = 100 # timeout for the response

    def _request_json(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        req = request.Request(
            url=f"{self.protocol}://{self.host}:{self.port}{self.endpoints[endpoint]}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def check_connection(self) -> bool:
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

    def predict(self, sample: Sample) -> Sample:
        payload : dict [str,Any] = {"sample": sample_to_json_payload(sample) }
        response = self._request_json("predict", payload)
        return sample_from_json_payload(response["samples"][0])

    def problem_definition(self):
        return self._request_json("problem_definition", {})

    def samples(self, sample_ids: list[int], split: str) -> list[Sample]:
        payload : dict [str,Any] = {"sample_ids": sample_ids, "split": split}
        response = self._request_json("samples", payload)
        return [sample_from_json_payload(sample_payload) for sample_payload in response["samples"]]