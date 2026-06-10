"""PLAID data server entry point for ParaView/client integrations.

This module provides both a CLI entry point and HTTP API entry points.

CLI entry point
---------------
- ``main(argv=None)`` (installed command: ``plaid-serve``)
- Arguments:
  - ``--host``: bind address (default ``0.0.0.0``)
  - ``--port``: bind port (default ``8000``)
  - ``ParaViewRun``: if true, launch ParaView plugin and run server in background

HTTP entry points
-----------------
All dataset API routes are handled by :meth:`_Handler.do_GET` and currently
expect a JSON request body (even though the verb is ``GET``).

- ``GET /health``
  - Input payload: none
  - Output payload:
    - ``{"status": "ok"}``

- ``GET /splits``
  - Input payload:
    - ``dataset`` or ``uri`` (string, required)
    - ``split`` (string, optional)
  - Output payload:
    - ``{"splits": {"<split_name>": <count_or_null>, ...}}``

- ``GET /timesteps``
  - Input payload:
    - ``dataset`` or ``uri`` (string, required)
    - ``split`` (string, optional; required if dataset has multiple splits)
    - ``sample_ids`` (list[int], required, non-empty)
    - ``include_features`` (list[str], optional; validated but not used)
  - Output payload:
    - ``{"time_times": [{"sample_id": int, "times": list[float], "count": int}, ...]}``

- ``GET /samples``
  - Input payload:
    - ``dataset`` or ``uri`` (string, required)
    - ``split`` (string, optional; required if dataset has multiple splits)
    - ``sample_ids`` (list[int], required, non-empty)
    - ``include_features`` (list[str], optional)
  - Output payload:
    - ``{"samples": [<serialized_sample_dict>, ...]}``

- ``GET /samples_time``
  - Input payload:
    - ``dataset`` or ``uri`` (string, required)
    - ``split`` (string, optional; required if dataset has multiple splits)
    - ``sample_ids`` (list[int], required, must contain exactly one id)
    - ``include_features`` (list[str], optional)
    - ``time`` (int | float, required)
  - Output payload:
    - ``{"samples": [<serialized_sample_dict>], "time": float}``

    methode to implement
- ``GET /entry_points``
  - Output payload:
    - ``{"samples_time": True, `predict`:False...}



Methods currently implemented
-----------------------------
- Implemented: ``GET`` for all above routes.
- Not implemented: ``PUT`` routes.

Error payloads
--------------
- Validation errors return ``400`` with ``{"error": "<message>"}``.
- Unexpected errors return ``500`` with
  ``{"error": "Internal server error: <message>"}``.

No Maestro runtime or prediction endpoint is used here.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np

from plaid.containers import Sample
from plaid.storage import init_from_disk
from plaid.storage.common.preprocessor import build_sample_dict
from plaid.storage.common.reader import (
    load_infos_from_disk,
    load_problem_definitions_from_disk,
)
from plaid.utils.sample_json import sample_to_json_payload

log = logging.getLogger(__name__)


def _to_jsonable(value: object) -> object:
    """Convert numpy/scalar values into JSON-serializable structures.

    Args:
        value: Input value to serialize.

    Returns:
        JSON-compatible Python object.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    return value


def _sample_to_payload(
    sample: Sample,
    include_features: list[str] | None,
) -> dict[str, object]:
    """Serialize a PLAID sample to a JSON-ready payload.

    Args:
        sample: PLAID sample to serialize.
        include_features: Optional list of feature paths to keep.

    Returns:
        Dictionary payload keyed by feature path.
    """
    sample_dict, _, _ = build_sample_dict(sample)
    serialized = {str(key): _to_jsonable(value) for key, value in sample_dict.items()}
    if not include_features:
        return serialized
    allowed = set(include_features)
    return {key: value for key, value in serialized.items() if key in allowed}


def _parse_request_payload(handler: BaseHTTPRequestHandler) -> dict[str, object]:
    """Parse and validate a JSON request payload from an HTTP handler.

    Args:
        handler: Request handler exposing headers and input stream.

    Returns:
        Parsed JSON payload.

    Raises:
        ValueError: If body is missing, invalid, or not a JSON object.
    """
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        raise ValueError("Request body must be a non-empty JSON object")

    raw = handler.rfile.read(content_length)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return cast(dict[str, object], payload)


def _parse_optional_request_payload(
    handler: BaseHTTPRequestHandler,
) -> dict[str, object]:
    """Parse an optional JSON request payload from an HTTP handler.

    Args:
        handler: Request handler exposing headers and input stream.

    Returns:
        Parsed JSON object, or an empty dictionary when no body is provided.

    Raises:
        ValueError: If a provided body is invalid or not a JSON object.
    """
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}

    raw = handler.rfile.read(content_length)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")
    return cast(dict[str, object], payload)


def _parse_dataset_uri(request: dict[str, object]) -> str:
    """Extract dataset location from a request payload.

    Supported keys are ``dataset`` and ``uri``.

    Args:
        request: Parsed JSON payload.

    Returns:
        Dataset location string.

    Raises:
        ValueError: If no valid dataset location is provided.
    """
    for key in ("dataset", "uri"):
        value = request.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError("dataset path/URI is required (dataset or uri)")


def _resolve_dataset_uri(
    request: dict[str, object],
    default_dataset_uri: str | None,
) -> str:
    """Resolve dataset URI from request payload or server default.

    Args:
        request: Parsed JSON payload.
        default_dataset_uri: Optional dataset configured at server start.

    Returns:
        Dataset path/URI string.

    Raises:
        ValueError: If no dataset can be resolved.
    """
    try:
        return _parse_dataset_uri(request)
    except ValueError:
        if default_dataset_uri is not None and default_dataset_uri.strip():
            return default_dataset_uri.strip()
        raise


def _parse_split(request: dict[str, object]) -> str | None:
    """Extract optional split from request payload.

    Args:
        request: Parsed JSON payload.

    Returns:
        The split name if provided, else ``None``.

    Raises:
        ValueError: If split is present but invalid.
    """
    split = request.get("split")
    if split is None:
        return None
    if not isinstance(split, str) or not split.strip():
        raise ValueError("split must be a non-empty string")
    return split.strip()


def _validate_sample_request(
    request: dict[str, object],
) -> tuple[list[int], list[str] | None]:
    """Validate and extract common sample request fields.

    Args:
        request: Parsed JSON payload.

    Returns:
        Tuple ``(sample_ids, include_features)``.

    Raises:
        ValueError: If request fields are invalid.
    """
    sample_ids = request.get("sample_ids")
    include_features = request.get("include_features")

    if not isinstance(sample_ids, list) or not all(
        isinstance(sample_id, int) for sample_id in sample_ids
    ):
        raise ValueError("sample_ids must be a list of integers")
    if len(sample_ids) == 0:
        raise ValueError("sample_ids must not be empty")
    if any(sample_id < 0 for sample_id in sample_ids):
        raise ValueError("sample_ids must be non-negative integers")

    if include_features is not None and (
        not isinstance(include_features, list)
        or not all(isinstance(feature, str) for feature in include_features)
    ):
        raise ValueError("include_features must be a list of strings")

    return sample_ids, cast(list[str] | None, include_features)


def _restrict_sample_to_time(sample: Sample, time: float | None) -> Sample:
    """Restrict a sample to a single time value when requested.

    Args:
        sample: Source sample.
        time: Requested time value, or ``None`` for full sample.

    Returns:
        Original sample if ``time is None`` else a sample containing only that time.
    """
    if time is None:
        return sample
    sample_tmp = type(sample)()
    sample_tmp.features.data[time] = sample.features.data[time]
    return sample_tmp


@dataclass(slots=True)
class _DatasetStore:
    """Cached dataset/converter dictionaries for a dataset URI."""

    datasetdict: dict[str, Any]
    converterdict: dict[str, Any]


@dataclass(slots=True)
class ServeContext:
    """Serving context shared by all HTTP handlers."""

    default_dataset_uri: str | None = None
    stores: dict[str, _DatasetStore] = field(default_factory=dict)

    def resolve_dataset_uri(self, request: dict[str, object]) -> str:
        """Resolve dataset URI from a request or the server default.

        Args:
            request: Parsed JSON payload.

        Returns:
            Dataset path/URI string.
        """
        return _resolve_dataset_uri(request, self.default_dataset_uri)

    def _get_store(self, dataset_uri: str) -> _DatasetStore:
        """Load and cache a PLAID dataset from disk.

        Args:
            dataset_uri: Dataset path/URI provided by the client.

        Returns:
            Loaded/cached dataset store.

        Raises:
            ValueError: If dataset cannot be loaded.
        """
        key = dataset_uri.strip()
        if key in self.stores:
            return self.stores[key]

        local_path = Path(key)
        if not local_path.exists() or not local_path.is_dir():
            raise ValueError(
                f"Dataset path does not exist or is not a directory: {key}"
            )

        try:
            datasetdict, converterdict = init_from_disk(str(local_path))
        except Exception as exc:  # pragma: no cover - defensive path
            raise ValueError(f"Failed to load dataset from {key}: {exc}") from exc

        if len(datasetdict) == 0:
            raise ValueError(f"Dataset has no splits: {key}")

        store = _DatasetStore(datasetdict=datasetdict, converterdict=converterdict)
        self.stores[key] = store
        return store

    @staticmethod
    def _resolve_split(store: _DatasetStore, split: str | None) -> str:
        """Resolve a split key from request.

        Args:
            store: Dataset store.
            split: Requested split, or ``None``.

        Returns:
            Resolved split key.

        Raises:
            ValueError: If split is ambiguous or unknown.
        """
        available = list(store.datasetdict.keys())
        if split is not None:
            if split not in store.datasetdict:
                raise ValueError(
                    f"Unknown split {split!r}; available splits: {sorted(available)}"
                )
            return split

        if len(available) == 1:
            return available[0]

        raise ValueError(
            "split is required when dataset has multiple splits; "
            f"available splits: {sorted(available)}"
        )

    def get_splits(self, dataset_uri: str) -> dict[str, int | None]:
        """Return available splits and sample counts for a dataset.

        Args:
            dataset_uri: Dataset path/URI provided by the client.

        Returns:
            Mapping ``split -> count``.
        """
        store = self._get_store(dataset_uri)
        counts: dict[str, int | None] = {}
        for split, ds in store.datasetdict.items():
            try:
                counts[split] = len(ds)
            except TypeError:
                counts[split] = None
        return counts

    def get_time_steps(
        self,
        dataset_uri: str,
        split: str | None,
        sample_ids: list[int],
    ) -> list[dict[str, object]]:
        """Return available time-step values for requested samples.

        Args:
            dataset_uri: Dataset path/URI.
            split: Requested split.
            sample_ids: Existing sample IDs.

        Returns:
            List of dictionaries with sample id, time values and count.
        """
        store = self._get_store(dataset_uri)
        split_key = self._resolve_split(store, split)
        print(f"get_time_steps : {split_key}")
        if split_key is None:
            return [
                {
                    "sample_id": sample_id,
                    "times": [],
                    "count": 0,
                }
                for sample_id in sample_ids
            ]

        dataset = store.datasetdict[split_key]
        converter = store.converterdict[split_key]

        results: list[dict[str, object]] = []
        for sample_id in sample_ids:
            sample = converter.to_plaid(dataset, sample_id)
            times = [float(time) for time in sample.get_all_time_values()]
            results.append(
                {
                    "sample_id": sample_id,
                    "times": times,
                    "count": len(times),
                }
            )
        return results

    def get_samples(
        self,
        dataset_uri: str,
        split: str | None,
        sample_ids: list[int],
        include_features: list[str] | None,
        time: float | None = None,
    ) -> list[dict[str, object]]:
        """Return serialized source samples from a PLAID dataset.

        Args:
            dataset_uri: Dataset path/URI.
            split: Requested split.
            sample_ids: Existing sample IDs.
            include_features: Optional list of feature paths to keep.
            time: Optional specific time value to read.

        Returns:
            Serialized sample payloads.
        """
        store = self._get_store(dataset_uri)
        split_key = self._resolve_split(store, split)
        dataset = store.datasetdict[split_key]
        converter = store.converterdict[split_key]

        payloads: list[dict[str, object]] = []
        for sample_id in sample_ids:
            sample = converter.to_plaid(dataset, sample_id)
            payloads.append(
                _sample_to_payload(
                    sample=_restrict_sample_to_time(sample, time),
                    include_features=include_features,
                )
            )
        return payloads

    def get_sample_objects(
        self,
        dataset_uri: str,
        split: str | None,
        sample_ids: list[int],
    ) -> list[Sample]:
        """Return source samples from a PLAID dataset.

        Args:
            dataset_uri: Dataset path/URI.
            split: Requested split.
            sample_ids: Existing sample IDs.

        Returns:
            PLAID sample objects.
        """
        store = self._get_store(dataset_uri)
        split_key = self._resolve_split(store, split)
        dataset = store.datasetdict[split_key]
        converter = store.converterdict[split_key]

        return [converter.to_plaid(dataset, sample_id) for sample_id in sample_ids]

    @staticmethod
    def get_infos(dataset_uri: str) -> dict[str, object]:
        """Return dataset infos in Maestro-compatible JSON shape.

        Args:
            dataset_uri: Dataset path/URI.

        Returns:
            Serialized dataset infos.
        """
        return cast(dict[str, object], load_infos_from_disk(dataset_uri).model_dump())

    @staticmethod
    def get_problem_definition(
        dataset_uri: str,
        name: str | None = None,
    ) -> dict[str, object]:
        """Return one problem definition in Maestro-compatible JSON shape.

        Args:
            dataset_uri: Dataset path/URI.
            name: Optional problem definition name to select.

        Returns:
            Serialized problem definition.

        Raises:
            ValueError: If the requested definition is unavailable.
        """
        problem_definitions = load_problem_definitions_from_disk(dataset_uri)
        if name is not None:
            if name not in problem_definitions:
                raise ValueError(
                    f"Problem definition {name!r} not found; available definitions: "
                    f"{sorted(problem_definitions)}"
                )
            return cast(dict[str, object], problem_definitions[name].model_dump())

        if "PLAID_benchmark" in problem_definitions:
            return cast(
                dict[str, object],
                problem_definitions["PLAID_benchmark"].model_dump(),
            )
        if len(problem_definitions) == 1:
            problem_definition = next(iter(problem_definitions.values()))
            return cast(dict[str, object], problem_definition.model_dump())

        first_name = sorted(problem_definitions)[0]
        return cast(dict[str, object], problem_definitions[first_name].model_dump())


class _Handler(BaseHTTPRequestHandler):
    """HTTP handler for PLAID dataset requests."""

    def _send_json(
        self,
        payload: dict[str, object],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


    def do_POST(self) -> None:  # noqa: N802
        """Handle Maestro-compatible POST serve API requests."""
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._send_json({"status": "ok"})
            return

        if parsed.path == "/predict":
            self._send_json(
                {"error": "Endpoint /predict is not supported by PLAID serve"},
                status=HTTPStatus.NOT_IMPLEMENTED,
            )
            return

        if parsed.path not in {
            "/samples",
            "/problem_definition",
            "/infos",
        }:
            self._send_json({"POST error": "Not Found"}, status=HTTPStatus.NOT_FOUND)
            return

        try:
            request_payload = _parse_optional_request_payload(self)
            server = cast("_ServeHTTPServer", self.server)
            dataset_uri = server.context.resolve_dataset_uri(request_payload)

            if parsed.path == "/problem_definition":
                problem_definition_name = request_payload.get(
                    "problem_definition",
                    request_payload.get("problem_definition_name"),
                )
                if problem_definition_name is not None and not isinstance(
                    problem_definition_name, str
                ):
                    raise ValueError("problem_definition must be a string")
                self._send_json(
                    server.context.get_problem_definition(
                        dataset_uri,
                        name=problem_definition_name,
                    )
                )
                return

            if parsed.path == "/infos":
                self._send_json(server.context.get_infos(dataset_uri))
                return

            sample_ids, _ = _validate_sample_request(request_payload)
            split = _parse_split(request_payload)
            samples = server.context.get_sample_objects(
                dataset_uri=dataset_uri,
                split=split,
                sample_ids=sample_ids,
            )
            self._send_json(
                {"samples": [sample_to_json_payload(sample) for sample in samples]}
            )
            return

        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - defensive path
            log.exception("Serve API request failed")
            self._send_json(
                {"error": f"Internal server error: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )


class _ServeHTTPServer(ThreadingHTTPServer):
    """HTTP server holding shared serving context."""

    def __init__(self, server_address: tuple[str, int], context: ServeContext):
        super().__init__(server_address, _Handler)
        self.context = context


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="plaid-serve",
        description="Run PLAID dataset server for client/ParaView integrations.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Default dataset path used by Maestro-compatible POST endpoints.",
    )
    parser.add_argument(
        "--ParaViewRun",
        action="store_true",
        help="Run ParaView before the server (path from the PARAVIEW_EXEC env varialbe).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run HTTP server for PLAID dataset access.

    Args:
        argv: Optional override of ``sys.argv[1:]`` for tests.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.ParaViewRun:
        import os

        os.environ["PLAID_PORT"] = str(args.port)
        from .paraview_plugin import run_paraview_with_plugin

        process = run_paraview_with_plugin()
    else:
        process = None

    context = ServeContext(default_dataset_uri=args.dataset)
    server = _ServeHTTPServer((str(args.host), int(args.port)), context)
    log.info("PLAID data API listening on http://%s:%s", args.host, args.port)

    if process is None:
        print("Runing server for ever")
        server.serve_forever()
    else:
        print("Launching the server for paraview")
        import threading

        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        try:
            print("Wainting for paraview to stop")
            process.wait()
        except:  # noqa: E722
            process.kill()
        finally:
            print("Killing server")
            server.shutdown()
            server.server_close()
            server_thread.join(timeout=5)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
