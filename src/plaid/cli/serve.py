"""Serve PLAID datasets over a small HTTP API.

The ``plaid-serve`` command exposes local PLAID datasets to client tools and
the ParaView plugin. It supports lightweight discovery routes (``GET /health``
and ``GET /entry_points``) plus JSON ``POST`` routes for dataset metadata,
problem definitions, and samples.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from subprocess import Popen
from typing import Any, cast
from urllib.parse import urlparse

from plaid.containers import Sample
from plaid.storage import init_from_disk
from plaid.storage.common.reader import (
    load_infos_from_disk,
    load_problem_definitions_from_disk,
)
from plaid.utils.sample_json import sample_to_json_payload

log = logging.getLogger(__name__)

HEALTH_PAYLOAD: dict[str, object] = {"status": "ok"}
ENTRY_POINTS_PAYLOAD: dict[str, object] = {
    "samples_step": True,
    "process": False,
    "splits": True,
    "timesteps": True,
    "samples": True,
}
POST_DATASET_ROUTES = {"/samples", "/problem_definition", "/infos"}
PROCESS_UNSUPPORTED_PAYLOAD: dict[str, object] = {
    "error": "Endpoint /process is not supported by PLAID serve"
}


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
    if content_length <= 0:  # pragma: no cover
        return {}

    raw = handler.rfile.read(content_length)
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):  # pragma: no cover
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
) -> list[int]:
    """Validate and extract common sample request fields.

    Args:
        request: Parsed JSON payload.

    Returns:
        Requested sample identifiers.

    Raises:
        ValueError: If request fields are invalid.
    """
    sample_ids = request.get("sample_ids")

    if not isinstance(sample_ids, list) or not all(
        isinstance(sample_id, int) for sample_id in sample_ids
    ):
        raise ValueError("sample_ids must be a list of integers")
    if len(sample_ids) == 0:
        raise ValueError("sample_ids must not be empty")
    if any(sample_id < 0 for sample_id in sample_ids):
        raise ValueError("sample_ids must be non-negative integers")

    return sample_ids


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
        """Return dataset infos as a JSON-serializable dictionary.

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
        """Return one problem definition JSON.

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

    def do_GET(self) -> None:  # noqa: N802
        """Handle GET serve API discovery requests."""
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._send_json(HEALTH_PAYLOAD)
            return

        if parsed.path == "/entry_points":
            self._send_json(ENTRY_POINTS_PAYLOAD)
            return

        self._send_json({"GET error": "Not Found"}, status=HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        """Handle POST serve API requests."""
        parsed = urlparse(self.path)

        if parsed.path == "/health":
            self._send_json(HEALTH_PAYLOAD)
            return

        if parsed.path == "/process":
            self._send_json(
                PROCESS_UNSUPPORTED_PAYLOAD,
                status=HTTPStatus.NOT_IMPLEMENTED,
            )
            return

        if parsed.path not in POST_DATASET_ROUTES:
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

            sample_ids = _validate_sample_request(request_payload)
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
        help="Default dataset path used by endpoints.",
    )
    parser.add_argument(
        "--ParaViewRun",
        action="store_true",
        help="Run ParaView before the server (path from the PARAVIEW_EXEC env variable).",
    )
    return parser


def _run_server_until_process_exits(
    server: ThreadingHTTPServer,
    process: Popen[Any],
) -> None:
    """Run the HTTP server until an external process exits.

    Args:
        server: HTTP server to run in a background thread.
        process: External process whose lifetime controls the server.
    """
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        log.info("Waiting for ParaView to stop")
        process.wait()
    except KeyboardInterrupt:
        log.info("Stopping ParaView after keyboard interrupt")
        process.terminate()
        process.wait(timeout=5)
    except Exception:
        log.exception("Stopping ParaView after unexpected server lifecycle error")
        process.kill()
        raise
    finally:
        log.info("Shutting down PLAID data API")
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)


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
        os.environ["PLAID_PORT"] = str(args.port)
        from .paraview_plugin import run_paraview_with_plugin

        process = run_paraview_with_plugin()
    else:
        process = None

    context = ServeContext(default_dataset_uri=args.dataset)
    server = _ServeHTTPServer((str(args.host), int(args.port)), context)
    log.info("PLAID data API listening on http://%s:%s", args.host, args.port)

    if process is None:
        log.info("Running PLAID data API until interrupted")
        server.serve_forever()
    else:
        log.info("Running PLAID data API for ParaView")
        _run_server_until_process_exits(server, process)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
