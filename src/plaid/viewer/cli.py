"""Command-line entry point for the dataset viewer.

Starts a single self-contained trame server. There is no FastAPI backend
and no separate port: dataset discovery, sample loading, CGNS export and
the 3D view are all served by the same trame process.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from plaid.viewer.cache import CacheRoot
from plaid.viewer.config import ViewerConfig
from plaid.viewer.preferences import get_last_datasets_root
from plaid.viewer.services import ParaviewArtifactService, PlaidDatasetService

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plaid-viewer",
        description="Launch the dataset viewer (trame + VTK).",
    )
    parser.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help=(
            "Directory containing one subdirectory per PLAID dataset. "
            "When omitted, the viewer starts without a root and the user "
            "selects one from the UI (unless --disable-root-change is set)."
        ),
    )
    parser.add_argument(
        "--browse-roots",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Directories the UI is allowed to expose through the datasets "
            "root text field and file browser. Defaults to the user home "
            "directory. Any path outside these roots is rejected."
        ),
    )
    parser.add_argument(
        "--disable-root-change",
        action="store_true",
        help=(
            "Hide the 'Datasets root' UI panel; the root stays fixed to "
            "--datasets-root for the lifetime of the server. Recommended "
            "for public deployments (e.g. Hugging Face Spaces)."
        ),
    )
    parser.add_argument(
        "--dataset-id",
        default=None,
        help=(
            "Dataset id selected when the viewer starts. Use together with "
            "--disable-dataset-change to pin the UI to that dataset."
        ),
    )
    parser.add_argument(
        "--disable-dataset-change",
        action="store_true",
        help=(
            "Hide the 'Dataset' dropdown; the selected dataset stays fixed "
            "for the lifetime of the server."
        ),
    )

    parser.add_argument("--host", default="127.0.0.1", help="Trame server host.")
    parser.add_argument("--port", type=int, default=8080, help="Trame server port.")
    parser.add_argument(
        "--backend-id",
        default="disk",
        help="PLAID backend identifier embedded in SampleRefs.",
    )
    parser.add_argument(
        "--hub-repo",
        action="append",
        default=None,
        metavar="NAMESPACE/NAME",
        help=(
            "Register a Hugging Face Hub repo id streamed through "
            "plaid.storage.init_streaming_from_hub. Repeat the flag to "
            "pre-register multiple repos. Additional repos can be added "
            "at runtime from the UI (unless --disable-root-change is set)."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the viewer until interrupted.

    Args:
        argv: Optional override of ``sys.argv[1:]`` for tests.

    Returns:
        Process exit code.
    """
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    # Permanently silence the process's file-descriptor 2 so the HDF5 /
    # CGNS C libraries (used by both VTK's ``vtkCGNSReader`` and PLAID's
    # pyCGNS loader) cannot pollute the console with messages like
    # ``Mismatch in number of children and child IDs read``. Python's
    # ``sys.stderr`` is preserved so tracebacks and the logger keep
    # working. See ``_reroute_c_stderr`` for the details.
    from plaid.viewer.trame_app.server import (  # noqa: PLC0415
        _reroute_c_stderr,
    )

    _reroute_c_stderr()

    # When no explicit ``--datasets-root`` is passed, fall back to the
    # last local root the user selected in a previous session (persisted
    # under ``$XDG_CONFIG_HOME/plaid/viewer.json``). This makes the
    # viewer "remember" the last dataset directory without requiring the
    # CLI flag on every launch.
    effective_datasets_root = args.datasets_root
    if effective_datasets_root is None:
        effective_datasets_root = get_last_datasets_root()
        if effective_datasets_root is not None:
            logger.info("Using persisted datasets root: %s", effective_datasets_root)
    browse_roots = tuple(args.browse_roots) if args.browse_roots else ()
    config = ViewerConfig(
        datasets_root=effective_datasets_root,
        backend_id=args.backend_id,
        browse_roots=browse_roots,
        allow_root_change=not args.disable_root_change,
        initial_dataset_id=args.dataset_id,
        allow_dataset_change=not args.disable_dataset_change,
    )

    with CacheRoot() as cache:
        dataset_service = PlaidDatasetService(config)
        for repo_id in args.hub_repo or []:
            try:
                dataset_service.add_hub_dataset(repo_id)
            except ValueError as exc:
                logger.warning("Ignoring --hub-repo %r: %s", repo_id, exc)
        artifact_service = ParaviewArtifactService(dataset_service, cache.path)

        # Deferred import so ``--help`` works without trame installed.
        from plaid.viewer.trame_app.server import build_server  # noqa: PLC0415

        server = build_server(dataset_service, artifact_service)
        server.start(host=args.host, port=args.port, open_browser=False)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
