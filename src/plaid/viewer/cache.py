"""Ephemeral-by-default artifact cache for the dataset viewer.

The cache lives under a per-process temporary directory by default and is
removed at shutdown. Four cleanup layers cover all practical failure modes:

1. ``atexit.register`` for normal Python exit.
2. Signal handlers for ``SIGINT`` / ``SIGTERM``.
3. A FastAPI lifespan context (provided by callers).
4. An orphan sweep at startup that removes directories left behind by
   previously-crashed processes (detected via ``os.kill(pid, 0)``).
"""

from __future__ import annotations

import atexit
import errno
import logging
import os
import re
import shutil
import signal
import tempfile
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

# Ephemeral tempdir naming: ``plaid-viewer-{pid}-{uuid4.hex}``.
_EPHEMERAL_PREFIX = "plaid-viewer-"
_EPHEMERAL_PATTERN = re.compile(r"^plaid-viewer-(?P<pid>\d+)-(?P<token>[0-9a-f]+)$")


def _process_is_alive(pid: int) -> bool:
    """Return ``True`` if a process with the given pid is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by someone else.
        return True
    except OSError as exc:  # pragma: no cover - defensive
        return exc.errno != errno.ESRCH
    return True


def sweep_orphans(temp_root: Path | None = None) -> list[Path]:
    """Remove viewer tempdirs whose owning process is no longer running.

    Args:
        temp_root: Base temp directory to scan. Defaults to
            :func:`tempfile.gettempdir`.

    Returns:
        List of directories that were removed.
    """
    root = Path(temp_root) if temp_root is not None else Path(tempfile.gettempdir())
    removed: list[Path] = []
    if not root.is_dir():
        return removed
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        match = _EPHEMERAL_PATTERN.match(entry.name)
        if match is None:
            continue
        pid = int(match.group("pid"))
        if _process_is_alive(pid):
            continue
        try:
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry)
            logger.info("Removed orphan viewer cache: %s", entry)
        except OSError as exc:  # pragma: no cover - defensive
            logger.warning("Could not remove orphan viewer cache %s: %s", entry, exc)
    return removed


class CacheRoot:
    """Context-manager-friendly artifact cache directory.

    When ``persistent_dir`` is ``None`` (the default), a new ephemeral tempdir
    named ``plaid-viewer-{pid}-{token}`` is created. The directory is
    removed at process exit (``atexit``), on ``SIGINT`` / ``SIGTERM``, and
    when the context manager is closed.

    When ``persistent_dir`` is provided, that directory is used as-is and is
    **not** removed. Callers wanting persistence pass this.
    """

    def __init__(
        self,
        persistent_dir: Path | None = None,
        *,
        install_signal_handlers: bool = True,
        run_orphan_sweep: bool = True,
    ) -> None:
        self._ephemeral = persistent_dir is None
        if self._ephemeral:
            if run_orphan_sweep:
                sweep_orphans()
            token = uuid.uuid4().hex[:12]
            base = Path(tempfile.gettempdir())
            self._path = base / f"{_EPHEMERAL_PREFIX}{os.getpid()}-{token}"
            self._path.mkdir(parents=True, exist_ok=False)
            atexit.register(self._safe_cleanup)
            if install_signal_handlers:
                self._install_signal_handlers()
        else:
            self._path = Path(persistent_dir)
            self._path.mkdir(parents=True, exist_ok=True)
        self._closed = False

    # ------------------------------------------------------------------ API

    @property
    def path(self) -> Path:
        """Root directory of the cache."""
        return self._path

    @property
    def is_ephemeral(self) -> bool:
        """Whether the cache directory is automatically cleaned up."""
        return self._ephemeral

    def close(self) -> None:
        """Remove the cache directory if it is ephemeral."""
        if self._closed:
            return
        self._closed = True
        if self._ephemeral:
            self._safe_cleanup()

    def __enter__(self) -> "CacheRoot":  # noqa: D105
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D105
        self.close()

    # -------------------------------------------------------------- Internals

    def _safe_cleanup(self) -> None:
        try:
            shutil.rmtree(self._path, ignore_errors=True)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to clean viewer cache %s: %s", self._path, exc)

    def _install_signal_handlers(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                previous = signal.getsignal(sig)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                continue

            def handler(signum, frame, _prev=previous):
                self._safe_cleanup()
                if callable(_prev) and _prev not in (signal.SIG_DFL, signal.SIG_IGN):
                    _prev(signum, frame)
                # Re-raise the default behaviour to keep expected exit codes.
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)

            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):  # pragma: no cover - non-main thread
                pass
