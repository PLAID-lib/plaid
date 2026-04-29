"""Runtime configuration for the dataset viewer."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ViewerConfig:
    """Static configuration for a viewer instance.

    Attributes:
        datasets_root: Directory scanned to discover datasets. A dataset is a
            subdirectory containing both ``data/`` and ``problem_definitions/``
            (or the root may itself be such a folder). When ``None``, the
            viewer starts without a root and the user is expected to pick one
            interactively (when ``allow_root_change`` is True).
        cache_dir: Root directory for ParaView artifacts. When ``None``, an
            ephemeral per-process directory is created under the OS temp root
            and cleaned up at shutdown.
        backend_id: PLAID backend identifier embedded in :class:`SampleRef`
            objects and in the artifact cache key.
        export_version: Opaque string mixed into the artifact cache key. Bump
            when export logic changes.
        extra_cache_key_fields: Extra fields serialised into the cache key.
        browse_roots: Directories the viewer is allowed to expose through the
            built-in file browser / datasets-root text field. Every candidate
            path must be a descendant of at least one of these roots. When
            empty, defaults to ``(Path.home(),)`` at the service level.
        allow_root_change: When ``True`` (default), the trame UI exposes a
            panel to change the datasets root at runtime. Set to ``False`` for
            public deployments (e.g. Hugging Face Spaces) where the root must
            remain fixed to what the operator configured.
    """

    datasets_root: Path | None = None
    cache_dir: Path | None = None
    backend_id: str = "disk"
    export_version: str = "1"
    extra_cache_key_fields: dict[str, str] = field(default_factory=dict)
    browse_roots: tuple[Path, ...] = ()
    allow_root_change: bool = True
