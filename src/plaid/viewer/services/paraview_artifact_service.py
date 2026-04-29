"""Produce ParaView-readable artifacts from PLAID samples.

This module is the one place in PLAID that writes CGNS files on disk. It
delegates the actual CGNS export to PLAID (``Sample.save_to_dir`` writes one
CGNS per timestep under ``meshes/``), then adds:

* A ``.cgns.series`` sidecar JSON file that ParaView's ``vtkCGNSReader`` /
  ``vtkCGNSFileSeriesReader`` understands for multi-timestep samples.
* A deterministic artifact id derived from a SHA256 cache key so the same
  inputs always resolve to the same folder.
* An optional ``scene.pvsm`` placeholder for future preset work.
* A ``metadata.json`` describing the artifact.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from plaid.viewer.models import ParaviewArtifact, SampleRef
from plaid.viewer.services.plaid_dataset_service import PlaidDatasetService

logger = logging.getLogger(__name__)

EXPORT_VERSION = "1"
ARTIFACT_TYPE = "raw"


@dataclass(frozen=True)
class _ArtifactLayout:
    """Internal paths for a single artifact folder."""

    root: Path
    meshes_dir: Path
    series_path: Path
    single_cgns_path: Path
    metadata_path: Path
    state_path: Path


def _plaid_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("pyplaid")
    except PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def _build_cache_key(
    ref: SampleRef, *, export_version: str, extra: dict[str, str] | None = None
) -> str:
    """Return a deterministic SHA256 cache key for a sample export."""
    payload = {
        "backend_id": ref.backend_id,
        "dataset_id": ref.dataset_id,
        "split": ref.split,
        "sample_id": ref.sample_id,
        "export_mode": "default",
        "artifact_type": ARTIFACT_TYPE,
        "plaid_version": _plaid_version(),
        "export_version": export_version,
    }
    if extra:
        payload["extra"] = dict(sorted(extra.items()))
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest


def _artifact_layout(
    cache_root: Path, ref: SampleRef, cache_key: str
) -> _ArtifactLayout:
    split = ref.split if ref.split is not None else "_default"
    root = (
        cache_root
        / "datasets"
        / ref.dataset_id
        / split
        / ref.sample_id
        / cache_key[:16]
    )
    return _ArtifactLayout(
        root=root,
        meshes_dir=root / "meshes",
        series_path=root / "meshes.cgns.series",
        single_cgns_path=root / "mesh.cgns",
        metadata_path=root / "metadata.json",
        state_path=root / "scene.pvsm",
    )


def _write_series_sidecar(
    series_path: Path, cgns_files: list[tuple[Path, float]]
) -> None:
    """Write a ParaView ``.cgns.series`` sidecar for the given file list.

    Each entry's ``name`` is stored as a POSIX-style path relative to the
    sidecar file so ``vtkCGNSFileSeriesReader`` can resolve it consistently
    across platforms. Notably, time-series CGNS files live in the
    ``meshes/`` subdirectory, so we keep that prefix instead of only the
    file name.
    """
    payload = {
        "file-series-version": "1.0",
        "files": [
            {"name": Path(path).as_posix(), "time": time} for path, time in cgns_files
        ],
    }
    series_path.write_text(json.dumps(payload, indent=2))


def _collect_time_values(sample) -> list[float]:
    data = getattr(sample.features, "data", None)
    if not data:
        return []
    return sorted(float(t) for t in data.keys())


class ParaviewArtifactService:
    """Create and look up ParaView-readable artifacts in a cache directory.

    Args:
        dataset_service: Used to load :class:`plaid.Sample` instances.
        cache_root: Root of the artifact cache. Usually owned by a
            :class:`plaid.viewer.cache.CacheRoot` instance.
        export_version: Opaque string included in the cache key. Bump this
            whenever the export logic changes in a backwards-incompatible way.
        extra_cache_key_fields: Extra fields to mix into the cache key (for
            example to invalidate artifacts when a preset template changes).
    """

    def __init__(
        self,
        dataset_service: PlaidDatasetService,
        cache_root: Path,
        *,
        export_version: str = EXPORT_VERSION,
        extra_cache_key_fields: dict[str, str] | None = None,
    ) -> None:
        self._dataset_service = dataset_service
        self._cache_root = Path(cache_root)
        self._cache_root.mkdir(parents=True, exist_ok=True)
        self._export_version = export_version
        self._extra = dict(extra_cache_key_fields or {})
        self._by_id: dict[str, ParaviewArtifact] = {}

    # ------------------------------------------------------------ Public API

    def ensure_artifact(
        self, ref: SampleRef, *, force: bool = False
    ) -> ParaviewArtifact:
        """Return a :class:`ParaviewArtifact` for ``ref``, creating it if needed."""
        cache_key = _build_cache_key(
            ref, export_version=self._export_version, extra=self._extra
        )
        layout = _artifact_layout(self._cache_root, ref, cache_key)

        if force and layout.root.exists():
            shutil.rmtree(layout.root)

        if layout.metadata_path.is_file() and not force:
            artifact = self._load_existing(layout, cache_key)
            self._by_id[artifact.artifact_id] = artifact
            return artifact

        layout.root.mkdir(parents=True, exist_ok=True)
        artifact = self._create(ref, layout, cache_key)
        self._by_id[artifact.artifact_id] = artifact
        return artifact

    def get(self, artifact_id: str) -> ParaviewArtifact:
        """Return a previously-created artifact by id.

        Raises:
            KeyError: If no artifact with this id has been created.
        """
        if artifact_id not in self._by_id:
            raise KeyError(f"Unknown artifact id: {artifact_id}")
        return self._by_id[artifact_id]

    # -------------------------------------------------------------- Internals

    def _create(
        self,
        ref: SampleRef,
        layout: _ArtifactLayout,
        cache_key: str,
    ) -> ParaviewArtifact:
        sample = self._dataset_service.load_sample(ref)
        times = _collect_time_values(sample)

        layout.meshes_dir.mkdir(exist_ok=True)
        # PLAID writes one CGNS per timestep as ``meshes/mesh_{i:09d}.cgns``.
        sample.save_to_dir(layout.root, overwrite=True)

        cgns_files = sorted(layout.meshes_dir.glob("mesh_*.cgns"))
        if not cgns_files:
            raise RuntimeError(
                f"PLAID produced no CGNS files for sample {ref.encode()}"
            )

        is_time_series = len(cgns_files) > 1 or len(times) > 1
        if is_time_series:
            pairs = [
                (layout.meshes_dir.relative_to(layout.root) / f.name, t)
                for f, t in zip(
                    cgns_files, times or range(len(cgns_files)), strict=False
                )
            ]
            # Reformat to full-path-relative-to-series-file entries.
            _write_series_sidecar(
                layout.series_path,
                [(Path("meshes") / pair[0].name, float(pair[1])) for pair in pairs],
            )
            cgns_path = layout.series_path
        else:
            # Move the single CGNS file up one level for convenience.
            cgns_files[0].replace(layout.single_cgns_path)
            cgns_path = layout.single_cgns_path

        metadata = {
            "artifact_type": ARTIFACT_TYPE,
            "cache_key": cache_key,
            "export_version": self._export_version,
            "plaid_version": _plaid_version(),
            "sample_ref": {
                "backend_id": ref.backend_id,
                "dataset_id": ref.dataset_id,
                "split": ref.split,
                "sample_id": ref.sample_id,
            },
            "cgns_path": str(cgns_path.relative_to(layout.root)),
            "is_time_series": is_time_series,
            "n_files": len(cgns_files),
            "time_values": list(times),
        }
        layout.metadata_path.write_text(json.dumps(metadata, indent=2))

        return ParaviewArtifact(
            artifact_id=cache_key[:16],
            cgns_path=cgns_path,
            state_path=None,
            metadata_path=layout.metadata_path,
            cache_key=cache_key,
            created=True,
        )

    @staticmethod
    def _load_existing(layout: _ArtifactLayout, cache_key: str) -> ParaviewArtifact:
        metadata = json.loads(layout.metadata_path.read_text())
        cgns_path = layout.root / metadata["cgns_path"]
        state_path = layout.state_path if layout.state_path.is_file() else None
        return ParaviewArtifact(
            artifact_id=cache_key[:16],
            cgns_path=cgns_path,
            state_path=state_path,
            metadata_path=layout.metadata_path,
            cache_key=cache_key,
            created=False,
        )


def ensure_paraview_artifact(
    sample_ref: SampleRef,
    *,
    cache_dir: Path,
    dataset_service: PlaidDatasetService,
    force: bool = False,
) -> ParaviewArtifact:
    """Functional wrapper around :meth:`ParaviewArtifactService.ensure_artifact`."""
    service = ParaviewArtifactService(dataset_service, cache_dir)
    return service.ensure_artifact(sample_ref, force=force)
