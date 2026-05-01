"""Data models for the gdataset viewer.

Contains both immutable dataclasses used by services (`SampleRef`,
`ParaviewArtifact`) and pydantic models used as FastAPI response payloads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class SampleRef:
    """Backend-agnostic reference to a PLAID sample.

    Attributes:
        backend_id: Identifier of the PLAID storage backend (e.g. ``"disk"``,
            ``"hf_datasets"``, ``"zarr"``).
        dataset_id: Identifier of the dataset (typically the dataset directory
            name).
        split: Optional split name (``"train"``, ``"test"``, ...). ``None``
            when the dataset is not split.
        sample_id: Identifier of the sample within the split. For disk-backed
            datasets this is the zero-based index rendered as a string.
    """

    backend_id: str
    dataset_id: str
    split: str | None
    sample_id: str

    def encode(self) -> str:
        """Return a URL-safe string identifier usable as a route parameter."""
        split = self.split if self.split is not None else "_"
        return f"{self.backend_id}:{self.dataset_id}:{split}:{self.sample_id}"

    @classmethod
    def decode(cls, value: str) -> "SampleRef":
        """Parse a string produced by :meth:`encode`."""
        parts = value.split(":")
        if len(parts) != 4:
            raise ValueError(f"Invalid sample reference: {value!r}")
        backend_id, dataset_id, split, sample_id = parts
        return cls(
            backend_id=backend_id,
            dataset_id=dataset_id,
            split=None if split == "_" else split,
            sample_id=sample_id,
        )


@dataclass(frozen=True)
class ParaviewArtifact:
    """A ParaView-readable artifact produced from a PLAID sample.

    For time-dependent samples, ``cgns_path`` points to a ``.cgns.series``
    sidecar file that groups multiple CGNS files into a single time sequence.
    For single-timestep samples, it points to the single CGNS file directly.

    Attributes:
        artifact_id: Stable identifier used in API routes. Derived from the
            cache key.
        cgns_path: Path to the file ParaView should open. Either a
            ``.cgns.series`` sidecar (multi-time) or a ``.cgns`` file.
        state_path: Optional ParaView state file (``.pvsm``) providing a
            reasonable default scene.
        metadata_path: Optional JSON metadata file describing the artifact.
        cache_key: Deterministic SHA256 key over the artifact inputs.
        created: ``True`` if the artifact was newly created, ``False`` if it
            was already present in the cache.
    """

    artifact_id: str
    cgns_path: Path
    state_path: Path | None
    metadata_path: Path | None
    cache_key: str
    created: bool


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class DatasetInfo(BaseModel):
    """Summary information about an available dataset.

    ``backend_id`` identifies the loading mode: ``"disk"`` for datasets
    opened with :func:`plaid.storage.init_from_disk` and ``"hub"`` for
    Hugging Face repositories streamed through
    :func:`plaid.storage.init_streaming_from_hub`. Streamed datasets do
    not always expose a total sample count and may need to be navigated
    sequentially through a streaming cursor.
    """

    dataset_id: str
    backend_id: str
    path: str
    has_infos: bool = False
    has_problem_definitions: bool = False


class DatasetDetail(DatasetInfo):
    """Full detail view of a dataset.

    ``splits`` maps each split name to its sample count. The count is
    ``None`` for streaming datasets where the total is unknown.
    """

    splits: dict[str, int | None] = Field(default_factory=dict)
    infos: dict | None = None
    problem_definitions: list[str] = Field(default_factory=list)


class SampleRefDTO(BaseModel):
    """Serializable form of :class:`SampleRef` used by the API."""

    backend_id: str
    dataset_id: str
    split: str | None
    sample_id: str
    encoded: str

    @classmethod
    def from_ref(cls, ref: SampleRef) -> "SampleRefDTO":
        """Build the DTO from a :class:`SampleRef`."""
        return cls(
            backend_id=ref.backend_id,
            dataset_id=ref.dataset_id,
            split=ref.split,
            sample_id=ref.sample_id,
            encoded=ref.encode(),
        )


class SampleSummary(BaseModel):
    """Minimal metadata describing a PLAID sample."""

    ref: SampleRefDTO
    n_times: int
    time_values: list[float]
    bases: list[str]
    zones_by_base: dict[str, list[str]] = Field(default_factory=dict)
    globals: dict[str, str] = Field(default_factory=dict)
    fields_by_base: dict[str, list[str]] = Field(default_factory=dict)


class ValidationResult(BaseModel):
    """Validation outcome for a PLAID sample."""

    ref: SampleRefDTO
    ok: bool
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class ArtifactInfo(BaseModel):
    """Public view of a :class:`ParaviewArtifact`."""

    artifact_id: str
    cache_key: str
    created: bool
    cgns_path: str
    state_path: str | None
    metadata_path: str | None
    is_time_series: bool
    n_files: int


class ViewerUrl(BaseModel):
    """Response model for the ``viewer-url`` endpoint."""

    artifact_id: str
    url: str
