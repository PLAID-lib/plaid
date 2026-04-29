"""Tests for the ParaView artifact service.

These tests only exercise the caching and file-layout logic. The real
``Sample.save_to_dir`` call is replaced by a fake service that writes fixture
CGNS files, so the tests do not depend on pyCGNS or a concrete PLAID sample.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from plaid.viewer.models import SampleRef
from plaid.viewer.services.paraview_artifact_service import (
    ParaviewArtifactService,
    _build_cache_key,
)


class _FakeSample:
    def __init__(self, meshes_dir: Path, n_times: int) -> None:
        self._meshes_dir = meshes_dir
        self.features = type(
            "F", (), {"data": {float(i): None for i in range(n_times)}}
        )()

    def save_to_dir(
        self,
        path: Path,
        overwrite: bool = False,  # noqa: ARG002
        memory_safe: bool = False,  # noqa: ARG002
    ) -> None:
        meshes = Path(path) / "meshes"
        meshes.mkdir(parents=True, exist_ok=True)
        for i in range(len(self.features.data)):
            (meshes / f"mesh_{i:09d}.cgns").write_bytes(b"CGNS_FAKE")


class _FakeDatasetService:
    def __init__(self, n_times: int = 1) -> None:
        self._n_times = n_times

    def load_sample(self, ref: SampleRef):  # noqa: ARG002 - interface match
        return _FakeSample(Path("."), self._n_times)


@pytest.fixture
def ref() -> SampleRef:
    return SampleRef(backend_id="disk", dataset_id="ds", split="train", sample_id="0")


def test_ensure_artifact_single_timestep_creates_single_cgns(
    tmp_path: Path, ref: SampleRef
) -> None:
    service = ParaviewArtifactService(_FakeDatasetService(n_times=1), tmp_path)
    artifact = service.ensure_artifact(ref)
    assert artifact.created is True
    assert artifact.cgns_path.suffix == ".cgns"
    assert artifact.cgns_path.exists()


def test_ensure_artifact_time_series_writes_series_sidecar(
    tmp_path: Path, ref: SampleRef
) -> None:
    service = ParaviewArtifactService(_FakeDatasetService(n_times=3), tmp_path)
    artifact = service.ensure_artifact(ref)
    assert artifact.cgns_path.name.endswith(".cgns.series")
    payload = json.loads(artifact.cgns_path.read_text())
    assert payload["file-series-version"] == "1.0"
    assert len(payload["files"]) == 3
    assert payload["files"][0]["time"] == 0.0
    # Each entry must reference an existing CGNS file relative to the
    # sidecar: CGNS files live in the ``meshes/`` subdirectory, so the
    # ``name`` field has to keep that prefix (regression: previously only
    # the file name was stored, which broke vtkFileSeriesReader).
    sidecar_dir = artifact.cgns_path.parent
    for entry in payload["files"]:
        assert entry["name"].startswith("meshes/"), entry
        assert (sidecar_dir / entry["name"]).is_file()


def test_ensure_artifact_is_idempotent(tmp_path: Path, ref: SampleRef) -> None:
    service = ParaviewArtifactService(_FakeDatasetService(), tmp_path)
    first = service.ensure_artifact(ref)
    assert first.created is True
    second = service.ensure_artifact(ref)
    assert second.created is False
    assert second.artifact_id == first.artifact_id


def test_force_recreates_artifact(tmp_path: Path, ref: SampleRef) -> None:
    service = ParaviewArtifactService(_FakeDatasetService(), tmp_path)
    first = service.ensure_artifact(ref)
    second = service.ensure_artifact(ref, force=True)
    assert second.created is True
    assert second.artifact_id == first.artifact_id  # cache key is deterministic


def test_cache_key_is_deterministic(ref: SampleRef) -> None:
    key_a = _build_cache_key(ref, export_version="1")
    key_b = _build_cache_key(ref, export_version="1")
    assert key_a == key_b
    key_c = _build_cache_key(ref, export_version="2")
    assert key_c != key_a


def test_get_unknown_artifact_raises(tmp_path: Path) -> None:
    service = ParaviewArtifactService(_FakeDatasetService(), tmp_path)
    with pytest.raises(KeyError):
        service.get("unknown")
