"""Tests for viewer data models."""

from __future__ import annotations

import pytest

from plaid.viewer.models import SampleRef, SampleRefDTO


def test_sample_ref_roundtrip_with_split() -> None:
    ref = SampleRef(backend_id="disk", dataset_id="ds", split="train", sample_id="0")
    assert SampleRef.decode(ref.encode()) == ref


def test_sample_ref_roundtrip_without_split() -> None:
    ref = SampleRef(backend_id="disk", dataset_id="ds", split=None, sample_id="42")
    encoded = ref.encode()
    assert "_" in encoded  # sentinel for missing split
    assert SampleRef.decode(encoded) == ref


def test_sample_ref_decode_invalid() -> None:
    with pytest.raises(ValueError):
        SampleRef.decode("too:few:parts")


def test_sample_ref_dto_round_trip() -> None:
    ref = SampleRef(backend_id="b", dataset_id="d", split=None, sample_id="s")
    dto = SampleRefDTO.from_ref(ref)
    assert dto.encoded == ref.encode()
    assert dto.split is None
