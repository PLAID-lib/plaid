"""Tests for runtime-checked storage backend lookup."""

from typing import get_type_hints

import pytest

from plaid.storage.backend_api import BackendModule
from plaid.storage.cgns import CgnsBackend
from plaid.storage.hf_datasets import HFBackend
from plaid.storage.registry import available_backends, get_backend
from plaid.storage.zarr import ZarrBackend


@pytest.mark.parametrize(
    ("name", "expected"),
    [("cgns", CgnsBackend), ("hf_datasets", HFBackend), ("zarr", ZarrBackend)],
)
def test_get_backend_returns_protocol_compatible_object(name, expected):
    """Lookup must retain backend classes without a protocol subclass check."""
    backend = get_backend(name)

    assert backend is expected
    assert backend.name == name
    assert isinstance(backend, BackendModule)


def test_get_backend_annotation_uses_object_protocol():
    """Data protocols require instance checks, not type[Protocol] checks."""
    assert get_type_hints(get_backend)["return"] is BackendModule


def test_get_backend_rejects_unknown_name():
    """Unknown backends must retain the existing descriptive error."""
    with pytest.raises(ValueError, match="backend 'missing' not available"):
        get_backend("missing")


def test_available_backends_order():
    """Backend names must retain their public lookup order."""
    assert available_backends() == ["cgns", "hf_datasets", "zarr"]
