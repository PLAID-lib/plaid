"""Backend registry for plaid.storage.

This module centralizes backend wiring so reader/writer code can use a single
source of truth for backend capabilities.
"""

from . import cgns, hf_datasets, zarr
from .backend_api import BackendModule

BACKENDS = {
    "cgns": cgns.CgnsBackend,
    "hf_datasets": hf_datasets.HFBackend,
    "zarr": zarr.ZarrBackend,
}


def get_backend(name: str) -> BackendModule:
    """Return the backend object implementing the storage protocol.

    Backends expose static methods, so their classes satisfy the protocol
    directly. Avoid ``type[BackendModule]``: its runtime subclass check is
    unsupported for protocols with data members such as ``name``.

    Args:
        name: Registered storage backend name.

    Returns:
        The backend class exposing the storage methods and backend name.

    Raises:
        ValueError: If the backend name is not registered.
    """
    if name not in BACKENDS:
        raise ValueError(
            f"Error! backend '{name}' not available, option are: {list(BACKENDS.keys())}"
        )
    return BACKENDS[name]


def available_backends() -> list[str]:
    """Return available backend names in stable order."""
    return list(BACKENDS.keys())
