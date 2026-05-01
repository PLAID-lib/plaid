"""Backend registry for plaid.storage.

This module centralizes backend wiring so reader/writer code can use a single
source of truth for backend capabilities.
"""

from . import cgns, hf_datasets, in_memory, zarr
from .backend_api import BackendModule

BACKENDS = {
    "in_memory": in_memory.InMemoryBackend,
    "cgns": cgns.CgnsBackend,
    "hf_datasets": hf_datasets.HFBackend,
    "zarr": zarr.ZarrBackend,
}


def get_backend(name: str) -> type[BackendModule]:
    if name not in BACKENDS:
        raise ValueError(
            f"Error! backend '{name}' not available, option are: {list(BACKENDS.keys())}"
        )
    return BACKENDS[name]


def available_backends() -> list[str]:
    """Return available backend names in stable order."""
    return list(BACKENDS.keys())
