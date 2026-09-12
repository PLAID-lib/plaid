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


def get_backend(name: str) -> type[BackendModule]:
    """Return backend module.

    Args:
        name (str): The backend name; see :func:`available_backends` for the
            available options.

    Returns:
        type[BackendModule]: The backend module class.

    Raises:
        ValueError: If ``name`` does not match any available backend.
    """
    if name not in BACKENDS:
        raise ValueError(
            f"Error! backend '{name}' not available, option are: {list(BACKENDS.keys())}"
        )
    return BACKENDS[name]


def available_backends() -> list[str]:
    """Return available backend names in stable order."""
    return list(BACKENDS.keys())
