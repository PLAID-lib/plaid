#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Backend registry for plaid.storage.

This module centralizes backend wiring so reader/writer code can use a single
source of truth for backend capabilities.
"""

from . import cgns, hf_datasets, zarr, in_memory
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
