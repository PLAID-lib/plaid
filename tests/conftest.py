"""This file defines shared pytest fixtures and test configurations."""

import numpy as np
import pytest

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample


@pytest.fixture()
def dataset():
    return Dataset()


def generate_samples(nb: int, zone_name: str, base_name: str) -> list[Sample]:
    """Generate a list of Sample objects with randomized scalar and field data."""
    sample_list = []
    for _ in range(nb):
        sample = Sample()
        sample.init_base(3, 3, base_name)
        sample.init_zone(np.array([0, 0, 0]), zone_name=zone_name, base_name=base_name)
        sample.add_scalar("test_scalar", np.random.randn())
        sample.add_field(
            "test_field_same_size", np.random.randn(17), zone_name, base_name
        )
        sample.add_field(
            f"test_field_{np.random.randint(1e8, 1e9)}",
            np.random.randn(np.random.randint(10, 20)),
            zone_name,
            base_name,
        )
        sample_list.append(sample)
    return sample_list


@pytest.fixture()
def nb_samples() -> int:
    """Number of samples to generate for tests."""
    return 11


@pytest.fixture()
def base_name() -> str:
    """Base name for initializing samples' base hierarchy."""
    return "TestBaseName"


@pytest.fixture()
def zone_name() -> str:
    """Zone name for initializing samples' zone hierarchy."""
    return "TestZoneName"


@pytest.fixture()
def samples(nb_samples: int, zone_name: str, base_name: str) -> list[Sample]:
    """A fixture providing a list of generated Sample objects."""
    return generate_samples(nb_samples, zone_name, base_name)


@pytest.fixture()
def other_samples(nb_samples: int, zone_name: str, base_name: str) -> list[Sample]:
    """An alternate fixture providing a different list of Sample objects."""
    return generate_samples(nb_samples, zone_name, base_name)


@pytest.fixture()
def infos() -> dict:
    """Legal metadata used in test datasets."""
    return {"legal": {"owner": "PLAID2", "license": "BSD-3"}}
