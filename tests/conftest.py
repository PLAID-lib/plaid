"""This file defines shared pytest fixtures and test configurations."""

import numpy as np
import pytest

from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import MeshCreationTools as MCT

from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import MeshCreationTools as MCT

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
def infos():
    return {"legal": {"owner": "PLAID2", "license": "BSD-3"}, "data_production": {"type":"simulation", "simulator":"Z-set"}}




@pytest.fixture()
def nodes():
    return np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.5, 1.5],
        ]
    )


@pytest.fixture()
def triangles():
    return np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [2, 4, 3],
        ]
    )


@pytest.fixture()
def nodal_tags():
    return np.array(
        [
            0,
            1,
        ]
    )


@pytest.fixture()
def vertex_field():
    return np.random.randn(5)


@pytest.fixture()
def cell_center_field():
    return np.random.randn(3)


@pytest.fixture()
def tree(nodes, triangles, vertex_field, cell_center_field, nodal_tags):
    Mesh = MCT.CreateMeshOfTriangles(nodes, triangles)
    Mesh.GetNodalTag("tag").AddToTag(nodal_tags)
    Mesh.nodeFields["test_node_field_1"] = vertex_field
    Mesh.nodeFields["big_node_field"] = np.random.randn(50)
    Mesh.elemFields["test_elem_field_1"] = cell_center_field
    tree = MeshToCGNS(Mesh)
    return tree


@pytest.fixture()
def sample_with_tree(sample, tree):
    sample.add_tree(tree)
    return sample


@pytest.fixture()
def sample():
    return Sample()
