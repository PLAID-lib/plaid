"""This file defines shared pytest fixtures and test configurations."""

import copy

import numpy as np
import pytest
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.types import CGNSTree


@pytest.fixture()
def dataset():
    return Dataset()


@pytest.fixture()
def sample():
    return Sample()


def generate_samples_no_string(nb: int, zone_name: str, base_name: str) -> list[Sample]:
    """Generate a list of Sample objects with randomized scalar and field data."""
    sample_list = []
    for i in range(nb):
        sample = Sample()
        sample.init_base(3, 3, base_name)
        sample.init_zone(np.array([0, 0, 0]), zone_name=zone_name, base_name=base_name)
        sample.add_scalar("test_scalar", float(i))
        sample.add_scalar("test_scalar_2", float(i**2))
        sample.add_global("global_0", 0.5 + np.ones((2, 3)))
        sample.add_global("global_1", 1.5 + i + np.ones((2, 3, 2)))
        sample.add_field(
            name="test_field_same_size",
            field=float(i**4) * np.ones(17),
            zone_name=zone_name,
            base_name=base_name,
        )
        sample.add_field(
            name="test_field_2785",
            field=float(i**5) * np.ones(3 * (i + 1)),
            zone_name=zone_name,
            base_name=base_name,
        )
        sample_list.append(sample)
    return sample_list


def generate_samples(nb: int, zone_name: str, base_name: str) -> list[Sample]:
    sample_list = generate_samples_no_string(nb, zone_name, base_name)
    for i, sample in enumerate(sample_list):
        sample.add_global("global_2", "a_string")
        sample.add_global("global_3", f"another_string_{i}")
    return sample_list


@pytest.fixture()
def nb_samples() -> int:
    """Number of samples to generate for tests."""
    return 4


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
def samples_no_string(nb_samples: int, zone_name: str, base_name: str) -> list[Sample]:
    """A fixture providing a list of generated Sample objects."""
    return generate_samples_no_string(nb_samples, zone_name, base_name)


@pytest.fixture()
def other_samples(nb_samples: int, zone_name: str, base_name: str) -> list[Sample]:
    """An alternate fixture providing a different list of Sample objects."""
    return generate_samples(nb_samples, zone_name, base_name)


@pytest.fixture()
def infos():
    return {
        "legal": {"owner": "PLAID2", "license": "BSD-3"},
        "data_production": {"type": "simulation", "simulator": "Z-set"},
    }


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
def sample_with_tree(tree: CGNSTree) -> Sample:
    """Generate a Sample objects with a tree."""
    sample = Sample()
    sample.features.add_tree(tree)
    return sample


@pytest.fixture()
def samples_with_tree(nb_samples: int, sample_with_tree: Sample) -> list[Sample]:
    """Generate a list of Sample objects with a tree."""
    sample_list = []
    for _ in range(nb_samples):
        sample_list.append(copy.deepcopy(sample_with_tree))
    return sample_list


@pytest.fixture()
def nb_scalars():
    return 5


@pytest.fixture()
def tabular(nb_samples, nb_scalars):
    return np.random.randn(nb_samples, nb_scalars)


@pytest.fixture()
def scalar_names(nb_scalars):
    return [f"test_scalar_{np.random.randint(1e8, 1e9)}" for _ in range(nb_scalars)]


@pytest.fixture
def empty_sample():
    return Sample()


@pytest.fixture()
def empty_dataset():
    return Dataset()


@pytest.fixture()
def dataset_with_samples(dataset, samples, infos):
    dataset.add_samples(samples)
    dataset.set_infos(infos)
    return dataset


@pytest.fixture()
def dataset_with_samples_with_tree(samples_with_tree, infos):
    dataset = Dataset()
    dataset.add_samples(samples_with_tree)
    dataset.set_infos(infos)
    return dataset


@pytest.fixture()
def other_dataset_with_samples(other_samples):
    other_dataset = Dataset()
    other_dataset.add_samples(other_samples)
    return other_dataset


@pytest.fixture()
def heterogeneous_dataset(dataset_with_samples_with_tree):
    dataset = dataset_with_samples_with_tree.copy()
    dataset.add_sample(Sample())
    sample_with_scalar = Sample()
    sample_with_scalar.add_scalar("scalar", 1.0)
    dataset.add_sample(sample_with_scalar)
    sample_with_ts = Sample()
    dataset.add_sample(sample_with_ts)
    return dataset


@pytest.fixture()
def scalar_dataset():
    dataset = Dataset()
    sample = Sample()
    sample.add_scalar("test_scalar", 0.0)
    dataset.add_sample(sample)
    sample2 = Sample()
    for i in range(8):
        sample2.add_scalar(f"scalar_{i}", float(i))
    dataset.add_sample(sample2)
    return dataset
