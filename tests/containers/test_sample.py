# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

from pathlib import Path

import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils as CGU
import numpy as np
import pytest
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.MeshTools import MeshCreationTools as MCT

from plaid.containers.sample import Sample
from plaid.containers.utils import (
    _check_names,
    _read_index,
    _read_index_array,
    _read_index_range,
)
from plaid.types.feature_types import FeatureIdentifier
from plaid.utils.cgns_helper import show_cgns_tree

# %% Fixtures


@pytest.fixture()
def topological_dim():
    return 2


@pytest.fixture()
def physical_dim():
    return 3


@pytest.fixture()
def zone_shape():
    return np.array([5, 3, 0])


@pytest.fixture()
def other_sample():
    return Sample()


@pytest.fixture()
def sample_with_scalar(sample):
    sample.add_scalar("test_scalar_1", np.random.randn())
    return sample


@pytest.fixture()
def nodes3d():
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.5, 1.0],
        ]
    )


@pytest.fixture()
def tree3d(nodes3d, triangles, vertex_field, cell_center_field):
    Mesh = MCT.CreateMeshOfTriangles(nodes3d, triangles)
    Mesh.nodeFields["test_node_field_1"] = vertex_field
    Mesh.nodeFields["big_node_field"] = np.random.randn(50)
    Mesh.elemFields["test_elem_field_1"] = cell_center_field
    tree = MeshToCGNS(Mesh)
    return tree


@pytest.fixture()
def sample_with_tree3d(sample, tree3d):
    sample.features.add_tree(tree3d)
    return sample


@pytest.fixture()
def sample_with_tree_and_scalar(
    sample_with_tree: Sample,
):
    sample_with_tree.add_scalar("r", np.random.randn())
    sample_with_tree.add_scalar("test_scalar_1", np.random.randn())
    return sample_with_tree


@pytest.fixture()
def full_sample(sample_with_tree_and_scalar: Sample, tree3d):
    sample_with_tree_and_scalar.add_scalar("r", np.random.randn())
    sample_with_tree_and_scalar.add_scalar("test_scalar_1", np.random.randn())
    sample_with_tree_and_scalar.add_field(
        name="test_field_1", field=np.random.randn(5, 3), location="CellCenter"
    )
    sample_with_tree_and_scalar.init_zone(
        zone_shape=np.array([5, 3]), zone_name="test_field_1"
    )
    sample_with_tree_and_scalar.init_base(
        topological_dim=2, physical_dim=3, base_name="test_base_1"
    )
    sample_with_tree_and_scalar.features.init_tree(time=1.0)
    sample_with_tree_and_scalar.features.add_tree(tree=tree3d)
    return sample_with_tree_and_scalar


# %% Test


def test_check_names():
    _check_names("test name")
    _check_names(["test name", "test_name_2"])
    with pytest.raises(ValueError):
        _check_names("test/name")
    with pytest.raises(ValueError):
        _check_names(["test/name"])
    with pytest.raises(ValueError):
        _check_names([r"test\/name"])


def test_read_index(tree, physical_dim):
    _read_index(tree, physical_dim)


def test_read_index_array(tree):
    _read_index_array(tree)


def test_read_index_range(tree, physical_dim):
    _read_index_range(tree, physical_dim)


@pytest.fixture()
def current_directory() -> Path:
    return Path(__file__).absolute().parent


# %% Tests


class Test_Sample:
    # -------------------------------------------------------------------------#
    def test___init__(self, current_directory):
        sample_path_1 = current_directory / "dataset" / "samples" / "sample_000000000"
        sample_path_2 = current_directory / "dataset" / "samples" / "sample_000000001"
        sample_path_3 = current_directory / "dataset" / "samples" / "sample_000000002"
        sample_already_filled_1 = Sample(path=sample_path_1)
        sample_already_filled_2 = Sample(path=sample_path_2)
        sample_already_filled_3 = Sample(path=sample_path_3)
        assert sample_already_filled_1.features
        assert sample_already_filled_2.features
        assert sample_already_filled_3.features

    def test__init__unknown_directory(self, current_directory):
        sample_path = current_directory / "dataset" / "samples" / "sample_000000298"
        with pytest.raises(FileNotFoundError):
            Sample(path=sample_path)

    def test__init__file_provided(self, current_directory):
        sample_path = current_directory / "dataset" / "samples" / "sample_000067392"
        with pytest.raises(FileExistsError):
            Sample(path=sample_path)

    def test__init__path(self, current_directory):
        sample_path = current_directory / "dataset" / "samples" / "sample_000000000"
        Sample(path=sample_path)

    # def test__init__directory_path(self, current_directory):
    #     sample_path = current_directory / "dataset" / "samples" / "sample_000000000"
    #     Sample(directory_path=sample_path)

    # def test__init__both_path_and_directory_path(self, current_directory):
    #     sample_path = current_directory / "dataset" / "samples" / "sample_000000000"
    #     with pytest.raises(ValueError):
    #         Sample(path=sample_path, directory_path=sample_path)

    def test_copy(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.copy()

    # -------------------------------------------------------------------------#
    def test_set_default_base(self, sample: Sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, time=0.5)

        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}", 0.5)
        # check dims getters
        assert sample.features.get_topological_dim() == topological_dim
        assert sample.features.get_physical_dim() == physical_dim
        assert (
            sample.features.get_base_assignment()
            == f"Base_{topological_dim}_{physical_dim}"
        )
        assert sample.features.get_time_assignment() == 0.5
        assert sample.features.get_base_assignment("test") == "test"

        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}")  # already set
        sample.set_default_base(None)  # will not assign to None
        assert (
            sample.features.get_base_assignment()
            == f"Base_{topological_dim}_{physical_dim}"
        )
        with pytest.raises(ValueError):
            sample.set_default_base("Unknown base name")

    def test_set_default_zone_with_default_base(
        self,
        sample: Sample,
        topological_dim,
        physical_dim,
        base_name,
        zone_name,
        zone_shape,
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=0.5)
        sample.set_default_base(base_name)
        # No zone provided
        assert sample.features.get_zone() is None

        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)
        # Look for the only zone in the default base
        assert sample.features.get_zone() is not None

        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)
        # There is more than one zone in this base
        with pytest.raises(KeyError):
            sample.features.get_zone()

    def test_set_default_zone(
        self,
        sample: Sample,
        topological_dim,
        physical_dim,
        base_name,
        zone_name,
        zone_shape,
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=0.5)
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)

        sample.set_default_zone_base(zone_name, base_name, 0.5)
        # check dims getters
        assert sample.features.get_topological_dim() == topological_dim
        assert sample.features.get_physical_dim() == physical_dim
        assert sample.features.get_base_assignment() == base_name
        assert sample.features.get_time_assignment() == 0.5

        sample.set_default_base(base_name)  # already set
        sample.set_default_base(None)  # will not assign to None
        assert sample.features.get_base_assignment() == base_name
        with pytest.raises(ValueError):
            sample.set_default_base("Unknown base name")

        assert sample.features.get_zone_assignment() == zone_name
        assert sample.features.get_time_assignment() == 0.5

        assert sample.features.get_zone() is not None
        sample.set_default_zone_base(zone_name, base_name)
        sample.set_default_zone_base(None, base_name)  # will not assign to None
        assert sample.features.get_zone_assignment() == zone_name
        with pytest.raises(ValueError):
            sample.set_default_zone_base("Unknown zone name", base_name)

    def test_set_default_time(self, sample: Sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, time=0.5)
        sample.init_base(topological_dim, physical_dim, "OK_name", time=1.5)

        assert sample.features.get_time_assignment() == 0.5
        sample.set_default_time(1.5)
        assert sample.features.get_time_assignment() == 1.5, "here"

        sample.set_default_time(1.5)  # already set
        sample.set_default_time(None)  # will not assign to None
        assert sample.features.get_time_assignment() == 1.5
        with pytest.raises(ValueError):
            sample.set_default_time(2.5)

    # -------------------------------------------------------------------------#

    def test_show_tree(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.show_tree()

    def test_init_tree(self, sample: Sample):
        sample.features.init_tree()
        sample.features.init_tree(0.5)

    def test_get_mesh_empty(self, sample: Sample):
        sample.get_mesh()
        sample.features.get_mesh()

    def test_get_mesh(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.get_mesh()

    def test_set_meshes_empty(self, sample, tree):
        sample.features.set_meshes({0.0: tree})

    def test_set_meshes(self, sample_with_tree: Sample, tree):
        with pytest.raises(KeyError):
            sample_with_tree.features.set_meshes({0.0: tree})

    def test_add_tree_empty(self, sample_with_tree: Sample):
        with pytest.raises(ValueError):
            sample_with_tree.features.add_tree([])

    def test_add_tree(self, sample: Sample, tree):
        sample.features.add_tree(tree)
        sample.features.add_tree(tree)
        sample.features.add_tree(tree, time=0.2)

    def test_del_tree(self, sample, tree):
        sample.features.add_tree(tree)
        sample.features.add_tree(tree, time=0.2)

        assert isinstance(sample.features.del_tree(0.2), list)
        assert list(sample.features.data.keys()) == [0.0]

        assert isinstance(sample.features.del_tree(0.0), list)
        assert list(sample.features.data.keys()) == []

    def test_on_error_del_tree(self, sample, tree):
        with pytest.raises(KeyError):
            sample.features.del_tree(0.0)

        sample.features.add_tree(tree)
        sample.features.add_tree(tree, time=0.2)
        with pytest.raises(KeyError):
            sample.features.del_tree(0.7)

    # -------------------------------------------------------------------------#
    def test_init_base(self, sample: Sample, base_name, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, base_name)
        # check dims getters
        assert sample.features.get_topological_dim(base_name) == topological_dim
        assert sample.features.get_physical_dim(base_name) == physical_dim

    def test_del_base_existing_base(
        self, sample: Sample, base_name, topological_dim, physical_dim
    ):
        second_base_name = base_name + "_2"
        sample.init_base(topological_dim, physical_dim, base_name)
        sample.init_base(topological_dim, physical_dim, second_base_name)

        # Delete first base
        updated_cgns_tree = sample.features.del_base(base_name, 0.0)
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Testing the resulting tree
        new_sample = Sample()
        new_sample.features.add_tree(updated_cgns_tree, 0.1)
        assert new_sample.features.get_topological_dim() == topological_dim
        assert new_sample.features.get_physical_dim() == physical_dim
        assert new_sample.features.get_base_names() == [second_base_name]

        # Add 2 bases and delete one base at time 0.2
        sample.init_base(topological_dim, physical_dim, "tree", 0.2)
        sample.init_base(topological_dim, physical_dim, base_name, 0.2)
        updated_cgns_tree = sample.features.del_base("tree", 0.2)
        assert sample.features.get_base("tree", 0.2) is None
        assert sample.features.get_base(base_name, 0.2) is not None
        assert sample.features.get_base(second_base_name) is not None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Testing the resulting from time 0.2
        new_sample = Sample()
        new_sample.features.add_tree(updated_cgns_tree)
        assert new_sample.features.get_topological_dim() == topological_dim
        assert new_sample.features.get_physical_dim() == physical_dim
        assert new_sample.features.get_base_names() == [base_name]

        # Deleting the last base at time 0.0
        updated_cgns_tree = sample.features.del_base(second_base_name, 0.0)
        assert sample.features.get_base(second_base_name) is None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Deleting the last base at time 0.2
        updated_cgns_tree = sample.features.del_base(base_name, 0.2)
        assert sample.features.get_base(base_name) is None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

    def test_del_base_nonexistent_base_nonexistent_time(
        self, sample: Sample, base_name, topological_dim, physical_dim
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=1.0)
        with pytest.raises(KeyError):
            sample.features.del_base(base_name, time=2.0)
        with pytest.raises(KeyError):
            sample.features.del_base("unknown", time=1.0)

    def test_del_base_no_cgns_tree(self, sample):
        with pytest.raises(KeyError):
            sample.features.del_base("unknwon", 0.0)

    def test_init_base_no_base_name(
        self, sample: Sample, topological_dim, physical_dim
    ):
        sample.init_base(topological_dim, physical_dim)

        # check dims getters
        assert (
            sample.features.get_topological_dim(
                f"Base_{topological_dim}_{physical_dim}"
            )
            == topological_dim
        )
        assert (
            sample.features.get_physical_dim(f"Base_{topological_dim}_{physical_dim}")
            == physical_dim
        )

        # check setting default base
        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}")
        assert sample.features.get_topological_dim() == topological_dim
        assert sample.features.get_physical_dim() == physical_dim

    def test_get_base_names(self, sample: Sample):
        assert sample.features.get_base_names() == []
        sample.init_base(3, 3, "base_name_1")
        sample.init_base(3, 3, "base_name_2")
        assert sample.features.get_base_names() == ["base_name_1", "base_name_2"]
        assert sample.features.get_base_names(full_path=True) == [
            "/base_name_1",
            "/base_name_2",
        ]
        # check dims getters
        assert sample.features.get_topological_dim("base_name_1") == 3
        assert sample.features.get_physical_dim("base_name_1") == 3
        assert sample.features.get_topological_dim("base_name_2") == 3
        assert sample.features.get_physical_dim("base_name_2") == 3

    def test_get_base(self, sample: Sample, base_name):
        sample.features.init_tree()
        assert sample.features.get_base() is None
        sample.init_base(3, 3, base_name)
        assert sample.features.get_base(base_name) is not None
        assert sample.features.get_base() is not None
        sample.init_base(3, 3, "other_base_name")
        assert sample.features.get_base(base_name) is not None
        assert sample.features.get_base(time=1.0) is None
        with pytest.raises(KeyError):
            sample.features.get_base()
        # check dims getters
        assert sample.features.get_topological_dim(base_name) == 3
        assert sample.features.get_physical_dim(base_name) == 3
        assert sample.features.get_topological_dim("other_base_name") == 3
        assert sample.features.get_physical_dim("other_base_name") == 3

    # -------------------------------------------------------------------------#
    def test_init_zone(self, sample: Sample, base_name, zone_name, zone_shape):
        with pytest.raises(KeyError):
            sample.init_zone(zone_shape, zone_name=zone_name, base_name=base_name)
        sample.init_base(3, 3, base_name)
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)
        sample.init_zone(zone_shape, CGK.Unstructured_s, zone_name, base_name=base_name)
        # check dims getters
        assert sample.features.get_topological_dim(base_name) == 3
        assert sample.features.get_physical_dim(base_name) == 3

    def test_init_zone_defaults_names(self, sample: Sample, zone_shape):
        sample.init_base(3, 3)
        sample.init_zone(zone_shape)

    def test_del_zone_existing_zone(
        self, sample: Sample, base_name, zone_name, zone_shape
    ):
        topological_dim, physical_dim = 3, 3
        sample.init_base(topological_dim, physical_dim, base_name)

        second_zone_name = zone_name + "_2"
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)
        sample.init_zone(
            zone_shape, CGK.Unstructured_s, second_zone_name, base_name=base_name
        )

        # Delete first zone
        updated_cgns_tree = sample.features.del_zone(zone_name, base_name, 0.0)
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Testing the resulting tree
        new_sample = Sample()
        new_sample.features.add_tree(updated_cgns_tree, 0.1)
        assert new_sample.features.get_zone_names() == [second_zone_name]

        # Add 2 zones and delete one zone at time 0.2
        sample.init_base(topological_dim, physical_dim, base_name, 0.2)
        sample.init_zone(
            zone_shape, CGK.Structured_s, zone_name, base_name=base_name, time=0.2
        )
        sample.init_zone(
            zone_shape, CGK.Unstructured_s, "test", base_name=base_name, time=0.2
        )

        updated_cgns_tree = sample.features.del_zone("test", base_name, 0.2)
        assert sample.features.get_zone("tree", base_name, 0.2) is None
        assert sample.features.get_zone(zone_name, base_name, 0.2) is not None
        assert sample.features.get_zone(second_zone_name, base_name) is not None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Testing the resulting from time 0.2
        new_sample = Sample()
        new_sample.features.add_tree(updated_cgns_tree)
        assert new_sample.features.get_zone_names(base_name) == [zone_name]

        # Deleting the last zone at time 0.0
        updated_cgns_tree = sample.features.del_zone(second_zone_name, base_name, 0.0)
        assert sample.features.get_zone(second_zone_name, base_name) is None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Deleting the last zone at time 0.2
        updated_cgns_tree = sample.features.del_zone(zone_name, base_name, 0.2)
        assert sample.features.get_zone(zone_name, base_name) is None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

    def test_del_zone_nonexistent_zone_nonexistent_time(
        self, sample: Sample, base_name, zone_shape, topological_dim, physical_dim
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=1.0)
        zone_name = "test123"
        sample.init_zone(
            zone_shape, CGK.Structured_s, zone_name, base_name=base_name, time=1.0
        )
        with pytest.raises(KeyError):
            sample.features.del_zone(zone_name, base_name, 2.0)
        with pytest.raises(KeyError):
            sample.features.del_zone("unknown", base_name, 1.0)

    def test_del_zone_no_cgns_tree(self, sample: Sample):
        sample.init_base(2, 3, "only_base")
        with pytest.raises(KeyError):
            sample.features.del_zone("unknwon", "only_base", 0.0)

    def test_has_zone(self, sample, base_name, zone_name):
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        sample.show_tree()
        assert sample.features.has_zone(zone_name, base_name)
        assert not sample.features.has_zone("not_present_zone_name", base_name)
        assert not sample.features.has_zone(zone_name, "not_present_base_name")
        assert not sample.features.has_zone(
            "not_present_zone_name", "not_present_base_name"
        )

    def test_get_zone_names(self, sample: Sample, base_name):
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name="zone_name_1",
            base_name=base_name,
        )
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name="zone_name_2",
            base_name=base_name,
        )
        assert sample.features.get_zone_names(base_name) == [
            "zone_name_1",
            "zone_name_2",
        ]
        assert sorted(sample.features.get_zone_names(base_name, unique=True)) == sorted(
            ["zone_name_1", "zone_name_2"]
        )
        assert sample.features.get_zone_names(base_name, full_path=True) == [
            f"{base_name}/zone_name_1",
            f"{base_name}/zone_name_2",
        ]

    def test_get_zone_type(self, sample: Sample, zone_name, base_name):
        with pytest.raises(KeyError):
            sample.features.get_zone_type(zone_name, base_name)
        sample.features.init_tree()
        with pytest.raises(KeyError):
            sample.features.get_zone_type(zone_name, base_name)
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.features.get_zone_type(zone_name, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        assert sample.features.get_zone_type(zone_name, base_name) == CGK.Unstructured_s

    def test_get_zone(self, sample: Sample, zone_name, base_name):
        assert sample.features.get_zone(zone_name, base_name) is None
        sample.init_base(3, 3, base_name)
        assert sample.features.get_zone(zone_name, base_name) is None
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        assert sample.features.get_zone() is not None
        assert sample.features.get_zone(zone_name, base_name) is not None
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name="other_zone_name",
            base_name=base_name,
        )
        assert sample.features.get_zone(zone_name, base_name) is not None
        with pytest.raises(KeyError):
            assert sample.features.get_zone() is not None

    # -------------------------------------------------------------------------#
    def test_get_scalar_names(self, sample: Sample):
        assert sample.get_scalar_names() == []

    def test_get_scalar_empty(self, sample):
        assert sample.get_scalar("missing_scalar_name") is None

    def test_get_scalar(self, sample_with_scalar):
        assert sample_with_scalar.get_scalar("missing_scalar_name") is None
        assert sample_with_scalar.get_scalar("test_scalar_1") is not None

    def test_scalars_add_empty(self, sample_with_scalar):
        assert isinstance(sample_with_scalar.get_scalar("test_scalar_1"), float)

    def test_scalars_add(self, sample_with_scalar):
        sample_with_scalar.add_scalar("test_scalar_2", np.random.randn())

    def test_del_scalar_unknown_scalar(self, sample_with_scalar):
        with pytest.raises(KeyError):
            sample_with_scalar.del_scalar("non_existent_scalar")

    def test_del_scalar_no_scalar(self):
        sample = Sample()
        with pytest.raises(KeyError):
            sample.del_scalar("non_existent_scalar")

    def test_del_scalar(self, sample_with_scalar):
        assert len(sample_with_scalar.get_scalar_names()) == 1

        sample_with_scalar.add_scalar("test_scalar_2", np.random.randn(5))
        assert len(sample_with_scalar.get_scalar_names()) == 2

        scalar = sample_with_scalar.del_scalar("test_scalar_1")
        assert len(sample_with_scalar.get_scalar_names()) == 1
        assert scalar is not None
        assert isinstance(scalar, float)

        scalar = sample_with_scalar.del_scalar("test_scalar_2")
        assert len(sample_with_scalar.get_scalar_names()) == 0
        assert scalar is not None
        assert isinstance(scalar, np.ndarray)

    def test_add_feature(self, sample_with_scalar):
        sample_with_scalar.add_feature(
            feature_identifier=FeatureIdentifier(
                {"type": "scalar", "name": "test_scalar_2"}
            ),
            feature=[3.1415],
        )

    def test_del_feature(self, sample_with_scalar: Sample, sample_with_tree3d: Sample):
        sample_with_scalar.del_feature(
            feature_identifier=FeatureIdentifier(
                {"type": "scalar", "name": "test_scalar_1"}
            ),
        )
        assert sample_with_scalar.get_all_features_identifiers_by_type("scalar") == []
        sample_with_tree3d.del_feature(
            feature_identifier=FeatureIdentifier(
                {"type": "field", "name": "test_node_field_1"}
            ),
        )
        sample_with_tree3d.del_feature(
            feature_identifier=FeatureIdentifier(
                {"type": "field", "name": "big_node_field"}
            ),
        )
        sample_with_tree3d.del_feature(
            feature_identifier=FeatureIdentifier(
                {"type": "field", "name": "test_elem_field_1", "location": "CellCenter"}
            ),
        )
        sample_with_tree3d.del_feature(
            feature_identifier=FeatureIdentifier(
                {"type": "field", "name": "OriginalIds"}
            ),
        )
        sample_with_tree3d.del_feature(
            feature_identifier=FeatureIdentifier(
                {"type": "field", "name": "OriginalIds", "location": "CellCenter"}
            ),
        )
        sample_with_tree3d.del_feature(
            feature_identifier=FeatureIdentifier(
                {"type": "field", "name": "OriginalIds", "location": "FaceCenter"}
            ),
        )
        with pytest.raises(NotImplementedError):
            sample_with_tree3d.del_feature(
                feature_identifier=FeatureIdentifier({"type": "nodes"}),
            )

    # -------------------------------------------------------------------------#
    def test_get_nodal_tags_empty(self, sample):
        assert sample.features.get_nodal_tags() == {}

    def test_get_nodal_tags(self, sample_with_tree, nodal_tags):
        assert np.all(sample_with_tree.features.get_nodal_tags()["tag"] == nodal_tags)

    # -------------------------------------------------------------------------#
    def test_get_nodes_empty(self, sample):
        assert sample.get_nodes() is None

    def test_get_nodes(self, sample_with_tree, nodes):
        assert np.all(sample_with_tree.get_nodes() == nodes)

    def test_get_nodes3d(self, sample_with_tree3d, nodes3d):
        assert np.all(sample_with_tree3d.get_nodes() == nodes3d)

    def test_set_nodes(self, sample, nodes, zone_name, base_name):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.set_nodes(nodes, zone_name, base_name)
        sample.init_zone(
            np.array([len(nodes), 0, 0]), zone_name=zone_name, base_name=base_name
        )
        sample.set_nodes(nodes, zone_name, base_name)

    # -------------------------------------------------------------------------#
    def test_get_elements_empty(self, sample: Sample):
        assert sample.features.get_elements() == {}

    def test_get_elements(self, sample_with_tree: Sample, triangles):
        assert list(sample_with_tree.features.get_elements().keys()) == ["TRI_3"]
        print(f"{triangles=}")
        print(f"{sample_with_tree.features.get_elements()=}")
        assert np.all(sample_with_tree.features.get_elements()["TRI_3"] == triangles)

    # -------------------------------------------------------------------------#
    def test_get_field_names(self, sample: Sample):
        assert sample.get_field_names() == []
        assert sample.get_field_names(location="CellCenter") == []

    def test_get_field_names_full(self, full_sample):
        full_sample.get_field_names()

    def test_get_field_names_several_bases(self):
        sample = Sample()
        sample.init_tree(time=-0.1)
        sample.init_tree(time=1.0)
        sample.init_base(
            topological_dim=1, physical_dim=2, base_name="Base_1_2", time=-0.1
        )
        sample.init_base(
            topological_dim=2, physical_dim=2, base_name="Base_2_2", time=-0.1
        )
        sample.init_base(
            topological_dim=1, physical_dim=3, base_name="Base_1_3", time=1.0
        )
        sample.init_base(
            topological_dim=3, physical_dim=3, base_name="Base_3_3", time=1.0
        )
        sample.init_zone(
            zone_shape=np.array([0, 0, 0]),
            zone_name="Zone_1",
            base_name="Base_1_2",
            time=-0.1,
        )
        sample.init_zone(
            zone_shape=np.array([0, 0, 0]),
            zone_name="Zone_2",
            base_name="Base_1_2",
            time=-0.1,
        )
        sample.init_zone(
            zone_shape=np.array([0, 0, 0]),
            zone_name="Zone_1",
            base_name="Base_2_2",
            time=-0.1,
        )
        sample.init_zone(
            zone_shape=np.array([0, 0, 0]),
            zone_name="Zone_2",
            base_name="Base_2_2",
            time=-0.1,
        )
        sample.init_zone(
            zone_shape=np.array([0, 0, 0]),
            zone_name="Zone_1",
            base_name="Base_1_3",
            time=1.0,
        )
        sample.init_zone(
            zone_shape=np.array([0, 0, 0]),
            zone_name="Zone_2",
            base_name="Base_1_3",
            time=1.0,
        )
        sample.init_zone(
            zone_shape=np.array([0, 0, 0]),
            zone_name="Zone_1",
            base_name="Base_3_3",
            time=1.0,
        )
        sample.init_zone(
            zone_shape=np.array([0, 0, 0]),
            zone_name="Zone_2",
            base_name="Base_3_3",
            time=1.0,
        )
        sample.add_field(
            name="test_vertex_Zone_1_Base_1_2_t_m0.1",
            field=np.random.randn(10),
            location="Vertex",
            zone_name="Zone_1",
            base_name="Base_1_2",
            time=-0.1,
        )
        sample.add_field(
            name="test_cell_Zone_1_Base_1_2_t_m0.1",
            field=np.random.randn(10),
            location="CellCenter",
            zone_name="Zone_1",
            base_name="Base_1_2",
            time=-0.1,
        )
        sample.add_field(
            name="test_vertex_Zone_2_Base_1_2_t_m0.1",
            field=np.random.randn(10),
            location="Vertex",
            zone_name="Zone_2",
            base_name="Base_1_2",
            time=-0.1,
        )
        sample.add_field(
            name="test_cell_Zone_2_Base_1_2_t_m0.1",
            field=np.random.randn(10),
            location="CellCenter",
            zone_name="Zone_2",
            base_name="Base_1_2",
            time=-0.1,
        )
        sample.add_field(
            name="test_vertex_Zone_1_Base_2_2_t_m0.1",
            field=np.random.randn(10),
            location="Vertex",
            zone_name="Zone_1",
            base_name="Base_2_2",
            time=-0.1,
        )
        sample.add_field(
            name="test_cell_Zone_1_Base_2_2_t_m0.1",
            field=np.random.randn(10),
            location="CellCenter",
            zone_name="Zone_1",
            base_name="Base_2_2",
            time=-0.1,
        )
        sample.add_field(
            name="test_vertex_Zone_2_Base_2_2_t_m0.1",
            field=np.random.randn(10),
            location="Vertex",
            zone_name="Zone_2",
            base_name="Base_2_2",
            time=-0.1,
        )
        sample.add_field(
            name="test_cell_Zone_2_Base_2_2_t_m0.1",
            field=np.random.randn(10),
            location="CellCenter",
            zone_name="Zone_2",
            base_name="Base_2_2",
            time=-0.1,
        )
        sample.add_field(
            name="test_vertex_Zone_1_Base_1_3_t_1.0",
            field=np.random.randn(10),
            location="Vertex",
            zone_name="Zone_1",
            base_name="Base_1_3",
            time=1.0,
        )
        sample.add_field(
            name="test_cell_Zone_1_Base_1_3_t_1.0",
            field=np.random.randn(10),
            location="CellCenter",
            zone_name="Zone_1",
            base_name="Base_1_3",
            time=1.0,
        )
        sample.add_field(
            name="test_vertex_Zone_2_Base_1_3_t_1.0",
            field=np.random.randn(10),
            location="Vertex",
            zone_name="Zone_2",
            base_name="Base_1_3",
            time=1.0,
        )
        sample.add_field(
            name="test_cell_Zone_2_Base_1_3_t_1.0",
            field=np.random.randn(10),
            location="CellCenter",
            zone_name="Zone_2",
            base_name="Base_1_3",
            time=1.0,
        )
        sample.add_field(
            name="test_vertex_Zone_1_Base_3_3_t_1.0",
            field=np.random.randn(10),
            location="Vertex",
            zone_name="Zone_1",
            base_name="Base_3_3",
            time=1.0,
        )
        sample.add_field(
            name="test_cell_Zone_1_Base_3_3_t_1.0",
            field=np.random.randn(10),
            location="CellCenter",
            zone_name="Zone_1",
            base_name="Base_3_3",
            time=1.0,
        )
        sample.add_field(
            name="test_vertex_Zone_2_Base_3_3_t_1.0",
            field=np.random.randn(10),
            location="Vertex",
            zone_name="Zone_2",
            base_name="Base_3_3",
            time=1.0,
        )
        sample.add_field(
            name="test_cell_Zone_2_Base_3_3_t_1.0",
            field=np.random.randn(10),
            location="CellCenter",
            zone_name="Zone_2",
            base_name="Base_3_3",
            time=1.0,
        )
        expected_field_names = [
            "test_vertex_Zone_1_Base_1_2_t_m0.1",
            "test_cell_Zone_1_Base_1_2_t_m0.1",
            "test_vertex_Zone_2_Base_1_2_t_m0.1",
            "test_cell_Zone_2_Base_1_2_t_m0.1",
            "test_vertex_Zone_1_Base_2_2_t_m0.1",
            "test_cell_Zone_1_Base_2_2_t_m0.1",
            "test_vertex_Zone_2_Base_2_2_t_m0.1",
            "test_cell_Zone_2_Base_2_2_t_m0.1",
            "test_vertex_Zone_1_Base_1_3_t_1.0",
            "test_cell_Zone_1_Base_1_3_t_1.0",
            "test_vertex_Zone_2_Base_1_3_t_1.0",
            "test_cell_Zone_2_Base_1_3_t_1.0",
            "test_vertex_Zone_1_Base_3_3_t_1.0",
            "test_cell_Zone_1_Base_3_3_t_1.0",
            "test_vertex_Zone_2_Base_3_3_t_1.0",
            "test_cell_Zone_2_Base_3_3_t_1.0",
        ]
        assert sample.get_field_names() == sorted(set(expected_field_names))

    def test_get_field_empty(self, sample: Sample):
        assert sample.get_field("missing_field_name") is None
        assert sample.get_field("missing_field_name", location="CellCenter") is None

    def test_get_field(self, sample_with_tree):
        assert sample_with_tree.get_field("missing_field") is None
        assert sample_with_tree.get_field("test_node_field_1").shape == (5,)
        assert sample_with_tree.get_field(
            "test_elem_field_1", location="CellCenter"
        ).shape == (3,)

    def test_add_field_vertex(self, sample: Sample, vertex_field, zone_name, base_name):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.add_field(
                name="test_node_field_2",
                field=vertex_field,
                zone_name=zone_name,
                base_name=base_name,
            )
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        sample.add_field(
            name="test_node_field_2",
            field=vertex_field,
            zone_name=zone_name,
            base_name=base_name,
        )

    def test_add_field_cell_center(
        self, sample: Sample, cell_center_field, zone_name, base_name
    ):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.add_field(
                name="test_elem_field_2",
                field=cell_center_field,
                location="CellCenter",
                zone_name=zone_name,
                base_name=base_name,
            )
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        sample.add_field(
            name="test_elem_field_2",
            location="CellCenter",
            field=cell_center_field,
            zone_name=zone_name,
            base_name=base_name,
        )

    def test_add_field_vertex_already_present(
        self, sample_with_tree: Sample, vertex_field
    ):
        # with pytest.raises(KeyError):
        sample_with_tree.show_tree()
        sample_with_tree.add_field(
            name="test_node_field_1",
            field=vertex_field,
            zone_name="Zone",
            base_name="Base_2_2",
        )

    def test_add_field_cell_center_already_present(
        self, sample_with_tree: Sample, cell_center_field
    ):
        # with pytest.raises(KeyError):
        sample_with_tree.show_tree()
        sample_with_tree.add_field(
            name="test_elem_field_1",
            field=cell_center_field,
            location="CellCenter",
            zone_name="Zone",
            base_name="Base_2_2",
        )

    def test_del_field_existing(self, sample_with_tree):
        with pytest.raises(KeyError):
            sample_with_tree.del_field(
                name="unknown",
                location="CellCenter",
                zone_name="Zone",
                base_name="Base_2_2",
            )
        with pytest.raises(KeyError):
            sample_with_tree.del_field(
                name="unknown",
                location="CellCenter",
                zone_name="unknown_zone",
                base_name="Base_2_2",
            )

    def test_del_field_nonexistent(self, base_name):
        sample = Sample()
        sample.init_base(2, 2, base_name)
        with pytest.raises(KeyError):
            sample.del_field(
                name="unknown",
                location="CellCenter",
                zone_name="unknown_zone",
                base_name=base_name,
            )

    def test_del_field_in_zone(self, zone_name, base_name, cell_center_field):
        sample = Sample()
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        sample.add_field(
            name="test_elem_field_1",
            field=cell_center_field,
            location="CellCenter",
            zone_name=zone_name,
            base_name=base_name,
        )

        # Add field 'test_elem_field_2'
        sample.add_field(
            name="test_elem_field_2",
            field=cell_center_field,
            location="CellCenter",
            zone_name=zone_name,
            base_name=base_name,
        )
        assert isinstance(
            sample.get_field(
                name="test_elem_field_2",
                location="CellCenter",
                zone_name=zone_name,
                base_name=base_name,
            ),
            np.ndarray,
        )

        # Del field 'test_elem_field_2'
        new_tree = sample.del_field(
            name="test_elem_field_2",
            location="CellCenter",
            zone_name=zone_name,
            base_name=base_name,
        )

        # Testing new tree on field 'test_elem_field_2'
        new_sample = Sample()
        new_sample.features.add_tree(new_tree)

        assert (
            new_sample.get_field(
                name="test_elem_field_2",
                location="CellCenter",
                zone_name=zone_name,
                base_name=base_name,
            )
            is None
        )
        fields = new_sample.get_field_names(
            location="CellCenter", zone_name=zone_name, base_name=base_name
        )

        assert "test_elem_field_2" not in fields
        assert "test_elem_field_1" in fields

        # Del field 'test_elem_field_1'
        new_tree = sample.del_field(
            name="test_elem_field_1",
            location="CellCenter",
            zone_name=zone_name,
            base_name=base_name,
        )

        # Testing new tree on field 'test_elem_field_1'
        new_sample = Sample()
        new_sample.features.add_tree(new_tree)

        assert (
            new_sample.get_field(
                name="test_elem_field_1",
                location="CellCenter",
                zone_name=zone_name,
                base_name=base_name,
            )
            is None
        )
        fields = new_sample.get_field_names(
            location="CellCenter", zone_name=zone_name, base_name=base_name
        )
        assert len(fields) == 0

    def test_del_all_fields(self, sample_with_tree):
        sample_with_tree.del_all_fields()

    # -------------------------------------------------------------------------#
    def test_get_feature_by_path(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.get_feature_by_path(
            "Base_2_2/Zone/Elements_TRI_3/ElementConnectivity", 0.0
        )

    def test_get_feature_from_string_identifier(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "scalar::test_scalar_1"
        )

        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "field::test_node_field_1"
        )
        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "field::test_node_field_1///Base_2_2"
        )
        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "field::test_node_field_1//Zone/Base_2_2"
        )
        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "field::test_node_field_1/Vertex/Zone/Base_2_2"
        )
        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "field::test_node_field_1/Vertex/Zone/Base_2_2/0"
        )

        sample_with_tree_and_scalar.get_feature_from_string_identifier("nodes::")
        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "nodes::/Base_2_2"
        )
        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "nodes::Zone/Base_2_2"
        )
        sample_with_tree_and_scalar.get_feature_from_string_identifier(
            "nodes::Zone/Base_2_2/0"
        )

    def test_get_feature_from_identifier(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "scalar", "name": "test_scalar_1"}
        )

        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "field", "name": "test_node_field_1"}
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "field", "name": "test_node_field_1", "base_name": "Base_2_2"}
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {
                "type": "field",
                "name": "test_node_field_1",
                "base_name": "Base_2_2",
                "zone_name": "Zone",
            }
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {
                "type": "field",
                "name": "test_node_field_1",
                "base_name": "Base_2_2",
                "zone_name": "Zone",
                "location": "Vertex",
            }
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {
                "type": "field",
                "name": "test_node_field_1",
                "base_name": "Base_2_2",
                "zone_name": "Zone",
                "location": "Vertex",
                "time": 0.0,
            }
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "field", "name": "test_node_field_1", "time": 0.0}
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {
                "type": "field",
                "name": "test_node_field_1",
                "base_name": "Base_2_2",
                "time": 0.0,
            }
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {
                "type": "field",
                "name": "test_node_field_1",
                "zone_name": "Zone",
                "location": "Vertex",
                "time": 0.0,
            }
        )

        sample_with_tree_and_scalar.get_feature_from_identifier({"type": "nodes"})
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "nodes", "base_name": "Base_2_2"}
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "nodes", "base_name": "Base_2_2", "zone_name": "Zone"}
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "nodes", "base_name": "Base_2_2", "zone_name": "Zone", "time": 0.0}
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "nodes", "zone_name": "Zone"}
        )
        sample_with_tree_and_scalar.get_feature_from_identifier(
            {"type": "nodes", "time": 0.0}
        )

    def test_get_features_from_identifiers(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.get_features_from_identifiers(
            [{"type": "scalar", "name": "test_scalar_1"}]
        )
        sample_with_tree_and_scalar.get_features_from_identifiers(
            [
                {"type": "scalar", "name": "test_scalar_1"},
            ]
        )

        sample_with_tree_and_scalar.get_features_from_identifiers(
            [
                {
                    "type": "field",
                    "name": "test_node_field_1",
                    "base_name": "Base_2_2",
                    "zone_name": "Zone",
                    "location": "Vertex",
                    "time": 0.0,
                },
                {"type": "scalar", "name": "test_scalar_1"},
                {"type": "nodes"},
            ]
        )

    def test_update_features_from_identifier(self, sample_with_tree_and_scalar):
        before = sample_with_tree_and_scalar.get_scalar("test_scalar_1")
        sample_ = sample_with_tree_and_scalar.update_features_from_identifier(
            feature_identifiers={"type": "scalar", "name": "test_scalar_1"},
            features=3.141592,
            in_place=False,
        )
        after = sample_.get_scalar("test_scalar_1")
        show_cgns_tree(sample_.features.data[0])
        assert after != before

        before = sample_with_tree_and_scalar.get_field(
            name="test_node_field_1",
            zone_name="Zone",
            base_name="Base_2_2",
            location="Vertex",
            time=0.0,
        )
        sample_ = sample_with_tree_and_scalar.update_features_from_identifier(
            feature_identifiers=FeatureIdentifier(
                {
                    "type": "field",
                    "name": "test_node_field_1",
                    "base_name": "Base_2_2",
                    "zone_name": "Zone",
                    "location": "Vertex",
                    "time": 0.0,
                }
            ),
            features=np.random.rand(*before.shape),
            in_place=False,
        )
        after = sample_.get_field(
            name="test_node_field_1",
            zone_name="Zone",
            base_name="Base_2_2",
            location="Vertex",
            time=0.0,
        )
        assert np.any(~np.isclose(after, before))

        before = sample_with_tree_and_scalar.get_nodes(
            zone_name="Zone", base_name="Base_2_2", time=0.0
        )
        sample_ = sample_with_tree_and_scalar.update_features_from_identifier(
            feature_identifiers=FeatureIdentifier(
                {
                    "type": "nodes",
                    "base_name": "Base_2_2",
                    "zone_name": "Zone",
                    "time": 0.0,
                }
            ),
            features=np.random.rand(*before.shape),
            in_place=False,
        )
        after = sample_.get_nodes(zone_name="Zone", base_name="Base_2_2", time=0.0)
        assert np.any(~np.isclose(after, before))

        before_1 = sample_with_tree_and_scalar.get_field("test_node_field_1")
        before_2 = sample_with_tree_and_scalar.get_nodes()
        sample_ = sample_with_tree_and_scalar.update_features_from_identifier(
            feature_identifiers=[
                {"type": "field", "name": "test_node_field_1"},
                {"type": "nodes"},
            ],
            features=[
                np.random.rand(*before_1.shape),
                np.random.rand(*before_2.shape),
            ],
            in_place=False,
        )
        after_1 = sample_.get_field("test_node_field_1")
        after_2 = sample_.get_nodes()
        assert np.any(~np.isclose(after_1, before_1))
        assert np.any(~np.isclose(after_2, before_2))

        sample_ = sample_with_tree_and_scalar.update_features_from_identifier(
            feature_identifiers=[{"type": "field", "name": "test_node_field_1"}],
            features=[np.random.rand(*before_1.shape)],
            in_place=True,
        )
        ref_1 = sample_with_tree_and_scalar.get_field("test_node_field_1")
        ref_2 = sample_.get_field("test_node_field_1")
        assert np.any(np.isclose(ref_1, ref_2))

    def test_extract_sample_from_identifier(self, sample_with_tree_and_scalar):
        sample_: Sample = sample_with_tree_and_scalar.extract_sample_from_identifier(
            feature_identifiers={"type": "scalar", "name": "test_scalar_1"},
        )
        assert sample_.get_scalar_names() == ["test_scalar_1"]
        assert len(sample_.get_field_names()) == 0

        sample_: Sample = sample_with_tree_and_scalar.extract_sample_from_identifier(
            feature_identifiers={
                "type": "field",
                "name": "test_node_field_1",
                "base_name": "Base_2_2",
                "zone_name": "Zone",
                "location": "Vertex",
                "time": 0.0,
            },
        )
        show_cgns_tree(sample_with_tree_and_scalar.features.data[0])
        assert len(sample_.get_scalar_names()) == 0
        assert sample_.get_field_names() == ["test_node_field_1"]

        sample_: Sample = sample_with_tree_and_scalar.extract_sample_from_identifier(
            feature_identifiers={
                "type": "nodes",
                "base_name": "Base_2_2",
                "zone_name": "Zone",
                "time": 0.0,
            },
        )
        assert len(sample_.get_scalar_names()) == 0
        assert len(sample_.get_field_names()) == 0

        sample_: Sample = sample_with_tree_and_scalar.extract_sample_from_identifier(
            feature_identifiers=[
                {"type": "field", "name": "test_node_field_1"},
                {"type": "nodes"},
            ],
        )
        assert len(sample_.get_scalar_names()) == 0
        assert sample_.get_field_names() == ["test_node_field_1"]

    def test_get_all_features_identifiers(self, sample_with_tree_and_scalar):
        feat_ids = sample_with_tree_and_scalar.get_all_features_identifiers()
        assert len(feat_ids) == 9
        assert {"type": "scalar", "name": "r"} in feat_ids
        assert {"type": "scalar", "name": "test_scalar_1"} in feat_ids
        assert {
            "type": "nodes",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "time": 0.0,
        } in feat_ids
        assert {
            "type": "field",
            "name": "big_node_field",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "location": "Vertex",
            "time": 0.0,
        } in feat_ids
        assert {
            "type": "field",
            "name": "test_node_field_1",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "location": "Vertex",
            "time": 0.0,
        } in feat_ids
        assert {
            "type": "field",
            "name": "OriginalIds",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "location": "Vertex",
            "time": 0.0,
        } in feat_ids
        assert {
            "type": "field",
            "name": "OriginalIds",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "location": "FaceCenter",
            "time": 0.0,
        } in feat_ids
        assert {
            "type": "field",
            "name": "test_elem_field_1",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "location": "CellCenter",
            "time": 0.0,
        } in feat_ids
        assert {
            "type": "field",
            "name": "OriginalIds",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "location": "CellCenter",
            "time": 0.0,
        } in feat_ids

    def test_get_all_features_identifiers_by_type(self, sample_with_tree_and_scalar):
        feat_ids = sample_with_tree_and_scalar.get_all_features_identifiers_by_type(
            "scalar"
        )
        assert len(feat_ids) == 2
        assert {"type": "scalar", "name": "r"} in feat_ids
        assert {"type": "scalar", "name": "test_scalar_1"} in feat_ids

        feat_ids = sample_with_tree_and_scalar.get_all_features_identifiers_by_type(
            "nodes"
        )
        assert {
            "type": "nodes",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "time": 0.0,
        } in feat_ids

        feat_ids = sample_with_tree_and_scalar.get_all_features_identifiers_by_type(
            "field"
        )
        assert len(feat_ids) == 6
        assert {
            "type": "field",
            "name": "big_node_field",
            "base_name": "Base_2_2",
            "zone_name": "Zone",
            "location": "Vertex",
            "time": 0.0,
        } in feat_ids

    def test_merge_features(self, sample_with_tree_and_scalar, sample_with_tree):
        feat_id = sample_with_tree_and_scalar.get_all_features_identifiers()
        feat_id = [fid for fid in feat_id if fid["type"] not in ["scalar"]]
        sample_1 = sample_with_tree_and_scalar.extract_sample_from_identifier(feat_id)
        feat_id = sample_with_tree.get_all_features_identifiers()
        feat_id = [fid for fid in feat_id if fid["type"] not in ["field"]]
        sample_2 = sample_with_tree.extract_sample_from_identifier(feat_id)
        sample_merge_1 = sample_1.merge_features(sample_2, in_place=False)
        sample_merge_2 = sample_2.merge_features(sample_1, in_place=False)
        assert (
            sample_merge_1.get_all_features_identifiers()
            == sample_merge_2.get_all_features_identifiers()
        )
        sample_2.merge_features(sample_1, in_place=True)
        sample_1.merge_features(sample_2, in_place=True)

    def test_merge_features2(self, sample_with_tree_and_scalar, sample_with_tree):
        feat_id = sample_with_tree_and_scalar.get_all_features_identifiers()
        feat_id = [fid for fid in feat_id if fid["type"] not in ["scalar"]]
        sample_1 = sample_with_tree_and_scalar.extract_sample_from_identifier(feat_id)
        feat_id = sample_with_tree.get_all_features_identifiers()
        feat_id = [fid for fid in feat_id if fid["type"] not in ["field", "nodes"]]
        sample_2 = sample_with_tree.extract_sample_from_identifier(feat_id)
        sample_merge_1 = sample_1.merge_features(sample_2, in_place=False)
        sample_merge_2 = sample_2.merge_features(sample_1, in_place=False)
        assert (
            sample_merge_1.get_all_features_identifiers()
            == sample_merge_2.get_all_features_identifiers()
        )
        sample_2.merge_features(sample_1, in_place=True)
        sample_1.merge_features(sample_2, in_place=True)

    # -------------------------------------------------------------------------#
    def test_save(self, sample_with_tree_and_scalar, tmp_path):
        save_dir = tmp_path / "test_dir"
        sample_with_tree_and_scalar.save(save_dir)
        assert save_dir.is_dir()
        with pytest.raises(ValueError):
            sample_with_tree_and_scalar.save(save_dir, memory_safe=False)
        sample_with_tree_and_scalar.save(save_dir, overwrite=True)
        sample_with_tree_and_scalar.save(save_dir, overwrite=True, memory_safe=True)

    def test_load_from_saved_file(self, sample_with_tree_and_scalar, tmp_path):
        save_dir = tmp_path / "test_dir"
        sample_with_tree_and_scalar.save(save_dir)
        new_sample = Sample()
        new_sample.load(save_dir)
        assert CGU.checkSameTree(
            sample_with_tree_and_scalar.get_mesh(),
            new_sample.get_mesh(),
        )

    def test_load_from_dir(self, sample_with_tree_and_scalar, tmp_path):
        save_dir = tmp_path / "test_dir"
        sample_with_tree_and_scalar.save(save_dir)
        new_sample = Sample.load_from_dir(save_dir)
        assert CGU.checkSameTree(
            sample_with_tree_and_scalar.get_mesh(),
            new_sample.get_mesh(),
        )

    # -------------------------------------------------------------------------#
    def test___repr___empty(self, sample):
        print(sample)

    def test___repr__with_scalar(self, sample_with_scalar):
        print(sample_with_scalar)

    def test___repr__with_tree(self, sample_with_tree):
        print(sample_with_tree)

    def test___repr__with_tree_and_scalar(self, sample_with_tree_and_scalar):
        print(sample_with_tree_and_scalar)

    def test___repr__full_sample(self, full_sample):
        print(full_sample)

    # -------------------------------------------------------------------------#

    def test_summarize_empty(self, sample):
        print(sample.summarize())

    def test_summarize_with_scalar(self, sample_with_scalar):
        print(sample_with_scalar.summarize())

    def test_summarize_with_tree(self, sample_with_tree):
        print(sample_with_tree.summarize())

    def test_summarize_with_tree_and_scalar(self, sample_with_tree_and_scalar):
        print(sample_with_tree_and_scalar.summarize())

    def test_check_completeness_empty(self, sample):
        print(sample.check_completeness())

    def test_check_completeness_with_scalar(self, sample_with_scalar):
        print(sample_with_scalar.check_completeness())

    def test_check_completeness_with_tree(self, sample_with_tree):
        print(sample_with_tree.check_completeness())

    def test_check_completeness_with_tree_and_scalar(self, sample_with_tree_and_scalar):
        print(sample_with_tree_and_scalar.check_completeness())
