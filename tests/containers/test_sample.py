# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import os

import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnsutils as CGU
import numpy as np
import pytest
from Muscat.Bridges.CGNSBridge import MeshToCGNS
from Muscat.Containers import MeshCreationTools as MCT

from plaid.containers.sample import (
    Sample,
    read_index,
    read_index_array,
    read_index_range,
)

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
def sample_with_time_series(sample):
    sample.add_time_series(
        "test_time_series_1", np.arange(111, dtype=float), np.random.randn(111)
    )
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
def sample_with_linked_tree(tree, tmp_path):
    sample_with_linked_tree = Sample()
    sample_with_linked_tree.add_tree(tree)
    path_linked_sample = tmp_path / "test_dir" / "meshes/mesh_000000000.cgns"
    sample_with_linked_tree.link_tree(
        path_linked_sample, sample_with_linked_tree, linked_time=0.0, time=1.0
    )
    return sample_with_linked_tree


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
    sample.add_tree(tree3d)
    return sample


@pytest.fixture()
def sample_with_tree_and_scalar_and_time_series(
    sample_with_tree,
):
    sample_with_tree.add_scalar("r", np.random.randn())
    sample_with_tree.add_scalar("test_scalar_1", np.random.randn())
    sample_with_tree.add_time_series(
        "test_time_series_1", np.arange(111, dtype=float), np.random.randn(111)
    )
    return sample_with_tree


# %% Test


def test_read_index(tree, physical_dim):
    read_index(tree, physical_dim)


def test_read_index_array(tree):
    read_index_array(tree)


def test_read_index_range(tree, physical_dim):
    read_index_range(tree, physical_dim)


@pytest.fixture()
def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


# %% Tests


class Test_Sample:
    # -------------------------------------------------------------------------#
    def test___init__(self, current_directory):
        dataset_path_1 = os.path.join(
            current_directory, "dataset", "samples", "sample_000000000"
        )
        dataset_path_2 = os.path.join(
            current_directory, "dataset", "samples", "sample_000000001"
        )
        dataset_path_3 = os.path.join(
            current_directory, "dataset", "samples", "sample_000000002"
        )
        sample_already_filled_1 = Sample(dataset_path_1)
        sample_already_filled_2 = Sample(dataset_path_2)
        sample_already_filled_3 = Sample(dataset_path_3)
        assert (
            sample_already_filled_1._meshes is not None
            and sample_already_filled_1._scalars is not None
        )
        assert (
            sample_already_filled_2._meshes is not None
            and sample_already_filled_2._scalars is not None
        )
        assert (
            sample_already_filled_3._meshes is not None
            and sample_already_filled_3._scalars is not None
        )

    def test__init__unknown_directory(self, current_directory):
        dataset_path = os.path.join(
            current_directory, "dataset", "samples", "sample_000000298"
        )
        with pytest.raises(FileNotFoundError):
            Sample(dataset_path)

    def test__init__file_provided(self, current_directory):
        dataset_path = os.path.join(
            current_directory, "dataset", "samples", "sample_000067392"
        )
        with pytest.raises(FileExistsError):
            Sample(dataset_path)

    # -------------------------------------------------------------------------#
    def test_set_default_base(self, sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, time=0.5)

        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}", 0.5)
        # check dims getters
        assert sample.get_topological_dim() == topological_dim
        assert sample.get_physical_dim() == physical_dim
        assert sample.get_base_assignment() == f"Base_{topological_dim}_{physical_dim}"
        assert sample.get_time_assignment() == 0.5
        assert sample.get_base_assignment("test") == "test"

        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}")  # already set
        sample.set_default_base(None)  # will not assign to None
        assert sample.get_base_assignment() == f"Base_{topological_dim}_{physical_dim}"
        with pytest.raises(ValueError):
            sample.set_default_base("Unknown base name")

    def test_set_default_zone_with_default_base(
        self, sample, topological_dim, physical_dim, base_name, zone_name, zone_shape
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=0.5)
        sample.set_default_base(base_name)
        # No zone provided
        assert sample.get_zone() is None

        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)
        # Look for the only zone in the default base
        assert sample.get_zone() is not None

        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)
        # There is more than one zone in this base
        with pytest.raises(KeyError):
            sample.get_zone()

    def test_set_default_zone(
        self, sample, topological_dim, physical_dim, base_name, zone_name, zone_shape
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=0.5)
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)

        sample.set_default_zone_base(zone_name, base_name, 0.5)
        # check dims getters
        assert sample.get_topological_dim() == topological_dim
        assert sample.get_physical_dim() == physical_dim
        assert sample.get_base_assignment() == base_name
        assert sample.get_time_assignment() == 0.5

        sample.set_default_base(base_name)  # already set
        sample.set_default_base(None)  # will not assign to None
        assert sample.get_base_assignment() == base_name
        with pytest.raises(ValueError):
            sample.set_default_base("Unknown base name")

        assert sample.get_zone_assignment() == zone_name
        assert sample.get_time_assignment() == 0.5

        assert sample.get_zone() is not None
        sample.set_default_zone_base(zone_name, base_name)
        sample.set_default_zone_base(None, base_name)  # will not assign to None
        assert sample.get_zone_assignment() == zone_name
        with pytest.raises(ValueError):
            sample.set_default_zone_base("Unknown zone name", base_name)

    def test_set_default_time(self, sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, time=0.5)
        sample.init_base(topological_dim, physical_dim, "OK_name", time=1.5)

        assert sample.get_time_assignment() == 0.5
        sample.set_default_time(1.5)
        assert sample.get_time_assignment() == 1.5, "here"

        sample.set_default_time(1.5)  # already set
        sample.set_default_time(None)  # will not assign to None
        assert sample.get_time_assignment() == 1.5
        with pytest.raises(ValueError):
            sample.set_default_time(2.5)

    # -------------------------------------------------------------------------#

    def test_show_tree(self, sample_with_tree_and_scalar_and_time_series):
        sample_with_tree_and_scalar_and_time_series.show_tree()

    def test_init_tree(self, sample):
        sample.init_tree()
        sample.init_tree(0.5)

    def test_get_mesh_empty(self, sample):
        sample.get_mesh()

    def test_get_mesh(self, sample_with_tree_and_scalar_and_time_series):
        sample_with_tree_and_scalar_and_time_series.get_mesh()

    def test_get_mesh_without_links(self, sample_with_linked_tree):
        sample_with_linked_tree.get_mesh(time=1.0, apply_links=False)

    def test_get_mesh_with_links_in_memory(self, sample_with_linked_tree):
        sample_with_linked_tree.get_mesh(time=1.0, apply_links=True, in_memory=True)

    def test_get_mesh_with_links(self, sample_with_linked_tree, tmp_path):
        sample_with_linked_tree.save(tmp_path / "test_dir")
        sample_with_linked_tree.get_mesh(time=1.0, apply_links=True)

    def test_set_meshes_empty(self, sample, tree):
        sample.set_meshes({0.0: tree})

    def test_set_meshes(self, sample_with_tree, tree):
        with pytest.raises(KeyError):
            sample_with_tree.set_meshes({0.0: tree})

    def test_add_tree_empty(self, sample_with_tree):
        with pytest.raises(ValueError):
            sample_with_tree.add_tree([])

    def test_add_tree(self, sample, tree):
        sample.add_tree(tree)
        sample.add_tree(tree)
        sample.add_tree(tree, time=0.2)

    def test_del_tree(self, sample, tree):
        sample.add_tree(tree)
        sample.add_tree(tree, time=0.2)

        assert isinstance(sample.del_tree(0.2), list)
        assert list(sample._meshes.keys()) == [0.0]
        assert list(sample._links.keys()) == [0.0]
        assert list(sample._paths.keys()) == [0.0]

        assert isinstance(sample.del_tree(0.0), list)
        assert list(sample._meshes.keys()) == []
        assert list(sample._links.keys()) == []
        assert list(sample._paths.keys()) == []

    def test_link_tree(self, sample_with_linked_tree):
        link_checks = [
            "/Base_2_2/Zone/Elements_Selections",
            "/Base_2_2/Zone/Points_Selections",
            "/Base_2_2/Zone/Points_Selections/tag",
            "/Base_2_2/Zone/Elements_TRI_3",
            "/Base_2_2/Zone/GridCoordinates",
            "/Base_2_2/Zone/ZoneBC",
        ]
        for link in sample_with_linked_tree._links[1]:
            assert link[1] == "mesh_000000000.cgns"
            assert link[2] == link[3]
            assert link[2] in link_checks

    def test_on_error_del_tree(self, sample, tree):
        with pytest.raises(KeyError):
            sample.del_tree(0.0)

        sample.add_tree(tree)
        sample.add_tree(tree, time=0.2)
        with pytest.raises(KeyError):
            sample.del_tree(0.7)

    # -------------------------------------------------------------------------#
    def test_init_base(self, sample, base_name, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, base_name)
        # check dims getters
        assert sample.get_topological_dim(base_name) == topological_dim
        assert sample.get_physical_dim(base_name) == physical_dim

    def test_del_base_existing_base(
        self, sample, base_name, topological_dim, physical_dim
    ):
        second_base_name = base_name + "_2"
        sample.init_base(topological_dim, physical_dim, base_name)
        sample.init_base(topological_dim, physical_dim, second_base_name)

        # Delete first base
        updated_cgns_tree = sample.del_base(base_name, 0.0)
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Testing the resulting tree
        new_sample = Sample()
        new_sample.add_tree(updated_cgns_tree, 0.1)
        assert new_sample.get_topological_dim() == topological_dim
        assert new_sample.get_physical_dim() == physical_dim
        assert new_sample.get_base_names() == [second_base_name]

        # Add 2 bases and delete one base at time 0.2
        sample.init_base(topological_dim, physical_dim, "tree", 0.2)
        sample.init_base(topological_dim, physical_dim, base_name, 0.2)
        updated_cgns_tree = sample.del_base("tree", 0.2)
        assert sample.get_base("tree", 0.2) is None
        assert sample.get_base(base_name, 0.2) is not None
        assert sample.get_base(second_base_name) is not None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Testing the resulting from time 0.2
        new_sample = Sample()
        new_sample.add_tree(updated_cgns_tree)
        assert new_sample.get_topological_dim() == topological_dim
        assert new_sample.get_physical_dim() == physical_dim
        assert new_sample.get_base_names() == [base_name]

        # Deleting the last base at time 0.0
        updated_cgns_tree = sample.del_base(second_base_name, 0.0)
        assert sample.get_base(second_base_name) is None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Deleting the last base at time 0.2
        updated_cgns_tree = sample.del_base(base_name, 0.2)
        assert sample.get_base(base_name) is None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

    def test_del_base_nonexistent_base_nonexistent_time(
        self, sample, base_name, topological_dim, physical_dim
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=1.0)
        with pytest.raises(KeyError):
            sample.del_base(base_name, time=2.0)
        with pytest.raises(KeyError):
            sample.del_base("unknown", time=1.0)

    def test_del_base_no_cgns_tree(self, sample):
        with pytest.raises(KeyError):
            sample.del_base("unknwon", 0.0)

    def test_init_base_no_base_name(self, sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim)

        # check dims getters
        assert (
            sample.get_topological_dim(f"Base_{topological_dim}_{physical_dim}")
            == topological_dim
        )
        assert (
            sample.get_physical_dim(f"Base_{topological_dim}_{physical_dim}")
            == physical_dim
        )

        # check setting default base
        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}")
        assert sample.get_topological_dim() == topological_dim
        assert sample.get_physical_dim() == physical_dim

    def test_get_base_names(self, sample):
        assert sample.get_base_names() == []
        sample.init_base(3, 3, "base_name_1")
        sample.init_base(3, 3, "base_name_2")
        assert sample.get_base_names() == ["base_name_1", "base_name_2"]
        assert sample.get_base_names(full_path=True) == ["/base_name_1", "/base_name_2"]
        # check dims getters
        assert sample.get_topological_dim("base_name_1") == 3
        assert sample.get_physical_dim("base_name_1") == 3
        assert sample.get_topological_dim("base_name_2") == 3
        assert sample.get_physical_dim("base_name_2") == 3

    def test_get_base(self, sample, base_name):
        sample.init_tree()
        assert sample.get_base() is None
        sample.init_base(3, 3, base_name)
        assert sample.get_base(base_name) is not None
        assert sample.get_base() is not None
        sample.init_base(3, 3, "other_base_name")
        assert sample.get_base(base_name) is not None
        with pytest.raises(KeyError):
            sample.get_base()
        # check dims getters
        assert sample.get_topological_dim(base_name) == 3
        assert sample.get_physical_dim(base_name) == 3
        assert sample.get_topological_dim("other_base_name") == 3
        assert sample.get_physical_dim("other_base_name") == 3

    # -------------------------------------------------------------------------#
    def test_init_zone(self, sample, base_name, zone_name, zone_shape):
        with pytest.raises(KeyError):
            sample.init_zone(zone_shape, zone_name=zone_name, base_name=base_name)
        sample.init_base(3, 3, base_name)
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)
        sample.init_zone(zone_shape, CGK.Unstructured_s, zone_name, base_name=base_name)
        # check dims getters
        assert sample.get_topological_dim(base_name) == 3
        assert sample.get_physical_dim(base_name) == 3

    def test_init_zone_defaults_names(self, sample, zone_shape):
        sample.init_base(3, 3)
        sample.init_zone(zone_shape)

    def test_del_zone_existing_zone(self, sample, base_name, zone_name, zone_shape):
        topological_dim, physical_dim = 3, 3
        sample.init_base(topological_dim, physical_dim, base_name)

        second_zone_name = zone_name + "_2"
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base_name=base_name)
        sample.init_zone(
            zone_shape, CGK.Unstructured_s, second_zone_name, base_name=base_name
        )

        # Delete first zone
        updated_cgns_tree = sample.del_zone(zone_name, base_name, 0.0)
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Testing the resulting tree
        new_sample = Sample()
        new_sample.add_tree(updated_cgns_tree, 0.1)
        assert new_sample.get_zone_names() == [second_zone_name]

        # Add 2 zones and delete one zone at time 0.2
        sample.init_base(topological_dim, physical_dim, base_name, 0.2)
        sample.init_zone(
            zone_shape, CGK.Structured_s, zone_name, base_name=base_name, time=0.2
        )
        sample.init_zone(
            zone_shape, CGK.Unstructured_s, "test", base_name=base_name, time=0.2
        )

        updated_cgns_tree = sample.del_zone("test", base_name, 0.2)
        assert sample.get_zone("tree", base_name, 0.2) is None
        assert sample.get_zone(zone_name, base_name, 0.2) is not None
        assert sample.get_zone(second_zone_name, base_name) is not None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Testing the resulting from time 0.2
        new_sample = Sample()
        new_sample.add_tree(updated_cgns_tree)
        assert new_sample.get_zone_names(base_name) == [zone_name]

        # Deleting the last zone at time 0.0
        updated_cgns_tree = sample.del_zone(second_zone_name, base_name, 0.0)
        assert sample.get_zone(second_zone_name, base_name) is None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

        # Deleting the last zone at time 0.2
        updated_cgns_tree = sample.del_zone(zone_name, base_name, 0.2)
        assert sample.get_zone(zone_name, base_name) is None
        assert updated_cgns_tree is not None and isinstance(updated_cgns_tree, list)

    def test_del_zone_nonexistent_zone_nonexistent_time(
        self, sample, base_name, zone_shape, topological_dim, physical_dim
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=1.0)
        zone_name = "test123"
        sample.init_zone(
            zone_shape, CGK.Structured_s, zone_name, base_name=base_name, time=1.0
        )
        with pytest.raises(KeyError):
            sample.del_zone(zone_name, base_name, 2.0)
        with pytest.raises(KeyError):
            sample.del_zone("unknown", base_name, 1.0)

    def test_del_zone_no_cgns_tree(self, sample):
        sample.init_base(2, 3, "only_base")
        with pytest.raises(KeyError):
            sample.del_zone("unknwon", "only_base", 0.0)

    def test_has_zone(self, sample, base_name, zone_name):
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        sample.show_tree()
        assert sample.has_zone(zone_name, base_name)
        assert not sample.has_zone("not_present_zone_name", base_name)
        assert not sample.has_zone(zone_name, "not_present_base_name")
        assert not sample.has_zone("not_present_zone_name", "not_present_base_name")

    def test_get_zone_names(self, sample, base_name):
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
        assert sample.get_zone_names(base_name) == ["zone_name_1", "zone_name_2"]
        assert sorted(sample.get_zone_names(base_name, unique=True)) == sorted(
            ["zone_name_1", "zone_name_2"]
        )
        assert sample.get_zone_names(base_name, full_path=True) == [
            f"{base_name}/zone_name_1",
            f"{base_name}/zone_name_2",
        ]

    def test_get_zone_type(self, sample, zone_name, base_name):
        with pytest.raises(KeyError):
            sample.get_zone_type(zone_name, base_name)
        sample.init_tree()
        with pytest.raises(KeyError):
            sample.get_zone_type(zone_name, base_name)
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.get_zone_type(zone_name, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        assert sample.get_zone_type(zone_name, base_name) == CGK.Unstructured_s

    def test_get_zone(self, sample, zone_name, base_name):
        assert sample.get_zone(zone_name, base_name) is None
        sample.init_base(3, 3, base_name)
        assert sample.get_zone(zone_name, base_name) is None
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        assert sample.get_zone() is not None
        assert sample.get_zone(zone_name, base_name) is not None
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name="other_zone_name",
            base_name=base_name,
        )
        assert sample.get_zone(zone_name, base_name) is not None
        with pytest.raises(KeyError):
            assert sample.get_zone() is not None

    # -------------------------------------------------------------------------#
    def test_get_scalar_names(self, sample):
        assert sample.get_scalar_names() == []

    def test_get_scalar_empty(self, sample):
        assert sample.get_scalar("missing_scalar_name") is None

    def test_get_scalar(self, sample_with_scalar):
        assert sample_with_scalar.get_scalar("missing_scalar_name") is None
        assert sample_with_scalar.get_scalar("test_scalar_1") is not None

    def test_add_scalar_empty(self, sample_with_scalar):
        assert isinstance(sample_with_scalar.get_scalar("test_scalar_1"), float)

    def test_add_scalar(self, sample_with_scalar):
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

    # -------------------------------------------------------------------------#
    def test_get_time_series_names_empty(self, sample):
        assert sample.get_time_series_names() == []

    def test_get_time_series_names(self, sample_with_time_series):
        assert sample_with_time_series.get_time_series_names() == ["test_time_series_1"]

    def test_get_time_series_empty(self, sample):
        assert sample.get_time_series("missing_time_series_name") is None

    def test_get_time_series(self, sample_with_time_series):
        assert (
            sample_with_time_series.get_time_series("missing_time_series_name") is None
        )
        assert sample_with_time_series.get_time_series("test_time_series_1") is not None

    def test_add_time_series_empty(self, sample_with_time_series):
        pass

    def test_add_time_series(self, sample_with_time_series):
        sample_with_time_series.add_time_series(
            "test_time_series_2", np.arange(111, dtype=float), np.random.randn(111)
        )

    def test_del_time_series_unknown_scalar(self, sample_with_time_series):
        with pytest.raises(KeyError):
            sample_with_time_series.del_time_series("non_existent_scalar")

    def test_del_time_series_no_scalar(self):
        sample = Sample()
        with pytest.raises(KeyError):
            sample.del_time_series("non_existent_scalar")

    def test_del_time_series(self, sample_with_time_series):
        assert len(sample_with_time_series.get_time_series_names()) == 1

        sample_with_time_series.add_time_series(
            "test_time_series_2", np.arange(222, dtype=float), np.random.randn(222)
        )
        assert len(sample_with_time_series.get_time_series_names()) == 2

        time_series = sample_with_time_series.del_time_series("test_time_series_1")
        assert len(sample_with_time_series.get_time_series_names()) == 1
        assert time_series is not None
        assert isinstance(time_series, tuple)
        assert isinstance(time_series[0], np.ndarray)
        assert isinstance(time_series[1], np.ndarray)

        time_series = sample_with_time_series.del_time_series("test_time_series_2")
        assert len(sample_with_time_series.get_time_series_names()) == 0
        assert time_series is not None
        assert isinstance(time_series, tuple)
        assert isinstance(time_series[0], np.ndarray)
        assert isinstance(time_series[1], np.ndarray)

    # -------------------------------------------------------------------------#
    def test_get_nodal_tags_empty(self, sample):
        assert sample.get_nodal_tags() == {}

    def test_get_nodal_tags(self, sample_with_tree, nodal_tags):
        assert np.all(sample_with_tree.get_nodal_tags()["tag"] == nodal_tags)

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
    def test_get_elements_empty(self, sample):
        assert sample.get_elements() == {}

    def test_get_elements(self, sample_with_tree, triangles):
        assert list(sample_with_tree.get_elements().keys()) == ["TRI_3"]
        print(f"{triangles=}")
        print(f"{sample_with_tree.get_elements()=}")
        assert np.all(sample_with_tree.get_elements()["TRI_3"] == triangles)

    # -------------------------------------------------------------------------#
    def test_get_field_names(self, sample):
        assert sample.get_field_names() == []
        assert sample.get_field_names(location="CellCenter") == []

    def test_get_field_empty(self, sample):
        assert sample.get_field("missing_field_name") is None
        assert sample.get_field("missing_field_name", location="CellCenter") is None

    def test_get_field(self, sample_with_tree):
        assert sample_with_tree.get_field("missing_field") is None
        assert sample_with_tree.get_field("test_node_field_1").shape == (5,)
        assert sample_with_tree.get_field(
            "test_elem_field_1", location="CellCenter"
        ).shape == (3,)

    def test_add_field_vertex(self, sample, vertex_field, zone_name, base_name):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.add_field("test_node_field_2", vertex_field, zone_name, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        sample.add_field("test_node_field_2", vertex_field, zone_name, base_name)

    def test_add_field_cell_center(
        self, sample, cell_center_field, zone_name, base_name
    ):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.add_field(
                "test_elem_field_2",
                cell_center_field,
                zone_name,
                base_name,
                location="CellCenter",
            )
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        sample.add_field(
            "test_elem_field_2",
            cell_center_field,
            zone_name,
            base_name,
            location="CellCenter",
        )

    def test_add_field_vertex_already_present(self, sample_with_tree, vertex_field):
        # with pytest.raises(KeyError):
        sample_with_tree.show_tree()
        sample_with_tree.add_field(
            "test_node_field_1", vertex_field, "Zone", "Base_2_2"
        )

    def test_add_field_cell_center_already_present(
        self, sample_with_tree, cell_center_field
    ):
        # with pytest.raises(KeyError):
        sample_with_tree.show_tree()
        sample_with_tree.add_field(
            "test_elem_field_1",
            cell_center_field,
            "Zone",
            "Base_2_2",
            location="CellCenter",
        )

    def test_del_field_existing(self, sample_with_tree):
        with pytest.raises(KeyError):
            sample_with_tree.del_field(
                "unknown", "Zone", "Base_2_2", location="CellCenter"
            )
        with pytest.raises(KeyError):
            sample_with_tree.del_field(
                "unknown", "unknown_zone", "Base_2_2", location="CellCenter"
            )

    def test_del_field_nonexistent(self, base_name):
        sample = Sample()
        sample.init_base(2, 2, base_name)
        with pytest.raises(KeyError):
            sample.del_field(
                "unknown", "unknown_zone", base_name, location="CellCenter"
            )

    def test_del_field_in_zone(self, zone_name, base_name, cell_center_field):
        sample = Sample()
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3), zone_name=zone_name, base_name=base_name
        )
        sample.add_field(
            "test_elem_field_1",
            cell_center_field,
            zone_name,
            base_name,
            location="CellCenter",
        )

        # Add field 'test_elem_field_2'
        sample.add_field(
            "test_elem_field_2",
            cell_center_field,
            zone_name,
            base_name,
            location="CellCenter",
        )
        assert isinstance(
            sample.get_field(
                "test_elem_field_2", zone_name, base_name, location="CellCenter"
            ),
            np.ndarray,
        )

        # Del field 'test_elem_field_2'
        new_tree = sample.del_field(
            "test_elem_field_2", zone_name, base_name, location="CellCenter"
        )

        # Testing new tree on field 'test_elem_field_2'
        new_sample = Sample()
        new_sample.add_tree(new_tree)

        assert (
            new_sample.get_field(
                "test_elem_field_2", zone_name, base_name, location="CellCenter"
            )
            is None
        )
        fields = new_sample.get_field_names(zone_name, base_name, location="CellCenter")

        assert "test_elem_field_2" not in fields
        assert "test_elem_field_1" in fields

        # Del field 'test_elem_field_1'
        new_tree = sample.del_field(
            "test_elem_field_1", zone_name, base_name, location="CellCenter"
        )

        # Testing new tree on field 'test_elem_field_1'
        new_sample = Sample()
        new_sample.add_tree(new_tree)

        assert (
            new_sample.get_field(
                "test_elem_field_1", zone_name, base_name, location="CellCenter"
            )
            is None
        )
        fields = new_sample.get_field_names(zone_name, base_name, location="CellCenter")
        assert len(fields) == 0

    # -------------------------------------------------------------------------#
    def test_save(self, sample_with_tree_and_scalar_and_time_series, tmp_path):
        save_dir = tmp_path / "test_dir"
        sample_with_tree_and_scalar_and_time_series.save(save_dir)
        assert save_dir.is_dir()
        with pytest.raises(ValueError):
            sample_with_tree_and_scalar_and_time_series.save(save_dir)

    def test_load_from_saved_file(
        self, sample_with_tree_and_scalar_and_time_series, tmp_path
    ):
        save_dir = tmp_path / "test_dir"
        sample_with_tree_and_scalar_and_time_series.save(save_dir)
        new_sample = Sample()
        new_sample.load(save_dir)
        assert CGU.checkSameTree(
            sample_with_tree_and_scalar_and_time_series.get_mesh(),
            new_sample.get_mesh(),
        )

    def test_load_from_dir(self, sample_with_tree_and_scalar_and_time_series, tmp_path):
        save_dir = tmp_path / "test_dir"
        sample_with_tree_and_scalar_and_time_series.save(save_dir)
        new_sample = Sample.load_from_dir(save_dir)
        assert CGU.checkSameTree(
            sample_with_tree_and_scalar_and_time_series.get_mesh(),
            new_sample.get_mesh(),
        )

    # -------------------------------------------------------------------------#
    def test___repr___empty(self, sample):
        print(sample)

    def test___repr__with_scalar(self, sample_with_scalar):
        print(sample_with_scalar)

    def test___repr__with_tree(self, sample_with_tree):
        print(sample_with_tree)

    def test___repr__with_tree_and_scalar(
        self, sample_with_tree_and_scalar_and_time_series
    ):
        print(sample_with_tree_and_scalar_and_time_series)
