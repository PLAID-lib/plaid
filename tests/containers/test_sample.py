# %% Imports

import copy
from pathlib import Path

import CGNS.PAT.cgnskeywords as CGK
import CGNS.PAT.cgnslib as CGL
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

# %% Fixtures


@pytest.fixture()
def topological_dim() -> int:
    return 2


@pytest.fixture()
def physical_dim():
    return 3


@pytest.fixture()
def coordinates_dim():
    return [5, 3, 2]


@pytest.fixture()
def zone_shape():
    return np.array([5, 3, 0])


@pytest.fixture()
def other_sample():
    return Sample()


@pytest.fixture()
def sample_with_scalar(sample):
    sample.add_global("test_scalar_1", np.random.randn())
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
    sample.add_tree(tree3d)
    return sample


@pytest.fixture()
def sample_from_muscat_mesh_dent3D(sample):
    from Muscat.IO.GmshReader import ReadGmsh
    from Muscat.TestData import GetTestDataPath

    filename = GetTestDataPath() + "dent3D.msh"
    mesh = ReadGmsh(filename)
    cgnsmesh = MeshToCGNS(mesh)
    from Muscat.Bridges.CGNSBridge import CGNSToMesh

    print(mesh)
    print(CGNSToMesh(cgnsmesh))
    sample.add_tree(cgnsmesh)
    return sample


@pytest.fixture()
def sample_with_tree_and_scalar(
    sample_with_tree: Sample,
):
    sample_with_tree.add_global("r", np.random.randn())
    sample_with_tree.add_global("test_scalar_1", np.random.randn())
    return sample_with_tree


@pytest.fixture()
def full_sample(sample_with_tree_and_scalar: Sample, tree3d):
    sample_with_tree_and_scalar.add_global("r", np.random.randn())
    sample_with_tree_and_scalar.add_global("test_scalar_1", np.random.randn())
    sample_with_tree_and_scalar.add_field(
        name="test_field_1", field=np.random.randn(3), location="CellCenter"
    )
    sample_with_tree_and_scalar.init_zone(
        zone_shape=np.array([[5, 3, 0]]), zone="test_field_1"
    )
    sample_with_tree_and_scalar.init_base(
        topological_dim=2, physical_dim=3, base="test_base_1"
    )
    sample_with_tree_and_scalar.init_tree(time=1.0)
    sample_with_tree_and_scalar.add_tree(tree=tree3d)
    return sample_with_tree_and_scalar


# %% Test


def test_check_names():
    _check_names("test name")
    _check_names(["test name", "test_name_2"])
    _check_names(None)
    _check_names([None, "short_name"])
    _check_names("a" * 31)
    with pytest.raises(ValueError):
        _check_names("test/name")
    with pytest.raises(ValueError):
        _check_names(["test/name"])
    with pytest.raises(ValueError):
        _check_names([r"test\/name"])
    with pytest.raises(ValueError):
        _check_names("a" * 33)
    with pytest.raises(ValueError):
        _check_names(["ok", "b" * 40])


def test_add_tree_invalid_name_length(sample: Sample, tree):
    invalid_tree = copy.deepcopy(tree)
    base_path = CGU.getPathsByTypeSet(invalid_tree, ["CGNSBase_t"])[0]
    base_node = CGU.getNodeByPath(invalid_tree, base_path)
    base_node[0] = "B" * 33

    with pytest.raises(ValueError):
        sample.add_tree(invalid_tree)


def test_read_index(tree, coordinates_dim):
    _read_index(tree, coordinates_dim)


def test_read_index_array(tree):
    _read_index_array(tree)


def test_read_index_range(tree, coordinates_dim):
    _read_index_range(tree, coordinates_dim)


@pytest.fixture()
def current_directory() -> Path:
    return Path(__file__).absolute().parent


# %% Tests


class Test_Sample:
    # -------------------------------------------------------------------------#
    def test___init__(self, current_directory):
        sample_path_1 = (
            current_directory / "dataset_cgns" / "data" / "test" / "sample_000000000"
        )
        sample_path_2 = (
            current_directory / "dataset_cgns" / "data" / "test" / "sample_000000001"
        )
        sample_path_3 = (
            current_directory / "dataset_cgns" / "data" / "test" / "sample_000000002"
        )
        sample_already_filled_1 = Sample(path=sample_path_1)
        sample_already_filled_2 = Sample(path=sample_path_2)
        sample_already_filled_3 = Sample(path=sample_path_3)
        assert sample_already_filled_1
        assert sample_already_filled_2
        assert sample_already_filled_3

    def test__init__unknown_directory(self, current_directory):
        sample_path = current_directory / "dataset" / "samples" / "sample_000000298"
        with pytest.raises(FileNotFoundError):
            Sample(path=sample_path)

    def test__init__file_provided(self, current_directory):
        sample_path = (
            current_directory
            / "dataset_cgns"
            / "data"
            / "test"
            / "sample_000000000"
            / "meshes"
            / "mesh_000000000.cgns"
        )
        with pytest.raises(FileExistsError):
            Sample(path=sample_path)

    def test__init__path(self, current_directory):
        sample_path = (
            current_directory / "dataset_cgns" / "data" / "test" / "sample_000000000"
        )
        Sample(path=sample_path)

    def test_copy(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.copy()

    # -------------------------------------------------------------------------#
    def test_set_default_base(self, sample: Sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, time=0.5)

        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}", 0.5)
        # check dims getters
        assert sample.get_topological_dim() == topological_dim
        assert sample.get_physical_dim() == physical_dim
        assert sample.resolve_base() == f"Base_{topological_dim}_{physical_dim}"
        assert sample.resolve_time() == 0.5
        assert sample.resolve_base("test") == "test"

        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}")  # already set
        assert sample.resolve_base() == f"Base_{topological_dim}_{physical_dim}"
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
        assert sample.get_zone() is None

        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base=base_name)
        # Look for the only zone in the default base
        assert sample.get_zone() is not None

        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base=base_name)
        # There is more than one zone in this base
        with pytest.raises(KeyError):
            sample.get_zone()

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
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base=base_name)

        sample.set_default_zone_base(zone_name, base_name, 0.5)
        # check dims getters
        assert sample.get_topological_dim() == topological_dim
        assert sample.get_physical_dim() == physical_dim
        assert sample.resolve_base() == base_name
        assert sample.resolve_time() == 0.5

        sample.set_default_base(base_name)  # already set
        assert sample.resolve_base() == base_name
        with pytest.raises(ValueError):
            sample.set_default_base("Unknown base name")

        assert sample.resolve_zone() == zone_name
        assert sample.resolve_time() == 0.5

        assert sample.get_zone() is not None
        sample.set_default_zone_base(zone_name, base_name)
        assert sample.resolve_zone() == zone_name
        with pytest.raises(ValueError):
            sample.set_default_zone_base("Unknown zone name", base_name)

    def test_set_default_time(self, sample: Sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, time=0.5)
        sample.init_base(topological_dim, physical_dim, "OK_name", time=1.5)

        assert sample.resolve_time() == 0.5
        sample.set_default_time(1.5)
        assert sample.resolve_time() == 1.5, "here"

        sample.set_default_time(1.5)  # already set
        assert sample.resolve_time() == 1.5
        with pytest.raises(ValueError):
            sample.set_default_time(2.5)

    # -------------------------------------------------------------------------#

    def test_show_tree(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.show_tree()

    def test_init_tree(self, sample: Sample):
        sample.init_tree()
        sample.init_tree(0.5)

    def test_get_tree_empty(self, sample: Sample):
        sample.get_tree()
        sample.get_tree()

    def test_get_tree(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.get_tree()
        sample_with_tree_and_scalar.get_tree(only_mesh=True)

    def test_set_trees_empty(self, sample, tree):
        sample.set_trees({0.0: tree})

    def test_set_trees(self, sample_with_tree: Sample, tree):
        with pytest.raises(KeyError):
            sample_with_tree.set_trees({0.0: tree})

    def test_add_tree_empty(self, sample_with_tree: Sample):
        with pytest.raises(ValueError):
            sample_with_tree.add_tree([])

    def test_add_tree(self, sample: Sample, tree):
        sample.add_tree(tree)
        sample.add_tree(tree)
        sample.add_tree(tree, time=0.2)

    def test_add_tree_in_place_true_mutates_input_tree(self, sample: Sample, tree):
        base_path = CGU.getPathsByTypeSet(tree, ["CGNSBase_t"])[0]
        base_node = CGU.getNodeByPath(tree, base_path)
        assert CGU.getValueByPath(base_node, "Time/TimeValues") is None

        sample.add_tree(tree, in_place=True)

        # With in_place=True, the same object may be reused internally.
        assert CGU.getValueByPath(base_node, "Time/TimeValues") is not None

    def test_add_tree_in_place_false_preserves_input_tree(self, sample: Sample, tree):
        base_path = CGU.getPathsByTypeSet(tree, ["CGNSBase_t"])[0]
        base_node = CGU.getNodeByPath(tree, base_path)
        assert CGU.getValueByPath(base_node, "Time/TimeValues") is None

        sample.add_tree(tree, in_place=False)

        # With in_place=False, the input tree is deep-copied first.
        assert CGU.getValueByPath(base_node, "Time/TimeValues") is None
        added_base_node = sample.get_base(time=0.0)
        assert CGU.getValueByPath(added_base_node, "Time/TimeValues") is not None

    def test_del_tree(self, sample, tree):
        sample.add_tree(tree)
        sample.add_tree(tree, time=0.2)

        assert isinstance(sample.del_tree(0.2), list)
        assert list(sample.data.keys()) == [0.0]

        assert isinstance(sample.del_tree(0.0), list)
        assert list(sample.data.keys()) == []

    def test_on_error_del_tree(self, sample, tree):
        with pytest.raises(KeyError):
            sample.del_tree(0.0)

        sample.add_tree(tree)
        sample.add_tree(tree, time=0.2)
        with pytest.raises(KeyError):
            sample.del_tree(0.7)

    # -------------------------------------------------------------------------#
    def test_init_base(self, sample: Sample, base_name, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, base_name)
        # check dims getters
        assert sample.get_topological_dim(base_name) == topological_dim
        assert sample.get_physical_dim(base_name) == physical_dim

    def test_del_base_existing_base(
        self, sample: Sample, base_name, topological_dim, physical_dim
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
        self, sample: Sample, base_name, topological_dim, physical_dim
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=1.0)
        with pytest.raises(KeyError):
            sample.del_base(base_name, time=2.0)
        with pytest.raises(KeyError):
            sample.del_base("unknown", time=1.0)

    def test_del_base_no_cgns_tree(self, sample):
        with pytest.raises(KeyError):
            sample.del_base("unknwon", 0.0)

    def test_init_base_no_base_name(
        self, sample: Sample, topological_dim, physical_dim
    ):
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

    def test_get_base_names(self, sample: Sample):
        assert sample.get_base_names() == []
        sample.init_base(3, 3, "base_name_1")
        sample.init_base(3, 3, "base_name_2")
        assert sample.get_base_names() == ["base_name_1", "base_name_2"]
        assert sample.get_base_names(full_path=True) == [
            "/base_name_1",
            "/base_name_2",
        ]
        # check dims getters
        assert sample.get_topological_dim("base_name_1") == 3
        assert sample.get_physical_dim("base_name_1") == 3
        assert sample.get_topological_dim("base_name_2") == 3
        assert sample.get_physical_dim("base_name_2") == 3

    def test_get_base(self, sample: Sample, base_name):
        sample.init_tree()
        assert sample.get_base() is None
        sample.init_base(3, 3, base_name)
        assert sample.get_base(base_name) is not None
        assert sample.get_base() is not None
        sample.init_base(3, 3, "other_base_name")
        assert sample.get_base(base_name) is not None
        assert sample.get_base(time=1.0) is None
        with pytest.raises(KeyError):
            sample.get_base()
        # check dims getters
        assert sample.get_topological_dim(base_name) == 3
        assert sample.get_physical_dim(base_name) == 3
        assert sample.get_topological_dim("other_base_name") == 3
        assert sample.get_physical_dim("other_base_name") == 3

    # -------------------------------------------------------------------------#
    def test_init_zone(self, sample: Sample, base_name, zone_name, zone_shape):
        with pytest.raises(KeyError):
            sample.init_zone(zone_shape, zone=zone_name, base=base_name)
        sample.init_base(3, 3, base_name)
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base=base_name)
        sample.init_zone(zone_shape, CGK.Unstructured_s, zone_name, base=base_name)
        # check dims getters
        assert sample.get_topological_dim(base_name) == 3
        assert sample.get_physical_dim(base_name) == 3

    def test_init_zone_defaults_names(self, sample: Sample, zone_shape):
        sample.init_base(3, 3)
        sample.init_zone(zone_shape)

    def test_del_zone_existing_zone(
        self, sample: Sample, base_name, zone_name, zone_shape
    ):
        topological_dim, physical_dim = 3, 3
        sample.init_base(topological_dim, physical_dim, base_name)

        second_zone_name = zone_name + "_2"
        sample.init_zone(zone_shape, CGK.Structured_s, zone_name, base=base_name)
        sample.init_zone(
            zone_shape, CGK.Unstructured_s, second_zone_name, base=base_name
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
            zone_shape, CGK.Structured_s, zone_name, base=base_name, time=0.2
        )
        sample.init_zone(
            zone_shape, CGK.Unstructured_s, "test", base=base_name, time=0.2
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
        self, sample: Sample, base_name, zone_shape, topological_dim, physical_dim
    ):
        sample.init_base(topological_dim, physical_dim, base_name, time=1.0)
        zone_name = "test123"
        sample.init_zone(
            zone_shape, CGK.Structured_s, zone_name, base=base_name, time=1.0
        )
        with pytest.raises(KeyError):
            sample.del_zone(zone_name, base_name, 2.0)
        with pytest.raises(KeyError):
            sample.del_zone("unknown", base_name, 1.0)

    def test_del_zone_no_cgns_tree(self, sample: Sample):
        sample.init_base(2, 3, "only_base")
        with pytest.raises(KeyError):
            sample.del_zone("unknwon", "only_base", 0.0)

    def test_has_zone(self, sample, base_name, zone_name):
        sample.init_base(3, 3, base_name)
        sample.init_zone(np.array([[5, 3, 0]]), zone=zone_name, base=base_name)
        sample.show_tree()
        assert sample.has_zone(zone_name, base_name)
        assert not sample.has_zone("not_present_zone_name", base_name)
        assert not sample.has_zone(zone_name, "not_present_base_name")
        assert not sample.has_zone("not_present_zone_name", "not_present_base_name")

    def test_get_zone_names(self, sample: Sample, base_name):
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            np.array([[5, 3, 0]]),
            zone="zone_name_1",
            base=base_name,
        )
        sample.init_zone(
            np.array([[5, 3, 0]]),
            zone="zone_name_2",
            base=base_name,
        )
        assert sample.get_zone_names(base_name) == [
            "zone_name_1",
            "zone_name_2",
        ]
        assert sorted(sample.get_zone_names(base_name, unique=True)) == sorted(
            ["zone_name_1", "zone_name_2"]
        )
        assert sample.get_zone_names(base_name, full_path=True) == [
            f"{base_name}/zone_name_1",
            f"{base_name}/zone_name_2",
        ]

    def test_get_zone_type(self, sample: Sample, zone_name, base_name):
        with pytest.raises(KeyError):
            sample.get_zone_type(zone_name, base_name)
        sample.init_tree()
        with pytest.raises(KeyError):
            sample.get_zone_type(zone_name, base_name)
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.get_zone_type(zone_name, base_name)
        sample.init_zone(np.array([[5, 3, 0]]), zone=zone_name, base=base_name)
        assert sample.get_zone_type(zone_name, base_name) == CGK.Unstructured_s

    def test_get_zone(self, sample: Sample, zone_name, base_name):
        assert sample.get_zone(zone_name, base_name) is None
        sample.init_base(3, 3, base_name)
        assert sample.get_zone(zone_name, base_name) is None
        sample.init_zone(np.array([[5, 3, 0]]), zone=zone_name, base=base_name)
        assert sample.get_zone() is not None
        assert sample.get_zone(zone_name, base_name) is not None
        sample.init_zone(
            np.array([[5, 3, 0]]),
            zone="other_zone_name",
            base=base_name,
        )
        assert sample.get_zone(zone_name, base_name) is not None
        with pytest.raises(KeyError):
            assert sample.get_zone() is not None

    # -------------------------------------------------------------------------#
    def test_get_scalar_names(self, sample: Sample):
        assert sample.get_global_names() == []

    def test_get_global_names_at_specific_time(self, sample: Sample):
        sample.add_global("g_t0", np.array([1.0]), time=0.0)
        sample.add_global("g_t1", np.array([2.0]), time=1.0)

        assert sample.get_global_names(time=0.0) == ["g_t0"]
        assert sample.get_global_names(time=1.0) == ["g_t1"]

    def test_add_global_string_and_update_existing(self, sample: Sample):
        sample.add_global("g_str", "abc")
        value = sample.get_global("g_str")
        assert isinstance(value, np.ndarray)
        assert value.tobytes().decode("ascii") == "abc"

        sample.add_global("g_str", np.array([7.0]))
        assert sample.get_global("g_str") == 7.0

    def test_get_global_names_excludes_time_arrays(self, sample: Sample):
        sample.init_base(2, 2, "Base_2_2", time=0.0)
        sample.add_global("kept_name", np.array([1.0]), time=0.0)
        names = sample.get_global_names(time=0.0)
        assert names == ["kept_name"]

    def test_get_scalar_empty(self, sample):
        assert sample.get_global("missing_scalar_name") is None

    def test_get_scalar(self, sample_with_scalar):
        assert sample_with_scalar.get_global("missing_scalar_name") is None
        assert sample_with_scalar.get_global("test_scalar_1") is not None
        assert isinstance(sample_with_scalar.get_global("test_scalar_1"), np.float64)

    def test_scalars_add_empty(self, sample_with_scalar):
        assert isinstance(sample_with_scalar.get_global("test_scalar_1"), float)

    def test_scalars_add(self, sample_with_scalar):
        sample_with_scalar.add_global("test_scalar_2", np.random.randn())

    def test_del_scalar_unknown_scalar(self, sample_with_scalar):
        with pytest.raises(KeyError):
            sample_with_scalar.del_global("non_existent_scalar")

    def test_del_scalar_no_scalar(self):
        sample = Sample()
        with pytest.raises(KeyError):
            sample.del_global("non_existent_scalar")

    def test_del_global(self, sample_with_scalar):
        assert len(sample_with_scalar.get_global_names()) == 1

        sample_with_scalar.add_global("test_scalar_2", np.random.randn(5))
        assert len(sample_with_scalar.get_global_names()) == 2

        scalar = sample_with_scalar.del_global("test_scalar_1")
        assert len(sample_with_scalar.get_global_names()) == 1
        assert scalar is not None
        assert isinstance(scalar, float)

        scalar = sample_with_scalar.del_global("test_scalar_2")
        assert len(sample_with_scalar.get_global_names()) == 0
        assert scalar is not None
        assert isinstance(scalar, np.ndarray)

    def test_add_feature(self, sample_with_tree3d):
        sample_with_tree3d.add_feature(
            feature_path="Global/test_scalar_2",
            feature=np.array([3.1415]),
        )

        sample_with_tree3d.add_feature(
            feature_path="Base_2_3/Zone/VertexFields/pressure",
            feature=np.arange(5),
        )

        sample_with_tree3d.add_feature(
            feature_path="Base_2_3/Zone/GridCoordinates",
            feature=np.zeros((5, 3)),
        )

    def test_del_feature(self, sample_with_scalar: Sample, sample_with_tree3d: Sample):
        sample_with_scalar.del_feature_by_path(path="Global/test_scalar_1")
        assert sample_with_scalar.get_all_features_identifiers_by_type("scalar") == []
        sample_with_tree3d.del_feature_by_path(
            "Base_2_3/Zone/VertexFields/test_node_field_1"
        )

    # -------------------------------------------------------------------------#
    def test_get_nodal_tags_empty(self, sample):
        assert sample.get_nodal_tags() == {}

    def test_get_nodal_tags(self, sample_with_tree, nodal_tags):
        assert np.all(sample_with_tree.get_nodal_tags()["tag"] == nodal_tags)

    def test_get_element_tags_empty(self, sample):
        assert sample.get_element_tags() == {}

    def test_get_element_tags(self, sample_from_muscat_mesh_dent3D: Sample):

        element_tags = sample_from_muscat_mesh_dent3D.get_element_tags()
        assert "Top" in element_tags
        assert np.all(element_tags["Top"] == np.arange(729, 753))
        assert "Vol" in element_tags
        assert np.all(element_tags["Vol"] == np.arange(9, 668))

    # -------------------------------------------------------------------------#
    def test_get_nodes_empty(self, sample):
        assert sample.get_nodes() is None

    def test_get_nodes(self, sample_with_tree, nodes):
        assert np.all(sample_with_tree.get_nodes() == nodes)

    def test_get_nodes3d(self, sample_with_tree3d, nodes3d):
        assert np.all(sample_with_tree3d.get_nodes() == nodes3d)

    def test_get_nodes_by_coordinate_name(self, sample_with_tree, nodes):
        assert np.all(sample_with_tree.get_nodes(name="CoordinateX") == nodes[:, 0])
        assert np.all(sample_with_tree.get_nodes(name="CoordinateY") == nodes[:, 1])
        sample_with_tree.get_nodes(name="CoordinateZ")

    def test_get_nodes_unknown_coordinate_name(self, sample_with_tree):
        with pytest.raises(ValueError):
            sample_with_tree.get_nodes(name="UnknownCoordinate")

    # def test_get_nodes_returns_none_without_gridcoordinates(
    #     self, sample: Sample, base_name: str, zone_name: str
    # ):
    #     sample.init_base(2, 2, base_name)
    #     sample.init_zone(np.array([3, 0, 0]), zone=zone_name, base=base_name)
    #     zone_node = sample.get_zone(zone=zone_name, base=base_name)
    #     gc_node = CGU.getNodeByPath(zone_node, "GridCoordinates")
    #     assert gc_node is not None
    #     CGU.nodeDelete(zone_node, gc_node)
    #    assert sample.get_nodes(zone=zone_name, base=base_name) is None

    def test_set_nodes(self, sample, nodes, zone_name, base_name):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.set_nodes(nodes, zone_name, base_name)
        sample.init_zone(np.array([len(nodes), 0, 0]), zone=zone_name, base=base_name)
        sample.set_nodes(nodes, zone_name, base_name)

    def test_set_nodes_replaces_existing_coordinates(
        self, sample: Sample, base_name: str, zone_name: str
    ):
        sample.init_base(3, 3, base_name)
        sample.init_zone(np.array([3, 0, 0]), zone=zone_name, base=base_name)

        nodes_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        nodes_b = np.array([[2.0, 2.0], [3.0, 2.0], [3.0, 3.0]])
        sample.set_nodes(nodes_a, zone=zone_name, base=base_name)
        sample.set_nodes(nodes_b, zone=zone_name, base=base_name)

        got = sample.get_nodes(zone=zone_name, base=base_name)
        assert np.allclose(got, nodes_b)

    # -------------------------------------------------------------------------#
    def test_get_elements_empty(self, sample: Sample):
        assert sample.get_elements() == {}

    def test_get_elements(self, sample_with_tree: Sample, triangles):
        assert list(sample_with_tree.get_elements().keys()) == ["TRI_3"]
        print(f"{triangles=}")
        print(f"{sample_with_tree.get_elements()=}")
        assert np.all(sample_with_tree.get_elements()["TRI_3"] == triangles)

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
        sample.init_base(topological_dim=1, physical_dim=2, base="Base_1_2", time=-0.1)
        sample.init_base(topological_dim=2, physical_dim=2, base="Base_2_2", time=-0.1)
        sample.init_base(topological_dim=1, physical_dim=3, base="Base_1_3", time=1.0)
        sample.init_base(topological_dim=3, physical_dim=3, base="Base_3_3", time=1.0)
        sample.init_zone(
            zone_shape=np.array([[5, 3, 0]]),
            zone="Zone_1",
            base="Base_1_2",
            time=-0.1,
        )
        sample.init_zone(
            zone_shape=np.array([[5, 3, 0]]),
            zone="Zone_2",
            base="Base_1_2",
            time=-0.1,
        )
        sample.init_zone(
            zone_shape=np.array([[5, 3, 0]]),
            zone="Zone_1",
            base="Base_2_2",
            time=-0.1,
        )
        sample.init_zone(
            zone_shape=np.array([[5, 3, 0]]),
            zone="Zone_2",
            base="Base_2_2",
            time=-0.1,
        )
        sample.init_zone(
            zone_shape=np.array([[5, 3, 0]]),
            zone="Zone_1",
            base="Base_1_3",
            time=1.0,
        )
        sample.init_zone(
            zone_shape=np.array([[5, 3, 0]]),
            zone="Zone_2",
            base="Base_1_3",
            time=1.0,
        )
        sample.init_zone(
            zone_shape=np.array([[5, 3, 0]]),
            zone="Zone_1",
            base="Base_3_3",
            time=1.0,
        )
        sample.init_zone(
            zone_shape=np.array([[5, 3, 0]]),
            zone="Zone_2",
            base="Base_3_3",
            time=1.0,
        )
        sample.add_field(
            name="vertex_Zone_1_Base_1_2_t_m0.1",
            field=np.random.randn(5),
            location="Vertex",
            zone="Zone_1",
            base="Base_1_2",
            time=-0.1,
        )
        sample.add_field(
            name="cell_Zone_1_Base_1_2_t_m0.1",
            field=np.random.randn(3),
            location="CellCenter",
            zone="Zone_1",
            base="Base_1_2",
            time=-0.1,
        )
        sample.add_field(
            name="vertex_Zone_2_Base_1_2_t_m0.1",
            field=np.random.randn(5),
            location="Vertex",
            zone="Zone_2",
            base="Base_1_2",
            time=-0.1,
        )
        sample.add_field(
            name="cell_Zone_2_Base_1_2_t_m0.1",
            field=np.random.randn(3),
            location="CellCenter",
            zone="Zone_2",
            base="Base_1_2",
            time=-0.1,
        )
        sample.add_field(
            name="vertex_Zone_1_Base_2_2_t_m0.1",
            field=np.random.randn(5),
            location="Vertex",
            zone="Zone_1",
            base="Base_2_2",
            time=-0.1,
        )
        sample.add_field(
            name="cell_Zone_1_Base_2_2_t_m0.1",
            field=np.random.randn(3),
            location="CellCenter",
            zone="Zone_1",
            base="Base_2_2",
            time=-0.1,
        )
        sample.add_field(
            name="vertex_Zone_2_Base_2_2_t_m0.1",
            field=np.random.randn(5),
            location="Vertex",
            zone="Zone_2",
            base="Base_2_2",
            time=-0.1,
        )
        sample.add_field(
            name="cell_Zone_2_Base_2_2_t_m0.1",
            field=np.random.randn(3),
            location="CellCenter",
            zone="Zone_2",
            base="Base_2_2",
            time=-0.1,
        )
        sample.add_field(
            name="vertex_Zone_1_Base_1_3_t_1.0",
            field=np.random.randn(5),
            location="Vertex",
            zone="Zone_1",
            base="Base_1_3",
            time=1.0,
        )
        sample.add_field(
            name="cell_Zone_1_Base_1_3_t_1.0",
            field=np.random.randn(3),
            location="CellCenter",
            zone="Zone_1",
            base="Base_1_3",
            time=1.0,
        )
        sample.add_field(
            name="vertex_Zone_2_Base_1_3_t_1.0",
            field=np.random.randn(5),
            location="Vertex",
            zone="Zone_2",
            base="Base_1_3",
            time=1.0,
        )
        sample.add_field(
            name="cell_Zone_2_Base_1_3_t_1.0",
            field=np.random.randn(3),
            location="CellCenter",
            zone="Zone_2",
            base="Base_1_3",
            time=1.0,
        )
        sample.add_field(
            name="vertex_Zone_1_Base_3_3_t_1.0",
            field=np.random.randn(5),
            location="Vertex",
            zone="Zone_1",
            base="Base_3_3",
            time=1.0,
        )
        sample.add_field(
            name="cell_Zone_1_Base_3_3_t_1.0",
            field=np.random.randn(3),
            location="CellCenter",
            zone="Zone_1",
            base="Base_3_3",
            time=1.0,
        )
        sample.add_field(
            name="vertex_Zone_2_Base_3_3_t_1.0",
            field=np.random.randn(5),
            location="Vertex",
            zone="Zone_2",
            base="Base_3_3",
            time=1.0,
        )
        sample.add_field(
            name="cell_Zone_2_Base_3_3_t_1.0",
            field=np.random.randn(3),
            location="CellCenter",
            zone="Zone_2",
            base="Base_3_3",
            time=1.0,
        )
        expected_field_names = [
            "vertex_Zone_1_Base_1_2_t_m0.1",
            "cell_Zone_1_Base_1_2_t_m0.1",
            "vertex_Zone_2_Base_1_2_t_m0.1",
            "cell_Zone_2_Base_1_2_t_m0.1",
            "vertex_Zone_1_Base_2_2_t_m0.1",
            "cell_Zone_1_Base_2_2_t_m0.1",
            "vertex_Zone_2_Base_2_2_t_m0.1",
            "cell_Zone_2_Base_2_2_t_m0.1",
            "vertex_Zone_1_Base_1_3_t_1.0",
            "cell_Zone_1_Base_1_3_t_1.0",
            "vertex_Zone_2_Base_1_3_t_1.0",
            "cell_Zone_2_Base_1_3_t_1.0",
            "vertex_Zone_1_Base_3_3_t_1.0",
            "cell_Zone_1_Base_3_3_t_1.0",
            "vertex_Zone_2_Base_3_3_t_1.0",
            "cell_Zone_2_Base_3_3_t_1.0",
        ]
        assert sample.get_field_names() == sorted(set(expected_field_names))
        sample.add_field(
            name="field_of_ints",
            field=np.arange(5),
            zone="Zone_2",
            base="Base_3_3",
            time=1.0,
        )
        field = sample.get_field(
            "field_of_ints", zone="Zone_2", base="Base_3_3", time=1.0
        )
        assert field.dtype == np.float64

    def test_get_field_empty(self, sample: Sample):
        assert sample.get_field("missing_field_name") is None
        assert sample.get_field("missing_field_name", location="CellCenter") is None

    def test_get_field(self, sample_with_tree):
        assert sample_with_tree.get_field("missing_field") is None
        assert sample_with_tree.get_field("test_node_field_1").shape == (5,)
        assert sample_with_tree.get_field(
            "test_elem_field_1", location="CellCenter"
        ).shape == (3,)

    def test_get_field_from_user_defined_data_lowercase_gridlocation(
        self, sample: Sample, base_name: str, zone_name: str
    ):
        sample.init_base(2, 2, base_name)
        sample.init_zone(np.array([3, 0, 0]), zone=zone_name, base=base_name)

        zone_node = sample.get_zone(zone=zone_name, base=base_name)
        udd = CGL.newUserDefinedData(zone_node, "ItgPointData")
        CGL.newDataArray(
            udd,
            "gridlocation",
            value=np.frombuffer("Vertex".encode("ascii"), dtype="S1"),
        )
        CGL.newDataArray(udd, "my_udd_field", value=np.array([1.0, 2.0, 3.0]))

        assert (
            sample.get_field("my_udd_field", zone=zone_name, base=base_name) is not None
        )
        assert "my_udd_field" in sample.get_field_names(
            zone=zone_name, base=base_name, location="Vertex"
        )

    def test_add_field_vertex(self, sample: Sample, vertex_field, zone_name, base_name):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.add_field(
                name="test_node_field_2",
                field=vertex_field,
                zone=zone_name,
                base=base_name,
            )
        with pytest.raises(ValueError):
            sample.add_field(
                name="test_node_field_2",
                field=np.zeros((5, 2)),
                zone=zone_name,
                base=base_name,
            )
        sample.init_zone(np.array([[5, 3, 0]]), zone=zone_name, base=base_name)
        sample.add_field(
            name="test_node_field_2",
            field=vertex_field,
            zone=zone_name,
            base=base_name,
        )
        with pytest.raises(ValueError):
            sample.add_field(
                name="test_node_field_2",
                field=np.zeros((13)),
                zone=zone_name,
                base=base_name,
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
                zone=zone_name,
                base=base_name,
            )
        sample.init_zone(np.array([[5, 3, 0]]), zone=zone_name, base=base_name)
        sample.add_field(
            name="test_elem_field_2",
            location="CellCenter",
            field=cell_center_field,
            zone=zone_name,
            base=base_name,
        )
        with pytest.raises(ValueError):
            sample.add_field(
                name="test_elem_field_2",
                location="CellCenter",
                field=np.zeros((13)),
                zone=zone_name,
                base=base_name,
            )

    def test_add_field_vertex_already_present(
        self, sample_with_tree: Sample, vertex_field
    ):
        # with pytest.raises(KeyError):
        sample_with_tree.show_tree()
        sample_with_tree.add_field(
            name="test_node_field_1",
            field=vertex_field,
            zone="Zone",
            base="Base_2_2",
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
            zone="Zone",
            base="Base_2_2",
        )

    def test_del_field_existing(self, sample_with_tree):
        with pytest.raises(KeyError):
            sample_with_tree.del_field(
                name="unknown",
                location="CellCenter",
                zone="Zone",
                base="Base_2_2",
            )
        with pytest.raises(KeyError):
            sample_with_tree.del_field(
                name="unknown",
                location="CellCenter",
                zone="unknown_zone",
                base="Base_2_2",
            )

    def test_del_field_nonexistent(self, base_name):
        sample = Sample()
        sample.init_base(2, 2, base_name)
        with pytest.raises(KeyError):
            sample.del_field(
                name="unknown",
                location="CellCenter",
                zone="unknown_zone",
                base=base_name,
            )

    def test_del_field_in_zone(self, zone_name, base_name, cell_center_field):
        sample = Sample()
        sample.init_base(3, 3, base_name)
        sample.init_zone(np.array([[5, 3, 0]]), zone=zone_name, base=base_name)
        sample.add_field(
            name="test_elem_field_1",
            field=cell_center_field,
            location="CellCenter",
            zone=zone_name,
            base=base_name,
        )

        # Add field 'test_elem_field_2'
        sample.add_field(
            name="test_elem_field_2",
            field=cell_center_field,
            location="CellCenter",
            zone=zone_name,
            base=base_name,
        )
        assert isinstance(
            sample.get_field(
                name="test_elem_field_2",
                location="CellCenter",
                zone=zone_name,
                base=base_name,
            ),
            np.ndarray,
        )

        # Del field 'test_elem_field_2'
        new_tree = sample.del_field(
            name="test_elem_field_2",
            location="CellCenter",
            zone=zone_name,
            base=base_name,
        )

        # Testing new tree on field 'test_elem_field_2'
        new_sample = Sample()
        new_sample.add_tree(new_tree)

        assert (
            new_sample.get_field(
                name="test_elem_field_2",
                location="CellCenter",
                zone=zone_name,
                base=base_name,
            )
            is None
        )
        fields = new_sample.get_field_names(
            location="CellCenter", zone=zone_name, base=base_name
        )

        assert "test_elem_field_2" not in fields
        assert "test_elem_field_1" in fields

        # Del field 'test_elem_field_1'
        new_tree = sample.del_field(
            name="test_elem_field_1",
            location="CellCenter",
            zone=zone_name,
            base=base_name,
        )

        # Testing new tree on field 'test_elem_field_1'
        new_sample = Sample()
        new_sample.add_tree(new_tree)

        assert (
            new_sample.get_field(
                name="test_elem_field_1",
                location="CellCenter",
                zone=zone_name,
                base=base_name,
            )
            is None
        )
        fields = new_sample.get_field_names(
            location="CellCenter", zone=zone_name, base=base_name
        )
        assert len(fields) == 0

    # -------------------------------------------------------------------------#
    def test_get_feature_by_path(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.get_feature_by_path(
            "Base_2_2/Zone/Elements_TRI_3/ElementConnectivity", 0.0
        )

    def test_get_feature_from_identifier(self, sample_with_tree_and_scalar):
        sample_with_tree_and_scalar.get_feature_by_path(
            "Base_2_2/Zone/GridCoordinates/CoordinateX"
        ) is not None
        print(sample_with_tree_and_scalar.show_tree())
        assert (
            sample_with_tree_and_scalar.get_feature_by_path(
                "Base_2_2/Zone/VertexFields/test_node_field_1"
            )
            is not None
        )
        assert (
            sample_with_tree_and_scalar.get_feature_by_path("Global/test_scalar_1")
            is not None
        )

    def test_update_value_by_path(self, sample_with_tree):
        path = "Base_2_2/Zone/VertexFields/test_node_field_1"
        new_field = np.linspace(0.0, 1.0, 5)

        sample_with_tree.update_value_by_path(path, new_field)

        assert np.allclose(sample_with_tree.get_feature_by_path(path), new_field)

    def test_update_value_by_path_warns_when_array_shape_differs(
        self, sample_with_tree, caplog, monkeypatch
    ):
        path = "Base_2_2/Zone/VertexFields/test_node_field_1"
        base_node = sample_with_tree.get_base("Base_2_2")
        node = CGU.getNodeByPath(base_node, "/Zone/VertexFields/test_node_field_1")
        original_get_value = CGU.getValue
        calls = 0

        def get_value_with_different_shape_once(current_node):
            nonlocal calls
            if current_node is node:
                calls += 1
                if calls == 1:
                    return np.zeros(5)
                return np.zeros((1, 5))
            return original_get_value(current_node)

        monkeypatch.setattr(CGU, "getValue", get_value_with_different_shape_once)

        with caplog.at_level("WARNING", logger="plaid.containers.sample"):
            sample_with_tree.update_value_by_path(path, np.linspace(0.0, 1.0, 5))

        assert "incomming data has shape" in caplog.text

    def test_update_value_by_path_at_specific_time(self, sample, tree):
        path = "Base_2_2/Zone/VertexFields/test_node_field_1"
        time_0_field = np.linspace(0.0, 1.0, 5)
        time_1_field = np.linspace(10.0, 14.0, 5)

        sample.add_tree(copy.deepcopy(tree), time=0.0)
        sample.add_tree(copy.deepcopy(tree), time=1.0)
        sample.update_value_by_path(path, time_0_field, time=0.0)
        sample.update_value_by_path(path, time_1_field, time=1.0)

        assert np.allclose(sample.get_feature_by_path(path, time=0.0), time_0_field)
        assert np.allclose(sample.get_feature_by_path(path, time=1.0), time_1_field)

    def test_update_value_by_path_rejects_unknown_path(self, sample_with_tree):
        with pytest.raises(KeyError, match="There is no node at path"):
            sample_with_tree.update_value_by_path(
                "Base_2_2/Zone/VertexFields/missing_field",
                np.zeros(5),
            )

    def test_update_value_by_path_rejects_incompatible_shape(self, sample_with_tree):
        with pytest.raises(ValueError, match="incomming data has shape"):
            sample_with_tree.update_value_by_path(
                "Base_2_2/Zone/VertexFields/test_node_field_1",
                np.zeros(6),
            )

    def test_update_features_by_path(self, sample_with_tree_and_scalar):
        original_value = sample_with_tree_and_scalar.get_feature_by_path(
            "Global/test_scalar_1"
        )

        updated_sample = sample_with_tree_and_scalar.update_features_by_path(
            "Global/test_scalar_1",
            features=3.141592,
            in_place=False,
        )

        assert updated_sample is not sample_with_tree_and_scalar
        assert updated_sample.get_feature_by_path("Global/test_scalar_1") == 3.141592
        assert (
            sample_with_tree_and_scalar.get_feature_by_path("Global/test_scalar_1")
            == original_value
        )

    def test_update_features_by_path_in_place(self, sample_with_tree_and_scalar):
        updated_sample = sample_with_tree_and_scalar.update_features_by_path(
            "Global/test_scalar_1",
            features=2.718281,
            in_place=True,
        )

        assert updated_sample is sample_with_tree_and_scalar
        assert sample_with_tree_and_scalar.get_feature_by_path(
            "Global/test_scalar_1"
        ) == pytest.approx(2.718281)

    def test_update_features_by_path_updates_multiple_features(
        self, sample_with_tree_and_scalar
    ):
        new_field = np.linspace(10.0, 14.0, 5)

        updated_sample = sample_with_tree_and_scalar.update_features_by_path(
            [
                "Global/test_scalar_1",
                "Base_2_2/Zone/VertexFields/test_node_field_1",
            ],
            [42.0, new_field],
            in_place=False,
        )

        assert updated_sample.get_feature_by_path("Global/test_scalar_1") == 42.0
        assert np.allclose(
            updated_sample.get_feature_by_path(
                "Base_2_2/Zone/VertexFields/test_node_field_1"
            ),
            new_field,
        )

    def test_update_features_by_path_rejects_mismatched_lengths(
        self, sample_with_tree_and_scalar
    ):
        with pytest.raises(AssertionError):
            sample_with_tree_and_scalar.update_features_by_path(
                ["Global/test_scalar_1", "Global/r"],
                [1.0],
            )

    def test_get_all_features_by_type(self, sample_with_tree_and_scalar):

        feat_paths = sample_with_tree_and_scalar.get_all_features_by_type("field")
        assert "Base_2_2/Zone/VertexFields/big_node_field" in feat_paths
        assert "Base_2_2/Zone/VertexFields/test_node_field_1" in feat_paths
        assert "Base_2_2/Zone/VertexFields/OriginalIds" in feat_paths
        assert "Base_2_2/Zone/CellCenterFields/test_elem_field_1" in feat_paths
        assert "Base_2_2/Zone/CellCenterFields/OriginalIds" in feat_paths

        feat_paths = sample_with_tree_and_scalar.get_all_features_by_type("global")
        assert "Global/r" in feat_paths
        assert "Global/test_scalar_1" in feat_paths

        feat_paths = sample_with_tree_and_scalar.get_all_features_by_type("coordinate")
        assert "Base_2_2/Zone/GridCoordinates/CoordinateX" in feat_paths
        assert "Base_2_2/Zone/GridCoordinates/CoordinateY" in feat_paths

        feat_paths = sample_with_tree_and_scalar.get_all_features_by_type(
            "boundary_condition"
        )
        assert "Base_2_2/Zone/ZoneBC/tag" in feat_paths
        assert "Base_2_2/Zone/ZoneBC/tag/PointList" in feat_paths
        assert "Base_2_2/Zone/ZoneBC/tag/GridLocation" in feat_paths

        feat_paths = sample_with_tree_and_scalar.get_all_features_by_type("elements")
        assert "Base_2_2/Zone/Elements_TRI_3/ElementRange" in feat_paths
        assert "Base_2_2/Zone/Elements_TRI_3/ElementConnectivity" in feat_paths

    def test_get_all_features_identifiers_by_type(self, sample_with_tree_and_scalar):
        feat_ids = sample_with_tree_and_scalar.get_all_features_identifiers_by_type(
            "scalar"
        )
        assert len(feat_ids) == 2
        assert "r" in feat_ids
        assert "test_scalar_1" in feat_ids

        feat_ids = sample_with_tree_and_scalar.get_all_features_identifiers_by_type(
            "field"
        )
        assert len(feat_ids) == 4

        feat_ids = sample_with_tree_and_scalar.get_all_features_identifiers_by_type(
            "nodes"
        )
        assert len(feat_ids) == 2

    # -------------------------------------------------------------------------#
    def test_save(self, sample_with_tree_and_scalar, tmp_path):
        save_dir = tmp_path / "test_dir"
        sample_with_tree_and_scalar.save_to_dir(save_dir)
        assert save_dir.is_dir()
        with pytest.raises(ValueError):
            sample_with_tree_and_scalar.save_to_dir(save_dir, memory_safe=False)
        sample_with_tree_and_scalar.save_to_dir(save_dir, overwrite=True)
        sample_with_tree_and_scalar.save_to_dir(
            save_dir, overwrite=True, memory_safe=True
        )

    def test_load_from_dir(self, sample_with_tree_and_scalar, tmp_path):
        save_dir = tmp_path / "test_dir"
        sample_with_tree_and_scalar.save_to_dir(save_dir)
        new_sample = Sample.load_from_dir(save_dir)
        assert CGU.checkSameTree(
            sample_with_tree_and_scalar.get_tree(),
            new_sample.get_tree(),
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
