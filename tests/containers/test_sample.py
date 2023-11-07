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
from Muscat.Containers import UnstructuredMeshCreationTools as UMCT

from plaid.containers.sample import Sample, show_cgns_tree

# %% Fixtures


@pytest.fixture()
def base_name():
    return 'TestBaseName'


@pytest.fixture()
def topological_dim():
    return 2


@pytest.fixture()
def physical_dim():
    return 3


@pytest.fixture()
def zone_name():
    return 'TestZoneName'


@pytest.fixture()
def zone_shape():
    return np.array([5, 3, 0])


@pytest.fixture()
def sample():
    return Sample()


@pytest.fixture()
def other_sample():
    return Sample()


@pytest.fixture()
def sample_with_scalar(sample):
    sample.add_scalar('test_scalar_1', np.random.randn())
    return sample


@pytest.fixture()
def sample_with_time_series(sample):
    sample.add_time_series(
        'test_time_series_1',
        np.arange(
            111,
            dtype=float),
        np.random.randn(111))
    return sample


@pytest.fixture()
def nodes():
    return np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.5],
    ])


@pytest.fixture()
def nodes3d():
    return np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 1.5, 1.0],
    ])


@pytest.fixture()
def triangles():
    return np.array([
        [0, 1, 2],
        [0, 2, 3],
        [2, 4, 3],
    ])


@pytest.fixture()
def vertex_field():
    return np.random.randn(5)


@pytest.fixture()
def cell_center_field():
    return np.random.randn(3)


@pytest.fixture()
def tree(nodes, triangles, vertex_field, cell_center_field):
    BTMesh = UMCT.CreateMeshOfTriangles(nodes, triangles)
    BTMesh.nodeFields['test_node_field_1'] = vertex_field
    BTMesh.nodeFields['big_node_field'] = np.random.randn(50)
    BTMesh.elemFields['test_elem_field_1'] = cell_center_field
    tree = MeshToCGNS(BTMesh)
    return tree


@pytest.fixture()
def tree3d(nodes3d, triangles, vertex_field, cell_center_field):
    BTMesh = UMCT.CreateMeshOfTriangles(nodes3d, triangles)
    BTMesh.nodeFields['test_node_field_1'] = vertex_field
    BTMesh.nodeFields['big_node_field'] = np.random.randn(50)
    BTMesh.elemFields['test_elem_field_1'] = cell_center_field
    tree = MeshToCGNS(BTMesh)
    return tree


@pytest.fixture()
def sample_with_tree(sample, tree):
    sample.add_tree(tree)
    return sample


@pytest.fixture()
def sample_with_tree3d(sample, tree3d):
    sample.add_tree(tree3d)
    return sample


@pytest.fixture()
def sample_with_tree_and_scalar_and_time_series(sample_with_tree, ):
    sample_with_tree.add_scalar('r', np.random.randn())
    sample_with_tree.add_scalar('test_scalar_1', np.random.randn())
    sample_with_tree.add_time_series(
        'test_time_series_1', np.arange(
            111, dtype=float), np.random.randn(111))
    return sample_with_tree

# %% Tests


def test_show_cgns_tree(tree):
    show_cgns_tree(tree)


def test_show_cgns_tree_not_a_list():
    with pytest.raises(TypeError):
        show_cgns_tree({1: 2})


@pytest.fixture()
def current_directory():
    return os.path.dirname(os.path.abspath(__file__))


class Test_Sample():

    # -------------------------------------------------------------------------#
    def test___init__(self, current_directory):
        dataset_path_1 = os.path.join(
            current_directory,
            "dataset",
            "samples",
            "sample_000000000")
        dataset_path_2 = os.path.join(
            current_directory,
            "dataset",
            "samples",
            "sample_000000001")
        dataset_path_3 = os.path.join(
            current_directory,
            "dataset",
            "samples",
            "sample_000000002")
        sample_already_filled_1 = Sample(dataset_path_1)
        sample_already_filled_2 = Sample(dataset_path_2)
        sample_already_filled_3 = Sample(dataset_path_3)
        assert sample_already_filled_1._meshes is not None and sample_already_filled_1._scalars is not None
        assert sample_already_filled_2._meshes is not None and sample_already_filled_2._scalars is not None
        assert sample_already_filled_3._meshes is not None and sample_already_filled_3._scalars is not None

    def test__init__unknown_directory(self, current_directory):
        dataset_path = os.path.join(
            current_directory,
            "dataset",
            "samples",
            "sample_000000298")
        with pytest.raises(FileNotFoundError):
            Sample(dataset_path)

    def test__init__file_provided(self, current_directory):
        dataset_path = os.path.join(
            current_directory,
            "dataset",
            "samples",
            "sample_000067392")
        with pytest.raises(FileExistsError):
            Sample(dataset_path)

    # -------------------------------------------------------------------------#
    def test_set_default_base(
            self, sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, time=0.5)

        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}", 0.5)
        # check dims getters
        assert sample.get_topological_dim() == topological_dim
        assert sample.get_physical_dim() == physical_dim
        assert sample.get_base_assignment() == f"Base_{topological_dim}_{physical_dim}"
        assert sample.get_time_assignment() == 0.5
        assert sample.get_base_assignment("test") == "test"

        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}") # already set
        sample.set_default_base(None) # will not assign to None
        assert sample.get_base_assignment() == f"Base_{topological_dim}_{physical_dim}"
        with pytest.raises(ValueError):
            sample.set_default_base(f"Unknown base name")

    def test_set_default_zone_with_default_base(
            self, sample, topological_dim, physical_dim, base_name, zone_name, zone_shape):
        sample.init_base(topological_dim, physical_dim, base_name, time=0.5)
        sample.set_default_base(base_name)
        # No zone provided
        assert sample.get_zone() is None

        sample.init_zone(
            zone_shape,
            CGK.Structured_s,
            zone_name,
            base_name=base_name)
        # Look for the only zone in the default base
        assert sample.get_zone() is not None

        sample.init_zone(
            zone_shape,
            CGK.Structured_s,
            zone_name,
            base_name=base_name)
        # There is more than one zone in this base
        with pytest.raises(KeyError):
            sample.get_zone()

    def test_set_default_zone(
            self, sample, topological_dim, physical_dim, base_name, zone_name, zone_shape):
        sample.init_base(topological_dim, physical_dim, base_name, time=0.5)
        sample.init_zone(
            zone_shape,
            CGK.Structured_s,
            zone_name,
            base_name=base_name)

        sample.set_default_base_zone(base_name, zone_name, 0.5)
        # check dims getters
        assert sample.get_topological_dim() == topological_dim
        assert sample.get_physical_dim() == physical_dim
        assert sample.get_base_assignment() == base_name
        assert sample.get_time_assignment() == 0.5

        sample.set_default_base(base_name) # already set
        sample.set_default_base(None) # will not assign to None
        assert sample.get_base_assignment() == base_name
        with pytest.raises(ValueError):
            sample.set_default_base(f"Unknown base name")

        assert sample.get_zone_assignment() == zone_name
        assert sample.get_time_assignment() == 0.5

        assert sample.get_zone() is not None
        sample.set_default_base_zone(base_name, zone_name)
        sample.set_default_base_zone(base_name, None) # will not assign to None
        assert sample.get_zone_assignment() == zone_name
        with pytest.raises(ValueError):
            sample.set_default_base_zone(base_name, f"Unknown zone name")

    def test_set_default_time(
            self, sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, time=0.5)
        sample.init_base(topological_dim, physical_dim, "OK_name", time=1.5)


        assert sample.get_time_assignment() == 0.5
        sample.set_default_time(1.5)
        assert sample.get_time_assignment() == 1.5, "here"

        sample.set_default_time(1.5) # already set
        sample.set_default_time(None) # will not assign to None
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

    def test_set_meshes_empty(self, sample, tree):
        sample.set_meshes(tree)

    def test_set_meshes(self, sample_with_tree, tree):
        with pytest.raises(KeyError):
            sample_with_tree.set_meshes(tree)

    def test_add_tree_empty(self, sample_with_tree, tree):
        pass

    def test_add_tree(self, sample_with_tree, tree):
        sample_with_tree.add_tree(tree)
        sample_with_tree.add_tree(tree, time=0.2)

    # -------------------------------------------------------------------------#
    def test_init_base(self, sample, base_name, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim, base_name)
        # check dims getters
        assert sample.get_topological_dim(base_name) == topological_dim
        assert sample.get_physical_dim(base_name) == physical_dim

    def test_init_base_no_base_name(
            self, sample, topological_dim, physical_dim):
        sample.init_base(topological_dim, physical_dim)

        # check dims getters
        assert sample.get_topological_dim(f"Base_{topological_dim}_{physical_dim}") == topological_dim
        assert sample.get_physical_dim(f"Base_{topological_dim}_{physical_dim}") == physical_dim

        # check setting default base
        sample.set_default_base(f"Base_{topological_dim}_{physical_dim}")
        assert sample.get_topological_dim() == topological_dim
        assert sample.get_physical_dim() == physical_dim

    def test_get_base_names(self, sample):
        assert (sample.get_base_names() == [])
        sample.init_base(3, 3, 'base_name_1')
        sample.init_base(3, 3, 'base_name_2')
        assert (sample.get_base_names() == ['base_name_1', 'base_name_2'])
        assert (
            sample.get_base_names(
                full_path=True) == [
                '/base_name_1',
                '/base_name_2'])
        # check dims getters
        assert sample.get_topological_dim('base_name_1') == 3
        assert sample.get_physical_dim('base_name_1') == 3
        assert sample.get_topological_dim('base_name_2') == 3
        assert sample.get_physical_dim('base_name_2') == 3

    def test_get_base(self, sample, base_name):
        sample.init_tree()
        assert (sample.get_base() is None)
        sample.init_base(3, 3, base_name)
        assert (sample.get_base(base_name) is not None)
        assert (sample.get_base() is not None)
        sample.init_base(3, 3, 'other_base_name')
        assert (sample.get_base(base_name) is not None)
        with pytest.raises(KeyError):
            sample.get_base()
        # check dims getters
        assert sample.get_topological_dim(base_name) == 3
        assert sample.get_physical_dim(base_name) == 3
        assert sample.get_topological_dim('other_base_name') == 3
        assert sample.get_physical_dim('other_base_name') == 3

    # -------------------------------------------------------------------------#
    def test_init_zone(self, sample, base_name, zone_name, zone_shape):
        with pytest.raises(KeyError):
            sample.init_zone(zone_shape, zone_name=zone_name, base_name=base_name)
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            zone_shape,
            CGK.Structured_s,
            zone_name,
            base_name=base_name)
        sample.init_zone(
            zone_shape,
            CGK.Unstructured_s,
            zone_name,
            base_name=base_name)
        # check dims getters
        assert sample.get_topological_dim(base_name) == 3
        assert sample.get_physical_dim(base_name) == 3

    def test_init_zone_defaults_names(self, sample, zone_shape):
        sample.init_base(3, 3)
        sample.init_zone(zone_shape)

    def test_has_zone(self, sample, base_name, zone_name):
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name=zone_name,
            base_name=base_name)
        sample.show_tree()
        assert (sample.has_zone(zone_name, base_name))
        assert (~sample.has_zone('not_present_zone_name', base_name))
        assert (~sample.has_zone(zone_name, 'not_present_base_name'))
        assert (
            ~sample.has_zone(
                'not_present_zone_name',
                'not_present_base_name'))

    def test_get_zone_names(self, sample, base_name):
        sample.init_base(3, 3, base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name='zone_name_1',
            base_name=base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name='zone_name_2',
            base_name=base_name)
        assert (
            sample.get_zone_names(base_name) == [
                'zone_name_1',
                'zone_name_2'])
        assert (
            sample.get_zone_names(
                base_name,
                full_path=True) == [
                f'{base_name}/zone_name_1',
                f'{base_name}/zone_name_2'])

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
            np.random.randint(0, 10, size=3),
            zone_name=zone_name,
            base_name=base_name)
        assert (
            sample.get_zone_type(
                zone_name,
                base_name) == CGK.Unstructured_s)

    def test_get_zone(self, sample, zone_name, base_name):
        assert (sample.get_zone(zone_name, base_name) is None)
        sample.init_base(3, 3, base_name)
        assert (sample.get_zone(zone_name, base_name) is None)
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name=zone_name,
            base_name=base_name)
        assert (sample.get_zone() is not None)
        assert (sample.get_zone(zone_name, base_name) is not None)
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name='other_zone_name',
            base_name=base_name)
        assert (sample.get_zone(zone_name, base_name) is not None)
        with pytest.raises(KeyError):
            assert (sample.get_zone() is not None)

    # -------------------------------------------------------------------------#
    def test_get_scalar_names(self, sample):
        assert (sample.get_scalar_names() == [])

    def test_get_scalar_empty(self, sample):
        assert (sample.get_scalar('missing_scalar_name') is None)

    def test_get_scalar(self, sample_with_scalar):
        assert (sample_with_scalar.get_scalar('missing_scalar_name') is None)
        assert (sample_with_scalar.get_scalar('test_scalar_1') is not None)

    def test_add_scalar_empty(self, sample_with_scalar):
        pass

    def test_add_scalar(self, sample_with_scalar):
        sample_with_scalar.add_scalar('test_scalar_2', np.random.randn())

    # -------------------------------------------------------------------------#
    def test_get_time_series_names_empty(self, sample):
        assert (sample.get_time_series_names() == [])

    def test_get_time_series_names(self, sample_with_time_series):
        assert (sample_with_time_series.get_time_series_names()
                == ['test_time_series_1'])

    def test_get_time_series_empty(self, sample):
        assert (sample.get_time_series('missing_time_series_name') is None)

    def test_get_time_series(self, sample_with_time_series):
        assert (sample_with_time_series.get_time_series(
            'missing_time_series_name') is None)
        assert (sample_with_time_series.get_time_series(
            'test_time_series_1') is not None)

    def test_add_time_series_empty(self, sample_with_time_series):
        pass

    def test_add_time_series(self, sample_with_time_series):
        sample_with_time_series.add_time_series(
            'test_time_series_2', np.arange(
                111, dtype=float), np.random.randn(111))

    # -------------------------------------------------------------------------#
    def test_get_nodes_empty(self, sample):
        assert (sample.get_nodes() is None)

    def test_get_nodes(self, sample_with_tree, nodes):
        assert (np.all(sample_with_tree.get_nodes() == nodes))

    def test_get_nodes3d(self, sample_with_tree3d, nodes3d):
        assert (np.all(sample_with_tree3d.get_nodes() == nodes3d))

    def test_set_nodes(self, sample, nodes, zone_name, base_name):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.set_nodes(nodes, zone_name, base_name)
        sample.init_zone(
            np.array([len(nodes), 0, 0]),
            zone_name=zone_name,
            base_name=base_name)
        sample.set_nodes(nodes, zone_name, base_name)

    # -------------------------------------------------------------------------#
    def test_get_elements_empty(self, sample):
        assert (sample.get_elements() == {})

    def test_get_elements(self, sample_with_tree, triangles):
        assert (list(sample_with_tree.get_elements().keys()) == ['TRI_3'])
        print(f"{triangles=}")
        print(f"{sample_with_tree.get_elements()=}")
        assert (np.all(sample_with_tree.get_elements()['TRI_3'] == triangles))

    # -------------------------------------------------------------------------#
    def test_get_field_names(self, sample):
        assert (sample.get_field_names() == [])
        assert (sample.get_field_names(location='CellCenter') == [])

    def test_get_field_empty(self, sample):
        assert (sample.get_field('missing_field_name') is None)
        assert (
            sample.get_field(
                'missing_field_name',
                location='CellCenter') is None)

    def test_get_field(self, sample_with_tree):
        assert (sample_with_tree.get_field('missing_field') is None)
        assert (sample_with_tree.get_field('test_node_field_1').shape == (5,))
        assert (
            sample_with_tree.get_field(
                'test_elem_field_1',
                location='CellCenter').shape == (
                3,
            ))

    def test_add_field_vertex(
            self, sample, vertex_field, zone_name, base_name):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.add_field(
                'test_node_field_2',
                vertex_field,
                zone_name,
                base_name)
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name=zone_name,
            base_name=base_name)
        sample.add_field(
            'test_node_field_2',
            vertex_field,
            zone_name,
            base_name)

    def test_add_field_cell_center(
            self, sample, cell_center_field, zone_name, base_name):
        sample.init_base(3, 3, base_name)
        with pytest.raises(KeyError):
            sample.add_field(
                'test_elem_field_2',
                cell_center_field,
                zone_name,
                base_name,
                location='CellCenter')
        sample.init_zone(
            np.random.randint(0, 10, size=3),
            zone_name=zone_name,
            base_name=base_name)
        sample.add_field(
            'test_elem_field_2',
            cell_center_field,
            zone_name,
            base_name,
            location='CellCenter')

    def test_add_field_vertex_already_present(
            self, sample_with_tree, vertex_field):
        # with pytest.raises(KeyError):
        sample_with_tree.show_tree()
        sample_with_tree.add_field(
            'test_node_field_1',
            vertex_field,
            'Zone',
            'Base_2_2')

    def test_add_field_cell_center_already_present(
            self, sample_with_tree, cell_center_field):
        # with pytest.raises(KeyError):
        sample_with_tree.show_tree()
        sample_with_tree.add_field(
            'test_elem_field_1',
            cell_center_field,
            'Zone',
            'Base_2_2',
            location='CellCenter')

    # -------------------------------------------------------------------------#
    def test_save(self, sample_with_tree_and_scalar_and_time_series, tmp_path):
        save_dir = tmp_path / 'test_dir'
        sample_with_tree_and_scalar_and_time_series.save(save_dir)
        assert (os.path.isdir(save_dir))
        with pytest.raises(ValueError):
            sample_with_tree_and_scalar_and_time_series.save(save_dir)

    def test_load_from_saved_file(
            self, sample_with_tree_and_scalar_and_time_series, tmp_path):
        save_dir = tmp_path / 'test_dir'
        sample_with_tree_and_scalar_and_time_series.save(save_dir)
        new_sample = Sample()
        new_sample.load(save_dir)
        assert (
            CGU.checkSameTree(
                sample_with_tree_and_scalar_and_time_series.get_mesh(),
                new_sample.get_mesh()))

    def test_load_from_dir(
            self, sample_with_tree_and_scalar_and_time_series, tmp_path):
        save_dir = tmp_path / 'test_dir'
        sample_with_tree_and_scalar_and_time_series.save(save_dir)
        new_sample = Sample.load_from_dir(save_dir)
        assert (
            CGU.checkSameTree(
                sample_with_tree_and_scalar_and_time_series.get_mesh(),
                new_sample.get_mesh()))

    # -------------------------------------------------------------------------#
    def test___repr___empty(self, sample):
        print(sample)

    def test___repr__with_scalar(self, sample_with_scalar):
        print(sample_with_scalar)

    def test___repr__with_tree(self, sample_with_tree):
        print(sample_with_tree)

    def test___repr__with_tree_and_scalar(
            self, sample_with_tree_and_scalar_and_time_series):
        print(sample_with_tree_and_scalar_and_time_series)
