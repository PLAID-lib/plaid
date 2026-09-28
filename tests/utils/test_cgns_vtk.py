"""Tests for direct CGNS-to-VTK conversion helpers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from plaid.utils import cgns_vtk
from plaid.utils.cgns_json import cgns_tree_from_json_payload, cgns_tree_to_json_payload


class _FakeVtkArray:
    def __init__(self, data):
        self.data = np.asarray(data)
        self.name = None
        self.number_of_components = None

    def SetName(self, name):  # noqa: N802
        self.name = name

    def SetNumberOfComponents(self, number_of_components):  # noqa: N802
        self.number_of_components = number_of_components


class _FakeAttributes:
    def __init__(self):
        self.arrays = []

    def AddArray(self, array):  # noqa: N802
        self.arrays.append(array)


class _FakeFieldData(_FakeAttributes):
    pass


class _FakePoints:
    def __init__(self):
        self.data = None

    def SetData(self, data):  # noqa: N802
        self.data = data


class _FakeCellArray:
    def __init__(self):
        self.offsets = None
        self.connectivity = None

    def SetData(self, offsets, connectivity):  # noqa: N802
        self.offsets = offsets
        self.connectivity = connectivity


class _FakeMetadata:
    def __init__(self):
        self.values = {}

    def Set(self, key, value):  # noqa: N802
        self.values[key] = value


class _FakeVtkObject:
    def __init__(self):
        self.points = None
        self.dimensions = None
        self.cell_types = None
        self.cell_array = None
        self.point_data = _FakeAttributes()
        self.cell_data = _FakeAttributes()
        self.field_data = _FakeFieldData()

    def SetPoints(self, points):  # noqa: N802
        self.points = points

    def SetDimensions(self, dimensions):  # noqa: N802
        self.dimensions = dimensions

    def SetCells(self, cell_types, cell_array):  # noqa: N802
        self.cell_types = cell_types
        self.cell_array = cell_array

    def GetNumberOfPoints(self):  # noqa: N802
        if self.points is None:
            return 0
        return len(self.points.data.data)

    def GetNumberOfCells(self):  # noqa: N802
        return 0 if self.cell_types is None else len(self.cell_types)

    def GetPointData(self):  # noqa: N802
        return self.point_data

    def GetCellData(self):  # noqa: N802
        return self.cell_data

    def GetFieldData(self):  # noqa: N802
        return self.field_data


class _FakeMultiBlock:
    @staticmethod
    def NAME():  # noqa: N802
        return "name"

    def __init__(self):
        self.blocks = []
        self.metadata = []

    def SetNumberOfBlocks(self, number_of_blocks):  # noqa: N802
        self.blocks = [None] * number_of_blocks
        self.metadata = [_FakeMetadata() for _ in range(number_of_blocks)]

    def SetBlock(self, index, block):  # noqa: N802
        self.blocks[index] = block

    def GetMetaData(self, index):  # noqa: N802
        return self.metadata[index]


class _FakeNumpySupport:
    @staticmethod
    def numpy_to_vtk(data, deep=False):
        _ = deep
        return _FakeVtkArray(data)

    @staticmethod
    def numpy_to_vtkIdTypeArray(data, deep=False):  # noqa: N802
        _ = deep
        return _FakeVtkArray(data)


class _FakeStringArray:
    def __init__(self):
        self.name = None
        self.values = []

    def SetName(self, name):  # noqa: N802
        self.name = name

    def SetNumberOfValues(self, number_of_values):  # noqa: N802
        self.values = [None] * number_of_values

    def SetValue(self, *args):  # noqa: N802
        if len(args) == 1:
            self.values.append(args[0])
            return
        index, value = args
        self.values[index] = value


def _patch_fake_vtk_import(monkeypatch):
    monkeypatch.setattr(
        cgns_vtk,
        "_import_vtk_for_direct_cgns",
        lambda: (
            _FakeVtkObject,
            _FakeVtkObject,
            _FakePoints,
            _FakeCellArray,
            _FakeMultiBlock,
            _FakeNumpySupport,
        ),
    )


def _node(name, value, children=None, label="DataArray_t"):
    return [name, value, children or [], label]


def test_cgns_child_helpers_find_children_by_label_and_name():
    parent = _node(
        "Parent",
        None,
        [
            _node("Grid", None, label="GridCoordinates_t"),
            _node("Flow", None, label="FlowSolution_t"),
        ],
        label="Zone_t",
    )

    assert cgns_vtk._cgns_children_by_label(parent, "FlowSolution_t") == [parent[2][1]]
    assert cgns_vtk._cgns_child_by_name(parent, "Grid") == parent[2][0]
    assert cgns_vtk._cgns_child_by_name(parent, "Missing") is None


def test_vtk_cell_conversion_validates_types_and_permutates_points():
    points = np.arange(15)
    cgns_type, converted = cgns_vtk._vtk_cell_to_cgns(26, points)
    assert cgns_type == 15
    np.testing.assert_array_equal(
        converted, points[np.argsort(cgns_vtk.CGNSNumberToVtkPermutation[15])]
    )
    with pytest.raises(NotImplementedError, match="999"):
        cgns_vtk._vtk_cell_to_cgns(999, points)


def test_vtk_string_arrays_and_binary_tag_contract():
    class StringArray:
        def IsA(self, name):
            return name == "vtkStringArray"

        def GetNumberOfValues(self):
            return 2

        def GetValue(self, index):
            return ("left", "right")[index]

    assert cgns_vtk._vtk_numpy_array(StringArray(), None).tolist() == ["left", "right"]

    class TagArray:
        def __init__(self, data, components=1, data_type="unsigned char"):
            self.data = np.asarray(data)
            self.components = components
            self.data_type = data_type

        def GetNumberOfComponents(self):
            return self.components

        def GetDataTypeAsString(self):
            return self.data_type

        def IsA(self, name):  # noqa: ARG002
            return False

    support = SimpleNamespace(vtk_to_numpy=lambda array: array.data)
    assert not cgns_vtk._vtk_array_is_binary_tag(None, 2, support)
    assert not cgns_vtk._vtk_array_is_binary_tag(TagArray([0], 2), 1, support)
    assert not cgns_vtk._vtk_array_is_binary_tag(
        TagArray([0], data_type="int"), 1, support
    )
    assert not cgns_vtk._vtk_array_is_binary_tag(TagArray([0]), 2, support)
    assert not cgns_vtk._vtk_array_is_binary_tag(TagArray([2]), 1, support)


def test_vtk_attributes_to_nodes_skips_missing_tags_and_metadata():
    class Array:
        def __init__(self, name, data):
            self.name = name
            self.data = np.asarray(data)

        def GetName(self):
            return self.name

        def IsA(self, name):  # noqa: ARG002
            return False

        def GetNumberOfComponents(self):
            return 1

        def GetDataTypeAsString(self):
            return "unsigned char" if self.data.dtype.kind in "iu" else "double"

    class Attributes:
        arrays = [
            None,
            Array(None, [3.0]),
            Array(cgns_vtk.PLAID_CGNS_ZONE_NAME, [4.0]),
            Array("Tag", [0, 1]),
        ]

        def GetNumberOfArrays(self):
            return len(self.arrays)

        def GetArray(self, index):
            return self.arrays[index]

    support = SimpleNamespace(vtk_to_numpy=lambda array: array.data)
    nodes = cgns_vtk._vtk_attributes_to_nodes(Attributes(), support, 2)
    assert [node[0] for node in nodes] == ["Array1"]


def test_vtk_coordinates_validates_and_pads_dimensions():
    support = SimpleNamespace(vtk_to_numpy=np.asarray)
    with pytest.raises(ValueError, match="no points"):
        cgns_vtk._vtk_coordinates(SimpleNamespace(GetPoints=lambda: None), support)
    bad_points = SimpleNamespace(GetData=lambda: np.ones(3))
    with pytest.raises(ValueError, match="two-dimensional"):
        cgns_vtk._vtk_coordinates(
            SimpleNamespace(GetPoints=lambda: bad_points), support
        )
    point_data = SimpleNamespace(GetData=lambda: np.ones((2, 1)))
    nodes = cgns_vtk._vtk_coordinates(
        SimpleNamespace(GetPoints=lambda: point_data), support, True
    )
    assert [node[0] for node in nodes] == ["CoordinateX", "CoordinateY", "CoordinateZ"]
    np.testing.assert_array_equal(nodes[1][1], [0.0, 0.0])


def test_vtk_uniform_elements_returns_none_for_non_grid_and_empty_grid():
    assert cgns_vtk._vtk_uniform_unstructured_elements(object(), False, None) is None

    class Grid:
        def IsA(self, name):
            return name == "vtkUnstructuredGrid"

        def GetNumberOfCells(self):
            return 0

    assert cgns_vtk._vtk_uniform_unstructured_elements(Grid(), False, None) is None


def test_vtk_uniform_elements_falls_back_for_unsupported_layouts(monkeypatch):
    support = SimpleNamespace(vtk_to_numpy=np.array)
    util = ModuleType("vtkmodules.util")
    util.numpy_support = support
    monkeypatch.setitem(sys.modules, "vtkmodules", ModuleType("vtkmodules"))
    monkeypatch.setitem(sys.modules, "vtkmodules.util", util)
    monkeypatch.setitem(sys.modules, "vtkmodules.util.numpy_support", support)
    monkeypatch.delitem(sys.modules, "paraview", raising=False)
    monkeypatch.delitem(sys.modules, "paraview.vtk", raising=False)

    class Cells:
        def __init__(self, offsets, connectivity):
            self.offsets = offsets
            self.connectivity = connectivity

        def GetOffsetsArray(self):
            return self.offsets

        def GetConnectivityArray(self):
            return self.connectivity

    class Grid:
        def __init__(self, types, offsets, connectivity):
            self.types = types
            self.cells = Cells(offsets, connectivity)

        def IsA(self, name):
            return name == "vtkUnstructuredGrid"

        def GetNumberOfCells(self):
            return len(self.types)

        def GetCellTypesArray(self):
            return self.types

        def GetCells(self):
            return self.cells

    assert (
        cgns_vtk._vtk_uniform_unstructured_elements(
            Grid([5, 9], [0, 3, 7], [0, 1, 2, 3, 4, 5, 6]), False, None
        )
        is None
    )
    assert (
        cgns_vtk._vtk_uniform_unstructured_elements(
            Grid([99], [0, 1], [0]), False, None
        )
        is None
    )
    assert (
        cgns_vtk._vtk_uniform_unstructured_elements(
            Grid([5], [0, 4], [0, 1, 2, 3]), False, None
        )
        is None
    )
    assert (
        cgns_vtk._vtk_uniform_unstructured_elements(
            Grid([5], [1, 4], [0, 1, 2]), False, None
        )
        is None
    )


def test_vtk_uniform_elements_permutates_and_returns_mapping(monkeypatch):
    support = SimpleNamespace(vtk_to_numpy=np.asarray)
    util = ModuleType("vtkmodules.util")
    util.numpy_support = support
    monkeypatch.setitem(sys.modules, "vtkmodules", ModuleType("vtkmodules"))
    monkeypatch.setitem(sys.modules, "vtkmodules.util", util)
    monkeypatch.setitem(sys.modules, "vtkmodules.util.numpy_support", support)
    monkeypatch.delitem(sys.modules, "paraview", raising=False)
    monkeypatch.delitem(sys.modules, "paraview.vtk", raising=False)

    class Cells:
        def GetOffsetsArray(self):
            return [0, 15]

        def GetConnectivityArray(self):
            return np.arange(15)

    class Grid:
        def IsA(self, name):
            return name == "vtkUnstructuredGrid"

        def GetNumberOfCells(self):
            return 1

        def GetCellTypesArray(self):
            return [26]

        def GetCells(self):
            return Cells()

    elements, cell_ids, dimensions = cgns_vtk._vtk_uniform_unstructured_elements(
        Grid(), True, {"15": "Penta15"}
    )
    assert elements[0][0] == "Penta15"
    np.testing.assert_array_equal(
        elements[0][2][1][1],
        (np.arange(15)[np.argsort(cgns_vtk.CGNSNumberToVtkPermutation[15])] + 1),
    )
    np.testing.assert_array_equal(cell_ids, [1])
    np.testing.assert_array_equal(dimensions, [3])
    elements_only = cgns_vtk._vtk_uniform_unstructured_elements(
        Grid(), False, {"15": "Penta15"}
    )
    assert elements_only[0][0] == "Penta15"


def test_cgns_value_as_string_decodes_supported_values():
    chars = np.array(list("Vertex\x00"), dtype="U1")

    assert cgns_vtk._cgns_value_as_string(None) is None
    assert (
        cgns_vtk._cgns_value_as_string(_node("Location", "CellCenter")) == "CellCenter"
    )
    assert cgns_vtk._cgns_value_as_string(_node("Location", chars)) == "Vertex"
    assert cgns_vtk._cgns_value_as_string(_node("Number", 3)) == "3"


def test_cgns_add_numpy_array_to_vtk_attributes_adds_scalar_and_vector_arrays():
    attributes = _FakeAttributes()

    assert cgns_vtk._cgns_add_numpy_array_to_vtk_attributes(
        attributes,
        "scalar",
        np.array([1.0, 2.0]),
        2,
        _FakeNumpySupport,
    )
    assert cgns_vtk._cgns_add_numpy_array_to_vtk_attributes(
        attributes,
        "vector",
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        2,
        _FakeNumpySupport,
    )
    assert [array.name for array in attributes.arrays] == ["scalar", "vector"]
    assert attributes.arrays[1].number_of_components == 2


@pytest.mark.parametrize(
    ("data", "number_of_tuples"),
    [(np.array(["a", "b"]), 2), (np.array([1, 2, 3]), 2), (np.array([1]), 0)],
)
def test_cgns_add_numpy_array_to_vtk_attributes_rejects_incompatible_arrays(
    data,
    number_of_tuples,
):
    attributes = _FakeAttributes()

    added = cgns_vtk._cgns_add_numpy_array_to_vtk_attributes(
        attributes,
        "bad",
        data,
        number_of_tuples,
        _FakeNumpySupport,
    )

    assert not added
    assert attributes.arrays == []


def test_cgns_insert_cells_from_elements_node_adds_linear_cells():
    elements = _node(
        "Triangles",
        np.array([5]),
        [_node("ElementConnectivity", np.array([1, 2, 3, 4, 5, 6]))],
        label="Elements_t",
    )
    cell_types = []
    offsets = [0]
    connectivity = []

    cgns_vtk._cgns_insert_cells_from_elements_node(
        elements,
        cell_types,
        offsets,
        connectivity,
    )

    assert cell_types == [5, 5]
    assert offsets == [0, 3, 6]
    assert connectivity == [0, 1, 2, 3, 4, 5]


def test_cgns_insert_cells_from_elements_node_supports_mixed_cells():
    elements = _node(
        "Mixed",
        np.array([20]),
        [_node("ElementConnectivity", np.array([3, 1, 2, 5, 3, 4, 5]))],
        label="Elements_t",
    )
    cell_types = []
    offsets = [0]
    connectivity = []

    cgns_vtk._cgns_insert_cells_from_elements_node(
        elements,
        cell_types,
        offsets,
        connectivity,
    )

    assert cell_types == [3, 5]
    assert offsets == [0, 2, 5]
    assert connectivity == [0, 1, 2, 3, 4]


def test_cgns_insert_cells_from_elements_node_applies_vtk_permutation():
    elements = _node(
        "Penta15",
        np.array([15]),
        [_node("ElementConnectivity", np.arange(1, 16))],
        label="Elements_t",
    )
    connectivity = []

    cgns_vtk._cgns_insert_cells_from_elements_node(elements, [], [0], connectivity)

    assert connectivity == [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11]


def test_cgns_insert_cells_from_elements_node_raises_for_unknown_type():
    elements = _node(
        "Unknown",
        np.array([99]),
        [_node("ElementConnectivity", np.array([1]))],
        label="Elements_t",
    )

    with pytest.raises(NotImplementedError, match="99"):
        cgns_vtk._cgns_insert_cells_from_elements_node(elements, [], [0], [])


def test_cgns_insert_cells_from_elements_node_uses_fallback_connectivity_name():
    elements = _node(
        "Triangles",
        np.array([5]),
        [_node("TrianglesElementConnectivity", np.array([1, 2, 3]))],
        label="Elements_t",
    )
    cell_types = []
    offsets = [0]
    connectivity = []

    cgns_vtk._cgns_insert_cells_from_elements_node(
        elements,
        cell_types,
        offsets,
        connectivity,
    )

    assert cell_types == [5]
    assert offsets == [0, 3]
    assert connectivity == [0, 1, 2]


def test_cgns_insert_cells_from_elements_node_ignores_missing_connectivity():
    cell_types = []
    offsets = [0]
    connectivity = []

    cgns_vtk._cgns_insert_cells_from_elements_node(
        _node("NoConnectivity", np.array([5]), [], label="Elements_t"),
        cell_types,
        offsets,
        connectivity,
    )

    assert cell_types == []
    assert offsets == [0]
    assert connectivity == []


def test_cgns_insert_cells_from_elements_node_raises_for_unknown_mixed_type():
    elements = _node(
        "Mixed",
        np.array([20]),
        [_node("ElementConnectivity", np.array([99, 1]))],
        label="Elements_t",
    )

    with pytest.raises(NotImplementedError, match="99"):
        cgns_vtk._cgns_insert_cells_from_elements_node(elements, [], [0], [])


def test_cgns_insert_cells_from_elements_node_applies_mixed_permutation():
    elements = _node(
        "Mixed",
        np.array([20]),
        [_node("ElementConnectivity", np.concatenate(([15], np.arange(1, 16))))],
        label="Elements_t",
    )
    connectivity = []

    cgns_vtk._cgns_insert_cells_from_elements_node(elements, [], [0], connectivity)

    assert connectivity == [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 9, 10, 11]

    cells = SimpleNamespace(
        GetNumberOfPoints=lambda: 15,
        GetPointId=lambda index: index,
    )
    grid = SimpleNamespace(
        GetNumberOfCells=lambda: 1,
        GetCellType=lambda _cell: 26,
        GetCell=lambda _cell: cells,
    )
    converted = cgns_vtk._vtk_unstructured_elements(grid)
    assert converted[0][0] == "Elements_15"


def test_cgns_index_ranges_support_descending_and_multidimensional_values():
    descending = _node(
        "Tag",
        None,
        [_node("Range", np.array([3, 1]), label="IndexRange_t")],
    )
    np.testing.assert_array_equal(
        cgns_vtk._cgns_index_values(descending, (3,)), [3, 2, 1]
    )
    rectangle = _node(
        "Tag",
        None,
        [_node("Range", np.array([[1, 2], [2, 3]]), label="IndexRange_t")],
    )
    np.testing.assert_array_equal(
        cgns_vtk._cgns_index_values(rectangle, (3, 4)), [2, 3, 6, 7]
    )
    ignored = _node(
        "Tag",
        None,
        [_node("Range", np.array([1, 2, 3]), label="IndexRange_t")],
    )
    assert cgns_vtk._cgns_index_values(ignored, (3,)).size == 0

    null_then_indices = _node(
        "Tag",
        None,
        [
            _node("Empty", None, label="IndexArray_t"),
            _node("Indices", np.array([2, 3]), label="IndexArray_t"),
        ],
    )
    np.testing.assert_array_equal(
        cgns_vtk._cgns_index_values(null_then_indices, (3,)), [2, 3]
    )


def test_cgns_tag_helpers_merge_masks_and_validate_indices():
    attributes = _FakeAttributes()

    class ExistingArray(_FakeVtkArray):
        pass

    attributes.arrays.append(ExistingArray([1, 0, 0]))
    attributes.arrays[0].SetName("merged")
    attributes.GetArray = lambda name: next(
        (array for array in attributes.arrays if array.name == name), None
    )
    attributes.RemoveArray = lambda name: attributes.arrays.__setitem__(
        slice(None), [array for array in attributes.arrays if array.name != name]
    )
    support = SimpleNamespace(
        vtk_to_numpy=lambda array: array.data,
        numpy_to_vtk=_FakeNumpySupport.numpy_to_vtk,
    )
    cgns_vtk._cgns_add_tag_array(
        attributes, "merged", np.array([False, True, False]), support
    )
    np.testing.assert_array_equal(attributes.arrays[0].data, [1, 1, 0])
    assert (
        cgns_vtk._cgns_tag_name(_node("Named_ZSR", None, [], label="ZoneSubRegion_t"))
        == "Named"
    )

    vtk_object = _FakeVtkObject()
    vtk_object.points = _FakePoints()
    vtk_object.points.SetData(_FakeVtkArray(np.zeros((2, 3))))
    point_tag = _node(
        "invalid",
        None,
        [_node("PointList", np.array([0]), label="IndexArray_t")],
        label="BC_t",
    )
    with pytest.raises(ValueError, match="invalid indices"):
        cgns_vtk._cgns_add_tags_to_vtk(
            _node(
                "Zone",
                None,
                [_node("ZoneBC", None, [point_tag], label="ZoneBC_t")],
                label="Zone_t",
            ),
            vtk_object,
            support,
        )


@pytest.mark.parametrize(
    ("topological_dim", "selected_dim", "location"),
    [(3, 2, "FaceCenter"), (2, 1, "EdgeCenter")],
)
def test_vtk_dataset_cell_boundary_tags_use_boundary_conditions(
    topological_dim, selected_dim, location
):
    class TagArray:
        def __init__(self):
            self.data = np.array([1, 0], dtype=np.uint8)

        def IsA(self, name):  # noqa: ARG002
            return False

        def GetNumberOfComponents(self):
            return 1

        def GetDataTypeAsString(self):
            return "unsigned char"

        def GetName(self):
            return "boundary"

    class Attributes:
        def GetNumberOfArrays(self):
            return 1

        def GetArray(self, _index):
            return TagArray()

    empty_attributes = SimpleNamespace(
        GetNumberOfArrays=lambda: 0,
        GetArray=lambda _index: None,
    )
    data = SimpleNamespace(
        GetPointData=lambda: empty_attributes,
        GetCellData=lambda: Attributes(),
        GetNumberOfPoints=lambda: 0,
        GetNumberOfCells=lambda: 2,
    )
    children = cgns_vtk._vtk_dataset_tag_nodes(
        data,
        SimpleNamespace(vtk_to_numpy=lambda array: array.data),
        np.array([1, 2]),
        np.array([selected_dim, topological_dim]),
    )
    assert children[0][0] == "ZoneBC"
    assert cgns_vtk._cgns_value_as_string(children[0][2][0][2][1]) == location

    # Both dimensions are explicitly supplied so this helper's topological
    # dimension is inferred from the cell dimension array as expected.
    assert topological_dim in (2, 3)


def test_vtk_grid_metadata_removes_old_values_and_decodes_bad_name_metadata():
    class FieldData(_FakeFieldData):
        def GetArray(self, name):
            return next((array for array in self.arrays if array.name == name), None)

        def RemoveArray(self, name):
            self.arrays = [array for array in self.arrays if array.name != name]

    vtk_object = _FakeVtkObject()
    vtk_object.field_data = FieldData()
    for name in (
        cgns_vtk.PLAID_CGNS_BASE_NAME,
        cgns_vtk.PLAID_CGNS_BASE_DIMENSIONS,
        cgns_vtk.PLAID_CGNS_ZONE_NAME,
    ):
        old = _FakeVtkArray([0])
        old.SetName(name)
        vtk_object.field_data.AddArray(old)

    cgns_vtk._vtk_add_cgns_grid_metadata(
        vtk_object, "Base", np.array([3, 2]), "Zone", _FakeNumpySupport
    )
    assert len(vtk_object.field_data.arrays) == 3

    class MetadataArray:
        def __init__(self, value):
            self.value = value

    class MetadataFieldData:
        def __init__(self, value):
            self.value = value

        def GetArray(self, _key):
            return self.value

    class MetadataObject:
        def __init__(self, value):
            self.field = MetadataFieldData(value)

        def GetFieldData(self):
            return self.field

    support = SimpleNamespace(vtk_to_numpy=lambda array: array.value)
    assert (
        cgns_vtk._vtk_read_names_metadata(
            MetadataObject(MetadataArray(np.array([255], dtype=np.uint8))),
            "names",
            support,
        )
        == {}
    )
    assert (
        cgns_vtk._vtk_read_names_metadata(
            MetadataObject(MetadataArray(np.frombuffer(b"[]", dtype=np.uint8))),
            "names",
            support,
        )
        == {}
    )
    assert cgns_vtk._vtk_read_names_metadata(
        MetadataObject(
            MetadataArray(np.frombuffer(b'{"a": 1, "b": "B"}', dtype=np.uint8))
        ),
        "names",
        support,
    ) == {"b": "B"}


def test_vtk_dataset_blocks_skips_empty_blocks_and_default_names():
    class Metadata:
        def Has(self, _key):
            return False

    class Leaf:
        def IsA(self, _name):
            return False

    class MultiBlock:
        def __init__(self):
            self.leaf = Leaf()

        def IsA(self, name):
            return name == "vtkMultiBlockDataSet"

        def GetNumberOfBlocks(self):
            return 2

        def GetBlock(self, index):
            return (None, self.leaf)[index]

        def GetMetaData(self, _index):
            return Metadata()

        def NAME(self):
            return "name"

    multiblock = MultiBlock()
    assert cgns_vtk._vtk_dataset_blocks(multiblock) == [
        ("Block_1_Zone", multiblock.GetBlock(1))
    ]


def test_vtk_to_cgns_tree_validates_type_empty_blocks_and_base_dimensions(
    monkeypatch,
):
    fake_numpy_support = SimpleNamespace(
        vtk_to_numpy=lambda array: np.asarray(array.data)
    )
    vtkmodules_util = ModuleType("vtkmodules.util")
    vtkmodules_util.numpy_support = fake_numpy_support
    monkeypatch.setitem(sys.modules, "vtkmodules.util", vtkmodules_util)
    monkeypatch.setitem(
        sys.modules, "vtkmodules.util.numpy_support", fake_numpy_support
    )
    monkeypatch.delitem(sys.modules, "paraview", raising=False)
    monkeypatch.delitem(sys.modules, "paraview.vtk", raising=False)
    with pytest.raises(TypeError, match="expects a VTK dataset"):
        cgns_vtk.VtkToCGNSTree(SimpleNamespace(IsA=lambda _name: False))

    class EmptyMultiBlock:
        def IsA(self, name):
            return name == "vtkMultiBlockDataSet"

        def GetNumberOfBlocks(self):
            return 0

    with pytest.raises(ValueError, match="contains no data sets"):
        cgns_vtk.VtkToCGNSTree(EmptyMultiBlock())

    class Metadata:
        def __init__(self, values):
            self.values = values

        def GetArray(self, name):
            value = self.values.get(name)
            return None if value is None else SimpleNamespace(data=value)

    class Grid:
        def __init__(self, dimensions):
            self.field = Metadata(
                {
                    cgns_vtk.PLAID_CGNS_BASE_NAME: np.frombuffer(
                        b"Base", dtype=np.uint8
                    ),
                    cgns_vtk.PLAID_CGNS_BASE_DIMENSIONS: np.asarray(dimensions),
                    cgns_vtk.PLAID_CGNS_ZONE_NAME: np.frombuffer(
                        b"Zone", dtype=np.uint8
                    ),
                }
            )

        def IsA(self, name):
            return name == "vtkDataSet"

        def GetFieldData(self):
            return self.field

    monkeypatch.setattr(
        cgns_vtk,
        "_vtk_dataset_to_zone",
        lambda *_args, **_kwargs: ["Zone", None, [], "Zone_t"],
    )
    multiblock = SimpleNamespace(
        IsA=lambda name: name == "vtkMultiBlockDataSet",
        GetNumberOfBlocks=lambda: 2,
        GetBlock=lambda index: (Grid([2, 2]), Grid([3, 2]))[index],
        GetMetaData=lambda _index: None,
        NAME=lambda: "name",
    )
    with pytest.raises(ValueError, match="inconsistent dimensions"):
        cgns_vtk.VtkToCGNSTree(multiblock)


def test_cgns_add_tags_rejects_unknown_cell_number():
    cell_tag = _node(
        "missing_cell",
        None,
        [
            _node("PointList", np.array([4]), label="IndexArray_t"),
            _node("GridLocation", "CellCenter", label="GridLocation_t"),
        ],
        label="ZoneSubRegion_t",
    )
    with pytest.raises(ValueError, match="invalid element number 4"):
        cgns_vtk._cgns_add_tags_to_vtk(
            _node("Zone", None, [cell_tag], label="Zone_t"),
            _FakeVtkObject(),
            _FakeNumpySupport,
            cgnsElementToVtkCell={},
        )


def test_cgns_add_flow_solutions_to_vtk_routes_point_and_cell_data():
    point_data = _FakeAttributes()
    cell_data = _FakeAttributes()
    vtk_object = SimpleNamespace(
        GetNumberOfPoints=lambda: 2,
        GetNumberOfCells=lambda: 1,
        GetPointData=lambda: point_data,
        GetCellData=lambda: cell_data,
    )
    zone = _node(
        "Zone",
        None,
        [
            _node(
                "PointFlow",
                None,
                [_node("pressure", np.array([1.0, 2.0]))],
                label="FlowSolution_t",
            ),
            _node(
                "CellFlow",
                None,
                [
                    _node("GridLocation", "CellCenter", label="GridLocation_t"),
                    _node("density", np.array([3.0])),
                ],
                label="FlowSolution_t",
            ),
        ],
        label="Zone_t",
    )

    cgns_vtk._cgns_add_flow_solutions_to_vtk(zone, vtk_object, _FakeNumpySupport)

    assert [array.name for array in point_data.arrays] == ["pressure"]
    assert [array.name for array in cell_data.arrays] == ["density"]


def test_cgns_add_flow_solutions_to_vtk_skips_unsupported_locations_and_empty_data():
    point_data = _FakeAttributes()
    cell_data = _FakeAttributes()
    vtk_object = SimpleNamespace(
        GetNumberOfPoints=lambda: 1,
        GetNumberOfCells=lambda: 1,
        GetPointData=lambda: point_data,
        GetCellData=lambda: cell_data,
    )
    zone = _node(
        "Zone",
        None,
        [
            _node(
                "BadLocation",
                None,
                [
                    _node("GridLocation", "Unknown", label="GridLocation_t"),
                    _node("ignored", np.array([1.0])),
                ],
                label="FlowSolution_t",
            ),
            _node(
                "NoData",
                None,
                [_node("empty", None)],
                label="FlowSolution_t",
            ),
        ],
        label="Zone_t",
    )

    cgns_vtk._cgns_add_flow_solutions_to_vtk(zone, vtk_object, _FakeNumpySupport)

    assert point_data.arrays == []
    assert cell_data.arrays == []


def test_cgns_zone_points_to_vtk_points_reads_coordinates():
    zone = _node(
        "Zone",
        None,
        [
            _node(
                "GridCoordinates",
                None,
                [
                    _node("CoordinateX", np.array([1.0, 2.0])),
                    _node("CoordinateY", np.array([3.0, 4.0])),
                ],
                label="GridCoordinates_t",
            )
        ],
        label="Zone_t",
    )

    points, shape = cgns_vtk._cgns_zone_points_to_vtk_points(
        zone,
        2,
        _FakeNumpySupport,
        _FakePoints,
    )

    assert shape == (2,)
    np.testing.assert_allclose(points.data.data, [[1.0, 3.0, 0.0], [2.0, 4.0, 0.0]])


def test_cgns_zone_points_to_vtk_points_requires_coordinates():
    zone = _node("Zone", None, [], label="Zone_t")

    with pytest.raises(ValueError, match="GridCoordinates_t"):
        cgns_vtk._cgns_zone_points_to_vtk_points(
            zone,
            3,
            _FakeNumpySupport,
            _FakePoints,
        )


def test_cgns_zone_points_to_vtk_points_requires_coordinate_x():
    zone = _node(
        "Zone",
        None,
        [_node("GridCoordinates", None, [], label="GridCoordinates_t")],
        label="Zone_t",
    )

    with pytest.raises(ValueError, match="CoordinateX"):
        cgns_vtk._cgns_zone_points_to_vtk_points(
            zone,
            3,
            _FakeNumpySupport,
            _FakePoints,
        )


def test_cgns_base_extract_globals_returns_non_empty_values():
    base = _node(
        "Global",
        None,
        [_node("labels", np.array([1])), _node("empty", None)],
        label="CGNSBase_t",
    )

    globals_ = cgns_vtk.CGNSBaseExtractGlobals(base)

    assert list(globals_) == ["labels"]
    np.testing.assert_array_equal(globals_["labels"], np.array([1]))


def test_cgns_base_to_vtk_validates_base_node():
    with pytest.raises(ValueError, match="CGNSBase_t"):
        cgns_vtk.CGNSBaseToVtk(_node("NotBase", None, label="Zone_t"))


def test_cgns_base_to_vtk_dispatches_zone_type(monkeypatch):
    structured_zone = _node(
        "StructuredZone",
        np.array([[2], [1], [1]]),
        [_node("ZoneType", "Structured", label="ZoneType_t")],
        label="Zone_t",
    )
    base = _node("Base", np.array([3, 3]), [structured_zone], label="CGNSBase_t")
    structured_result = object()

    monkeypatch.setattr(
        cgns_vtk,
        "_cgns_structured_zone_to_vtk",
        lambda zone, physical_dim: (zone, physical_dim, structured_result),
    )

    assert cgns_vtk.CGNSBaseToVtk(base) == (structured_zone, 3, structured_result)


def test_cgns_base_to_vtk_converts_structured_zone(monkeypatch):
    _patch_fake_vtk_import(monkeypatch)
    zone = _node(
        "StructuredZone",
        np.array([[2], [1], [1]]),
        [
            _node("ZoneType", "Structured", label="ZoneType_t"),
            _node(
                "GridCoordinates",
                None,
                [_node("CoordinateX", np.array([1.0, 2.0]))],
                label="GridCoordinates_t",
            ),
        ],
        label="Zone_t",
    )
    base = _node("Base", np.array([3, 2]), [zone], label="CGNSBase_t")

    output = cgns_vtk.CGNSBaseToVtk(base)

    assert output.dimensions == [2, 1, 1]
    np.testing.assert_allclose(output.points.data.data[:, 0], [1.0, 2.0])


def test_cgns_base_to_vtk_converts_unstructured_zone(monkeypatch):
    _patch_fake_vtk_import(monkeypatch)
    zone = _node(
        "UnstructuredZone",
        np.array([[3, 1, 0]]),
        [
            _node(
                "GridCoordinates",
                None,
                [_node("CoordinateX", np.array([1.0, 2.0, 3.0]))],
                label="GridCoordinates_t",
            ),
            _node(
                "Triangles",
                np.array([5]),
                [_node("ElementConnectivity", np.array([1, 2, 3]))],
                label="Elements_t",
            ),
        ],
        label="Zone_t",
    )
    base = _node("Base", np.array([3, 3]), [zone], label="CGNSBase_t")

    output = cgns_vtk.CGNSBaseToVtk(base)

    assert output.cell_types == [5]
    np.testing.assert_array_equal(output.cell_array.connectivity.data, [0, 1, 2])


def test_unstructured_zone_mixed_section_omits_ambiguous_element_names(monkeypatch):
    _patch_fake_vtk_import(monkeypatch)
    zone = _node(
        "Zone",
        np.array([[3, 2, 0]]),
        [
            _node(
                "GridCoordinates",
                None,
                [_node("CoordinateX", np.array([0.0, 1.0, 0.0]))],
                label="GridCoordinates_t",
            ),
            _node(
                "Triangles",
                np.array([5]),
                [
                    _node("ElementRange", np.array([1, 1]), label="IndexRange_t"),
                    _node("ElementConnectivity", np.array([1, 2, 3])),
                ],
                label="Elements_t",
            ),
            _node(
                "Mixed",
                np.array([20]),
                [
                    _node("ElementRange", np.array([2, 2]), label="IndexRange_t"),
                    _node("ElementConnectivity", np.array([5, 1, 2, 3])),
                ],
                label="Elements_t",
            ),
        ],
        label="Zone_t",
    )

    output = cgns_vtk._cgns_unstructured_zone_to_vtk(zone, 2)

    assert output.cell_types == [5, 5]
    assert all(
        array.name != cgns_vtk.PLAID_CGNS_ELEMENT_NAMES
        for array in output.field_data.arrays
    )


def test_unstructured_zone_repeated_element_types_omit_original_names(monkeypatch):
    _patch_fake_vtk_import(monkeypatch)
    coordinates = _node(
        "GridCoordinates",
        None,
        [_node("CoordinateX", np.array([0.0, 1.0, 0.0]))],
        label="GridCoordinates_t",
    )
    sections = [
        _node(
            name,
            np.array([5]),
            [
                _node("ElementRange", np.array([index, index]), label="IndexRange_t"),
                _node("ElementConnectivity", np.array([1, 2, 3])),
            ],
            label="Elements_t",
        )
        for index, name in enumerate(("TrianglesA", "TrianglesB"), start=1)
    ]
    zone = _node(
        "Zone",
        np.array([[3, 2, 0]]),
        [coordinates, *sections],
        label="Zone_t",
    )

    output = cgns_vtk._cgns_unstructured_zone_to_vtk(zone, 2)

    assert output.cell_types == [5, 5]
    assert all(
        array.name != cgns_vtk.PLAID_CGNS_ELEMENT_NAMES
        for array in output.field_data.arrays
    )


def test_cgns_base_to_vtk_returns_multiblock_for_multiple_zones(monkeypatch):
    monkeypatch.setattr(
        cgns_vtk, "_cgns_unstructured_zone_to_vtk", lambda zone, _dim: zone[0]
    )
    _patch_fake_vtk_import(monkeypatch)
    zones = [
        _node("ZoneA", np.array([[0]]), label="Zone_t"),
        _node("ZoneB", np.array([[0]]), label="Zone_t"),
    ]
    base = _node("Base", np.array([3, 3]), zones, label="CGNSBase_t")

    output = cgns_vtk.CGNSBaseToVtk(base)

    assert output.blocks == ["ZoneA", "ZoneB"]
    assert output.metadata[0].values == {"name": "ZoneA"}


def test_cgns_base_to_vtk_rejects_missing_data_and_unknown_zone_type():
    with pytest.raises(ValueError, match="dimensionality"):
        cgns_vtk.CGNSBaseToVtk(_node("Base", None, [], label="CGNSBase_t"))
    with pytest.raises(ValueError, match="no Zone_t"):
        cgns_vtk.CGNSBaseToVtk(_node("Base", np.array([3, 3]), [], label="CGNSBase_t"))

    zone = _node(
        "Zone",
        np.array([[0]]),
        [_node("ZoneType", "Unsupported", label="ZoneType_t")],
        label="Zone_t",
    )
    with pytest.raises(NotImplementedError, match="Unsupported"):
        cgns_vtk.CGNSBaseToVtk(
            _node("Base", np.array([3, 3]), [zone], label="CGNSBase_t")
        )


def test_cgns_tree_to_vtk_adds_global_field_data(monkeypatch):
    _patch_fake_vtk_import(monkeypatch)
    vtk_object = _FakeVtkObject()
    monkeypatch.setattr(cgns_vtk, "CGNSBaseToVtk", lambda _base: vtk_object)
    tree = _node(
        "CGNSTree",
        None,
        [
            _node(
                "Global",
                None,
                [_node("ids", np.array([1, 2, 3]))],
                label="CGNSBase_t",
            ),
            _node("Base", np.array([3, 3]), [], label="CGNSBase_t"),
        ],
        label="CGNSTree_t",
    )

    output = cgns_vtk.CGNSTreeToVtk(tree)

    assert output is vtk_object
    assert [array.name for array in output.field_data.arrays] == ["ids"]
    np.testing.assert_array_equal(output.field_data.arrays[0].data, [1, 2, 3])


def test_cgns_tree_to_vtk_adds_global_string_field_data(monkeypatch):
    _patch_fake_vtk_import(monkeypatch)
    vtk_object = _FakeVtkObject()
    monkeypatch.setattr(cgns_vtk, "CGNSBaseToVtk", lambda _base: vtk_object)

    fake_core_module = ModuleType("vtkmodules.vtkCommonCore")
    fake_core_module.vtkStringArray = _FakeStringArray
    monkeypatch.setitem(sys.modules, "vtkmodules.vtkCommonCore", fake_core_module)
    labels = np.array([b"A", b"B"], dtype="|S1")
    tree = _node(
        "CGNSTree",
        None,
        [
            _node(
                "Global",
                None,
                [_node("labels", labels)],
                label="CGNSBase_t",
            ),
            _node("Base", np.array([3, 3]), [], label="CGNSBase_t"),
        ],
        label="CGNSTree_t",
    )

    cgns_vtk.CGNSTreeToVtk(tree)

    string_array = vtk_object.field_data.arrays[0]
    assert string_array.name == "labels"
    assert string_array.values == [np.bytes_(b"A"), np.bytes_(b"B")]


def test_cgns_tree_to_vtk_returns_multiblock_for_multiple_bases(monkeypatch):
    _patch_fake_vtk_import(monkeypatch)
    outputs = {"BaseA": _FakeVtkObject(), "BaseB": _FakeVtkObject()}
    monkeypatch.setattr(cgns_vtk, "CGNSBaseToVtk", lambda base: outputs[base[0]])
    tree = _node(
        "CGNSTree",
        None,
        [
            _node("BaseA", np.array([3, 3]), [], label="CGNSBase_t"),
            _node("BaseB", np.array([3, 3]), [], label="CGNSBase_t"),
        ],
        label="CGNSTree_t",
    )

    output = cgns_vtk.CGNSTreeToVtk(tree)

    assert output.blocks == [outputs["BaseA"], outputs["BaseB"]]
    assert output.metadata[1].values == {"name": "BaseB"}


def test_import_vtk_for_direct_cgns_uses_vtkmodules_fallback(monkeypatch):
    fake_numpy_support = object()
    vtkmodules = ModuleType("vtkmodules")
    vtkmodules_util = ModuleType("vtkmodules.util")
    vtkmodules_util.numpy_support = fake_numpy_support
    vtk_common_core = ModuleType("vtkmodules.vtkCommonCore")
    vtk_common_core.vtkPoints = "vtkPoints"
    vtk_common_data_model = ModuleType("vtkmodules.vtkCommonDataModel")
    vtk_common_data_model.vtkCellArray = "vtkCellArray"
    vtk_common_data_model.vtkMultiBlockDataSet = "vtkMultiBlockDataSet"
    vtk_common_data_model.vtkStructuredGrid = "vtkStructuredGrid"
    vtk_common_data_model.vtkUnstructuredGrid = "vtkUnstructuredGrid"
    modules = {
        "paraview": None,
        "paraview.vtk": None,
        "vtkmodules": vtkmodules,
        "vtkmodules.util": vtkmodules_util,
        "vtkmodules.util.numpy_support": fake_numpy_support,
        "vtkmodules.vtkCommonCore": vtk_common_core,
        "vtkmodules.vtkCommonDataModel": vtk_common_data_model,
    }
    for name, module in modules.items():
        if module is None:
            monkeypatch.delitem(sys.modules, name, raising=False)
        else:
            monkeypatch.setitem(sys.modules, name, module)

    assert cgns_vtk._import_vtk_for_direct_cgns() == (
        "vtkStructuredGrid",
        "vtkUnstructuredGrid",
        "vtkPoints",
        "vtkCellArray",
        "vtkMultiBlockDataSet",
        fake_numpy_support,
    )


def test_vtk_to_cgns_tree_round_trips_unstructured_data():
    """VTK geometry and point/cell data survive the CGNS JSON round trip."""
    vtk = pytest.importorskip("vtk")
    from vtk.util import numpy_support

    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.SetData(
        numpy_support.numpy_to_vtk(
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            deep=True,
        )
    )
    grid.SetPoints(points)
    triangle = vtk.vtkTriangle()
    for index, point_id in enumerate((0, 1, 2)):
        triangle.GetPointIds().SetId(index, point_id)
    grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())

    pressure = numpy_support.numpy_to_vtk(np.asarray([1.0, 2.0, 3.0]), deep=True)
    pressure.SetName("Pressure")
    grid.GetPointData().AddArray(pressure)
    density = numpy_support.numpy_to_vtk(np.asarray([4.0]), deep=True)
    density.SetName("Density")
    grid.GetCellData().AddArray(density)

    tree = cgns_vtk.VtkToCGNSTree(grid)
    restored_tree = cgns_tree_from_json_payload(cgns_tree_to_json_payload(tree))
    restored = cgns_vtk.CGNSTreeToVtk(restored_tree)

    assert restored.GetNumberOfPoints() == 3
    assert restored.GetNumberOfCells() == 1
    assert restored.GetPointData().GetArray("Pressure").GetTuple1(2) == 3.0
    assert restored.GetCellData().GetArray("Density").GetTuple1(0) == 4.0


@pytest.mark.parametrize("cell_types", [(5, 5, 5), (5, 9, 5), (13, 13)])
def test_bulk_unstructured_elements_match_per_cell_conversion(cell_types):
    """Preserve CGNS connectivity, grouping, IDs, and dimensions for VTK grids."""
    vtk = pytest.importorskip("vtk")

    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    for index in range(30):
        points.InsertNextPoint(index, 0, 0)
    grid.SetPoints(points)
    for cell_index, cell_type in enumerate(cell_types):
        cgns_type = cgns_vtk.VtkNumberToCGNSNumber[cell_type]
        ids = vtk.vtkIdList()
        for point_id in range(cgns_vtk.CGNSNumberOfNodes[cgns_type]):
            ids.InsertNextId(point_id + cell_index)
        grid.InsertNextCell(cell_type, ids)

    class PerCellGrid:
        """Proxy that disables the fast path without altering cell access."""

        def __init__(self, wrapped):
            self.wrapped = wrapped

        def IsA(self, name):  # noqa: N802, ARG002
            return False

        def __getattr__(self, name):
            return getattr(self.wrapped, name)

    fast = cgns_vtk._vtk_unstructured_elements(
        grid, returnCellMapping=True, elementNames={"5": "Triangles"}
    )
    slow = cgns_vtk._vtk_unstructured_elements(
        PerCellGrid(grid), returnCellMapping=True, elementNames={"5": "Triangles"}
    )
    for fast_element, slow_element in zip(fast[0], slow[0], strict=True):
        assert fast_element[0] == slow_element[0]
        np.testing.assert_array_equal(fast_element[1], slow_element[1])
        for fast_child, slow_child in zip(
            fast_element[2], slow_element[2], strict=True
        ):
            np.testing.assert_array_equal(fast_child[1], slow_child[1])
    np.testing.assert_array_equal(fast[1], slow[1])
    np.testing.assert_array_equal(fast[2], slow[2])


def test_metadata_free_vtk_retains_legacy_third_coordinate_override():
    """Metadata-free VTK input can still request a third coordinate array."""
    vtk = pytest.importorskip("vtk")

    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 4.0)
    grid.SetPoints(points)
    vertex = vtk.vtkVertex()
    vertex.GetPointIds().SetId(0, 0)
    grid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())

    tree = cgns_vtk.VtkToCGNSTree(grid, ensure_3D_points=True)
    zone = tree[2][0][2][0]
    gridCoordinates = next(
        child for child in zone[2] if child[3] == "GridCoordinates_t"
    )

    assert [coordinate[0] for coordinate in gridCoordinates[2]] == [
        "CoordinateX",
        "CoordinateY",
        "CoordinateZ",
    ]
    np.testing.assert_array_equal(gridCoordinates[2][2][1], np.array([4.0]))


def test_vtk_to_cgns_tree_preserves_structured_dimensions_and_field_data():
    """Structured dimensions and global field arrays are converted correctly."""
    vtk = pytest.importorskip("vtk")
    from vtk.util import numpy_support

    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(2, 2, 1)
    points = vtk.vtkPoints()
    points.SetData(
        numpy_support.numpy_to_vtk(
            np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
            ),
            deep=True,
        )
    )
    grid.SetPoints(points)
    field = numpy_support.numpy_to_vtk(np.asarray([7], dtype=np.int32), deep=True)
    field.SetName("CaseId")
    grid.GetFieldData().AddArray(field)

    tree = cgns_vtk.VtkToCGNSTree(grid)

    assert tree[2][0][0] == "Global"
    assert tree[2][0][2][0][0] == "CaseId"
    assert tree[2][1][2][0][1].tolist() == [[2, 2, 1]]


def test_vtk_to_cgns_tree_converts_vertex_cells():
    """Supported VTK vertex cells retain their connectivity."""
    vtk = pytest.importorskip("vtk")

    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.InsertNextPoint(0.0, 0.0, 0.0)
    grid.SetPoints(points)
    vertex = vtk.vtkVertex()
    vertex.GetPointIds().SetId(0, 0)
    grid.InsertNextCell(vertex.GetCellType(), vertex.GetPointIds())

    tree = cgns_vtk.VtkToCGNSTree(grid)

    elements = tree[2][0][2][0][2][1]
    connectivity = elements[2][1]
    assert connectivity[1].tolist() == [1]


def test_cgns_tree_to_vtk_transfers_boundary_and_subregion_tags():
    """CGNS point and element tags become signed-character VTK masks."""
    pytest.importorskip("vtk")
    coordinates = _node(
        "GridCoordinates",
        None,
        [
            _node("CoordinateX", np.array([0.0, 1.0, 0.0, 1.0])),
            _node("CoordinateY", np.array([0.0, 0.0, 1.0, 1.0])),
        ],
        label="GridCoordinates_t",
    )
    elements = _node(
        "Elements_5",
        np.array([5], dtype=np.int32),
        [
            _node(
                "ElementRange",
                np.array([1, 2], dtype=np.int32),
                label="IndexRange_t",
            ),
            _node(
                "ElementConnectivity",
                np.array([1, 2, 3, 2, 4, 3], dtype=np.int32),
            ),
        ],
        label="Elements_t",
    )
    zone_bc = _node(
        "ZoneBC",
        None,
        [
            _node(
                "wall_nodes",
                np.array(list("Null"), dtype="|S1"),
                [
                    _node(
                        "PointList",
                        np.array([[1, 3]], dtype=np.int32),
                        label="IndexArray_t",
                    ),
                    _node(
                        "GridLocation",
                        np.array(list("Vertex"), dtype="|S1"),
                        label="GridLocation_t",
                    ),
                ],
                label="BC_t",
            )
        ],
        label="ZoneBC_t",
    )
    subregion = _node(
        "selected_cells_ZSR",
        np.array([[1, 2]], dtype=np.int32),
        [
            _node(
                "PointList",
                np.array([2], dtype=np.int32),
                label="IndexArray_t",
            ),
            _node(
                "GridLocation",
                np.array(list("CellCenter"), dtype="|S1"),
                label="GridLocation_t",
            ),
            _node(
                "FamilyName",
                np.array(list("selected_cells"), dtype="|S1"),
                label="FamilyName_t",
            ),
        ],
        label="ZoneSubRegion_t",
    )
    zone = _node(
        "Zone",
        np.array([[4, 2, 0]], dtype=np.int32),
        [
            coordinates,
            elements,
            _node("ZoneType", "Unstructured", label="ZoneType_t"),
            subregion,
            zone_bc,
        ],
        label="Zone_t",
    )
    tree = _node(
        "CGNSTree",
        None,
        [_node("Base", np.array([2, 2]), [zone], label="CGNSBase_t")],
        label="CGNSTree_t",
    )

    output = cgns_vtk.CGNSTreeToVtk(tree)
    point_tag = output.GetPointData().GetArray("wall_nodes")
    cell_tag = output.GetCellData().GetArray("selected_cells")

    assert point_tag.GetDataTypeAsString() == "signed char"
    assert cell_tag.GetDataTypeAsString() == "signed char"
    assert [point_tag.GetTuple1(i) for i in range(4)] == [1, 0, 1, 0]
    assert [cell_tag.GetTuple1(i) for i in range(2)] == [0, 1]


def test_vtk_to_cgns_tree_separates_char_tags_from_binary_float_fields():
    """Only binary character arrays are reconstructed as CGNS tags."""
    vtk = pytest.importorskip("vtk")
    from vtk.util import numpy_support

    grid = vtk.vtkUnstructuredGrid()
    points = vtk.vtkPoints()
    points.SetData(
        numpy_support.numpy_to_vtk(
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            deep=True,
        )
    )
    grid.SetPoints(points)
    for point_ids in [(0, 1, 2), (1, 3, 2)]:
        triangle = vtk.vtkTriangle()
        for index, point_id in enumerate(point_ids):
            triangle.GetPointIds().SetId(index, point_id)
        grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())

    point_tag = numpy_support.numpy_to_vtk(
        np.array([1, 0, 1, 0], dtype=np.int8), deep=True
    )
    point_tag.SetName("wall_nodes")
    grid.GetPointData().AddArray(point_tag)
    cell_tag = numpy_support.numpy_to_vtk(np.array([0, 1], dtype=np.uint8), deep=True)
    cell_tag.SetName("selected_cells")
    grid.GetCellData().AddArray(cell_tag)
    binary_field = numpy_support.numpy_to_vtk(np.array([0.0, 1.0, 0.0, 1.0]), deep=True)
    binary_field.SetName("binary_field")
    grid.GetPointData().AddArray(binary_field)

    tree = cgns_vtk.VtkToCGNSTree(grid)
    zone = tree[2][0][2][0]
    zone_bc = next(child for child in zone[2] if child[3] == "ZoneBC_t")
    subregion = next(child for child in zone[2] if child[3] == "ZoneSubRegion_t")
    vertex_flow = next(
        child
        for child in zone[2]
        if child[3] == "FlowSolution_t" and child[0] == "VertexData"
    )

    assert [child[0] for child in zone_bc[2]] == ["wall_nodes"]
    np.testing.assert_array_equal(
        zone_bc[2][0][2][0][1], np.array([[1, 3]], dtype=np.int32)
    )
    assert subregion[0] == "selected_cells_ZSR"
    np.testing.assert_array_equal(subregion[2][0][1], np.array([[2]], dtype=np.int32))
    assert [child[0] for child in vertex_flow[2] if child[3] == "DataArray_t"] == [
        "binary_field"
    ]

    restored = cgns_vtk.CGNSTreeToVtk(tree)
    restored_point_tag = restored.GetPointData().GetArray("wall_nodes")
    restored_cell_tag = restored.GetCellData().GetArray("selected_cells")
    assert restored_point_tag.GetDataTypeAsString() == "signed char"
    assert restored_cell_tag.GetDataTypeAsString() == "signed char"
    assert [restored_point_tag.GetTuple1(i) for i in range(4)] == [1, 0, 1, 0]
    assert [restored_cell_tag.GetTuple1(i) for i in range(2)] == [0, 1]
    assert restored.GetPointData().GetArray("binary_field") is not None


def test_cgns_tree_to_vtk_transfers_zone_family_as_full_cell_tag():
    """A zone family becomes a tag on all top-dimensional cells."""
    pytest.importorskip("vtk")
    zone = _node(
        "Zone",
        np.array([[3, 1, 0]], dtype=np.int32),
        [
            _node(
                "GridCoordinates",
                None,
                [
                    _node("CoordinateX", np.array([0.0, 1.0, 0.0])),
                    _node("CoordinateY", np.array([0.0, 0.0, 1.0])),
                ],
                label="GridCoordinates_t",
            ),
            _node(
                "Elements_5",
                np.array([5], dtype=np.int32),
                [
                    _node(
                        "ElementRange",
                        np.array([1, 1], dtype=np.int32),
                        label="IndexRange_t",
                    ),
                    _node("ElementConnectivity", np.array([1, 2, 3])),
                ],
                label="Elements_t",
            ),
            _node("ZoneType", "Unstructured", label="ZoneType_t"),
            _node(
                "FamilyName",
                np.array(list("fluid"), dtype="|S1"),
                label="FamilyName_t",
            ),
        ],
        label="Zone_t",
    )
    tree = _node(
        "CGNSTree",
        None,
        [_node("Base", np.array([2, 2]), [zone], label="CGNSBase_t")],
        label="CGNSTree_t",
    )

    output = cgns_vtk.CGNSTreeToVtk(tree)
    family_tag = output.GetCellData().GetArray("fluid")

    assert family_tag.GetDataTypeAsString() == "signed char"
    assert family_tag.GetTuple1(0) == 1


def _metadata_test_zone(name: str, z_coordinate: float = 0.0) -> list:
    """Build a triangle zone for CGNS hierarchy metadata tests.

    Args:
        name: Name of the CGNS zone.
        z_coordinate: Constant third coordinate assigned to each point.

    Returns:
        A pyCGNS-style unstructured ``Zone_t`` node.
    """
    return _node(
        name,
        np.array([[3, 1, 0]], dtype=np.int32),
        [
            _node(
                "GridCoordinates",
                None,
                [
                    _node("CoordinateX", np.array([0.0, 1.0, 0.0])),
                    _node("CoordinateY", np.array([0.0, 0.0, 1.0])),
                    _node("CoordinateZ", np.full(3, z_coordinate)),
                ],
                label="GridCoordinates_t",
            ),
            _node(
                "Elements_5",
                np.array([5], dtype=np.int32),
                [
                    _node(
                        "ElementRange",
                        np.array([1, 1], dtype=np.int32),
                        label="IndexRange_t",
                    ),
                    _node("ElementConnectivity", np.array([1, 2, 3])),
                ],
                label="Elements_t",
            ),
            _node("ZoneType", "Unstructured", label="ZoneType_t"),
        ],
        label="Zone_t",
    )


def test_cgns_vtk_round_trip_restores_training_flow_and_element_paths():
    """A pre-existing VertexFields and TRI_3 section retain their CGNS paths."""
    pytest.importorskip("vtk")
    zone = _metadata_test_zone("TrainingZone")
    elements = next(child for child in zone[2] if child[3] == "Elements_t")
    elements[0] = "Elements_TRI_3"
    zone[2].extend(
        [
            _node(
                "VertexFields",
                None,
                [_node("existing_field", np.array([1.0, 2.0, 3.0]))],
                label="FlowSolution_t",
            ),
            _node(
                "CellFields",
                None,
                [
                    _node("GridLocation", "CellCenter", label="GridLocation_t"),
                    _node("cell_field", np.array([4.0])),
                ],
                label="FlowSolution_t",
            ),
        ]
    )
    tree = _node(
        "CGNSTree",
        None,
        [_node("Base", np.array([2, 2]), [zone], label="CGNSBase_t")],
        label="CGNSTree_t",
    )

    grid = cgns_vtk.CGNSTreeToVtk(tree)
    from vtk.util import numpy_support

    distance = numpy_support.numpy_to_vtk(np.array([0.2, 0.3, 0.4]), deep=True)
    distance.SetName("distance_to_boundary")
    grid.GetPointData().AddArray(distance)
    restored = cgns_vtk.VtkToCGNSTree(grid)[2][0][2][0]

    flows = {node[0]: node for node in restored[2] if node[3] == "FlowSolution_t"}
    assert set(flows) == {"VertexFields", "CellFields"}
    assert {
        node[0] for node in flows["VertexFields"][2] if node[3] == "DataArray_t"
    } == {"existing_field", "distance_to_boundary"}
    np.testing.assert_array_equal(
        next(
            node[1] for node in flows["VertexFields"][2] if node[0] == "existing_field"
        ),
        [1.0, 2.0, 3.0],
    )
    assert [node[0] for node in restored[2] if node[3] == "Elements_t"] == [
        "Elements_TRI_3"
    ]


def test_cgns_vtk_names_survive_vtk_serialization():
    """Reserved field data survives transport through a VTK XML grid."""
    vtk = pytest.importorskip("vtk")
    zone = _metadata_test_zone("Zone")
    next(child for child in zone[2] if child[3] == "Elements_t")[0] = "Elements_TRI_3"
    zone[2].append(
        _node(
            "VertexFields",
            None,
            [_node("field", np.array([1.0, 2.0, 3.0]))],
            label="FlowSolution_t",
        )
    )
    base = _node("Base", np.array([2, 2]), [zone], label="CGNSBase_t")
    writer = vtk.vtkXMLUnstructuredGridWriter()
    writer.SetWriteToOutputString(True)
    writer.SetInputData(cgns_vtk.CGNSBaseToVtk(base))
    assert writer.Write() == 1
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.ReadFromInputStringOn()
    reader.SetInputString(writer.GetOutputString())
    reader.Update()

    restored = cgns_vtk.VtkToCGNSTree(reader.GetOutput())[2][0][2][0]
    assert "VertexFields" in [node[0] for node in restored[2]]
    assert "Elements_TRI_3" in [node[0] for node in restored[2]]


def test_cgns_vtk_round_trip_ambiguous_flow_names_use_generated_name():
    """Two vertex solutions cannot map unambiguously to one VTK point data set."""
    pytest.importorskip("vtk")
    zone = _metadata_test_zone("Zone")
    for name in ("First", "Second"):
        zone[2].append(
            _node(
                name,
                None,
                [_node(name.lower(), np.array([1.0, 2.0, 3.0]))],
                label="FlowSolution_t",
            )
        )
    base = _node("Base", np.array([2, 2]), [zone], label="CGNSBase_t")

    restored = cgns_vtk.VtkToCGNSTree(cgns_vtk.CGNSBaseToVtk(base))[2][0][2][0]
    assert [node[0] for node in restored[2] if node[3] == "FlowSolution_t"] == [
        "VertexData"
    ]


def test_cgns_vtk_round_trip_ambiguous_element_names_use_generated_name():
    """Separate same-type sections merged into VTK cannot retain both names."""
    pytest.importorskip("vtk")
    zone = _metadata_test_zone("Zone")
    first = next(child for child in zone[2] if child[3] == "Elements_t")
    first[0] = "FirstTriangles"
    second = _node(
        "SecondTriangles",
        np.array([5], dtype=np.int32),
        [
            _node("ElementRange", np.array([2, 2]), label="IndexRange_t"),
            _node("ElementConnectivity", np.array([1, 3, 2])),
        ],
        label="Elements_t",
    )
    zone[2].append(second)
    base = _node("Base", np.array([2, 2]), [zone], label="CGNSBase_t")

    restored = cgns_vtk.VtkToCGNSTree(cgns_vtk.CGNSBaseToVtk(base))[2][0][2][0]
    assert [node[0] for node in restored[2] if node[3] == "Elements_t"] == [
        "Elements_5"
    ]


def test_cgns_vtk_round_trip_preserves_base_dimensions_and_names():
    """Grid metadata restores base dimensions, base name, and zone name."""
    pytest.importorskip("vtk")
    tree = _node(
        "CGNSTree",
        None,
        [
            _node(
                "SurfaceBase",
                np.array([2, 3], dtype=np.int32),
                [_metadata_test_zone("SurfaceZone", z_coordinate=2.0)],
                label="CGNSBase_t",
            )
        ],
        label="CGNSTree_t",
    )

    vtk_grid = cgns_vtk.CGNSTreeToVtk(tree)
    field_data = vtk_grid.GetFieldData()
    assert field_data.GetArray(cgns_vtk.PLAID_CGNS_BASE_NAME) is not None
    assert field_data.GetArray(cgns_vtk.PLAID_CGNS_BASE_DIMENSIONS) is not None
    assert field_data.GetArray(cgns_vtk.PLAID_CGNS_ZONE_NAME) is not None

    restored = cgns_vtk.VtkToCGNSTree(vtk_grid)

    assert [base[0] for base in restored[2]] == ["SurfaceBase"]
    np.testing.assert_array_equal(restored[2][0][1], np.array([2, 3]))
    restoredZone = restored[2][0][2][0]
    assert restoredZone[0] == "SurfaceZone"
    gridCoordinates = next(
        child for child in restoredZone[2] if child[3] == "GridCoordinates_t"
    )
    assert [coordinate[0] for coordinate in gridCoordinates[2]] == [
        "CoordinateX",
        "CoordinateY",
        "CoordinateZ",
    ]
    np.testing.assert_array_equal(gridCoordinates[2][2][1], np.full(3, 2.0))


def test_stored_planar_base_dimensions_override_legacy_third_coordinate():
    """Stored physical dimension two suppresses the legacy third coordinate."""
    pytest.importorskip("vtk")
    tree = _node(
        "CGNSTree",
        None,
        [
            _node(
                "PlanarBase",
                np.array([2, 2], dtype=np.int32),
                [_metadata_test_zone("PlanarZone", z_coordinate=7.0)],
                label="CGNSBase_t",
            )
        ],
        label="CGNSTree_t",
    )

    vtk_grid = cgns_vtk.CGNSTreeToVtk(tree)
    restored = cgns_vtk.VtkToCGNSTree(vtk_grid, ensure_3D_points=True)
    restoredZone = restored[2][0][2][0]
    gridCoordinates = next(
        child for child in restoredZone[2] if child[3] == "GridCoordinates_t"
    )

    np.testing.assert_array_equal(restored[2][0][1], np.array([2, 2]))
    assert [coordinate[0] for coordinate in gridCoordinates[2]] == [
        "CoordinateX",
        "CoordinateY",
    ]


def test_cgns_vtk_round_trip_groups_zones_by_stored_base_metadata():
    """Leaf-grid metadata reconstructs multiple bases from VTK multiblocks."""
    pytest.importorskip("vtk")
    tree = _node(
        "CGNSTree",
        None,
        [
            _node(
                "SurfaceBase",
                np.array([2, 3], dtype=np.int32),
                [
                    _metadata_test_zone("SurfaceA"),
                    _metadata_test_zone("SurfaceB", z_coordinate=1.0),
                ],
                label="CGNSBase_t",
            ),
            _node(
                "PlanarBase",
                np.array([2, 2], dtype=np.int32),
                [_metadata_test_zone("PlanarZone")],
                label="CGNSBase_t",
            ),
        ],
        label="CGNSTree_t",
    )

    vtk_multiblock = cgns_vtk.CGNSTreeToVtk(tree)
    restored = cgns_vtk.VtkToCGNSTree(vtk_multiblock)

    assert [base[0] for base in restored[2]] == ["SurfaceBase", "PlanarBase"]
    np.testing.assert_array_equal(restored[2][0][1], np.array([2, 3]))
    np.testing.assert_array_equal(restored[2][1][1], np.array([2, 2]))
    assert [zone[0] for zone in restored[2][0][2]] == ["SurfaceA", "SurfaceB"]
    assert [zone[0] for zone in restored[2][1][2]] == ["PlanarZone"]
