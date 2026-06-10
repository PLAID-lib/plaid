"""Tests for direct CGNS-to-VTK conversion helpers."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from plaid.utils import cgns_vtk


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
        _ = number_of_values
        self.values = []

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
