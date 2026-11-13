"""Exercise vector component selection with real VTK arrays and mappers."""

import numpy as np
import pytest
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from plaid.viewer.trame_app import server as srv


def _dataset(offset: float = 0.0):
    """Return a triangle carrying named point and unnamed cell vectors."""
    points = vtk.vtkPoints()
    for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        points.InsertNextPoint(*point)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    triangle = vtk.vtkTriangle()
    for index in range(3):
        triangle.GetPointIds().SetId(index, index)
    grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())
    array = numpy_to_vtk(
        np.array([[-3, 4, 0], [0, -2, 0], [1, 0, 0]], dtype=float) + offset,
        deep=True,
    )
    array.SetName("U")
    for index, name in enumerate(("x", "y", "dummy")):
        array.SetComponentName(index, name)
    grid.GetPointData().AddArray(array)
    cell = numpy_to_vtk(np.array([[-4.0, 2.0]]), deep=True)
    cell.SetName("force")
    grid.GetCellData().AddArray(cell)
    return grid


def test_component_options_and_multiblock_ranges():
    """Expose real components, hide padding, and aggregate signed ranges."""
    blocks = vtk.vtkMultiBlockDataSet()
    blocks.SetBlock(0, _dataset())
    blocks.SetBlock(1, _dataset(-1))
    assert srv._list_point_and_cell_fields(blocks) == (
        ["U", "U_x", "U_y"],
        ["force", "force_0", "force_1"],
    )
    assert srv._compute_field_range(blocks, "U_x", "point") == (-4, 1)
    assert srv._compute_field_range(blocks, "U_y", "point") == (-3, 4)
    assert srv._compute_field_range(blocks, "force_0", "cell") == (-4, -4)
    assert srv._compute_field_range(blocks, "U", "point") == (1, 26**0.5)
    assert srv._compute_field_range(blocks, "absent", "point") == (0, 1)
    assert srv._resolve_field_selection(blocks, "U_y", "point") == ("U", 1)
    assert srv._resolve_field_selection(blocks, "absent", "point") == ("absent", -1)
    assert srv._resolve_field_selection(None, "U", "point") == ("U", -1)


def test_non_numeric_arrays_do_not_create_components():
    """Keep string array names without attempting numeric component access."""
    attributes = vtk.vtkPointData()
    strings = vtk.vtkStringArray()
    strings.SetName("labels")
    strings.InsertNextValue("triangle")
    attributes.AddArray(strings)
    assert srv._field_array_options(attributes) == {"labels": ("labels", -1)}


def test_native_names_take_precedence():
    """Do not replace a real scalar with a generated component alias."""
    grid = _dataset()
    scalar = numpy_to_vtk(np.array([10.0, 11.0, 12.0]), deep=True)
    scalar.SetName("U_x")
    grid.GetPointData().AddArray(scalar)
    assert srv._field_array_options(grid.GetPointData())["U_x"] == ("U_x", -1)
    assert srv._compute_field_range(grid, "U_x", "point") == (10, 12)


@pytest.mark.parametrize(
    "association,field,source,component",
    [
        ("point", "U_x", "U", 0),
        ("point", "U_y", "U", 1),
        ("cell", "force_1", "force", 1),
    ],
)
def test_mapper_component_and_magnitude(association, field, source, component):
    """Select signed components and reset magnitude mode when switching back."""
    producer = vtk.vtkTrivialProducer()
    producer.SetOutput(_dataset())
    pipeline = srv._VtkPipeline.__new__(srv._VtkPipeline)
    pipeline.reader = producer
    pipeline.mapper = vtk.vtkCompositePolyDataMapper()
    pipeline.actor = vtk.vtkActor()
    pipeline.scalar_bar = vtk.vtkScalarBarActor()
    options = dict(
        association=association,
        cmap="viridis",
        show_edges=False,
        color_range=(-4, 4),
    )
    pipeline.update(field=field, **options)
    assert pipeline.mapper.GetArrayName() == source
    assert pipeline.lut.GetVectorComponent() == component
    assert pipeline.lut.GetVectorMode() == 1
    pipeline.update(field=source, **options)
    assert pipeline.lut.GetVectorMode() == 0
    pipeline.update(field=None, **options)
    assert not pipeline.mapper.GetScalarVisibility()


def test_component_ranges_follow_new_timestep():
    """Re-read current values instead of retaining a previous frame's range."""
    grid = _dataset()
    assert srv._compute_field_range(grid, "U_x", "point") == (-3, 1)
    array = grid.GetPointData().GetArray("U")
    array.SetComponent(0, 0, -8)
    array.Modified()
    assert srv._compute_field_range(grid, "U_x", "point") == (-8, 1)
