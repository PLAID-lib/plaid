"""Exercise bundled Sample readers with real VTK and on-disk Sample data."""

import ast
import builtins
import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vtk
from vtk.util.vtkAlgorithm import VTKPythonAlgorithmBase

from plaid import Sample
from plaid.cli.paraview_plugin import get_ParaView_plugin_path_one_file
from plaid.utils.cgns_helper import compare_cgns_trees
from plaid.utils.cgns_vtk import VtkToCGNSTree
from plaid.utils.sample_json import (
    sample_from_json,
    sample_to_json,
    sample_to_json_payload,
)


@pytest.fixture
def load_plugin(tmp_path):
    """Load the full bundled plugin, replacing only unavailable PV decorators."""
    directory = get_ParaView_plugin_path_one_file(tmp_path)
    path = Path(directory) / "PlaidParaViewPlugin.py"

    def load(with_plaid=True):
        def decorator(*_args, **_kwargs):
            return lambda obj: obj

        decorators = SimpleNamespace(
            **{
                name: decorator
                for name in (
                    "reader",
                    "source",
                    "filter",
                    "stringvector",
                    "intvector",
                    "doublevector",
                    "xml",
                    "filelist",
                    "filechooser",
                    "input",
                    "datatype",
                    "dataarrayselection",
                )
            }
        )

        def filechooser(extensions, file_description):
            return decorator(extensions, file_description)

        decorators.filechooser = filechooser
        algorithm = SimpleNamespace(
            VTKPythonAlgorithmBase=VTKPythonAlgorithmBase,
            smproxy=decorators,
            smproperty=decorators,
            smdomain=decorators,
            smhint=decorators,
        )

        def import_module(name, *args, **kwargs):
            if name == "paraview.util.vtkAlgorithm":
                return algorithm
            if not with_plaid and (
                name in ("plaid", "CGNS") or name.startswith(("plaid.", "CGNS."))
            ):
                raise ImportError("PLAID deliberately unavailable")
            return builtins.__import__(name, *args, **kwargs)

        namespace = {"__builtins__": dict(vars(builtins), __import__=import_module)}
        exec(compile(path.read_text(), str(path), "exec"), namespace)
        return namespace

    return load


@pytest.fixture
def timed_sample():
    """Build a real triangular mesh with two deliberately unsorted times."""
    points = vtk.vtkPoints()
    for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0)):
        points.InsertNextPoint(*point)
    grid = vtk.vtkUnstructuredGrid()
    grid.SetPoints(points)
    triangle = vtk.vtkTriangle()
    for index in range(3):
        triangle.GetPointIds().SetId(index, index)
    grid.InsertNextCell(triangle.GetCellType(), triangle.GetPointIds())
    tree = VtkToCGNSTree(grid)
    sample = Sample()
    sample.add_tree(tree, time=2.0, in_place=False)
    sample.add_tree(tree, time=0.5, in_place=False)
    return sample_from_json(sample_to_json(sample))


@pytest.mark.parametrize("with_plaid", [True, False])
def test_json_reader_bundled_pipeline(load_plugin, tmp_path, timed_sample, with_plaid):
    """JSON decoding, caching and VTK time navigation work without PLAID."""
    plugin = load_plugin(with_plaid)
    if not with_plaid:
        assert "PlaidSampleReader" not in plugin
        assert "PlaidDataSetReader" not in plugin
        assert "PlaidExplorer" in plugin
    reader = plugin["PlaidSampleJSONReader"]()
    assert reader.GetSampleData() == {}
    path = tmp_path / "sample.json"
    path.write_text(sample_to_json(timed_sample))
    reader.SetFileName(str(path))
    assert reader.GetTimestepValues() == [0.5, 2.0]
    cached = reader.GetSampleData()
    reader.SetFileName(str(path))
    assert reader.GetSampleData() is cached
    for time, tree in cached.items():
        assert compare_cgns_trees(timed_sample.data[time], tree)
    reader.UpdateInformation()
    info = reader.GetOutputInformation(0)
    executive = reader.GetExecutive()
    assert info.Get(executive.TIME_STEPS()) == (0.5, 2.0)
    assert info.Get(executive.TIME_RANGE()) == (0.5, 2.0)
    reader.UpdateTimeStep(1.8)
    output = reader.GetOutputDataObject(0)
    assert output.GetNumberOfPoints() == 3
    assert output.GetNumberOfCells() == 1
    assert output.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()) == 2.0
    info.Remove(executive.UPDATE_TIME_STEP())
    _, time = reader._get_requested_output(info)
    assert time == 0.5

    empty = tmp_path / "empty.json"
    empty.write_text(sample_to_json(Sample()))
    reader.SetFileName(str(empty))
    assert reader._sample_cache is None
    reader.Update()
    assert reader.GetTimestepValues() == []
    assert not info.Has(executive.TIME_STEPS())
    assert not info.Has(executive.TIME_RANGE())
    assert reader.GetOutputDataObject(0).GetNumberOfPoints() == 0


def test_directory_reader_roundtrip(load_plugin, tmp_path, timed_sample):
    """The optional directory reader loads real files through Sample.load_from_dir."""
    reader = load_plugin()["PlaidSampleReader"]()
    path = tmp_path / "sample"
    timed_sample.save_to_dir(path)
    reader.SetFileName(str(path))
    assert reader.GetTimestepValues() == [0.5, 2.0]
    for time, tree in reader.GetSampleData().items():
        assert compare_cgns_trees(timed_sample.data[time], tree)
    reader.UpdateTimeStep(0.5)
    assert reader.GetOutputDataObject(0).GetNumberOfCells() == 1
    reader.SetFileName(str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        reader.GetSampleData()


@pytest.mark.parametrize(
    "payload, message",
    [
        ([], "format"),
        ({"format": "other"}, "format"),
        ({"format": "plaid-sample-json", "version": 2}, "version"),
        ({"format": "plaid-sample-json", "version": 1, "trees": {}}, "list"),
        ({"format": "plaid-sample-json", "version": 1, "trees": [None]}, "entry"),
        ({"format": "plaid-sample-json", "version": 1, "trees": [{}]}, "entry"),
        (
            {
                "format": "plaid-sample-json",
                "version": 1,
                "trees": [{"time": "bad", "tree": {}}],
            },
            "numeric",
        ),
    ],
)
def test_json_reader_rejects_invalid_payload(load_plugin, tmp_path, payload, message):
    """Malformed Sample envelopes produce explicit errors instead of cached data."""
    reader = load_plugin(False)["PlaidSampleJSONReader"]()
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload))
    reader.SetFileName(str(path))
    with pytest.raises(ValueError, match=message):
        reader.GetSampleData()
    assert reader._sample_cache is None


def test_json_reader_io_errors(load_plugin, tmp_path):
    """Missing files and malformed JSON preserve their original exceptions."""
    reader = load_plugin(False)["PlaidSampleJSONReader"]()
    path = tmp_path / "missing.json"
    reader.SetFileName(str(path))
    with pytest.raises(FileNotFoundError):
        reader.GetSampleData()
    path.write_text("not JSON")
    with pytest.raises(json.JSONDecodeError):
        reader.GetSampleData()


def test_json_reader_single_time_and_multiblock(load_plugin, tmp_path, timed_sample):
    """One timestamp and multi-base data retain their actual VTK output type."""
    tree = timed_sample.data[0.5]
    second_base = next(node for node in tree[2] if node[3] == "CGNSBase_t")
    second_base = copy.deepcopy(second_base)
    second_base[0] = "SecondBase"
    tree[2].append(second_base)
    path = tmp_path / "multiblock.json"
    path.write_text(json.dumps(sample_to_json_payload(Sample(data={3.0: tree}))))
    reader = load_plugin(False)["PlaidSampleJSONReader"]()
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutputDataObject(0)
    assert output.IsA("vtkMultiBlockDataSet")
    assert output.GetNumberOfBlocks() == 2
    assert reader.GetOutputInformation(0).Get(reader.GetExecutive().TIME_STEPS()) == (
        3.0,
    )
    assert np.isclose(
        output.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()), 3.0
    )


def test_json_reader_structured_output(load_plugin, tmp_path):
    """Structured CGNS zones keep their structured VTK output type."""
    grid = vtk.vtkStructuredGrid()
    grid.SetDimensions(2, 2, 1)
    points = vtk.vtkPoints()
    for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)):
        points.InsertNextPoint(*point)
    grid.SetPoints(points)
    path = tmp_path / "structured.json"
    path.write_text(sample_to_json(Sample(data={0.0: VtkToCGNSTree(grid)})))
    reader = load_plugin(False)["PlaidSampleJSONReader"]()
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutputDataObject(0)
    assert output.IsA("vtkStructuredGrid")
    assert output.GetNumberOfPoints() == 4
    assert output.GetNumberOfCells() == 1


@pytest.fixture
def vtk_compatible_sample(timed_sample):
    """Use standard two-value Elements_t descriptors required by VTK's reader.

    The existing VTK-to-CGNS converter emits only the element type. Keep its
    output unchanged in production, and test its rejection separately below.
    """
    sample = timed_sample.copy()

    def normalize(node):
        if node[3] == "Elements_t":
            node[1] = np.array([node[1][0], 0], dtype=np.int32)
        for child in node[2]:
            normalize(child)

    for tree in sample.data.values():
        normalize(tree)
    return sample


@pytest.mark.parametrize("with_plaid", [True, False])
def test_experimental_vtk_reader_times_fields_and_cache(
    load_plugin, tmp_path, vtk_compatible_sample, with_plaid
):
    """The standalone VTK reader loads physical times and real mesh fields."""
    timed_sample = vtk_compatible_sample
    for time in timed_sample.data:
        timed_sample.add_field("pressure", np.full(3, time), time=time)
        timed_sample.add_field(
            "density", np.array([time + 10]), time=time, location="CellCenter"
        )
    path = tmp_path / "vtk_sample"
    timed_sample.save_to_dir(path)
    plugin = load_plugin(with_plaid)
    reader = plugin["PlaidSampleVTKReader"]()
    assert reader.GetTimestepValues() == []
    reader.SetFileName(str(path))
    assert reader.GetTimestepValues() == [0.5, 2.0]
    assert reader._vtk_cache is None
    cached = reader.GetSampleData()
    reader.SetFileName(str(path))
    assert reader.GetSampleData() is cached
    reader.Update()
    info = reader.GetOutputInformation(0)
    assert info.Get(reader.GetExecutive().TIME_RANGE()) == (0.5, 2.0)
    for requested, selected in ((1.8, 2.0), (0.6, 0.5), (3.0, 2.0)):
        reader.UpdateTimeStep(requested)
        output = reader.GetOutputDataObject(0)
        assert output.IsA("vtkMultiBlockDataSet")
        assert output.GetNumberOfPoints() == 3
        assert output.GetNumberOfCells() == 1
        assert (
            output.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()) == selected
        )
        leaf = output.GetBlock(0).GetBlock(0)
        assert leaf.GetPointData().GetArray("pressure").GetValue(0) == selected
        assert leaf.GetCellData().GetArray("density").GetValue(0) == selected + 10
    cache = reader._vtk_cache
    reader._get_requested_output(info)
    assert reader._vtk_cache is cache
    info.Remove(reader.GetExecutive().UPDATE_TIME_STEP())
    assert reader._get_requested_output(info)[1] == 0.5

    empty = tmp_path / "empty_vtk_sample"
    Sample().save_to_dir(empty)
    reader.SetFileName(str(empty))
    assert reader._sample_cache is None
    assert reader._vtk_cache is None
    reader.Update()
    assert reader.GetOutputDataObject(0).GetNumberOfBlocks() == 0
    assert not info.Has(reader.GetExecutive().TIME_STEPS())
    assert not info.Has(reader.GetExecutive().TIME_RANGE())


@pytest.mark.parametrize("kind", ["structured", "multibase"])
def test_experimental_vtk_reader_layout(
    load_plugin, tmp_path, vtk_compatible_sample, kind
):
    """Native VTK hierarchy is retained for structured and multi-base Samples."""
    timed_sample = vtk_compatible_sample
    if kind == "structured":
        grid = vtk.vtkStructuredGrid()
        grid.SetDimensions(2, 2, 1)
        points = vtk.vtkPoints()
        for point in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)):
            points.InsertNextPoint(*point)
        grid.SetPoints(points)
        sample = Sample()
        sample.add_tree(VtkToCGNSTree(grid), time=3.5)
    else:
        sample = Sample(data={3.5: copy.deepcopy(timed_sample.data[0.5])})
        base = copy.deepcopy(sample.data[3.5][2][0])
        base[0] = "Base_second"
        sample.data[3.5][2].append(base)
    path = tmp_path / "layout"
    sample.save_to_dir(path)
    reader = load_plugin(False)["PlaidSampleVTKReader"]()
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutputDataObject(0)
    assert output.IsA("vtkMultiBlockDataSet")
    if kind == "structured":
        assert output.GetBlock(0).GetBlock(0).IsA("vtkStructuredGrid")
        assert reader.GetTimestepValues() == [3.5]
    else:
        assert output.GetNumberOfBlocks() == 2
        assert output.GetNumberOfPoints() == 6


@pytest.mark.parametrize(
    "kind, error, message",
    [
        ("missing", FileNotFoundError, "does not exist"),
        ("file", NotADirectoryError, "not a directory"),
        ("no_meshes", ValueError, "no meshes"),
        ("no_cgns", ValueError, "No Sample CGNS"),
        ("corrupt", ValueError, "Unable to read"),
        ("duplicate", ValueError, "Duplicate physical"),
    ],
)
def test_experimental_vtk_reader_invalid_input(
    load_plugin, tmp_path, timed_sample, kind, error, message
):
    """Invalid directories and CGNS files fail explicitly without caching."""
    path = tmp_path / "invalid_vtk"
    if kind == "file":
        path.write_text("not a directory")
    elif kind != "missing":
        path.mkdir()
        if kind == "no_meshes":
            (path / "unrelated.txt").write_text("not a sample")
        elif kind == "no_cgns":
            (path / "meshes").mkdir()
        elif kind == "corrupt":
            (path / "meshes").mkdir()
            (path / "meshes/mesh_000000000.cgns").write_text("not CGNS")
        elif kind == "duplicate":
            path.rmdir()
            timed_sample.save_to_dir(path)
            shutil.copyfile(
                path / "meshes/mesh_000000000.cgns",
                path / "meshes/mesh_000000001.cgns",
            )
    reader = load_plugin(False)["PlaidSampleVTKReader"]()
    reader.SetFileName(str(path))
    with pytest.raises(error, match=message):
        reader.GetSampleData()
    assert reader._sample_cache is None


def test_experimental_vtk_reader_rejects_incompatible_elements(
    load_plugin, tmp_path, timed_sample
):
    """VTK read errors must not silently return an incomplete mesh."""
    path = tmp_path / "incompatible"
    timed_sample.save_to_dir(path)
    reader = load_plugin(False)["PlaidSampleVTKReader"]()
    reader.SetFileName(str(path))
    info = reader.GetOutputInformation(0)
    with pytest.raises(ValueError, match="Unable to read CGNS"):
        reader._get_requested_output(info)
    assert reader._vtk_cache is None


def test_experimental_vtk_reader_vtk_default_time(
    load_plugin, tmp_path, vtk_compatible_sample
):
    """Document VTK's synthetic zero time when a file lacks time metadata."""
    tree = copy.deepcopy(vtk_compatible_sample.data[0.5])
    for base in tree[2]:
        base[2] = [child for child in base[2] if child[3] != "BaseIterativeData_t"]
    path = tmp_path / "no_times"
    Sample(data={0.5: tree}).save_to_dir(path)
    reader = load_plugin(False)["PlaidSampleVTKReader"]()
    reader.SetFileName(str(path))
    assert reader.GetTimestepValues() == [0.0]


@pytest.mark.parametrize(
    "reader_name",
    ["PlaidSampleJSONReader", "PlaidSampleVTKReader", "PlaidSampleReader"],
)
def test_readers_accept_paraview_unset_filename(load_plugin, reader_name):
    """ParaView's string None placeholder must not trigger disk access."""
    reader = load_plugin()[reader_name]()
    reader.SetFileName("None")
    assert reader.GetTimestepValues() == []
    reader.Update()
    assert reader.GetOutputDataObject(0).GetNumberOfPoints() == 0


@pytest.mark.parametrize("single_time", [True, False])
@pytest.mark.parametrize("input_directory", ["sample", "meshes", "direct"])
def test_experimental_vtk_reader_directory_selection(
    load_plugin, tmp_path, vtk_compatible_sample, single_time, input_directory
):
    """Accept a Sample root or the directory directly containing its CGNS files."""
    sample = vtk_compatible_sample
    if single_time:
        sample = Sample(data={0.5: sample.data[0.5]})
    root = tmp_path / "sample"
    sample.save_to_dir(root)
    selected = root
    if input_directory == "meshes":
        selected = root / "meshes"
    elif input_directory == "direct":
        selected = tmp_path / "exported_meshes"
        (root / "meshes").rename(selected)
    reader = load_plugin(False)["PlaidSampleVTKReader"]()
    reader.SetFileName(str(selected))
    expected = [0.5] if single_time else [0.5, 2.0]
    assert reader.GetTimestepValues() == expected
    reader.UpdateTimeStep(expected[-1])
    output = reader.GetOutputDataObject(0)
    assert output.GetNumberOfPoints() == 3
    assert output.GetNumberOfCells() == 1
    assert (
        output.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()) == expected[-1]
    )


def test_experimental_globals_output(load_plugin, tmp_path, vtk_compatible_sample):
    """Globals are a synchronized table, never a spatial base or time bookkeeping."""
    sample = vtk_compatible_sample
    for time in sample.data:
        sample.add_global("scalar", time, time=time)
        sample.add_global("matrix", np.array([[1, 2], [3, 4]]), time=time)
        sample.add_global("description", "sample text", time=time)
    path = tmp_path / "globals_sample"
    sample.save_to_dir(path)
    reader = load_plugin(False)["PlaidSampleVTKReader"]()
    reader.SetFileName(str(path))
    assert reader.GetNumberOfOutputPorts() == 2
    reader.UpdateInformation()
    for port in range(2):
        assert reader.GetOutputInformation(port).Get(
            reader.GetExecutive().TIME_STEPS()
        ) == (0.5, 2.0)
    for time in (2.0, 0.5):
        reader.UpdateTimeStep(time, 1)
        mesh = reader.GetOutputDataObject(0)
        table = reader.GetOutputDataObject(1)
        assert mesh.GetNumberOfPoints() == 3
        assert mesh.GetNumberOfBlocks() == 1
        assert table.IsA("vtkTable")
        assert table.GetNumberOfRows() == 6
        rows = {
            (
                table.GetValue(i, 0).ToString(),
                table.GetValue(i, 1).ToString(),
            ): table.GetValue(i, 2).ToString()
            for i in range(table.GetNumberOfRows())
        }
        assert rows[("scalar", "(0,)")] == str(time)
        assert rows[("matrix", "(1, 0)")] == "3"
        assert rows[("description", "")] == "sample text"
        assert not any(name == "Time" for name, _ in rows)
        for output in (mesh, table):
            assert (
                output.GetInformation().Get(vtk.vtkDataObject.DATA_TIME_STEP()) == time
            )
        cgns = reader._read_cgns(reader.GetSampleData()[time], time)
        assert cgns.GetBaseSelection().ArrayIsEnabled("Global") == 0
    empty = tmp_path / "empty_globals"
    Sample().save_to_dir(empty)
    reader.SetFileName(str(empty))
    reader.Update()
    assert reader.GetOutputDataObject(1).GetNumberOfRows() == 0
    assert reader._globals_cache is None


def test_experimental_globals_output_port_name():
    """The ParaView proxy labels its second output Globals."""
    path = (
        Path(__file__).parents[2]
        / "src/plaid/cli/paraview_plugin/PlaidParaViewPlugin.py"
    )
    module = ast.parse(path.read_text())
    reader = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "PlaidSampleVTKReader"
    )
    assert any(
        isinstance(decorator, ast.Call)
        and isinstance(decorator.func, ast.Attribute)
        and decorator.func.attr == "xml"
        and decorator.args
        and ast.literal_eval(decorator.args[0])
        == '<OutputPort index="1" name="Globals" />'
        for decorator in reader.decorator_list
    )
