"""Exercise bundled Sample readers with real VTK and on-disk Sample data."""

import builtins
import copy
import json
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
            if not with_plaid and (name == "plaid" or name.startswith("plaid.")):
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
