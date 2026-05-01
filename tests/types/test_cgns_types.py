# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#

import importlib.util
import runpy
from pathlib import Path

import pytest


def _load_cgns_types_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "src" / "plaid" / "types" / "cgns_types.py"
    )
    spec = importlib.util.spec_from_file_location("cgns_types_for_test", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cgns_node_and_tree_alias():
    cgns_types = _load_cgns_types_module()
    child = cgns_types.CGNSNode(name="Child", value=1, label="DataArray_t")
    root = cgns_types.CGNSNode(
        name="Root",
        value=None,
        children=[child],
        label="CGNSTree_t",
    )

    assert root.name == "Root"
    assert root.children[0].name == "Child"
    assert cgns_types.CGNSTree is cgns_types.CGNSNode


def test_cgns_path_properties_and_zone_method():
    cgns_types = _load_cgns_types_module()
    path = cgns_types.CGNSPath("Base_1_0/Zone/GridCoordinates")

    assert path.root == "Base_1_0/Zone/GridCoordinates"
    assert path.path == "Base_1_0/Zone/GridCoordinates"
    assert path.base == "Base_1_0"
    assert path.zone() == "Zone"


def test_cgns_path_rejects_invalid_pattern():
    cgns_types = _load_cgns_types_module()
    with pytest.raises(ValueError, match="Invalid CGNS variable format"):
        cgns_types.CGNSPath("InvalidPath")


def test_module_main_example_runs(capsys):
    module_path = (
        Path(__file__).resolve().parents[2] / "src" / "plaid" / "types" / "cgns_types.py"
    )
    runpy.run_path(str(module_path), run_name="__main__")

    out = capsys.readouterr().out
    assert "Valid path: Base_1_0/Zone/GridCoordinates" in out
    assert "Valid path: Base_0_0/Normal/Normals" in out
    assert "Invalid path error:" in out