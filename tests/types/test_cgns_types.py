import importlib.util
from pathlib import Path

def _load_cgns_types_module():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "plaid"
        / "types"
        / "cgns_types.py"
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
