"""Utility functions for working with CGNS trees and nodes."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import CGNS.PAT.cgnsutils as CGU
import numpy as np

from plaid.types import CGNSTree


def get_base_names(
    tree: CGNSTree, full_path: bool = False, unique: bool = False
) -> list[str]:
    """Get a list of base names from a CGNSTree.

    Args:
        tree (CGNSTree): The CGNSTree containing the CGNSBase_t nodes.
        full_path (bool, optional): If True, return full base paths including '/' separators. Defaults to False.
        unique (bool, optional): If True, return unique base names. Defaults to False.

    Returns:
        list[str]: A list of base names.
    """
    base_paths = []
    if tree is not None:
        b_paths = CGU.getPathsByTypeSet(tree, "CGNSBase_t")
        for pth in b_paths:
            s_pth = pth.split("/")
            assert len(s_pth) == 2
            assert s_pth[0] == ""
            if full_path:
                base_paths.append(pth)
            else:
                base_paths.append(s_pth[1])

    if unique:
        return list(set(base_paths))
    else:
        return base_paths


def get_time_values(tree: CGNSTree) -> np.ndarray:
    """Get consistent time values from CGNSBase_t nodes in a CGNSTree.

    Args:
        tree (CGNSTree): The CGNSTree containing CGNSBase_t nodes.

    Returns:
        np.ndarray: An array of consistent time values.

    Raises:
        AssertionError: If the time values across bases are not consistent.
    """
    base_paths = get_base_names(tree, unique=True)  # TODO full_path=True ??
    time_values = []
    for bp in base_paths:
        base_node = CGU.getNodeByPath(tree, bp)
        time_values.append(CGU.getValueByPath(base_node, "Time/TimeValues")[0])
    assert time_values.count(time_values[0]) == len(time_values), (
        "times values are not consistent in bases"
    )
    return time_values[0]


def show_cgns_tree(pyTree: list, pre: str = ""):
    """Pretty print for CGNS Tree.

    Args:
        pyTree (list): CGNS tree to print
        pre (str, optional): indentation of print. Defaults to ''.
    """
    if not (isinstance(pyTree, list)):
        if pyTree is None:  # pragma: no cover
            return True
        else:
            raise TypeError(f"{type(pyTree)=}, but should be a list or None")

    np.set_printoptions(threshold=5, edgeitems=1)

    def printValue(node):
        if node[1].dtype == "|S1":
            return CGU.getValueAsString(node)
        else:
            return f"{node[1]}".replace("\n", "")

    for child in pyTree[2]:
        try:
            print(
                pre,
                child[0],
                ":",
                child[1].shape,
                printValue(child),
                child[1].dtype,
                child[3],
            )
        except AttributeError:
            print(pre, child[0], ":", child[1], child[3])

        if child[2]:
            show_cgns_tree(child, " " * len(pre) + "|_ ")
    np.set_printoptions(edgeitems=3, threshold=1000)


def flatten_cgns_tree(pyTree: list):
    """Flatten CGNS tree into dict of primitives (for HF)."""
    flat = {}
    dtypes = {}
    extras = {}

    def visit(tree, path=""):
        for node in tree[2]:
            name, data, children, extra = node
            new_path = f"{path}/{name}" if path else name

            # Flatten values for HF: always primitive types
            if isinstance(data, np.ndarray):
                if data.dtype.kind == "S":  # string arrays
                    flat[new_path] = [x.decode("utf-8") for x in data.tolist()]
                else:
                    flat[new_path] = data.tolist()
                dtypes[new_path] = str(data.dtype)
            elif data is None:
                flat[new_path] = None
                dtypes[new_path] = None
            else:
                flat[new_path] = data
                dtypes[new_path] = str(np.array(data).dtype)

            extras[new_path] = extra

            if children:
                visit(node, new_path)

    visit(pyTree)
    return flat, dtypes, extras


def unflatten_cgns_tree(flat: dict, dtypes: dict, cgns_types: dict):
    """Reconstruct a CGNS tree.

      From
      - flat: dict with Arrow-compatible primitives
      - dtypes: dict with dtype strings
      - cgns_types: dict with CGNS type names (ending in '_t')
    Returns the root CGNSTree node.
    """
    # Build all nodes from paths
    nodes = {}

    for path, value in flat.items():
        dtype = np.dtype(dtypes.get(path))
        cgns_type = cgns_types.get(path)

        # reconstruct data as numpy array or None
        if value is None:
            data = None
        else:
            if dtype is None:
                data = None
            else:
                data = np.array(value, dtype=dtype)

        # empty children for now
        nodes[path] = [path.split("/")[-1], data, [], cgns_type]

    # Re-link nodes into tree structure
    root = None
    for path, node in nodes.items():
        parts = path.split("/")
        if len(parts) == 1:
            # root-level node
            if root is None:
                root = ["CGNSTree", None, [node], "CGNSTree_t"]
            else:
                root[2].append(node)
        else:
            parent_path = "/".join(parts[:-1])
            parent = nodes[parent_path]
            parent[2].append(node)

    return root


def compare_cgns_trees(tree1, tree2, path="CGNSTree"):
    """Recursively compare two CGNS trees ignoring the order of children.

    Checks:
      - Node name
      - Data (numpy arrays or scalars) with exact dtype
      - Number of children
      - CGNS type (extra field)
    Returns True if identical, False otherwise.
    """
    # Compare node name
    if tree1[0] != tree2[0]:
        print(f"Name mismatch at {path}: {tree1[0]} != {tree2[0]}")
        return False

    # Compare data
    data1, data2 = tree1[1], tree2[1]

    if data1 is None and data2 is None:
        pass
    elif isinstance(data1, np.ndarray) and isinstance(data2, np.ndarray):
        if data1.dtype != data2.dtype:
            print(
                f"Dtype mismatch at {path}/{tree1[0]}: {data1.dtype} != {data2.dtype}"
            )
            return False
        if not np.array_equal(data1, data2):
            print(f"Data mismatch at {path}/{tree1[0]}")
            return False
    else:
        if isinstance(data1, np.ndarray) or isinstance(data2, np.ndarray):
            print(f"Data type mismatch at {path}/{tree1[0]}")
            return False
        if data1 != data2:
            print(f"Data mismatch at {path}/{tree1[0]}: {data1} != {data2}")
            return False

    # Compare extra (CGNS type)
    extra1, extra2 = tree1[3], tree2[3]
    if extra1 != extra2:
        print(f"Type mismatch at {path}/{tree1[0]}: {extra1} != {extra2}")
        return False

    # Compare children ignoring order
    children1_dict = {c[0]: c for c in tree1[2] or []}
    children2_dict = {c[0]: c for c in tree2[2] or []}

    if set(children1_dict.keys()) != set(children2_dict.keys()):
        print(
            f"Children name mismatch at {path}/{tree1[0]}: {set(children1_dict.keys())} != {set(children2_dict.keys())}"
        )
        return False

    # Recursively compare children
    for name in children1_dict:
        if not compare_cgns_trees(
            children1_dict[name], children2_dict[name], path=f"{path}/{tree1[0]}"
        ):
            return False

    return True


def summarize_cgns_tree(pyTree: list, verbose=True) -> str:
    """Provide a summary of a CGNS tree's contents.

    Args:
        pyTree (list): The CGNS tree to summarize.
        verbose (bool, optional): If True, include detailed field information. Defaults to True.

    Example:
        >>> summarize_cgns_tree(pyTree)
        Number of Bases: 2
        Number of Zones: 5
        Number of Nodes: 20
        Number of Elements: 10
        Number of Fields: 8

        Fields:
          'Base1/Zone1/Solution1/Field1'
          'Base1/Zone1/Solution1/Field2'
          'Base2/Zone2/Solution2/Field1'
          ...
    """
    summary = []
    base_paths = CGU.getPathsByTypeSet(pyTree, "CGNSBase_t")
    nb_base = len(base_paths)
    nb_zones = 0
    nb_nodes = 0
    nb_elements = 0
    nb_fields = 0
    fields = []

    # Bases
    for base_path in base_paths:
        base_node = CGU.getNodeByPath(pyTree, base_path)
        base_name = base_node[0]

        zone_paths = CGU.getPathsByTypeSet(base_node, "Zone_t")
        nb_zones += len(zone_paths)

        # Zones
        for zone_path in zone_paths:
            zone_node = CGU.getNodeByPath(base_node, zone_path)
            zone_name = zone_node[0]
            # Read number of nodes and elements from the Zone node
            nb_nodes += zone_node[1][0][0]
            nb_elements += zone_node[1][0][1]

            # Flow Solutions (Fields)
            sol_paths = CGU.getPathsByTypeSet(zone_node, "FlowSolution_t")
            if sol_paths:
                for sol_path in sol_paths:
                    sol_node = CGU.getNodeByPath(zone_node, sol_path)
                    sol_name = sol_node[0]
                    field_names = [n[0] for n in sol_node[2]]
                    nb_fields += len(field_names)
                    fields.append(((field_names, sol_name, zone_name, base_name)))

    summary.append(f"Number of Bases: {nb_base}")
    summary.append(f"Number of Zones: {nb_zones}")
    summary.append(f"Number of Nodes: {nb_nodes}")
    summary.append(f"Number of Elements: {nb_elements}")
    summary.append(f"Number of Fields: {nb_fields}")
    summary.append("")

    if verbose:
        summary.append("Fields :")
        for field_names, sol_name, zone_name, base_name in fields:
            for field_name in field_names:
                summary.append(f"  {base_name}/{zone_name}/{sol_name}/{field_name}")

    print("\n".join(summary))
