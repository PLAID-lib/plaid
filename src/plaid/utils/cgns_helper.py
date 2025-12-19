"""Utility functions for working with CGNS trees and nodes."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Any

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


def show_cgns_tree(pyTree: CGNSTree, pre: str = ""):
    """Pretty print for CGNS Tree.

    Args:
        pyTree (CGNSTree): CGNS tree to print
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


def fix_cgns_tree_types(tree: CGNSTree) -> CGNSTree:
    """Recursively fix the data types of a CGNS tree node and its children.

    This function ensures that data arrays match the expected CGNS types:
      - "IndexArray_t": converted to integer arrays and stacked
      - "Zone_t": stacked as numpy arrays
      - "Elements_t", "CGNSBase_t", "BaseIterativeData_t": converted to integer arrays

    Args:
        tree (CGNSTree): A CGNS tree of the form
            [name: str, data: Any, children: List[CGNSTree], cgns_type: str].

    Returns:
        CGNSTree: A new CGNS tree node with corrected data types and recursively
        fixed children.

    Example:
        >>> node = ["Zone1", [[1, 2], [3, 4]], [], "Zone_t"]
        >>> fixed_node = fix_cgns_tree_types(node)
        >>> fixed_node[1].shape
        (2, 2)
    """
    name, data, children, cgns_type = tree

    # Fix data types according to CGNS type
    if data is not None:
        if cgns_type == "IndexArray_t":
            data = CGU.setIntegerAsArray(*data)
            data = np.stack(data)
        elif cgns_type == "Zone_t":
            data = np.stack(data)
        elif cgns_type in ["Elements_t", "CGNSBase_t", "BaseIterativeData_t"]:
            data = CGU.setIntegerAsArray(*data)

    # Recursively fix children
    new_children = []
    if children:
        for child in children:
            new_children.append(fix_cgns_tree_types(child))

    return [name, data, new_children, cgns_type]


def compare_cgns_trees(
    tree1: CGNSTree,
    tree2: CGNSTree,
    path: str = "CGNSTree",
) -> bool:
    """Recursively compare two CGNS trees for exact equality, ignoring the order of children.

    This function checks:
      - Node names
      - Node data (numpy arrays or scalars) with exact dtype and values
      - Number and names of children nodes
      - CGNS type (stored as the extra field)

    It prints informative messages whenever a mismatch is found, including the
    path in the tree where the mismatch occurs.

    Args:
        tree1 (CGNSTree): The first CGNS tree node to compare.
        tree2 (CGNSTree): The second CGNS tree node to compare.
        path (str, optional): The current path in the tree for error messages.
            Defaults to "CGNSTree".

    Returns:
        bool: True if the trees are identical (including node names, data, types,
              and children), False otherwise.

    Example:
        >>> identical = compare_cgns_trees(tree1, tree2)
        >>> if identical:
        >>>     print("The trees are identical")
        >>> else:
        >>>     print("The trees differ")
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
        if len(data1) == 0 and len(data2) == 0:
            pass
        elif not np.array_equal(data1, data2):
            print(f"Data mismatch at {path}/{tree1[0]}")
            return False
    else:
        if isinstance(data1, np.ndarray) or isinstance(data2, np.ndarray):
            print(f"Data type mismatch at {path}/{tree1[0]}")
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


def compare_leaves(d1: Any, d2: Any) -> bool:
    """Compare two leaf values in a CGNS tree or flattened structure, handling arrays and scalars.

    This function supports:
      - NumPy arrays, including byte arrays (converted to str)
      - Floating-point arrays or scalars (compared with tolerance)
      - Integer arrays or scalars (exact comparison)
      - Strings and None

    Args:
        d1 (Any): First value to compare (scalar or np.ndarray).
        d2 (Any): Second value to compare (scalar or np.ndarray).

    Returns:
        bool: True if the values are considered equal, False otherwise.

    Note:
        - Floating-point comparisons use `np.allclose` or `np.isclose` with `rtol=1e-7` and `atol=0`.
        - Byte arrays (`dtype.kind == "S"`) are converted to string before comparison.

    Examples:
        >>> compare_leaves(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        True
        >>> compare_leaves(3.0, 3.00000001)
        True
        >>> compare_leaves(np.array([1, 2]), np.array([2, 1]))
        False
    """
    # Convert bytes arrays to str
    if isinstance(d1, np.ndarray) and d1.dtype.kind == "S":
        d1 = d1.astype(str)
    if isinstance(d2, np.ndarray) and d2.dtype.kind == "S":
        d2 = d2.astype(str)

    # Both arrays
    if isinstance(d1, np.ndarray) and isinstance(d2, np.ndarray):
        if np.issubdtype(d1.dtype, np.floating) or np.issubdtype(d2.dtype, np.floating):
            return np.allclose(d1, d2, rtol=1e-7, atol=0)
        else:
            return np.array_equal(d1, d2)

    # Scalars (int/float/str/None)
    if isinstance(d1, float) or isinstance(d2, float):
        return np.isclose(d1, d2, rtol=1e-7, atol=0)
    return d1 == d2


def compare_cgns_trees_no_types(
    tree1: CGNSTree, tree2: CGNSTree, path: str = "CGNSTree"
) -> bool:
    """Recursively compare two CGNS trees ignoring the order of children and relaxing strict type checks.

    This function is useful for heterogeneous or nested CGNS samples,
    such as those encountered in Hugging Face Arrow datasets. It compares:
    - Node names
    - Node data using `compare_leaves` (supports arrays, scalars, strings)
    - CGNS type (extra field)
    - Children nodes by name, ignoring their order

    Args:
        tree1 (CGNSTree): The first CGNS tree node to compare.
        tree2 (CGNSTree): The second CGNS tree node to compare.
        path (str, optional): Path for error reporting. Defaults to "CGNSTree".

    Returns:
        bool: True if the trees are considered equivalent, False otherwise.

    Example:
        >>> identical = compare_cgns_trees_no_types(tree1, tree2)
        >>> if identical:
        >>>     print("The trees match ignoring types")
        >>> else:
        >>>     print("The trees differ")
    """
    if tree1[0] != tree2[0]:
        print(f"Name mismatch at {path}: {tree1[0]} != {tree2[0]}")
        return False

    # Compare data using recursive helper
    data1, data2 = tree1[1], tree2[1]
    if not compare_leaves(data1, data2):
        print(f"Data mismatch at {path}/{tree1[0]}")
        return False

    # Compare extra (CGNS type)
    if tree1[3] != tree2[3]:
        print(f"Type mismatch at {path}/{tree1[0]}: {tree1[3]} != {tree2[3]}")
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
        if not compare_cgns_trees_no_types(
            children1_dict[name], children2_dict[name], path=f"{path}/{tree1[0]}"
        ):
            return False

    return True


def summarize_cgns_tree(pyTree: CGNSTree, verbose=True) -> str:
    """Provide a summary of a CGNS tree's contents.

    Args:
        pyTree (CGNSTree): The CGNS tree to summarize.
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
