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


def summarize_cgns_tree(pyTree: list, verbose=True) -> str:
    """Provide a summary of a CGNS tree's contents.

    Args:
        pyTree (list): The CGNS tree to summarize.
        verbose (bool, optional): If True, include detailed field information. Defaults to True.

    Example:
        >>> summarize_cgns_tree(pyTree)
        Number of Bases: 2
        Number of Zones: 5
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

            # Elements
            elem_paths = CGU.getPathsByTypeSet(zone_node, "Elements_t")
            if elem_paths:
                for elem_path in elem_paths:
                    elem_node = CGU.getNodeByPath(zone_node, elem_path)
                    nb_elements += elem_node[1][0]

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
    summary.append(f"Number of Elements: {nb_elements}")
    summary.append(f"Number of Fields: {nb_fields}")
    summary.append("")

    if verbose:
        summary.append("Fields :")
        for field_names, sol_name, zone_name, base_name in fields:
            for field_name in field_names:
                if field_name.startswith("Coordinate"):
                    continue
                summary.append(f"  {base_name}/{zone_name}/{sol_name}/{field_name}")

    print("\n".join(summary))
