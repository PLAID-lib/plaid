# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Common tree handling utilities.

This module provides utilities for flattening and unflattening CGNS trees,
converting between hierarchical tree structures and flat dictionaries for
storage and serialization purposes.
"""

from typing import Optional

from plaid.types import CGNSTree


def flatten_cgns_tree(
    pyTree: CGNSTree,
) -> tuple[dict[str, object], dict[str, str]]:
    """Flatten a CGNS tree into dictionaries of primitives.

    Args:
        pyTree (CGNSTree): The CGNS tree to flatten.

    Returns:
        tuple[dict[str, object], dict[str, str]]:
            - flat: dict of paths to primitive values
            - cgns_types: dict of paths to CGNS type strings
    """
    flat = {}
    cgns_types = {}

    def visit(tree, path=""):
        for node in tree[2]:
            name, data, children, cgns_type = node
            new_path = f"{path}/{name}" if path else name

            flat[new_path] = data
            cgns_types[new_path] = cgns_type

            if children:
                visit(node, new_path)

    visit(pyTree)
    return flat, cgns_types


def nodes_to_tree(nodes: dict[str, CGNSTree]) -> Optional[CGNSTree]:
    """Reconstruct a CGNS tree from a dictionary of nodes keyed by their paths.

    Each node is assumed to follow the CGNSTree format:
    [name: str, data: Any, children: List[CGNSTree], cgns_type: str]

    The dictionary keys are the full paths to each node, e.g. "Base1/Zone1/Field1".

    Args:
        nodes (Dict[str, CGNSTree]): A dictionary mapping node paths to CGNSTree nodes.

    Returns:
        Optional[CGNSTree]: The root CGNSTree node with all children linked,
        or None if the input dictionary is empty.

    Note:
        - Nodes with a path of length 1 are treated as root-level nodes.
        - The root node is named "CGNSTree" with type "CGNSTree_t".
        - Parent-child relationships are reconstructed using path prefixes.
    """
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


def unflatten_cgns_tree(
    flat: dict[str, object],
    cgns_types: dict[str, str],
) -> CGNSTree:
    """Reconstruct a CGNS tree from flattened dictionaries of data and types.

    This function takes a "flat" representation of a CGNS tree, where each node
    is stored in a dictionary keyed by its full path (e.g., "Base1/Zone1/Field1"),
    and another dictionary mapping each path to its CGNS type. It rebuilds the
    original tree structure by creating nodes and linking them according to their paths.

    Args:
        flat (dict[str, object]): Dictionary mapping node paths to their data values.
            The data can be a scalar, list, numpy array, or None.
        cgns_types (dict[str, str]): Dictionary mapping node paths to CGNS type strings
            (e.g., "Zone_t", "FlowSolution_t").

    Returns:
        CGNSTree: The reconstructed CGNS tree with nodes properly nested according
        to their paths. Each node is a list in the format:
        [name: str, data: Any, children: List[CGNSTree], cgns_type: str]

    Example:
        >>> flat = {
        >>>     "Base1": None,
        >>>     "Base1/Zone1": [10, 20],
        >>>     "Base1/Zone1/Field1": [1.0, 2.0]
        >>> }
        >>> cgns_types = {
        >>>     "Base1": "CGNSBase_t",
        >>>     "Base1/Zone1": "Zone_t",
        >>>     "Base1/Zone1/Field1": "FlowSolution_t"
        >>> }
        >>> tree = unflatten_cgns_tree(flat, cgns_types)
    """
    # Build all nodes from paths
    nodes = {}

    for path, value in flat.items():
        cgns_type = cgns_types.get(path)
        nodes[path] = [path.split("/")[-1], value, [], cgns_type]

    # Re-link nodes into tree structure
    tree = nodes_to_tree(nodes)
    return tree
