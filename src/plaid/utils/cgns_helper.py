# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import CGNS.PAT.cgnsutils as CGU
import numpy as np

CGNSTree = list
"""A CGNSTree is a list
"""


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
