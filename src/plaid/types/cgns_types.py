"""Custom types for CGNS data structures."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Tuple, Union

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias

from plaid.types.common import Array

# CGNS types inside a node
NodeName: TypeAlias = str
NodeLabel: TypeAlias = NodeName
NodeValue: TypeAlias = Union[
    None,
    str,
    bytes,
    int,
    float,
    Array,
]

# A CGNSNode is a list of: [name, value, children, label]
CGNSNode: TypeAlias = list[
    Union[
        NodeName,
        NodeValue,
        list["CGNSNode"],
        NodeLabel,
    ]
]

# A CGNSTree is simply the root CGNSNode
CGNSTree: TypeAlias = CGNSNode

# CGNS links and paths
LinkType: TypeAlias = list[str]  # [dir, filename, source_path, target_path]
PathType: TypeAlias = Tuple[str, ...]  # a path in the CGNS tree
