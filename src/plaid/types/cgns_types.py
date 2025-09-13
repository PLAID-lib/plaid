"""Custom types for CGNS data structures."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Any, Union

from pydantic import BaseModel

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias

# from plaid.types.common import Array

# CGNS types inside a node
# CGNSNodeName: TypeAlias = str
# CGNSNodeLabel: TypeAlias = str
# CGNSNodeValue: TypeAlias = Union[
#     None,
#     str,
#     bytes,
#     int,
#     float,
#     Array,
# ]

# A CGNSNode is a list of: [name, value, children, label]
# CGNSNode: TypeAlias = list[
#     Union[
#         CGNSNodeName,
#         CGNSNodeValue,
#         list["CGNSNode"],
#         CGNSNodeLabel,
#     ]
# ]


class CGNSNode(BaseModel):
    """Custom type that represents a CGNS node."""
    name: str
    value: Any | None = None
    children: list["CGNSNode"]
    label: str


# A CGNSTree is simply the root CGNSNode
CGNSTree: TypeAlias = CGNSNode

# CGNS links and paths
CGNSLink: TypeAlias = list[str]  # [dir, filename, source_path, target_path]
CGNSPath: TypeAlias = tuple[str, ...]  # a path in the CGNS tree
