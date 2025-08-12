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
except ImportError:
    from typing_extensions import TypeAlias

from plaid.types.common import Array

# CGNS types inside a node
NodeName: TypeAlias = str
NodeLabel: TypeAlias = str
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

# Physical data types
ScalarType: TypeAlias = Union[float, int]
FieldType: TypeAlias = Array
TimeSequenceType: TypeAlias = Array
TimeSeriesType: TypeAlias = Tuple[TimeSequenceType, FieldType]

# Feature data types
FeatureType: TypeAlias = Union[ScalarType, FieldType, TimeSeriesType, Array]
