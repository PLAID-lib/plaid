"""Custom types for features.

This module defines type aliases for different kinds of features and their identifiers.
The types are used across the codebase to properly type hint feature identifiers.
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Literal, Sequence, Tuple, TypedDict, Union

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:
    from typing_extensions import TypeAlias

from plaid.types import Array

# Physical data types
ScalarType: TypeAlias = Union[float, int]
FieldType: TypeAlias = Array
TimeSequenceType: TypeAlias = Array
TimeSeriesType: TypeAlias = Tuple[TimeSequenceType, FieldType]

# Feature data types
FeatureType: TypeAlias = Union[ScalarType, FieldType, TimeSeriesType, Array]

# Basic feature fields
FeatureTypeKey = Literal["scalar", "field", "nodes", "time_series"]


# Base identifiers
class BaseScalarIdentifier(TypedDict):
    """Type definition for scalar feature identifiers.

    A scalar feature is identified by its type and name.
    """

    type: Literal["scalar"]
    name: str


class BaseFieldIdentifier(TypedDict):
    """Type definition for field feature identifiers.

    A field feature is identified by:
    - type: Always 'field'
    - name: Field name
    - base_name: Name of the base containing the field
    - zone_name: Name of the zone containing the field
    - location: Field location
    """

    type: Literal["field"]
    name: str
    base_name: str
    zone_name: str
    location: str


class BaseNodesIdentifier(TypedDict):
    """Type definition for nodes feature identifiers.

    A nodes feature is identified by:
    - type: Always 'nodes'
    - base_name: Name of the base containing the nodes
    - zone_name: Name of the zone containing the nodes
    """

    type: Literal["nodes"]
    base_name: str
    zone_name: str


class BaseTimeSeriesIdentifier(TypedDict):
    """Type definition for time series feature identifiers.

    A time series feature is identified by its type and name.
    """

    type: Literal["time_series"]
    name: str


# Time stamped identifiers
class TimedFieldIdentifier(BaseFieldIdentifier, total=False):
    """Type definition for field feature identifiers with time.

    Extends BaseFieldIdentifier with an optional time field.
    """

    time: float


class TimedNodesIdentifier(BaseNodesIdentifier, total=False):
    """Type definition for nodes feature identifiers with time.

    Extends BaseNodesIdentifier with an optional time field.
    """

    time: float


# Feature identifiers - individual types
ScalarIdentifier = BaseScalarIdentifier
FieldIdentifier = Union[BaseFieldIdentifier, TimedFieldIdentifier]
NodesIdentifier = Union[BaseNodesIdentifier, TimedNodesIdentifier]
TimeSeriesIdentifier = BaseTimeSeriesIdentifier

# Overall feature identifier types
FeatureIdentifier = Union[
    ScalarIdentifier, FieldIdentifier, NodesIdentifier, TimeSeriesIdentifier
]
FeatureIdentifierSequence = Sequence[FeatureIdentifier]

# Feature name types
FeatureNames = list[str]  # Use list for mutability to support append/sort
