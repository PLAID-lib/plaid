"""Custom types for features."""

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

from plaid.types import Array

# Physical data types
ScalarType: TypeAlias = Union[float, int]
FieldType: TypeAlias = Array
TimeSequenceType: TypeAlias = Array
TimeSeriesType: TypeAlias = Tuple[TimeSequenceType, FieldType]

# Feature data types
FeatureType: TypeAlias = Union[ScalarType, FieldType, TimeSeriesType, Array]

# Identifiers
FeatureIdentifier: TypeAlias = dict[str, Union[str, float]]
