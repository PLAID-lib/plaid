"""Custom types for features."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Union

try:
    from typing import TypeAlias  # Python 3.10+
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias

from plaid.types.common import Array

# Physical data types
Scalar: TypeAlias = Union[float, int]
Field: TypeAlias = Array
TimeSequence: TypeAlias = Array
TimeSeries: TypeAlias = tuple[TimeSequence, Field]

# Feature data types
Feature: TypeAlias = Union[Scalar, Field, TimeSeries, Array]

# Identifiers
FeatureIdentifier: TypeAlias = dict[str, Union[str, float]]
