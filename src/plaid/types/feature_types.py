"""Custom types for features."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import sys
from typing import Union

if sys.version_info >= (3, 11):
    from typing import TypeAlias
else:  # pragma: no cover
    from typing_extensions import TypeAlias


from plaid.types.common import Array

# Physical data types
Scalar: TypeAlias = Union[float, int]
Field: TypeAlias = Array
TimeSequence: TypeAlias = Array

# Feature data types
Feature: TypeAlias = Union[Scalar, Field, Array]
