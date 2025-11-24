"""Custom types for PLAID library."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from plaid.types.cgns_types import (
    CGNSNode,
    CGNSTree,
)
from plaid.types.common import Array, ArrayDType, IndexType
from plaid.types.feature_types import (
    Feature,
    Field,
    Scalar,
    TimeSequence,
)
from plaid.types.sklearn_types import SklearnBlock

__all__ = [
    "Array",
    "ArrayDType",
    "IndexType",
    "CGNSNode",
    "CGNSTree",
    "Scalar",
    "Field",
    "TimeSequence",
    "Feature",
    "FeatureIdentifier",
    "SklearnBlock",
]

# Re-export FeatureIdentifier from containers to maintain backwards compatibility
# Import is done at the bottom to avoid circular import issues
from plaid.containers.feature_identifier import FeatureIdentifier
