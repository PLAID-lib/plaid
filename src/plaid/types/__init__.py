"""Custom types for PLAID library."""

from .cgns_types import (
    CGNSNode,
    CGNSTree,
)
from .common import (
    Array,
    IndexArrayType,
    ScalarDType,
    ScalarOrArray,
    ScalarOrArrayOrStr,
)

__all__ = [
    "ScalarOrArray",
    "ScalarOrArrayOrStr",
    "Array",
    "ScalarDType",
    "IndexArrayType",
    "CGNSNode",
    "CGNSTree",
]
