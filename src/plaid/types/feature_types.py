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
TimeSeries: TypeAlias = tuple[TimeSequence, Field]

# Feature data types
Feature: TypeAlias = Union[Scalar, Field, TimeSeries, Array]


# Identifiers
# FeatureIdentifier: TypeAlias = dict[str, Union[str, float]]
class FeatureIdentifier(dict[str, Union[str, float]]):
    """Feature identifier for a specific feature."""

    def __init__(self, *args, **kwargs) -> None:
        return super().__init__(*args, **kwargs)

    def __hash__(self) -> int:  # pyright: ignore[reportIncompatibleVariableOverride]
        """Compute a hash for the feature identifier.

        Returns:
            int: The hash value.
        """
        return hash(frozenset(sorted(self.items())))
        # return hash(tuple(sorted(self.items())))

    def __lt__(self, other: "FeatureIdentifier") -> bool:
        """Compare two feature identifiers for ordering.

        Args:
            other (FeatureIdentifier): The other feature identifier to compare against.

        Returns:
            bool: True if this feature identifier is less than the other, False otherwise.
        """
        return sorted(self.items()) < sorted(other.items())
