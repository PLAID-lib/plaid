"""Feature identifier class for PLAID containers."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Union


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
