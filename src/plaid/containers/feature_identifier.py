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

    @classmethod
    def from_string(cls, string_identifier: str) -> "FeatureIdentifier":
        """Create a FeatureIdentifier from a string representation.

        Args:
            string_identifier (str): The string representation of the feature identifier.

        The `string_identifier` must follow the format:
            "<feature_type>::<detail1>/<detail2>/.../<detailN>"

        Supported feature types:
            - "scalar": expects 1 detail → ["name"],
            - "field": up to 5 details → ["name", "location", "zone_name", "base_name", "time"],
            - "nodes":  up to 3 details → ["zone_name", "base_name", "time"],

        Returns:
            FeatureIdentifier


        Warnings:
            - If "time" is present in a field/nodes identifier, it is cast to float.
            - `name` is required for scalar and field features.
        """
        splitted_identifier = string_identifier.split("::")

        feature_type = splitted_identifier[0]
        feature_details = [detail for detail in splitted_identifier[1].split("/")]

        from plaid.constants import AUTHORIZED_FEATURE_TYPES, AUTHORIZED_FEATURE_INFOS

        assert feature_type in AUTHORIZED_FEATURE_TYPES, "feature_type not known"

        arg_names = AUTHORIZED_FEATURE_INFOS[feature_type]
        assert len(arg_names) >= len(feature_details), "Too much details provided"
        data = {"type": feature_type}
        data.update({ arg_name : feature_detail for arg_name, feature_detail in zip(arg_names, feature_details)})
        return FeatureIdentifier(data)
