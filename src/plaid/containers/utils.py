"""Utility functions for PLAID containers."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

from pathlib import Path
from typing import Union

from plaid.constants import (
    AUTHORIZED_FEATURE_INFOS,
    AUTHORIZED_FEATURE_TYPES,
)
from plaid.types import FeatureType
from plaid.utils.base import safe_len

# %% Functions


def get_sample_ids(savedir: Union[str, Path]) -> list[int]:
    """Return list of sample ids in a dataset on disk.

    Args:
        savedir (Union[str,Path]): The path to the directory where sample files are stored.

    Returns:
        list[int]: List of sample ids.
    """
    savedir = Path(savedir)
    return sorted(
        [
            int(d.stem.split("_")[-1])
            for d in (savedir / "samples").glob("sample_*")
            if d.is_dir()
        ]
    )


def get_number_of_samples(savedir: Union[str, Path]) -> int:
    """Return number of samples in a dataset on disk.

    Args:
        savedir (Union[str,Path]): The path to the directory where sample files are stored.

    Returns:
        int: number of samples.
    """
    return len(get_sample_ids(savedir))


def get_feature_type_and_details_from_identifier(
    feature_identifier: dict[str, Union[str, float]],
) -> tuple[str, dict[str, Union[str, float]]]:
    """Extract and validate the feature type and its associated metadata from a feature identifier.

    This utility function ensures that the `feature_identifier` dictionary contains a valid
    "type" key (e.g., "scalar", "time_series", "field", "node") and returns the type along
    with the remaining identifier keys, which are specific to the feature type.

    Args:
        feature_identifier (dict): A dictionary with a "type" key, and
            other keys (some optional) depending on the feature type. For example:
                - {"type": "scalar", "name": "Mach"}
                - {"type": "time_series", "name": "AOA"}
                - {"type": "field", "name": "pressure"}

    Returns:
        tuple[str, dict]: A tuple `(feature_type, feature_details)` where:
            - `feature_type` is the value of the "type" key (e.g., "scalar").
            - `feature_details` is a dictionary of the remaining keys.

    Raises:
        AssertionError:
            - If "type" is missing.
            - If the type is not in `AUTHORIZED_FEATURE_TYPES`.
            - If any unexpected keys are present for the given type.
    """
    assert "type" in feature_identifier, (
        "feature type not specified in feature_identifier"
    )
    feature_type = feature_identifier["type"]
    feature_details = {k: v for k, v in feature_identifier.items() if k != "type"}

    assert feature_type in AUTHORIZED_FEATURE_TYPES, "feature type not known"

    assert all(
        key in AUTHORIZED_FEATURE_INFOS[feature_type] for key in feature_details
    ), "Unexpected key(s) in feature_identifier"

    return feature_type, feature_details


def check_features_type_homogeneity(
    feature_identifiers: list[dict[str, Union[str, float]]],
) -> None:
    """Check type homogeneity of features, for tabular conversion.

    Args:
        feature_identifiers (list[dict]): dict with a "type" key, and
            other keys (some optional) depending on the feature type. For example:
                - {"type": "scalar", "name": "Mach"}
                - {"type": "time_series", "name": "AOA"}
                - {"type": "field", "name": "pressure"}

    Raises:
        AssertionError: if types are not consistent
    """
    feat_type = feature_identifiers[0]["type"]
    for i, feat_id in enumerate(feature_identifiers):
        assert feat_id["type"] in AUTHORIZED_FEATURE_TYPES, "feature type not known"
        assert feat_id["type"] == feat_type, (
            f"Inconsistent feature types: {i}-th feature type is {feat_id['type']}, while the first one is {feat_type}"
        )


def check_features_size_homogeneity(
    feature_identifiers: list[dict[str, Union[str, float]]],
    features: dict[int, list[FeatureType]],
) -> int:
    """Check size homogeneity of features, for tabular conversion.

    Size homogeneity is check through samples for each feature, and through features for each sample.
    To be converted to tabular data, each sample must have the same number of features and each feature
    must have the same dimension

    Args:
        feature_identifiers (list[dict]): dict with a "type" key, and
            other keys (some optional) depending on the feature type. For example:
                - {"type": "scalar", "name": "Mach"}
                - {"type": "time_series", "name": "AOA"}
                - {"type": "field", "name": "pressure"}
        features (dict): dict with sample index as keys and one or more features as values.

    Returns:
        int: the common feature dimension

    Raises:
        AssertionError: if sizes are not consistent
    """
    features_values = list(features.values())
    nb_samples = len(features_values)
    nb_features = len(feature_identifiers)
    for i in range(nb_features):
        size = safe_len(features_values[0][i])
        for j in range(nb_samples):
            size_j = safe_len(features_values[j][i])
            assert size_j == size, (
                f"Inconsistent feature sizes for feature {i} (name {feature_identifiers[i]['name']}): has size {size_j} in sample {j}, while having size {size} in sample 0"
            )

    for j in range(nb_samples):
        size = safe_len(features_values[j][0])
        for i in range(nb_features):
            size_i = safe_len(features_values[j][i])
            assert size_i == size, (
                f"Inconsistent feature sizes in sample {j}: feature {i} (name {feature_identifiers[i]['name']}) size {size_i}, while feature 0 (name {feature_identifiers[0]['name']}) is of size {size}"
            )
    return size
