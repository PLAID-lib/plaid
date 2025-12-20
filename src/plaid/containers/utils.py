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

import CGNS.PAT.cgnsutils as CGU
import numpy as np

from plaid.constants import (
    AUTHORIZED_FEATURE_INFOS,
    AUTHORIZED_FEATURE_TYPES,
)
from plaid.containers.feature_identifier import FeatureIdentifier
from plaid.types import Feature
from plaid.utils.base import safe_len


def _check_names(names: Union[str, list[str]]):
    """Check that names do not contain invalid character ``/``.

    Args:
        names (Union[str, list[str]]): The names to check.

    Raises:
        ValueError: If any name contains the invalid character ``/``.
    """
    if isinstance(names, str):
        names = [names]
    for name in names:
        if (name is not None) and ("/" in name):
            raise ValueError(
                f"feature_names containing `/` are not allowed, but {name=}, you should first replace any occurence of `/` with something else, for example: `name.replace('/','__')`"
            )


def _read_index(pyTree: list, dim: list[int]):
    """Read Index Array or Index Range from CGNS.

    Args:
        pyTree (list): CGNS node which has a child Index to read
        dim (list): dimensions of the coordinates

    Returns:
        indices
    """
    a = _read_index_array(pyTree)
    b = _read_index_range(pyTree, dim)
    return np.hstack((a, b))


def _read_index_array(pyTree: list):
    """Read Index Array from CGNS.

    Args:
        pyTree (list): CGNS node which has a child of type IndexArray_t to read

    Returns:
        indices
    """
    indexArrayPaths = CGU.getPathsByTypeSet(pyTree, ["IndexArray_t"])
    res = []
    for indexArrayPath in indexArrayPaths:
        data = CGU.getNodeByPath(pyTree, indexArrayPath)
        if data[1] is None:  # pragma: no cover
            continue
        else:
            res.extend(data[1].ravel())
    return np.array(res, dtype=int).ravel()


def _read_index_range(pyTree: list, dim: list[int]):
    """Read Index Range from CGNS.

    Args:
        pyTree (list): CGNS node which has a child of type IndexRange_t to read
        dim (list[str]): dimensions of the coordinates

    Returns:
        indices
    """
    indexRangePaths = CGU.getPathsByTypeSet(pyTree, ["IndexRange_t"])
    res = []

    for indexRangePath in indexRangePaths:  # Is it possible there are several ?
        indexRange = CGU.getValueByPath(pyTree, indexRangePath)

        if indexRange.shape == (3, 2):  # 3D  # pragma: no cover
            for k in range(indexRange[:, 0][2], indexRange[:, 1][2] + 1):
                for j in range(indexRange[:, 0][1], indexRange[:, 1][1] + 1):
                    global_id = (
                        np.arange(indexRange[:, 0][0], indexRange[:, 1][0] + 1)
                        + dim[0] * (j - 1)
                        + dim[0] * dim[1] * (k - 1)
                    )
                    res.extend(global_id)

        elif indexRange.shape == (2, 2):  # 2D  # pragma: no cover
            for j in range(indexRange[:, 0][1], indexRange[:, 1][1]):
                for i in range(indexRange[:, 0][0], indexRange[:, 1][0]):
                    global_id = i + dim[0] * (j - 1)
                    res.append(global_id)
        else:
            begin = indexRange[0]
            end = indexRange[1]
            res.extend(np.arange(begin, end + 1).ravel())

    return np.array(res, dtype=int).ravel()


def get_sample_ids(savedir: Union[str, Path]) -> list[int]:
    """Return list of sample ids from a dataset on disk.

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


def get_feature_type_and_details_from(
    feature_identifier: FeatureIdentifier,
) -> tuple[str, FeatureIdentifier]:
    """Extract and validate the feature type and its associated metadata from a feature identifier.

    This utility function ensures that the `feature_identifier` dictionary contains a valid
    "type" key (e.g., "scalar", "field", "node") and returns the type along
    with the remaining identifier keys, which are specific to the feature type.

    Args:
        feature_identifier (dict): A dictionary with a "type" key, and
            other keys (some optional) depending on the feature type. For example:
            - {"type": "scalar", "name": "Mach"}
            - {"type": "field", "name": "pressure"}
            - {"type": "field", "name": "pressure", "time":0.}
            - {"type": "nodes", "base_name": "Base_2_2"}

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
    feature_details = feature_identifier.copy()
    feature_type = feature_details.pop("type")

    assert feature_type in AUTHORIZED_FEATURE_TYPES, (
        f"feature type {feature_type} not known"
    )

    assert all(
        key in AUTHORIZED_FEATURE_INFOS[feature_type] for key in feature_details
    ), (
        f"Unexpected key(s) in feature_identifier {feature_details=} | {feature_type=} -> {AUTHORIZED_FEATURE_INFOS[feature_type]}"
    )

    return feature_type, feature_details


def check_features_type_homogeneity(
    feature_identifiers: list[FeatureIdentifier],
) -> None:
    """Check type homogeneity of features, for tabular conversion.

    Args:
        feature_identifiers (list[dict]): dict with a "type" key, and
            other keys (some optional) depending on the feature type. For example:
            - {"type": "scalar", "name": "Mach"}
            - {"type": "field", "name": "pressure"}

    Raises:
        AssertionError: if types are not consistent
    """
    assert feature_identifiers and isinstance(feature_identifiers, list), (
        "feature_identifiers must be a non-empty list"
    )
    feat_type = feature_identifiers[0]["type"]
    for i, feat_id in enumerate(feature_identifiers):
        assert feat_id["type"] in AUTHORIZED_FEATURE_TYPES, "feature type not known"
        assert feat_id["type"] == feat_type, (
            f"Inconsistent feature types: {i}-th feature type is {feat_id['type']}, while the first one is {feat_type}"
        )


def check_features_size_homogeneity(
    feature_identifiers: list[FeatureIdentifier],
    features: dict[int, list[Feature]],
) -> int:
    """Check size homogeneity of features, for tabular conversion.

    Size homogeneity is check through samples for each feature, and through features for each sample.
    To be converted to tabular data, each sample must have the same number of features and each feature
    must have the same dimension

    Args:
        feature_identifiers (list[dict]): dict with a "type" key, and
            other keys (some optional) depending on the feature type. For example:
            - {"type": "scalar", "name": "Mach"}
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
        name_feature = feature_identifiers[i].get("name", "nodes")
        size = safe_len(features_values[0][i])
        for j in range(nb_samples):
            size_j = safe_len(features_values[j][i])
            assert size_j == size, (
                f"Inconsistent feature sizes for feature {i} (name {name_feature}): has size {size_j} in sample {j}, while having size {size} in sample 0"
            )

    for j in range(nb_samples):
        size = safe_len(features_values[j][0])
        for i in range(nb_features):
            name_feature = feature_identifiers[i].get("name", "nodes")
            size_i = safe_len(features_values[j][i])
            assert size_i == size, (
                f"Inconsistent feature sizes in sample {j}: feature {i} (name {name_feature}) size {size_i}, while feature 0 (name {feature_identifiers[0]['name']}) is of size {size}"
            )
    return size


def has_duplicates_feature_ids(feature_identifiers: list[FeatureIdentifier]):
    """Check whether a list of feature identifier contains duplicates.

    Args:
        feature_identifiers (list[FeatureIdentifier]):
            A list of dictionaries representing feature identifiers.

    Returns:
        bool: True if a duplicate is found in the list, False otherwise.
    """
    seen = set()
    for d in feature_identifiers:
        frozen = frozenset(d.items())
        if frozen in seen:
            return True
        seen.add(frozen)
    return False
