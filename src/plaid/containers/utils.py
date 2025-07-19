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
