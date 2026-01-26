"""WebDataset bridge utilities.

This module provides utility functions for bridging between PLAID samples and WebDataset storage format.
It includes functions for sample data conversion and feature extraction.
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Any, Optional


def to_var_sample_dict(
    wds_dataset, idx: int, features: Optional[list[str]]
) -> dict[str, Any]:
    """Extracts variable features from a WebDataset.

    Args:
        wds_dataset: The WebDataset wrapper object.
        idx: The sample index to extract.
        features: Optional list of feature names to extract. If None, all features are returned.

    Returns:
        dict[str, Any]: Dictionary of variable features for the sample.
    """
    # Get the sample from the dataset
    wds_sample = wds_dataset[idx]

    if features is None:
        # Return only features that exist in the sample (non-None features)
        # This matches zarr behavior
        return wds_sample

    # Return requested features
    # For features not in the sample, return None
    # But for _times features, only return if base feature exists OR both are missing
    result = {}
    for feat in features:
        if feat in wds_sample:
            result[feat] = wds_sample[feat]
        elif feat.endswith("_times"):
            # For _times, only add if the base feature is also requested
            base_feat = feat[:-6]
            if base_feat in features:
                # Both requested, return None for _times
                result[feat] = None
        else:
            # Feature not in sample, return None
            result[feat] = None

    return result


def sample_to_var_sample_dict(wds_sample: dict[str, Any]) -> dict[str, Any]:
    """Converts a WebDataset sample to a variable sample dictionary.

    This is a pass-through function since WebDataset samples are already in the correct format.

    Args:
        wds_sample: The raw WebDataset sample data.

    Returns:
        dict[str, Any]: The processed variable sample dictionary (same as input).
    """
    return wds_sample
