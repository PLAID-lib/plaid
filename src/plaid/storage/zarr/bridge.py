"""Zarr bridge utilities.

This module provides utility functions for bridging between PLAID samples and Zarr storage format.
It includes functions for key transformation and sample data conversion.
"""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

from typing import Any, Optional

import zarr

from plaid.storage.common.bridge import flatten_path, unflatten_path


def to_var_sample_dict(
    zarr_dataset: zarr.Group, idx: int, features: Optional[list[str]]
) -> dict[str, Any]:
    """Extracts a sample dictionary from a Zarr dataset by index.

    Args:
        zarr_dataset (zarr.Group): The Zarr group containing the dataset.
        idx (int): The sample index to extract.
        features: Iterable of feature names (keys) to extract from the dataset.

    Returns:
        dict[str, Any]: Dictionary of variable features for the sample.
    """
    zarr_sample = zarr_dataset.zarr_group[f"sample_{idx:09d}"]

    if features is None:
        features = [unflatten_path(p) for p in zarr_sample.array_keys()]

    flattened = {feat: flatten_path(feat) for feat in features}
    missing = set(flattened.values()) - set(zarr_sample.array_keys())
    if missing:  # pragma: no cover
        raise KeyError(f"Missing features in sample {idx}: {sorted(missing)}")

    return {feat: zarr_sample[flat_feat] for feat, flat_feat in flattened.items()}


def sample_to_var_sample_dict(zarr_sample: dict[str, Any]) -> dict[str, Any]:
    """Converts a Zarr sample to a variable sample dictionary.

    Args:
        zarr_sample (dict[str, Any]): The raw Zarr sample data.

    Returns:
        dict[str, Any]: The processed variable sample dictionary.
    """
    return zarr_sample
