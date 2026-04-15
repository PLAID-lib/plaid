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

import numpy as np

from plaid.storage.common.bridge import flatten_path, unflatten_path


def to_var_sample_dict(
    zarr_dataset: Any,
    idx: int,
    features: Optional[list[str]],
    indexers: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Extracts a sample dictionary from a Zarr dataset by index.

    Args:
        zarr_dataset (zarr.Group): The Zarr group containing the dataset.
        idx (int): The sample index to extract.
        features: Iterable of feature names (keys) to extract from the dataset.
        indexers: Optional mapping ``feature_path -> indexer`` used to select
            feature values along the last axis.

    Returns:
        dict[str, Any]: Dictionary of variable features for the sample.
    """
    zarr_sample = zarr_dataset.zarr_group[f"sample_{idx:09d}"]

    if features is None:
        features = [unflatten_path(p) for p in zarr_sample.array_keys()]

    flattened = {feat: flatten_path(feat) for feat in features}
    # missing = set(flattened.values()) - set(zarr_sample.array_keys())
    # if missing:  # pragma: no cover
    #     raise KeyError(f"Missing features in sample {idx}: {sorted(missing)}")

    indexers = indexers or {}
    out = {}
    for feat, flat_feat in flattened.items():
        if flat_feat not in zarr_sample.array_keys():
            continue

        arr = zarr_sample[flat_feat]
        if feat in indexers:
            out[feat] = _apply_indexer(arr, indexers[feat], feat)
        else:
            out[feat] = arr

    return out


def _apply_indexer(arr: Any, indexer: Any, feat_name: str) -> np.ndarray:
    """Apply a last-axis indexer to a Zarr array-like object."""
    if indexer is None:  # pragma: no cover
        return np.asarray(arr)

    if arr.ndim == 0:
        raise ValueError(f"Cannot apply indexer to scalar feature '{feat_name}'")

    selector_prefix = (slice(None),) * (arr.ndim - 1)

    if isinstance(indexer, slice):
        return np.asarray(arr[selector_prefix + (indexer,)])

    idx = np.asarray(indexer, dtype=np.int64)
    if idx.ndim != 1:
        raise ValueError(
            f"Indexer for feature '{feat_name}' must be a 1D sequence or slice"
        )

    axis_size = int(arr.shape[-1])
    if np.any(idx >= axis_size) or np.any(idx < -axis_size):
        raise IndexError(
            f"Indexer for feature '{feat_name}' contains out-of-bounds values "
            f"for last axis of size {axis_size}"
        )

    return np.asarray(arr.oindex[selector_prefix + (idx,)])


def sample_to_var_sample_dict(zarr_sample: dict[str, Any]) -> dict[str, Any]:
    """Converts a Zarr sample to a variable sample dictionary.

    Args:
        zarr_sample (dict[str, Any]): The raw Zarr sample data.

    Returns:
        dict[str, Any]: The processed variable sample dictionary.
    """
    return zarr_sample
