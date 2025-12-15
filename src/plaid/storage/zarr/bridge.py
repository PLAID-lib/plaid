# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

"""Zarr bridge utilities.

This module provides utility functions for bridging between PLAID samples and Zarr storage format.
It includes functions for key transformation and sample data conversion.
"""

from typing import Any

import zarr


def unflatten_zarr_key(key: str) -> str:
    """Unflattens a Zarr key by replacing underscores with slashes.

    Args:
        key (str): The flattened key with underscores.

    Returns:
        str: The unflattened key with slashes.
    """
    return key.replace("__", "/")


def to_var_sample_dict(zarr_dataset: zarr.Group, idx: int) -> dict[str, Any]:
    """Extracts a sample dictionary from a Zarr dataset by index.

    Args:
        zarr_dataset (zarr.Group): The Zarr group containing the dataset.
        idx (int): The sample index to extract.

    Returns:
        dict[str, Any]: Dictionary of variable features for the sample.
    """
    return zarr_dataset[idx]


def sample_to_var_sample_dict(zarr_sample: dict[str, Any]) -> dict[str, Any]:
    """Converts a Zarr sample to a variable sample dictionary.

    Args:
        zarr_sample (dict[str, Any]): The raw Zarr sample data.

    Returns:
        dict[str, Any]: The processed variable sample dictionary.
    """
    return zarr_sample
