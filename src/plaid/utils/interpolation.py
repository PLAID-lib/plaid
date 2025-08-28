"""Interpolation utilities for working with ordered lists and vectors."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import bisect
from typing import Union

import numpy as np

# %% Functions


def binary_search(
    ordered_list: Union[list, np.ndarray], item: Union[float, int]
) -> int:
    """Find the rank of the largest element smaller or equal to the given item in a sorted list.

    Inspects the sorted list "ordered_list" and returns:
        - 0 if item <= ordered_list[0]
        - the rank of the largest element smaller or equal than item otherwise

    Args:
        ordered_list (Union[list, np.ndarray]): The data sorted in increasing order from which the previous rank is searched.
        item (Union[float, int]): The item for which the previous rank is searched.

    Returns:
        int: 0 or the rank of the largest element smaller or equal than item in "ordered_list".
    """
    return max(bisect.bisect_right(ordered_list, item) - 1, 0)


def binary_search_vectorized(
    ordered_list: Union[list, np.ndarray], items: Union[list, np.ndarray]
) -> np.ndarray:
    """Vectorized binary search for multiple items in a sorted list (items is now a list or one-dimensional np.ndarray).

    Args:
        ordered_list (Union[list, np.ndarray]): The data sorted in increasing order.
        items (Union[list, np.ndarray]): The items for which ranks are searched.

    Returns:
        np.ndarray: An array containing the ranks of the largest elements smaller or equal to each item.
    """
    return np.fromiter(
        map(lambda item: binary_search(ordered_list, item), items), dtype=int
    )


def piece_wise_linear_interpolation(
    item: float,
    item_indices: np.ndarray,
    vectors: Union[np.ndarray, dict],
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Computes a item interpolation for temporal vectors defined either by item_indices  and vectors at these indices.

    Args:
        item (float): The input item at which the interpolation is required.
        item_indices (np.ndarray): The items where the available data is defined, of size (numberOfTimeIndices).
        vectors (Union[np.ndarray, dict]): The available data, of size (numberOfVectors, numberOfDofs).
        tolerance (float): Tolerance for deciding when using the closest timestep value instead of carrying out the linear interpolation, default to 1e-4.

    Returns:
        np.ndarray: Interpolated vector, of size (numberOfDofs).
    """
    if item <= item_indices[0]:
        return vectors[0]
    if item >= item_indices[-1]:
        return vectors[-1]

    prev = binary_search(item_indices, item)
    coef = (item - item_indices[prev]) / (item_indices[prev + 1] - item_indices[prev])

    if 0.5 - abs(coef - 0.5) < tolerance:
        coef = round(coef)

    return coef * vectors[prev + 1] + (1 - coef) * vectors[prev]


def piece_wise_linear_interpolation_with_map(
    item: float,
    item_indices: np.ndarray,
    vectors: Union[np.ndarray, dict],
    vectors_map: list = None,
    tolerance: float = 1e-4,
) -> np.ndarray:
    """Computes a item interpolation for temporal vectors defined either by item_indices, some tags at these item indices (vectors_map), and vectors at those tags.

    Args:
        item (float): The input item at which the interpolation is required.
        item_indices (np.ndarray): The items where the available data is defined, of size (numberOfTimeIndices).
        vectors (Union[np.ndarray, dict]): The available data, of size (numberOfVectors, numberOfDofs).
        vectors_map (list, optional): List containing the mapping from the numberOfTimeIndices items indices to the numberOfVectors vectors, of size (numberOfTimeIndices,). Defaults to None.
        tolerance (float, optional): Tolerance for deciding when using the closest timestep value instead of carrying out the linear interpolation, default to 1e-4.

    Returns:
        np.ndarray: Interpolated vector, of size (numberOfDofs).
    """
    # TODO What if vectorsMap = None ??? it will crash
    if item <= item_indices[0]:
        return vectors[vectors_map[0]]
    if item >= item_indices[-1]:
        return vectors[vectors_map[-1]]

    prev = binary_search(item_indices, item)
    coef = (item - item_indices[prev]) / (item_indices[prev + 1] - item_indices[prev])

    if 0.5 - abs(coef - 0.5) < tolerance:
        coef = round(coef)

    return (
        coef * vectors[vectors_map[prev + 1]] + (1 - coef) * vectors[vectors_map[prev]]
    )


def piece_wise_linear_interpolation_vectorized(
    items: list[float], item_indices: np.ndarray, vectors: Union[np.ndarray, str]
) -> list[np.ndarray]:
    """piece_wise_linear_interpolation for more than one call (items is now a list or one-dimensional np.ndarray).

    Args:
        items (list[float]): The input items at which interpolations are required.
        item_indices (np.ndarray): The items where the available data is defined, of size (numberOfTimeIndices).
        vectors (np.ndarray or dict): The available data, of size (numberOfVectors, numberOfDofs).

    Returns:
        list[np.ndarray]: List of interpolated vectors, each of size (numberOfDofs).
    """
    return [
        piece_wise_linear_interpolation(item, item_indices, vectors) for item in items
    ]
    # return np.fromiter(map(lambda item: piece_wise_linear_interpolation(item,
    # item_indices, vectors), items), dtype = type(vectors[0]))


def piece_wise_linear_interpolation_vectorized_with_map(
    items: list[float],
    item_indices: np.ndarray,
    vectors: Union[np.ndarray, dict],
    vectors_map: list = None,
) -> list[np.ndarray]:
    """piece_wise_linear_interpolation_with_map for more than one call (items is now a list or one-dimensional np.ndarray).

    Args:
        items (list[float]): The input items at which interpolations are required.
        item_indices (np.ndarray): The items where the available data is defined, of size (numberOfTimeIndices).
        vectors (np.ndarray or dict): The available data, of size (numberOfVectors, numberOfDofs).
        vectors_map (list): List containing the mapping from the numberOfTimeIndices items indices to the numberOfVectors vectors, of size (numberOfTimeIndices,). Default is None, in which case numberOfVectors = numberOfTimeIndices.

    Returns:
        list[np.ndarray]: List of interpolated vectors, each of size (numberOfDofs).
    """
    return [
        piece_wise_linear_interpolation_with_map(
            item, item_indices, vectors, vectors_map
        )
        for item in items
    ]
    # return np.fromiter(map(lambda item:
    # piece_wise_linear_interpolation_with_map(item, item_indices, vectors,
    # vectors_map), items), dtype = np.float)
