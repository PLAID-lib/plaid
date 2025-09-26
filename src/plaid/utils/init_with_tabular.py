"""Utility functions to initialize a Dataset with tabular data."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import logging

import numpy as np

from plaid import Dataset, Sample

# from plaid.quantity import QuantityValueType

logger = logging.getLogger(__name__)


# %% Functions


def initialize_dataset_with_tabular_data(
    tabular_data: dict[str, np.ndarray],
) -> Dataset:
    """Initialize a Dataset with tabular data.

    This function takes a dictionary of tabular data where keys represent scalar names,
    and values are numpy arrays of the same length. It creates a Dataset and adds samples
    to it based on the provided tabular data.

    Args:
        tabular_data (dict[str,np.ndarray]): A dictionary of scalar names and corresponding numpy arrays.

    Returns:
        Dataset: A Dataset initialized with the tabular data.

    Raises:
        AssertionError: If the lengths of the numpy arrays in tabular data are not identical.

    Example:
        .. code-block:: python

            import numpy as np
            from plaid.utils.init import initialize_dataset_with_tabular_data
            tabular_data = {'feature1': np.array([1, 2, 3]), 'feature2': np.array([4, 5, 6])}
            dataset = initialize_dataset_with_tabular_data(tabular_data)
    """
    lengths = [len(value) for value in tabular_data.values()]
    assert len(list(set(lengths))) == 1, "sizes not identical in tabular data"

    dataset = Dataset()

    nb_samples = lengths[0]
    for i in range(nb_samples):
        sample = Sample()
        for scalar_name, value in tabular_data.items():
            sample.add_scalar(scalar_name, value[i])
        dataset.add_sample(sample)

    # TODO:
    # logger.info("Pour l'instant on boucle sur les samples, il y a probablement mieux à faire, mais l'API est simple")

    return dataset


# def initialize_quantity_dataset_with_tabular_data(tabular_data:dict[str,Union[list[QuantityValueType],np.ndarray]]) -> Dataset:
#     """_summary_

#     Args:
# tabular_data (dict[str,Union[list[QuantityValueType],np.ndarray]]):
# `feature_name` -> tabular values

#     Returns:
#         Dataset
#     """
#     lengths = [len(value) for value in tabular_data.values()]
#     assert len(list(set(lengths))) == 1, "sizes not identical in tabular data"

#     #---# Adds data to collection
#     data_collection = DataCollection()
#     for name in tabular_data:
#         storage = data_collection.add_storage('quantity', name)
#         storage.add_values(tabular_data[name])

#     #---# Link samples to data in collection
#     dataset = Dataset()
#     nb_samples = lengths[0]
#     for i_samp in range(nb_samples):
#         sample = Sample(data_collection = data_collection)
#         for feature_name in tabular_data:
#             sample.link_to_value("quantity", feature_name, i_samp)
#         dataset.add_sample(sample)

#     return dataset

# %% Classes
