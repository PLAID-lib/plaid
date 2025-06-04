"""Utility functions for computing statistics on datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import logging
from typing import Union

import numpy as np

from plaid.containers.dataset import Dataset
from plaid.containers.sample import Sample
from plaid.utils.base import ShapeError

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)

# %% Functions


def aggregate_stats(
    sizes: np.ndarray, means: np.ndarray, vars: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute aggregated statistics of a batch of already computed statistics (without original samples information).

    This function calculates aggregated statistics, such as the total number of samples, mean, and variance, by taking into account the statistics computed for each batch of data.

    cf: https://fr.wikipedia.org/wiki/Variance_(math%C3%A9matiques)

    Args:
        sizes (np.ndarray): An array containing the sizes (number of samples) of each batch.
        means (np.ndarray): An array containing the means of each batch.
        vars (np.ndarray): An array containing the variances of each batch.

    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray]: A tuple containing the aggregated statistics in the following order:
        - Total number of samples in all batches.
        - Weighted mean calculated from the batch means.
        - Weighted variance calculated from the batch variances, considering the means.
    """
    total_n_samples = np.sum(sizes, keepdims=True)
    total_mean = np.sum(sizes * means, keepdims=True) / total_n_samples
    total_var = (
        np.sum(sizes * (vars + (total_mean - means) ** 2), keepdims=True)
        / total_n_samples
    )
    return total_n_samples, total_mean, total_var


# %% Classes


class OnlineStatistics(object):
    """OnlineStatistics is a class for computing online statistics (e.g., min, max, mean, variance, and standard deviation) of numpy arrays."""

    def __init__(self) -> None:
        """Initialize an empty OnlineStatistics object."""
        self.n_samples: int = 0
        self.min: np.ndarray = None
        self.max: np.ndarray = None
        self.mean: np.ndarray = None
        self.var: np.ndarray = None
        self.std: np.ndarray = None

    def add_samples(self, x: np.ndarray) -> None:
        """Add samples to compute statistics for.

        Args:
            x (np.ndarray): The input numpy array containing samples data.

        Raises:
            ShapeError: Raised when there is an inconsistency in the shape of the input array.
        """
        if x.ndim == 1:
            if self.min is not None:
                if self.min.size == 1:
                    # n_samples x 1
                    x = x.reshape((-1, 1))
                else:
                    # 1 x n_features
                    x = x.reshape((1, -1))
            else:  # pragma: no cover
                raise ShapeError(
                    "can't determine if input array with ndim=1, is 1 x n_features or n_samples x 1"
                )
        elif x.ndim > 2:
            # suppose last dim is features dim, all previous dims are space
            # dims and are aggregated
            x = x.reshape((-1, x.shape[-1]))

        added_n_samples = len(x)
        added_min = np.min(x, axis=0, keepdims=True)
        added_max = np.max(x, axis=0, keepdims=True)
        added_mean = np.mean(x, axis=0, keepdims=True)
        added_var = np.var(x, axis=0, keepdims=True)

        if (
            (self.n_samples == 0)
            or (self.min is None)
            or (self.max is None)
            or (self.mean is None)
            or (self.var is None)
        ):
            self.n_samples = added_n_samples
            self.min = added_min
            self.max = added_max
            self.mean = added_mean
            self.var = added_var
        else:
            self.min = np.min(np.concatenate((self.min, added_min), axis=0), axis=0)
            self.max = np.max(np.concatenate((self.max, added_max), axis=0), axis=0)
            # new_n_samples = self.n_samples + added_n_samples
            # new_mean = (
            #     self.n_samples * self.mean + added_n_samples * added_mean
            # ) / new_n_samples
            self.n_samples, self.mean, self.var = aggregate_stats(
                np.concatenate(
                    [
                        self.n_samples + np.zeros(self.mean.shape, dtype=int),
                        added_n_samples + np.zeros(added_mean.shape, dtype=int),
                    ]
                ),
                np.concatenate([self.mean, added_mean]),
                np.concatenate([self.var, added_var]),
            )

            # # cf: https://fr.wikipedia.org/wiki/Variance_(math%C3%A9matiques)
            # self.var = (self.n_samples * (self.var + (new_mean - self.mean)**2) + added_n_samples*(added_var + (new_mean - added_mean)**2)) / new_n_samples
            # self.n_samples = new_n_samples
            # self.mean = new_mean

        self.std = np.sqrt(self.var)

    def flatten_array(self) -> None:
        """When a shape incoherence is detected, you should call this function."""
        self.min = np.min(self.min, keepdims=True)
        self.max = np.max(self.max, keepdims=True)
        assert self.mean.shape == self.var.shape
        self.n_samples, self.mean, self.var = aggregate_stats(
            np.zeros(self.mean.shape, dtype=int) + self.n_samples, self.mean, self.var
        )
        self.std = np.sqrt(self.var)

    def get_stats(self) -> dict[str, np.ndarray]:
        """Get computed statistics.

        Returns:
            dict[str,np.ndarray]:  A dictionary containing computed statistics.
        """
        return {
            "n_samples": self.n_samples,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "var": self.var,
            "std": self.std,
        }


class Stats(object):
    """Stats is a class for aggregating and computing statistics for datasets."""

    def __init__(self):
        """Initialize an empty Stats object."""
        self._stats = {}

    def add_dataset(self, dset: Dataset) -> None:
        """Add a dataset to compute statistics for.

        Args:
            dset (Dataset): The dataset to add.
        """
        self.add_samples(dset)

    def add_samples(self, samples: Union[list[Sample], Dataset]) -> None:
        """Add samples (or a dataset) to compute statistics for.

        Args:
            samples (Union[list[Sample],Dataset]): The list of samples or a dataset to add.
        """
        # ---# Aggregate
        new_data = {}
        for sample in samples:
            # ---# Scalars
            for s_name in sample.get_scalar_names():
                if s_name not in new_data:
                    new_data[s_name] = []
                new_data[s_name].append(sample.get_scalar(s_name))

            # ---# Fields
            # TODO

            # ---# Categorical
            # TODO

            # ---# SpatialSupport (Meshes)
            # TODO

            # ---# TemporalSupport
            # TODO

        # ---# Process
        for name in new_data:
            # new_shapes = [value.shape for value in new_data[name] if value.shape!=new_data[name][0].shape]
            # has_same_shape = (len(new_shapes)==0)
            has_same_shape = True

            if has_same_shape:
                new_data[name] = np.array(new_data[name])
            else:  # pragma: no cover  ### remove "no cover" when "has_same_shape = True" is no longer used
                if name in self._stats:
                    self._stats[name].flatten_array()
                new_data[name] = np.concatenate(
                    [np.ravel(value) for value in new_data[name]]
                )

            if new_data[name].ndim == 1:
                new_data[name] = new_data[name].reshape((-1, 1))

            if name not in self._stats:
                self._stats[name] = OnlineStatistics()

            self._stats[name].add_samples(new_data[name])

    def get_stats(self) -> dict[str, dict[str, np.ndarray]]:
        """Get computed statistics for different data identifiers.

        Returns:
            dict[str,dict[str,np.ndarray]]: A dictionary containing computed statistics for different data identifiers.
        """
        stats = {}
        for identifier in self._stats:
            stats[identifier] = {}
            for stat_name, stat_value in self._stats[identifier].get_stats().items():
                stats[identifier][stat_name] = np.squeeze(stat_value)

        return stats

    # TODO : FAIRE DEUX FONCTIONS :
    # - compute_stats(samples) -> stats
    # - aggregate_stats(list[stats])

    # TODO: reuse this ? more adapted to heterogenous data
    # def _compute_scalars_stats_(self) -> None:
    #     nb_samples_with_scalars = 0
    #     scalars_have_timestamps = False
    #     full_scalars = []
    #     full_scalars_timestamps = []
    #     for sample in self.samples:
    #         if 'scalars' in sample._data:
    #             nb_samples_with_scalars += 1
    #             if isinstance(sample._data['scalars'], dict):
    #                 scalars_have_timestamps = True
    #                 for k in sample._data['scalars']:
    #                     full_scalars_timestamps.append(k)
    #                 for val in sample._data['scalars'].values():
    #                     full_scalars.append(val)
    #             elif isinstance(sample._data['scalars'], tuple):
    #                 scalars_have_timestamps = True
    #                 full_scalars_timestamps.append(sample._data['scalars'][0])
    #                 full_scalars.append(sample._data['scalars'][1])
    #             else:
    #                 full_scalars.append(sample._data['scalars'])
    #     if nb_samples_with_scalars>0:
    #         full_scalars = np.array(full_scalars)
    #         logger.debug("full_scalars.shape: {}".format(full_scalars.shape))
    #         self._stats['scalars'] = {
    #             'min': np.min(full_scalars, axis=0),
    #             'max': np.max(full_scalars, axis=0),
    #             'mean': np.mean(full_scalars, axis=0),
    #             'std': np.std(full_scalars, axis=0),
    #             'var': np.var(full_scalars, axis=0),
    #         }
    #         if scalars_have_timestamps:
    #             full_scalars_timestamps = np.array(full_scalars_timestamps)
    #             logger.debug("full_scalars_timestamps.shape: {}".format(full_scalars_timestamps.shape))
    #             self._stats['scalars_timestamps'] = {
    #                 'min': np.min(full_scalars_timestamps),
    #                 'max': np.max(full_scalars_timestamps),
    #                 'mean': np.mean(full_scalars_timestamps),
    #                 'std': np.std(full_scalars_timestamps),
    #                 'var': np.var(full_scalars_timestamps),
    #             }

    # def _compute_fields_stats_(self) -> None:
    #     nb_samples_with_fields = 0
    #     fields_have_timestamps = False
    #     full_fields = []
    #     full_fields_timestamps = []
    #     for sample in self.samples:
    #         if 'fields' in sample._data:
    #             nb_samples_with_fields += 1
    #             if isinstance(sample._data['fields'], dict):
    #                 fields_have_timestamps = True
    #                 for k in sample._data['fields']:
    #                     full_fields_timestamps.append(k)
    #                 for val in sample._data['fields'].values():
    #                     full_fields.append(val)
    #             elif isinstance(sample._data['fields'], tuple):
    #                 fields_have_timestamps = True
    #                 full_fields_timestamps.append(sample._data['fields'][0])
    #                 full_fields.append(sample._data['fields'][1])
    #             else:
    #                 full_fields.append(sample._data['fields'])
    #     if nb_samples_with_fields>0:
    #         full_fields = np.concatenate(full_fields, axis=0)
    #         logger.debug("full_fields.shape: {}".format(full_fields.shape))
    #         self._stats['fields'] = {
    #             'min': np.min(full_fields, axis=0),
    #             'max': np.max(full_fields, axis=0),
    #             'mean': np.mean(full_fields, axis=0),
    #             'std': np.std(full_fields, axis=0),
    #             'var': np.var(full_fields, axis=0),
    #         }
    #         if fields_have_timestamps:
    #             full_fields_timestamps = np.array(full_fields_timestamps)
    #             logger.debug("full_fields_timestamps.shape: {}".format(full_fields_timestamps.shape))
    #             self._stats['fields_timestamps'] = {
    #                 'min': np.min(full_fields_timestamps),
    #                 'max': np.max(full_fields_timestamps),
    #                 'mean': np.mean(full_fields_timestamps),
    #                 'std': np.std(full_fields_timestamps),
    #                 'var': np.var(full_fields_timestamps),
    #             }

    # def _compute_mesh_stats_(self) -> None:
    #     nb_samples_with_mesh = 0
    #     mesh_have_timestamps = False
    #     full_mesh = []
    #     full_mesh_timestamps = []
    #     for sample in self.samples:
    #         if 'mesh' in sample._data:
    #             nb_samples_with_mesh += 1
    #             if isinstance(sample._data['mesh'], dict):
    #                 mesh_have_timestamps = True
    #                 for k in sample._data['mesh']:
    #                     full_mesh_timestamps.append(k)
    #                 for val in sample._data['mesh'].values():
    #                     full_mesh.append(val)
    #             elif isinstance(sample._data['mesh'], tuple):
    #                 mesh_have_timestamps = True
    #                 full_mesh_timestamps.append(sample._data['mesh'][0])
    #                 full_mesh.append(sample._data['mesh'][1])
    #             else:
    #                 full_mesh.append(sample._data['mesh'])
    #     if nb_samples_with_mesh>0:
    #         full_mesh = np.array(full_mesh)
    #         logger.debug("full_mesh.shape: {}".format(full_mesh.shape))
    #         self._stats['mesh'] = {
    #             'min': np.min(full_mesh, axis=0),
    #             'max': np.max(full_mesh, axis=0),
    #             'mean': np.mean(full_mesh, axis=0),
    #             'std': np.std(full_mesh, axis=0),
    #             'var': np.var(full_mesh, axis=0),
    #         }
    #         if mesh_have_timestamps:
    #             full_mesh_timestamps = np.array(full_mesh_timestamps)
    #             logger.debug("full_mesh_timestamps.shape: {}".format(full_mesh_timestamps.shape))
    #             self._stats['mesh_timestamps'] = {
    #                 'min': np.min(full_mesh_timestamps),
    #                 'max': np.max(full_mesh_timestamps),
    #                 'mean': np.mean(full_mesh_timestamps),
    #                 'std': np.std(full_mesh_timestamps),
    #                 'var': np.var(full_mesh_timestamps),
    #             }
