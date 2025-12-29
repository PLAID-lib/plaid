"""Utility functions for computing statistics on datasets."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import copy
import logging
import sys
from typing import Union

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")


import numpy as np

from plaid import Dataset, Sample
from plaid.constants import CGNS_FIELD_LOCATIONS

logger = logging.getLogger(__name__)


# %% Functions


def aggregate_stats(
    sizes: np.ndarray, means: np.ndarray, vars: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute aggregated statistics of a batch of already computed statistics (without original samples information).

    This function calculates aggregated statistics, such as the total number of samples, mean, and variance, by taking into account the statistics computed for each batch of data.

    cf: [Variance from (cardinal,mean,variance) of several statistical series](https://fr.wikipedia.org/wiki/Variance_(math%C3%A9matiques)#Formules)

    Args:
        sizes (np.ndarray): An array containing the sizes (number of samples) of each batch. Expect shape (n_batches,1).
        means (np.ndarray): An array containing the means of each batch. Expect shape (n_batches, n_features).
        vars (np.ndarray): An array containing the variances of each batch. Expect shape (n_batches, n_features).

    Returns:
        tuple[np.ndarray,np.ndarray,np.ndarray]: A tuple containing the aggregated statistics in the following order:
        - Total number of samples in all batches.
        - Weighted mean calculated from the batch means.
        - Weighted variance calculated from the batch variances, considering the means.
    """
    assert sizes.ndim == 1
    assert means.ndim == 2
    assert len(sizes) == len(means)
    assert means.shape == vars.shape
    sizes = sizes.reshape((-1, 1))
    total_n_samples = np.sum(sizes)
    total_mean = np.sum(sizes * means, axis=0, keepdims=True) / total_n_samples
    total_var = (
        np.sum(sizes * (vars + (total_mean - means) ** 2), axis=0, keepdims=True)
        / total_n_samples
    )
    return total_n_samples, total_mean, total_var


# %% Classes


class OnlineStatistics(object):
    """OnlineStatistics is a class for computing online statistics of numpy arrays.

    This class computes running statistics (min, max, mean, variance, std) for streaming data
    without storing all samples in memory.

    Example:
        >>> stats = OnlineStatistics()
        >>> stats.add_samples(np.array([[1, 2], [3, 4]]))
        >>> stats.add_samples(np.array([[5, 6]]))
        >>> print(stats.get_stats()['mean'])
        [[3. 4.]]
    """

    def __init__(self) -> None:
        """Initialize an empty OnlineStatistics object."""
        self.n_samples: int = 0
        self.n_features: int = None
        self.n_points: int = None
        self.min: np.ndarray = None
        self.max: np.ndarray = None
        self.mean: np.ndarray = None
        self.var: np.ndarray = None
        self.std: np.ndarray = None

    def add_samples(self, x: np.ndarray, n_samples: int = None) -> None:
        """Add samples to compute statistics for.

        Args:
            x (np.ndarray): The input numpy array containing samples data. Expect 2D arrays with shape (n_samples, n_features).
            n_samples (int, optional): The number of samples in the input array. If not provided, it will be inferred from the shape of `x`. Use this argument when the input array has already been flattened because of shape inconsistencies.

        Raises:
            ValueError: Raised when input contains NaN or Inf values.
        """
        # Validate input
        if not isinstance(x, np.ndarray):
            raise TypeError("Input must be a numpy array")

        if np.any(~np.isfinite(x)):
            raise ValueError("Input contains NaN or Inf values")

        # Handle 1D arrays
        if x.ndim == 1:
            if self.min is not None:
                if self.min.shape[1] == 1:
                    x = x.reshape((-1, 1))
                else:
                    x = x.reshape((1, -1))
            else:
                x = x.reshape((-1, 1))  # Default to column vector

        # Handle n-dimensional arrays
        elif x.ndim > 2:
            # if we have array of shape (n_samples, n_points, n_features)
            # it will be reshaped to (n_samples * n_points, n_features)
            x = x.reshape((-1, x.shape[-1]))

        if self.n_features is None:
            self.n_features = x.shape[1]

        if x.shape[1] != self.n_features:
            # it means that stats where previously on a per-point mode,
            # but it is no longer possible as the new added samples have a different shape
            # so we need to shift the stats to a per-sample mode, and then flatten the stats array
            self.flatten_array()
            n_samples = x.shape[0]
            x = x.reshape((-1, 1))

        added_n_samples = len(x) if n_samples is None else n_samples
        added_n_points = x.size
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
            self.n_points = added_n_points
            self.min = added_min
            self.max = added_max
            self.mean = added_mean
            self.var = added_var
        else:
            self.min = np.min(
                np.concatenate((self.min, added_min), axis=0), axis=0, keepdims=True
            )
            self.max = np.max(
                np.concatenate((self.max, added_max), axis=0), axis=0, keepdims=True
            )
            if self.n_features > 1:
                # feature not flattened, we are on a per-sample mode
                self.n_points += added_n_points
                self.n_samples, self.mean, self.var = aggregate_stats(
                    np.array([self.n_samples, added_n_samples]),
                    np.concatenate([self.mean, added_mean]),
                    np.concatenate([self.var, added_var]),
                )
            else:
                # feature flattened, we are on a per-point mode
                self.n_samples += added_n_samples
                self.n_points, self.mean, self.var = aggregate_stats(
                    np.array([self.n_points, added_n_points]),
                    np.concatenate([self.mean, added_mean]),
                    np.concatenate([self.var, added_var]),
                )

        self.std = np.sqrt(self.var)

    def merge_stats(self, other: Self) -> None:
        """Merge statistics from another instance.

        Args:
            other (Self): The other instance to merge statistics from.
        """
        if not isinstance(other, self.__class__):
            raise TypeError("Can only merge with another instance of the same class")

        if self.n_features != other.n_features:
            # flatten both
            self.flatten_array()
            other = copy.deepcopy(other)
            other.flatten_array()
            assert self.min.shape == other.min.shape, (
                "Shape mismatch in OnlineStatistics merging"
            )

        self.min = np.min(
            np.concatenate((self.min, other.min), axis=0), axis=0, keepdims=True
        )
        self.max = np.max(
            np.concatenate((self.max, other.max), axis=0), axis=0, keepdims=True
        )
        self.n_points += other.n_points
        self.n_samples, self.mean, self.var = aggregate_stats(
            np.array([self.n_samples, other.n_samples]),
            np.concatenate([self.mean, other.mean]),
            np.concatenate([self.var, other.var]),
        )
        self.std = np.sqrt(self.var)

    def flatten_array(self) -> None:
        """When a shape incoherence is detected, you should call this function."""
        self.min = np.min(self.min, keepdims=True).reshape(1, 1)
        self.max = np.max(self.max, keepdims=True).reshape(1, 1)
        self.n_points = self.n_samples * self.n_features
        assert self.mean.shape == self.var.shape
        self.n_points, self.mean, self.var = aggregate_stats(
            np.array([self.n_samples] * self.n_features),
            self.mean.reshape(-1, 1),
            self.var.reshape(-1, 1),
        )
        self.std = np.sqrt(self.var)

        self.n_features = 1

    def get_stats(self) -> dict[str, Union[int, np.ndarray]]:
        """Get computed statistics.

        Returns:
            dict[str, Union[int, np.ndarray]]: A dictionary containing computed statistics.
            The shapes of the arrays depend on the input data and may vary.
        """
        return {
            "n_samples": self.n_samples,
            "n_points": self.n_points,
            "n_features": self.n_features,
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "var": self.var,
            "std": self.std,
        }


class Stats:
    """Class for aggregating and computing statistics across datasets.

    The Stats class processes both scalar and field data from samples or datasets,
    computing running statistics like min, max, mean, variance and standard deviation.

    Attributes:
        _stats (dict[str, OnlineStatistics]): Dictionary mapping data identifiers to their statistics
    """

    def __init__(self):
        """Initialize an empty Stats object."""
        self._stats: dict[str, OnlineStatistics] = {}
        self._feature_is_flattened: dict[str, bool] = {}

    def add_dataset(self, dset: Dataset) -> None:
        """Add a dataset to compute statistics for.

        Args:
            dset (Dataset): The dataset to add.
        """
        self.add_samples(dset)

    def add_samples(self, samples: Union[list[Sample], Dataset]) -> None:
        """Add samples or a dataset to compute statistics for.

        Compute stats for each features present in the samples among scalars and fields.
        For fields, as long as the added samples have the same shape as the existing ones,
        the stats will be computed per-coordinates (n_features=x.shape[-1]).
        But as soon as the shapes differ, the stats and added fields will be flattened (n_features=1),
        then stats will be computed over all values of the field.

        Args:
            samples (Union[list[Sample], Dataset]): List of samples or dataset to process

        Raises:
            TypeError: If samples is not a list[Sample] or Dataset
            ValueError: If a sample contains invalid data
        """
        # Input validation
        if not isinstance(samples, (list, Dataset)):
            raise TypeError("samples must be a list[Sample] or Dataset")

        # Process each sample
        new_data: dict[str, list] = {}

        for sample in samples:
            # Process scalars
            self._process_scalar_data(sample, new_data)

            # Process fields
            self._process_field_data(sample, new_data)

            # ---# SpatialSupport (Meshes)
            # TODO

            # ---# TemporalSupport
            # TODO

            # ---# Categorical
            # TODO

        # Update statistics
        self._update_statistics(new_data)

    def get_stats(
        self, identifiers: list[str] = None
    ) -> dict[str, dict[str, np.ndarray]]:
        """Get computed statistics for specified data identifiers.

        Args:
            identifiers (list[str], optional): List of data identifiers to retrieve.
                If None, returns statistics for all identifiers.

        Returns:
            dict[str, dict[str, np.ndarray]]: Dictionary mapping identifiers to their statistics
        """
        if identifiers is None:
            identifiers = self.get_available_statistics()

        stats = {}
        for identifier in identifiers:
            if identifier in self._stats:
                stats[identifier] = {}
                for stat_name, stat_value in (
                    self._stats[identifier].get_stats().items()
                ):
                    stats[identifier][stat_name] = stat_value
                    # stats[identifier][stat_name] = np.squeeze(stat_value)

        return stats

    def get_available_statistics(self) -> list[str]:
        """Get list of data identifiers with computed statistics.

        Returns:
            list[str]: List of data identifiers
        """
        return sorted(self._stats.keys())

    def clear_statistics(self) -> None:
        """Clear all computed statistics."""
        self._stats.clear()

    def merge_stats(self, other: Self) -> None:
        """Merge statistics from another Stats object.

        Args:
            other (Stats): Stats object to merge with
        """
        for name, stats in other._stats.items():
            if name not in self._stats:
                self._stats[name] = copy.deepcopy(stats)
            else:
                self._stats[name].merge_stats(stats)

    def _process_scalar_data(self, sample: Sample, data_dict: dict[str, list]) -> None:
        """Process scalar data from a sample.

        Args:
            sample (Sample): Sample containing scalar data
            data_dict (dict[str, list]): Dictionary to store processed data
        """
        for name in sample.get_scalar_names():
            if name not in data_dict:
                data_dict[name] = []
            value = sample.get_scalar(name)
            if value is not None:
                data_dict[name].append(np.array(value).reshape((1, -1)))

    def _process_field_data(self, sample: Sample, data_dict: dict[str, list]) -> None:
        """Process field data from a sample.

        Args:
            sample (Sample): Sample containing field data
            data_dict (dict[str, list]): Dictionary to store processed data
        """
        for time in sample.features.get_all_time_values():
            for base_name in sample.features.get_base_names(time=time):
                for zone_name in sample.features.get_zone_names(
                    base_name=base_name, time=time
                ):
                    for location in CGNS_FIELD_LOCATIONS:
                        for field_name in sample.get_field_names(
                            location=location,
                            zone_name=zone_name,
                            base_name=base_name,
                            time=time,
                        ):
                            stat_key = (
                                f"{base_name}/{zone_name}/{location}/{field_name}"
                            )
                            if stat_key not in data_dict:
                                data_dict[stat_key] = []
                            field = sample.get_field(
                                field_name,
                                location=location,
                                zone_name=zone_name,
                                base_name=base_name,
                                time=time,
                            ).reshape((1, -1))
                            if field is not None:
                                # check if all previous arrays are the same shape as the new one that will be added to data_dict[stat_key]
                                if len(
                                    data_dict[stat_key]
                                ) > 0 and not self._feature_is_flattened.get(
                                    stat_key, False
                                ):
                                    prev_shape = data_dict[stat_key][0].shape
                                    if field.shape != prev_shape:
                                        # set this stat as flattened
                                        self._feature_is_flattened[stat_key] = True
                                        # flatten corresponding stat
                                        if stat_key in self._stats:
                                            self._stats[stat_key].flatten_array()

                                if self._feature_is_flattened.get(stat_key, False):
                                    field = field.reshape((-1, 1))

                                data_dict[stat_key].append(field)

    def _update_statistics(self, new_data: dict[str, list]) -> None:
        """Update running statistics with new data.

        Args:
            new_data (dict[str, list]): Dictionary containing new data to process
        """
        for name, list_of_arrays in new_data.items():
            if len(list_of_arrays) > 0:
                if name not in self._stats:
                    self._stats[name] = OnlineStatistics()

                # internal check, should never happen if self._process_* functions work correctly
                for sample_id in range(len(list_of_arrays)):
                    assert isinstance(list_of_arrays[sample_id], np.ndarray)
                    assert list_of_arrays[sample_id].ndim == 2, (
                        f"for feature <{name}> -> {sample_id=}: {list_of_arrays[sample_id].ndim=} should be 2"
                    )

                if self._feature_is_flattened.get(name, False):
                    # flatten all arrays in list_of_arrays
                    n_samples = len(list_of_arrays)
                    for i in range(len(list_of_arrays)):
                        list_of_arrays[i] = list_of_arrays[i].reshape((-1, 1))
                else:
                    n_samples = None

                # Convert to numpy array and reshape if needed
                data = np.concatenate(list_of_arrays)
                assert data.ndim == 2

                self._stats[name].add_samples(data, n_samples=n_samples)

        # # old version of _update_statistics logic
        # for name in new_data:
        #     # new_shapes = [value.shape for value in new_data[name] if value.shape!=new_data[name][0].shape]
        #     # has_same_shape = (len(new_shapes)==0)
        #     has_same_shape = True

        #     if has_same_shape:
        #         new_data[name] = np.array(new_data[name])
        #     else:  # pragma: no cover  ### remove "no cover" when "has_same_shape = True" is no longer used
        #         if name in self._stats:
        #             self._stats[name].flatten_array()
        #         new_data[name] = np.concatenate(
        #             [np.ravel(value) for value in new_data[name]]
        #         )

        #     if new_data[name].ndim == 1:
        #         new_data[name] = new_data[name].reshape((-1, 1))

        #     if name not in self._stats:
        #         self._stats[name] = OnlineStatistics()

        #     self._stats[name].add_samples(new_data[name])

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
