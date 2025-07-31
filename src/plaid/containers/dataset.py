"""Implementation of the `Dataset` container."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports
import copy
import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import logging
import os
import shutil
import subprocess
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, Union

import numpy as np
import yaml
from tqdm import tqdm

from plaid.constants import AUTHORIZED_INFO_KEYS
from plaid.containers.sample import Sample
from plaid.containers.utils import check_features_size_homogeneity
from plaid.types import Array, FeatureType
from plaid.utils.base import DeprecatedError, ShapeError, generate_random_ASCII

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)


# %% Functions


def process_sample(sample_path: Union[str, Path]) -> tuple:  # pragma: no cover
    """Load Sample from path.

    Args:
        sample_path (Union[str,Path]): The path of the Sample.

    Returns:
        tuple: The loaded Sample and its ID.
    """
    sample_path = Path(sample_path)
    id = int(sample_path.stem.split("_")[-1])
    return id, Sample(sample_path)


# %% Classes


class Dataset(object):
    """A set of samples, and optionnaly some other informations about the Dataset."""

    def __init__(
        self,
        directory_path: Union[str, Path] = None,
        verbose: bool = False,
        processes_number: int = 0,
    ) -> None:
        """Initialize a :class:`Dataset <plaid.containers.dataset.Dataset>`.

        If `directory_path` is not specified it initializes an empty :class:`Dataset <plaid.containers.dataset.Dataset>` that should be fed with :class:`Samples <plaid.containers.sample.Sample>`.

        Use :meth:`add_sample <plaid.containers.dataset.Dataset.add_sample>` or :meth:`add_samples <plaid.containers.dataset.Dataset.add_samples>` to feed the :class:`Dataset`

        Args:
            directory_path (Union[str,Path], optional): The path from which to load PLAID dataset files.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset

                # 1. Create empty instance of Dataset
                dataset = Dataset()
                print(dataset)
                >>> Dataset(0 samples, 0 scalars, 0 fields)
                print(len(dataset))
                >>> 0

                # 2. Load dataset and create Dataset instance
                dataset = Dataset("path_to_plaid_dataset") # .plaid or directory
                print(dataset)
                >>> Dataset(3 samples, 2 scalars, 5 fields)
                print(len(dataset))
                >>> 3
                for sample in dataset:
                    print(sample)
                >>> Sample(1 scalar, 0 time series, 1 timestamp, 2 fields)
                    Sample(1 scalar, 0 time series, 0 timestamps, 0 fields)
                    Sample(2 scalars, 0 time series, 1 timestamp, 2 fields)

        Caution:
            It is assumed that you provided a compatible PLAID dataset.
        """
        self._samples: dict[int, Sample] = {}  # sample_id -> sample
        # info_name -> description
        self._infos: dict[str, dict[str, str]] = {}

        if directory_path is not None:
            directory_path = Path(directory_path)

            if directory_path.suffix == ".plaid":
                self.load(
                    directory_path, verbose=verbose, processes_number=processes_number
                )
            else:
                self._load_from_dir_(
                    directory_path, verbose=verbose, processes_number=processes_number
                )

    def copy(self) -> Self:
        """Create a deep copy of the dataset.

        Returns:
            A new `Dataset` instance with all internal data (samples, infos)
            deeply copied to ensure full isolation from the original.

        Notes:
            This operation may be memory-intensive for large datasets.
        """
        return copy.deepcopy(self)

    # -------------------------------------------------------------------------#
    def get_samples(
        self, ids: list[int] = None, as_list: bool = False
    ) -> dict[int, Sample]:
        """Return dictionnary of samples with ids corresponding to :code:`ids` if specified, else all samples.

        Args:
            ids (list[int], optional): If None, take all samples. Defaults to None.
            as_list (bool, optional): If False, return a dict ``id -> sample``, else return a list on ``Sample`` in the same order as ``ids``. Defaults to False.

        Returns:
            dict[int,Sample]: Samples with corresponding ids.
        """
        if ids is None:
            ids = sorted(list(self._samples.keys()))
        if as_list:
            return [self._samples[id] for id in ids]
        else:
            return {id: self._samples[id] for id in ids}

    def add_sample(self, sample: Sample, id: int = None) -> int:
        """Add a new :class:`Sample <plaid.containers.sample.Sample>` to the :class:`Dataset <plaid.containers.dataset.Dataset>.`.

        Args:
            sample (Sample): The sample to add.
            id (int, optional): An optional ID for the new sample. If not provided, the ID will be automatically generated based on the current number of samples in the dataset.

        Raises:
            TypeError: If ``sample`` is not a :class:`Sample <plaid.containers.sample.Sample>`.

        Returns:
            int: Id of the new added :class:`Sample <plaid.containers.sample.Sample>`.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                dataset.add_sample(sample)
                print(dataset)
                >>> Dataset(3 samples, 0 scalars, 2 fields)
        """
        if not (isinstance(sample, Sample)):
            raise TypeError(f"sample should be of type Sample but {type(sample)=}")

        if id is None:
            id = len(self)
        self.set_sample(id=id, sample=sample)
        return id

    def del_sample(self, sample_id: int) -> None:
        """Delete a :class:`Sample <plaid.containers.sample.Sample>` from the :class:`Dataset <plaid.containers.dataset.Dataset>` and reorganize the remaining sample IDs to eliminate gaps.

        Args:
            sample_id (int): The ID of the sample to delete.

        Raises:
            ValueError: If the provided sample ID is not present in the dataset.

        Returns:
            list[int]: The new list of sample ids.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                dataset.add_samples(samples)
                print(dataset)
                >>> Dataset(1 samples, y scalars, x fields)
                dataset.del_sample(0)
                print(dataset)
                >>> Dataset(0 samples, 0 scalars, 0 fields)
        """
        if sample_id < 0 or sample_id >= len(self._samples):
            raise ValueError(
                f"Invalid ID {sample_id}, it must be within [0, len(dataset)]"
            )

        if sample_id == len(self) - 1:
            return self._samples.pop(sample_id)

        deleted_sample = self._samples[sample_id]
        keys_to_move = np.arange(sample_id + 1, len(self._samples))

        # Move each key one position back
        for key in keys_to_move:
            self._samples[key - 1] = self._samples.pop(key)

        return deleted_sample

    def add_samples(self, samples: list[Sample], ids: list[int] = None) -> list[int]:
        """Add new :class:`Samples <plaid.containers.sample.Sample>` to the :class:`Dataset <plaid.containers.dataset.Dataset>`.

        Args:
            samples (list[Sample]): The list of samples to add.
            ids (list[int], optional): An optional list of IDs for the new samples. If not provided, the IDs will be automatically generated based on the current number of samples in the dataset.

        Raises:
            TypeError: If ``samples`` is not a list or if one of the ``samples`` is not a :class:`Sample <plaid.containers.sample.Sample>`.
            ValueError: If samples list is empty.
            ValueError: If the length of ids list (if provided) is not equal to the length of samples list.
            ValueError: If provided ids are not unique.

        Returns:
            list[int]: Ids of added :class:`Samples <plaid.containers.sample.Sample>`.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                dataset.add_samples(samples)
                print(len(samples))
                >>> n
                print(dataset)
                >>> Dataset(n samples, 0 scalars, x fields)
        """
        if not (isinstance(samples, list)):
            raise TypeError(f"samples should be of type list but {type(samples)=}")
        if samples == []:
            raise ValueError("The list of samples to add is empty")

        for i_sample, sample in enumerate(samples):
            if not (isinstance(sample, Sample)):
                raise TypeError(
                    f"element {i_sample} of samples should be of type Sample but {type(sample)=}"
                )

        if ids is None:
            ids = list(range(len(self), len(self) + len(samples)))
        else:
            if len(samples) != len(ids):
                raise ValueError(
                    "The length of the list of samples to add and the list of IDs are different"
                )
            if len(set(ids)) != len(ids):
                raise ValueError("IDS must be unique")

        self._samples.update(dict(zip(ids, samples)))
        return ids

    def del_samples(self, sample_ids: list[int]) -> None:
        """Delete  :class:`Sample <plaid.containers.sample.Sample>` from the :class:`Dataset <plaid.containers.dataset.Dataset>` and reorganize the remaining sample IDs to eliminate gaps.

        Args:
            sample_ids (list[int]): The list of IDs of samples to delete.

        Raises:
            TypeError: If ``sample_ids`` is not a list.
            ValueError: If sample_ids list is empty.
            ValueError: If any of the sample_ids does not exist in the dataset.
            ValueError: If the provided IDs are not unique.

        Returns:
            list[int]: The new list of sample ids.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                # Assume samples are already added to the dataset
                print(dataset)
                >>> Dataset(6 samples, y scalars, x fields)
                dataset.del_samples([1, 3, 5])
                print(dataset)
                >>> Dataset(3 samples, y scalars, x fields)
        """
        if not isinstance(sample_ids, list):
            raise TypeError(
                f"sample_ids should be of type list but {type(sample_ids)=}"
            )

        if sample_ids == []:
            raise ValueError("The list of sample IDs to delete is empty")

        for id in sample_ids:
            if id < 0 or id >= len(self._samples):
                raise ValueError(
                    f"Invalid ID {id}, it must be within [0, len(dataset)]"
                )

        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("Sample with IDs must be unique")

        # Delete samples
        deleted_samples = []
        for id in sample_ids:
            deleted_samples.append(self._samples[id])
            del self._samples[id]

        # Reorganize remaining sample IDs to eliminate gaps
        # from the min index of sample_ids to delete
        del_idx_min = min(sample_ids)
        remaining_ids = list(self._samples.keys())
        for new_id, old_id in enumerate(remaining_ids[del_idx_min:], start=del_idx_min):
            if new_id != old_id:
                self._samples[new_id] = self._samples.pop(old_id)

        return deleted_samples

    # -------------------------------------------------------------------------#
    def get_sample_ids(self) -> list[int]:
        """Return list of sample ids.

        Returns:
            list[int]: List of sample ids.
        """
        return list(self._samples.keys())

    # -------------------------------------------------------------------------#
    def get_scalar_names(self, ids: list[int] = None) -> list[str]:
        """Return union of scalars names in all samples with id in ids.

        Args:
            ids (list[int], optional): Select scalars depending on sample id. If None, take all samples. Defaults to None.

        Returns:
            list[str]: List of all scalars names
        """
        if ids is not None and len(set(ids)) != len(ids):
            logger.warning("Provided ids are not unique")

        scalars_names = []
        for sample in self.get_samples(ids, as_list=True):
            s_names = sample.get_scalar_names()
            for s_name in s_names:
                if s_name not in scalars_names:
                    scalars_names.append(s_name)
        scalars_names.sort()
        return scalars_names

    # -------------------------------------------------------------------------#
    def get_time_series_names(self, ids: list[int] = None) -> list[str]:
        """Return union of time series names in all samples with id in ids.

        Args:
            ids (list[int], optional): Select time series depending on sample id. If None, take all samples. Defaults to None.

        Returns:
            list[str]: List of all time series names
        """
        if ids is not None and len(set(ids)) != len(ids):
            logger.warning("Provided ids are not unique")

        time_series_names = []
        for sample in self.get_samples(ids, as_list=True):
            ts_names = sample.get_time_series_names()
            for ts_name in ts_names:
                if ts_name not in time_series_names:
                    time_series_names.append(ts_name)
        time_series_names.sort()
        return time_series_names

    # -------------------------------------------------------------------------#
    def get_field_names(
        self, ids: list[int] = None, zone_name: str = None, base_name: str = None
    ) -> list[str]:
        """Return union of fields names in all samples with id in ids.

        Args:
            ids (list[int], optional): Select fields depending on sample id. If None, take all samples. Defaults to None.
            zone_name (str, optional): If provided, only field names from this zone will be included. Defaults to None.
            base_name (str, optional): If provided, only field names containing this base name will be included. Defaults to None.

        Returns:
            list[str]: List of all fields names.
        """
        if ids is not None and len(set(ids)) != len(ids):  # pragma: no cover
            logger.warning("Provided ids are not unique")

        fields_names = []
        for sample in self.get_samples(ids, as_list=True):
            times = sample.get_all_mesh_times()
            for time in times:
                f_names = sample.get_field_names(
                    zone_name=zone_name, base_name=base_name, time=time
                )
                for f_name in f_names:
                    if f_name not in fields_names:
                        fields_names.append(f_name)
        fields_names.sort()
        return fields_names

    # -------------------------------------------------------------------------#
    def get_all_features_identifiers(
        self, ids: list[int] = None
    ) -> list[dict[str, Union[str, float]]]:
        """Get all features identifiers from the dataset.

        Args:
            ids (list[int], optional): Sample id from which returning feature identifiers. If None, take all samples. Defaults to None.

        Returns:
            list[dict[str, Union[str, float]]]: A list of dictionaries containing the identifiers of all features in the dataset.
        """
        if ids is not None and len(set(ids)) != len(ids):
            logger.warning("Provided ids are not unique")

        all_features_identifiers = []
        for sample in self.get_samples(ids, as_list=True):
            features_identifiers = sample.get_all_features_identifiers()
            for feat_id in features_identifiers:
                if feat_id not in all_features_identifiers:
                    all_features_identifiers.append(feat_id)
        all_features_identifiers
        return all_features_identifiers

    def get_all_features_identifiers_by_type(
        self, feature_type: str, ids: list[int] = None
    ) -> list[dict[str, Union[str, float]]]:
        """Get all features identifiers from the dataset.

        Args:
            feature_type (str): Type of features to return
            ids (list[int], optional): Sample id from which returning feature identifiers. If None, take all samples. Defaults to None.

        Returns:
            list[dict[str, Union[str, float]]]: A list of dictionaries containing the identifiers of all features of a given type  in the dataset.
        """
        all_features_identifiers = self.get_all_features_identifiers(ids)
        return [
            feat_id
            for feat_id in all_features_identifiers
            if feat_id["type"] == feature_type
        ]

    # -------------------------------------------------------------------------#
    def add_tabular_scalars(self, tabular: np.ndarray, names: list[str] = None) -> None:
        """Add tabular scalar data to the summary.

        Args:
            tabular (np.ndarray): A 2D NumPy array containing tabular scalar data.
            names (list[str], optional): A list of column names for the tabular data. Defaults to None.

        Raises:
            ShapeError: Raised if the input tabular array does not have the correct shape (2D).
            ShapeError: Raised if the number of columns in the tabular data does not match the number of names provided.

        Notes:
            If no names are provided, it will automatically create names based on the pattern 'X{number}'
        """
        nb_samples = len(tabular)

        if tabular.ndim != 2:
            raise ShapeError(f"{tabular.ndim=}!=2, should be == 2")
        if names is None:
            names = [f"X{i}" for i in range(tabular.shape[1])]
        if tabular.shape[1] != len(names):
            raise ShapeError(
                f"tabular should have as many columns as there are names, but {tabular.shape[1]=} and {len(names)=}"
            )

        # ---# For efficiency, first add values to storage
        name_to_ids = {}
        for col, name in zip(tabular.T, names):
            name_to_ids[name] = col

        # ---# Then add data in sample
        for i_samp in range(nb_samples):
            sample = Sample()
            for name in names:
                sample.add_scalar(name, name_to_ids[name][i_samp])
            self.add_sample(sample)

    def get_scalars_to_tabular(
        self,
        scalar_names: list[str] = None,
        sample_ids: list[int] = None,
        as_nparray=False,
    ) -> Union[dict[str, np.ndarray], np.ndarray]:
        """Return a dict containing scalar values as tabulars/arrays.

        Args:
            scalar_names (str, optional): Scalars to work on. If None, all scalars will be returned. Defaults to None.
            sample_ids (list[int], optional): Filter by sample id. If None, take all samples. Defaults to None.
            as_nparray (bool, optional): If True, return the data as a single numpy ndarray. If False, return a dictionary mapping scalar names to their respective tabular values. Defaults to False.

        Returns:
            np.ndarray: if as_nparray is True.
            dict[str,np.ndarray]: if as_nparray is False, scalar name -> tabular values.
        """
        if scalar_names is None:
            scalar_names = self.get_scalar_names(sample_ids)
        elif len(set(scalar_names)) != len(scalar_names):
            logger.warning("Provided scalar names are not unique")

        if sample_ids is None:
            sample_ids = self.get_sample_ids()
        elif len(set(sample_ids)) != len(sample_ids):
            logger.warning("Provided sample ids are not unique")
        nb_samples = len(sample_ids)

        named_tabular = {}
        for s_name in scalar_names:
            res = np.empty(nb_samples)
            res.fill(None)
            for i_, id in enumerate(sample_ids):
                val = self[id].get_scalar(s_name)
                if val is not None:
                    res[i_] = val
            named_tabular[s_name] = res

        if as_nparray:
            named_tabular = np.array(list(named_tabular.values())).T
        return named_tabular

    # -------------------------------------------------------------------------#
    def get_feature_from_string_identifier(
        self, feature_string_identifier: str
    ) -> dict[int, FeatureType]:
        """Get a list of features from the dataset based on the provided feature string identifier.

        Args:
            feature_string_identifier (str): A string identifier for the feature.

        Returns:
            dict[int, FeatureType]: A list of features matching the provided string identifier.
        """
        return {
            id: self[id].get_feature_from_string_identifier(feature_string_identifier)
            for id in self.get_sample_ids()
        }

    def get_feature_from_identifier(
        self, feature_identifier: dict[str : Union[str, float]]
    ) -> dict[int, FeatureType]:
        """Get a list of features from the dataset based on the provided feature identifier.

        Args:
            feature_identifier (dict[str, Union[str, float]]): A dictionary containing the feature identifier.

        Returns:
            ldict[int, FeatureType]: A list of features matching the provided identifier.
        """
        return {
            id: self[id].get_feature_from_identifier(feature_identifier)
            for id in self.get_sample_ids()
        }

    def get_features_from_identifiers(
        self, feature_identifiers: list[dict[str : Union[str, float]]]
    ) -> dict[int, list[FeatureType]]:
        """Get a list of features from the dataset based on the provided feature identifiers.

        Args:
            feature_identifiers (dict[str, Union[str, float]]): A dictionary containing the feature identifier.

        Returns:
            dict[int, list[FeatureType]]: A list of features matching the provided identifier.
        """
        return {
            id: self[id].get_features_from_identifiers(feature_identifiers)
            for id in self.get_sample_ids()
        }

    def update_features_from_identifier(
        self,
        feature_identifiers: Union[
            dict[str, Union[str, float]], list[dict[str, Union[str, float]]]
        ],
        features: dict[int, Union[FeatureType, list[FeatureType]]],
        in_place: bool = False,
    ) -> Self:
        """Update one or several features of the dataset by their identifier(s).

        This method applies updates to scalars, time series, fields, or nodes
        using feature identifiers, and corresponding feature data. When `in_place=False`, a deep copy of the dataset is created
        before applying updates, ensuring full isolation from the original.

        Args:
            feature_identifiers (dict or list of dict): one or more feature identifiers.
            features (dict): dict with sample index as keys and one or more features as values.
            in_place (bool, optional): If True, modifies the current dataset in place.
                If False, returns a deep copy with updated features.

        Returns:
            Self: The updated dataset (either the current instance or a new copy).

        Raises:
            AssertionError: If types are inconsistent or identifiers contain unexpected keys.
        """
        assert set(features.keys()) == set(self.get_sample_ids()), (
            "Must provide the same sample indices in features as in the dataset"
        )

        dataset = self if in_place else self.copy()

        for id in self.get_sample_ids():
            dataset[id].update_features_from_identifier(
                feature_identifiers, features[id], in_place=True
            )
        return dataset

    def from_features_identifier(
        self,
        feature_identifiers: Union[
            dict[str, Union[str, float]], list[dict[str, Union[str, float]]]
        ],
    ) -> Self:
        """Extract features of the dataset by their identifier(s) and return a new dataset containing these features.

        This method applies updates to scalars, time series, fields, or nodes
        using feature identifiers

        Args:
            feature_identifiers (dict or list of dict): One or more feature identifiers.

        Returns:
            Self: New dataset containing the provided feature identifiers

        Raises:
            AssertionError: If types are inconsistent or identifiers contain unexpected keys.
        """
        dataset = Dataset()
        dataset.set_infos(copy.deepcopy(self.get_infos()))

        for id in self.get_sample_ids():
            extracted_sample = self[id].from_features_identifier(feature_identifiers)
            dataset.add_sample(sample=extracted_sample, id=id)
        return dataset

    def get_tabular_from_homogeneous_identifiers(
        self,
        feature_identifiers: list[dict[str, Union[str, float]]],
    ) -> Array:
        """Extract features of the dataset by their identifier(s) and return an array containing these features.

        Features must have identic sizes to be casted in an array. The first dimension of the array is the number of samples in the dataset.
        This method applies updates to scalars, time series, fields, or nodes using feature identifiers.

        Args:
            feature_identifiers (list of dict): Feature identifiers.

        Returns:
            Array: An containing the provided feature identifiers, size (nb_sample, nb_features, dim_features)

        Notes:
            Not working with time_series for the moment (time series have 2 elements: time_sequence and values)

        Raises:
            AssertionError: If feature sizes are inconsistent.
        """
        features = self.get_features_from_identifiers(feature_identifiers)
        dim_features = check_features_size_homogeneity(feature_identifiers, features)

        tabular = np.stack(list(features.values()))
        if dim_features == 0:
            tabular = np.expand_dims(tabular, axis=-1)
        assert tabular.ndim == 3, (
            "tabular must be constructed to have 3 dimensions: (nb_sample, nb_features, dim_features)"
        )

        return tabular

    def get_tabular_from_stacked_identifiers(
        self,
        feature_identifiers: list[dict[str, Union[str, float]]],
    ) -> Array:
        """Extract features of the dataset by their identifier(s), stack them and return an array containing these features.

        After stacking, each sample has one feature of dimension dim_stacked_features

        Args:
            feature_identifiers (list of dict): Feature identifiers.

        Returns:
            Array: An containing the provided feature identifiers, size (nb_sample, dim_stacked_features)
        """
        features = self.get_features_from_identifiers(feature_identifiers)

        tabular = []
        for local_feats in features.values():
            tabular.append(np.hstack([np.atleast_1d(e) for e in local_feats]))
        tabular = np.stack(tabular)

        return tabular

    def from_tabular(
        self,
        tabular: Array,
        feature_identifiers: Union[
            dict[str, Union[str, float]], list[dict[str, Union[str, float]]]
        ],
        restrict_to_features=True,
    ) -> Self:
        """Generates a dataset from tabular data and feature_identifiers.

        Parameters:
            tabular (Array): of size (nb_sample, nb_features) or (nb_sample, nb_features, dim_feature) if dim_feature>1
            feature_identifiers (dict or list of dict): One or more feature identifiers.
            extract_features (bool, optional): If True, only returns the features from feature identifiers, otherwise keep the other features as well

        Returns:
            Self
                A new dataset defined from tabular data and feature_identifiers.

        Raises:
            AssertionError
                If the number of rows in `tabular` does not match the number of samples in the dataset,
                or if the number of feature identifiers does not match the number of columns in `tabular`.
        """
        if isinstance(feature_identifiers, dict):
            feature_identifiers = [feature_identifiers]

        assert tabular.shape[0] == len(self)
        # assert tabular.shape[1] == len(feature_identifiers)

        features = {id: tabular[i] for i, id in enumerate(self.get_sample_ids())}

        if restrict_to_features:
            dataset = self.from_features_identifier(feature_identifiers)
            dataset.update_features_from_identifier(
                feature_identifiers=feature_identifiers,
                features=features,
                in_place=True,
            )
        else:
            dataset = self.update_features_from_identifier(
                feature_identifiers=feature_identifiers,
                features=features,
                in_place=False,
            )

        return dataset

    # -------------------------------------------------------------------------#
    def add_info(self, cat_key: str, info_key: str, info: str) -> None:
        """Add information to the :class:`Dataset <plaid.containers.dataset.Dataset>`, overwriting existing information if there's a conflict.

        Args:
            cat_key (str): Category key, choose among "legal," "data_production," and "data_description".
            info_key (str): Information key, depending on the chosen category key, choose among "owner", "license", "type", "physics", "simulator", "hardware", "computation_duration", "script", "contact", "location", "number_of_samples", "number_of_splits", "DOE", "inputs" and "outputs".
            info (str): Information content.

        Raises:
            KeyError: Invalid category key.
            KeyError: Invalid info key.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                infos = {"legal":{"owner":"CompX", "license":"li_X"}}
                dataset.set_infos(infos)
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}}
                dataset.add_info("data_production", "type", "simulation")
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}, 'data_production': {'type': 'simulation'}}

        """
        if cat_key not in AUTHORIZED_INFO_KEYS:
            raise KeyError(
                f"{cat_key=} not among authorized keys. Maybe you want to try among these keys {list(AUTHORIZED_INFO_KEYS.keys())}"
            )
        if info_key not in AUTHORIZED_INFO_KEYS[cat_key]:
            raise KeyError(
                f"{info_key=} not among authorized keys. Maybe you want to try among these keys {AUTHORIZED_INFO_KEYS[cat_key]}"
            )

        if cat_key not in self._infos:
            self._infos[cat_key] = {}
        elif info_key in self._infos[cat_key]:
            logger.warning(
                f"{cat_key=} and {info_key=} already set, replacing it anyway"
            )
        self._infos[cat_key][info_key] = info

    def add_infos(self, cat_key: str, infos: dict[str, str]) -> None:
        """Add information to the :class:`Dataset <plaid.containers.dataset.Dataset>`, overwriting existing information if there's a conflict.

        Args:
            cat_key (str): Category key, choose among "legal," "data_production," and "data_description".
            infos (str): Information key with its related content.

        Raises:
            KeyError: Invalid category key.
            KeyError: Invalid info key.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                infos = {"legal":{"owner":"CompX", "license":"li_X"}}
                dataset.set_infos(infos)
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}}
                new_info = {"type":"simulation", "simulator":"Z-set"}
                dataset.add_infos("data_production", new_info)
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}, 'data_production': {'type': 'simulation', 'simulator': 'Z-set'}}

        """
        if cat_key not in AUTHORIZED_INFO_KEYS:  # Format checking on "infos"
            raise KeyError(
                f"{cat_key=} not among authorized keys. Maybe you want to try among these keys {list(AUTHORIZED_INFO_KEYS.keys())}"
            )
        for info_key in infos.keys():
            if info_key not in AUTHORIZED_INFO_KEYS[cat_key]:
                raise KeyError(
                    f"{info_key=} not among authorized keys. Maybe you want to try among these keys {AUTHORIZED_INFO_KEYS[cat_key]}"
                )

        if cat_key not in self._infos:
            self._infos[cat_key] = {}
        elif info_key in self._infos[cat_key]:
            logger.warning(
                f"{cat_key=} and {info_key=} already set, replacing it anyway"
            )

        for key, value in infos.items():
            self._infos[cat_key][key] = value

    def set_infos(self, infos: dict[str, dict[str, str]]) -> None:
        """Set information to the :class:`Dataset <plaid.containers.dataset.Dataset>`, overwriting the existing one.

        Args:
            infos (dict[str,dict[str,str]]): Information to associate with this data set (Dataset).

        Raises:
            KeyError: Invalid category key format in provided infos.
            KeyError: Invalid info key format in provided infos.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                infos = {"legal":{"owner":"CompX", "license":"li_X"}}
                dataset.set_infos(infos)
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}}
        """
        for cat_key in infos.keys():  # Format checking on "infos"
            if cat_key not in AUTHORIZED_INFO_KEYS:
                raise KeyError(
                    f"{cat_key=} not among authorized keys. Maybe you want to try among these keys {list(AUTHORIZED_INFO_KEYS.keys())}"
                )
            for info_key in infos[cat_key].keys():
                if info_key not in AUTHORIZED_INFO_KEYS[cat_key]:
                    raise KeyError(
                        f"{info_key=} not among authorized keys. Maybe you want to try among these keys {AUTHORIZED_INFO_KEYS[cat_key]}"
                    )

        if len(self._infos) > 0:
            logger.warning("infos not empty, replacing it anyway")
        self._infos = infos

    def get_infos(self) -> dict[str, dict[str, str]]:
        """Get information from an instance of :class:`Dataset <plaid.containers.dataset.Dataset>`.

        Returns:
            dict[str,dict[str,str]]: Information associated with this data set (Dataset).

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                infos = {"legal":{"owner":"CompX", "license":"li_X"}}
                dataset.set_infos(infos)
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}}
        """
        return self._infos

    def print_infos(self) -> None:
        """Prints information in a readable format (pretty print)."""
        infos_cats = list(self._infos.keys())
        tf = "*********************** \x1b[34;1mdataset infos\x1b[0m **********************\n"
        for cat in infos_cats:
            tf += "\x1b[33;1m" + str(cat) + "\x1b[0m\n"
            infos = list(self._infos[cat].keys())
            for info in infos:
                tf += (
                    "  \x1b[32;1m"
                    + str(cat)
                    + "\x1b[0m:"
                    + str(self._infos[cat][info])
                    + "\n"
                )
        tf += "************************************************************\n"
        print(tf)

    # -------------------------------------------------------------------------#
    def merge_dataset(self, dataset: Self) -> list[int]:
        """Merges samples of another dataset into this one.

        Args:
            dataset (Dataset): The data set to be merged into this one (self).
            in_place (bool, option): If True, modifies the current dataset in place.

        Returns:
            list[int]: ids of added :class:`Samples <plaid.containers.sample.Sample>` from input :class:`Dataset <plaid.containers.dataset.Dataset>`.

        Raises:
            ValueError: If the provided dataset value is not an instance of Dataset
        """
        if dataset is None:
            return
        if not isinstance(dataset, Dataset):
            raise ValueError("dataset must be an instance of Dataset")
        return self.add_samples(dataset.get_samples(as_list=True))

    def merge_features(self, dataset: Self, in_place: bool = False) -> Self:
        """Merge features of another dataset into this one.

        Args:
            dataset (Dataset): The dataset to be merged into this one (self).
            in_place (bool, option): If True, modifies the current dataset in place.
                If False, returns a deep copy with the merged features.

        Returns:
            Dataset: A dataset containing all samples from the input datasets.
        """
        if dataset is None:
            return
        if not isinstance(dataset, Dataset):
            raise ValueError("dataset must be an instance of Dataset")

        merged_dataset = self if in_place else self.copy()
        assert dataset.get_sample_ids() == merged_dataset.get_sample_ids(), (
            "Cannot merge features of datasets with different sample ids. "
        )
        for id in dataset.get_sample_ids():
            merged_sample = merged_dataset[id].merge_features(
                dataset[id], in_place=in_place
            )
            merged_dataset.set_sample(
                id=id, sample=merged_sample, warning_overwrite=False
            )

        return merged_dataset

    @classmethod
    def merge_dataset_by_features(cls, datasets_list: list) -> Self:
        """Merge features a list of datasets.

        Args:
            datasets_list (list(Dataset)): The list of datasets to be merged.

        Returns:
            Dataset: A new dataset containing all samples from the input datasets.
        """
        assert len(datasets_list) > 1, "Provide more than one dataset"
        merged_dataset = datasets_list[0]
        for dataset in datasets_list[1:]:
            merged_dataset = merged_dataset.merge_features(dataset, in_place=False)
        return merged_dataset

    def save(self, fname: Union[str, Path]) -> None:
        """Saves the data set to a TAR (Tape Archive) file.

        It creates a temporary intermediate directory to store temporary files during the loading process.

        Args:
            fname (Union[str,Path]): The path to which the data set will be saved.

        Raises:
            ValueError: If the randomly generated temporary dir name is already used (extremely unlikely!).
        """
        fname = Path(fname)

        # First : creates a directory <savedir> to save everything in an
        # arborescence on disk
        savedir = fname.parent / f"tmpsavedir_{generate_random_ASCII()}"
        if savedir.is_dir():  # pragma: no cover
            raise ValueError(
                f"temporary intermediate directory <{savedir}> already exits"
            )
        savedir.mkdir(parents=True)

        self._save_to_dir_(savedir)

        # Then : tar dir in file <fname>
        # TODO: avoid using subprocess by using lib tarfile
        ARGUMENTS = ["tar", "-cf", fname, "-C", savedir, "."]
        subprocess.call(ARGUMENTS)

        # Finally : removes directory <savedir>
        shutil.rmtree(savedir)

    @classmethod
    def from_list_of_samples(cls, list_of_samples: list[Sample]) -> Self:
        """Initialise a dataset from a list of samples.

        Args:
            list_of_samples (list[Sample]): The list of samples.

        Returns:
            Self: The intialized dataset (Dataset).
        """
        instance = cls()
        instance.add_samples(list_of_samples)
        return instance

    @classmethod
    def load_from_file(
        cls, fname: Union[str, Path], verbose: bool = False, processes_number: int = 0
    ) -> Self:
        """Load data from a specified TAR (Tape Archive) file.

        Args:
            fname (Union[str,Path]): The path to the data file to be loaded.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.

        Returns:
            Self: The loaded dataset (Dataset).
        """
        fname = Path(fname)
        instance = cls()
        instance.load(fname, verbose, processes_number)
        return instance

    @classmethod
    def load_from_dir(
        cls,
        dname: Union[str, Path],
        ids: list[int] = None,
        verbose: bool = False,
        processes_number: int = 0,
    ) -> Self:
        """Load data from a specified directory.

        Args:
            dname (Union[str,Path]): The path from which to load files.
            ids (list, optional): The specific sample IDs to load from the dataset. Defaults to None.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.

        Returns:
            Self: The loaded dataset (Dataset).
        """
        dname = Path(dname)
        instance = cls()
        instance._load_from_dir_(
            dname, ids=ids, verbose=verbose, processes_number=processes_number
        )
        return instance

    def load(
        self, fname: Union[str, Path], verbose: bool = False, processes_number: int = 0
    ) -> None:
        """Load data from a specified TAR (Tape Archive) file.

        It creates a temporary intermediate directory to store temporary files during the loading process.

        Args:
            fname (Union[str,Path]): The path to the data file to be loaded.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.

        Raises:
            ValueError: If a randomly generated temporary directory already exists,
            indicating a potential conflict during the loading process (extremely unlikely).
        """
        fname = Path(fname)

        inputdir = fname.parent / f"tmploaddir_{generate_random_ASCII()}"
        if inputdir.is_dir():  # pragma: no cover
            raise ValueError(
                f"temporary intermediate directory <{inputdir}> already exits"
            )
        inputdir.mkdir(parents=True)

        # First : untar file <fname> to a directory <inputdir>
        # TODO: avoid using subprocess by using a lib tarfile
        arguments = ["tar", "-xf", fname, "-C", inputdir]
        subprocess.call(arguments)

        # Then : load data from directory <inputdir>
        self._load_from_dir_(
            inputdir, verbose=verbose, processes_number=processes_number
        )

        # Finally : removes directory <inputdir>
        shutil.rmtree(inputdir)

    # -------------------------------------------------------------------------#
    def _save_to_dir_(self, savedir: Union[str, Path], verbose: bool = False) -> None:
        """Saves the dataset into a created sample directory and creates an 'infos.yaml' file to store additional information about the dataset.

        Args:
            savedir (Union[str,Path]): The path in which to save the files.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
        """
        savedir = Path(savedir)
        if not (savedir.is_dir()):
            savedir.mkdir(parents=True)

        if verbose:  # pragma: no cover
            print(f"Saving database to: {savedir}")

        samples_dir = savedir / "samples"
        if not (samples_dir.is_dir()):
            samples_dir.mkdir(parents=True)

        # ---# save samples
        for i_sample, sample in tqdm(self._samples.items(), disable=not (verbose)):
            sample_fname = samples_dir / f"sample_{i_sample:09d}"
            sample.save(sample_fname)

        # ---# save infos
        if len(self._infos) > 0:
            infos_fname = savedir / "infos.yaml"
            with open(infos_fname, "w") as file:
                yaml.dump(self._infos, file, default_flow_style=False, sort_keys=False)

        # #---# save stats
        # stats_fname = savedir / 'stats.yaml'
        # self._stats.save(stats_fname)

        # #---# save flags
        # flags_fname = savedir / 'flags.yaml'
        # self._flags.save(flags_fname)

    def _load_from_dir_(
        self,
        savedir: Union[str, Path],
        ids: list[int] = None,
        verbose: bool = False,
        processes_number: int = 0,
    ) -> None:
        """Loads a dataset from a sample directory and retrieves additional information about the dataset from an 'infos.yaml' file, if available.

        Args:
            savedir (Union[str,Path]): The path from which to load files.
            ids (list, optional): The specific sample IDs to load from the dataset. Defaults to None.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.

        Raises:
            FileNotFoundError: Triggered if the provided directory does not exist.
            FileExistsError: Triggered if the provided path is a file instead of a directory.
            ValueError: Triggered if the number of processes is < -1.
        """
        savedir = Path(savedir)
        if not savedir.is_dir():
            raise FileNotFoundError(
                f'"{savedir}" is not a directory or does not exist. Abort'
            )

        if processes_number < -1:
            raise ValueError("Number of processes cannot be < -1")

        if verbose:  # pragma: no cover
            print(f"Reading database located at: {savedir}")

        sample_paths = sorted(
            [path for path in (savedir / "samples").glob("sample_*") if path.is_dir()]
        )

        if ids is not None:
            filtered_sample_paths = []
            for sample_path in sample_paths:
                id = int(sample_path.stem.split("_")[-1])
                if id in ids:
                    filtered_sample_paths.append(sample_path)
            sample_paths = filtered_sample_paths

            if len(sample_paths) != len(set(ids)):  # pragma: no cover
                raise ValueError(
                    "The length of the list of samples to add and the list of IDs are different"
                )

        if processes_number == -1:
            logger.info(
                f"Number of processes set to maximum available: {os.cpu_count()}"
            )
            processes_number = os.cpu_count()

        if processes_number == 0 or processes_number == 1:
            for sample_path in tqdm(sample_paths, disable=not (verbose)):
                id = int(sample_path.stem.split("_")[-1])
                sample = Sample(sample_path)
                self.add_sample(sample, id)
        else:
            with Pool(processes_number) as p:
                for id, sample in list(
                    tqdm(
                        p.imap(process_sample, sample_paths),
                        total=len(sample_paths),
                        disable=not (verbose),
                    )
                ):
                    self.set_sample(id, sample)

            """
            samples_pool = Pool(processes_number)
            pbar = tqdm(total=len(sample_paths), disable=not (verbose))

            def update(self, *a):
                pbar.update()

            samples = [
                samples_pool.apply_async(
                    process_sample,
                    args=(
                        sample_paths[i],
                        i),
                    callback=update) for i in range(
                    len(sample_paths))]

            samples_pool.close()
            samples_pool.join()

            for s in samples:
                id, sample = s.get()
                self.set_sample(id, sample)
            """

        infos_fname = savedir / "infos.yaml"
        if os.path.isfile(infos_fname):
            with open(infos_fname, "r") as file:
                self._infos = yaml.safe_load(file)

        if len(self) == 0:  # pragma: no cover
            print("Warning: dataset contains no sample")

    @staticmethod
    def _load_number_of_samples_(_savedir: Union[str, Path]) -> int:  # pragma: no cover
        """Warning: This method is deprecated, use instead :meth:`plaid.get_number_of_samples <plaid.containers.utils.get_number_of_samples>`.

        This function counts the number of sample files in a specified directory, which is
        useful for determining the total number of samples in a dataset.

        Args:
            savedir (Union[str,Path]): The path to the directory where sample files are stored.

        Returns:
            int: The number of sample files found in the specified directory.
        """
        raise DeprecatedError(
            'use instead: plaid.get_number_of_samples("path-to-my-dataset")'
        )

    # -------------------------------------------------------------------------#
    def set_samples(self, samples: dict[int, Sample]) -> None:
        """Set the samples of the data set, overwriting the existing ones.

        Args:
            samples (dict[int,Sample]): A dictionary of samples to set inside the dataset.

        Raises:
            TypeError: If the 'samples' parameter is not of type dict[int, Sample].
            TypeError: If the 'id' inside a sample is not of type int.
            ValueError: If the 'id' inside a sample is negative (id >= 0 is required).
            TypeError: If the values inside the 'samples' dictionary are not of type Sample.
        """
        if not (isinstance(samples, dict)):
            raise TypeError(
                f"samples should be of type dict[int,Sample] but is {type(samples)=}"
            )

        ids = list(samples.keys())
        for id in ids:
            if not (isinstance(id, int)):
                raise TypeError(f"id should be of type {int.__class__} but {type(id)=}")
            if not (id >= 0):
                raise ValueError(f"id should be positive (id>=0) but {id=}")
            if not (isinstance(samples[id], Sample)):
                raise TypeError(
                    f"samples[{id=}] should be of type {Sample.__class__} but {type(samples[id])=}"
                )

        if len(self._samples) > 0:
            logger.warning(
                f"{len(self._samples)} samples are already present in dataset, replacing them anyway"
            )
        self._samples = samples

    # TODO: on veut vraiment faire ça ?
    # laisser l’utilisateur faire joujou avec les id des samples ?
    # le laisser placer des samples n’importe où ?
    #  - avec des ids potentiellement négatifs,
    #  - potentiellement loin après le dernier id déjà présent...
    def set_sample(self, id: int, sample: Sample, warning_overwrite=True) -> None:
        """Set a :class:`sample` with :code:`id` in the Dataset, overwriting existing samples if there's a conflict.

        Args:
            id (int): The choosen id of the sample.
            sample (Sample): The sample to set inside the dataset.
            warning_overwrite (bool, optional): Show warning if an preexisting field is being overwritten

        Raises:
            TypeError: If the 'id' inside the sample is not of type int.
            ValueError: If the 'id' inside a sample is negative (id >= 0 is required).
            TypeError: If 'sample' parameter is not of type Sample.

        Caution:
            In case of conflict, the existing samples will be overwritten.
        """
        if not (isinstance(id, int)):
            raise TypeError(f"id should be of type {int.__class__} but {type(id)=}")
        if not (id >= 0):
            raise ValueError(f"id should be positive (id>=0) but {id=}")
        if not (isinstance(sample, Sample)):
            raise TypeError(
                f"sample should be of type {Sample.__class__} but {type(sample)=}"
            )

        if warning_overwrite:
            if id in self._samples:
                logger.warning(
                    f"sample with {id=} already present in dataset, replacing it anyway"
                )
        self._samples[id] = sample

    # -------------------------------------------------------------------------#
    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                len(dataset)
                >>> 10  # Assuming there are 10 samples in the dataset
        """
        return len(self._samples)

    def __iter__(self) -> Iterator[Sample]:
        """Iterate over the samples in the dataset.

        Yields:
            Iterator[Sample]: An iterator over the Sample objects stored in the dataset.

        Example:
            >>> for sample in dataset:
            ...     process(sample)

        Notes:
            The samples are yielded in ascending order of their IDs.
            Only samples that have been explicitly added to the dataset are returned.
        """
        return (self._samples[k] for k in sorted(list(self._samples.keys())))

    def __getitem__(
        self, id: Union[int, slice, range, list[int], np.ndarray]
    ) -> Union[Sample, Self]:
        """Retrieve a specific sample by its ID int this dataset.

        Args:
            id (Union[int, slice, list[int], np.ndarray]): The ID(s) of the sample to retrieve.

        Raises:
            IndexError: If the provided ID is out of bounds or does not exist in the dataset.

        Returns:
            Union[Sample, Dataset]: The sample with the specified ID or a dataset in the specified IDs.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                sample = dataset[3]  # Retrieve the sample with ID 3

        Seealso:
            This function can also be called using `__call__()`.
        """
        if isinstance(id, (slice, range, list, np.ndarray)):
            if isinstance(id, slice):
                id = list(range(*id.indices(len(self))))
            dataset = Dataset()
            for i in id:
                dataset.add_sample(self[int(i)], int(i))
            dataset.set_infos(self.get_infos())
            return dataset
        else:
            if id in self._samples:
                return self._samples[id]
            else:
                raise IndexError(
                    f"sample with {id=} not set -> use 'Dataset.add_sample' or 'Dataset.add_samples'"
                )

    __call__ = __getitem__

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            str: A string representation of the overview of dataset content.

        Example:
            .. code-block:: python

                from plaid.containers.dataset import Dataset
                dataset = Dataset()
                print(dataset)
                >>> Dataset(0 samples, 0 scalars, 0 fields)
        """
        str_repr = "Dataset("

        # samples
        nb_samples = len(self._samples)
        str_repr += f"{nb_samples} sample{'' if nb_samples == 1 else 's'}, "

        # scalars
        nb_scalars = len(self.get_scalar_names())
        str_repr += f"{nb_scalars} scalar{'' if nb_scalars == 1 else 's'}, "

        # time series
        nb_time_series = len(self.get_time_series_names())
        str_repr += f"{nb_time_series} time_series, "

        # fields
        nb_fields = len(self.get_field_names())
        str_repr += f"{nb_fields} field{'' if nb_fields == 1 else 's'}, "

        if str_repr[-2:] == ", ":
            str_repr = str_repr[:-2]
        str_repr = str_repr + ")"
        return str_repr
