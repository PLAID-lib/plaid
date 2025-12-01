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
from typing import Iterator, Literal, Optional, Union

import numpy as np
import yaml
from packaging.version import Version
from tqdm import tqdm

import plaid
from plaid.constants import AUTHORIZED_INFO_KEYS, CGNS_FIELD_LOCATIONS
from plaid.containers.feature_identifier import FeatureIdentifier
from plaid.containers.sample import Sample
from plaid.containers.utils import check_features_size_homogeneity
from plaid.types import Array, Feature
from plaid.utils.base import DeprecatedError, ShapeError, generate_random_ASCII
from plaid.utils.deprecation import deprecated, deprecated_argument

logger = logging.getLogger(__name__)


# %% Functions


def _process_sample(path: Union[str, Path]) -> tuple:  # pragma: no cover
    """Load Sample from path.

    Args:
        path (Union[str, Path]): The path to the Sample.

    Returns:
        tuple: The loaded Sample and its ID.
    """
    path = Path(path)
    id = int(path.stem.split("_")[-1])
    return id, Sample(path=path)


# %% Classes


class Dataset(object):
    """A set of samples, and optionnaly some other informations about the Dataset."""

    @deprecated_argument("directory_path", "path", version="0.1.8", removal="0.2.0")
    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
        processes_number: int = 0,
        samples: Optional[list[Sample]] = None,
        sample_ids: Optional[list[int]] = None,
    ) -> None:
        """Initialize a :class:`Dataset <plaid.containers.dataset.Dataset>`.

        If `path` is not specified it initializes an empty :class:`Dataset <plaid.containers.dataset.Dataset>` that should be fed with :class:`Samples <plaid.containers.sample.Sample>`.

        Use :meth:`add_sample <plaid.containers.dataset.Dataset.add_sample>` or :meth:`add_samples <plaid.containers.dataset.Dataset.add_samples>` to feed the :class:`Dataset`

        Args:
            path (Union[str, Path], optional): The path from which to load PLAID dataset files.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.
            samples (list[Sample], optional): A list of :class:`Samples <plaid.containers.sample.Sample>` to initialize the :class:`Dataset <plaid.containers.dataset.Dataset>`. Defaults to None.
            sample_ids (list[int], optional): An optional list of IDs for the new samples. If not provided, the IDs will be automatically generated based on the current number of samples in the dataset.

        Example:
            .. code-block:: python

                from plaid import Dataset
                from plaid import Sample

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
                >>> Sample(1 scalar, 1 timestamp, 2 fields)
                    Sample(1 scalar, 0 timestamps, 0 fields)
                    Sample(2 scalars, 1 timestamp, 2 fields)

                # 3. Create Dataset instance from a list of Samples
                dataset = Dataset(samples=[sample1, sample2, sample3])
                print(dataset)
                >>> Dataset(3 samples, 0 scalars, 2 fields)

                # 4. Create Dataset instance from a list of Samples with specific ids
                dataset = Dataset(samples=[sample1, sample2, sample3], sample_ids=[3, 5, 7])
                print(dataset)
                >>> Dataset(3 samples, 0 scalars, 2 fields)


        Caution:
            It is assumed that you provided a compatible PLAID dataset.
        """
        self._samples: dict[int, Sample] = {}  # sample_id -> sample
        # info_name -> description
        self._infos: dict[str, dict[str, Union[str, Version]]] = {
            "plaid": {"version": Version(plaid.__version__)}
        }

        if samples is not None and (path is not None):
            raise ValueError("'samples' and 'path' are mutually exclusive")

        if path is not None:
            path = Path(path)

            if path.suffix == ".plaid":
                self.load(path, verbose=verbose, processes_number=processes_number)
            else:
                self._load_from_dir_(
                    path, verbose=verbose, processes_number=processes_number
                )
        elif samples is not None:
            if sample_ids is None:
                self.add_samples(samples)
            else:
                self.add_samples(samples, sample_ids)

    def copy(self) -> Self:
        """Create a deep copy of the dataset.

        Returns:
            A new `Dataset` instance with all internal data (samples, infos)
            deeply copied to ensure full isolation from the original.

        Note:
            This operation may be memory-intensive for large datasets.
        """
        return copy.deepcopy(self)

    # -------------------------------------------------------------------------#
    def get_samples(
        self, ids: Optional[list[int]] = None, as_list: bool = False
    ) -> Union[list[Sample], dict[int, Sample]]:
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

    def add_sample(self, sample: Sample, id: Optional[int] = None) -> int:
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

                from plaid import Dataset
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

                from plaid import Dataset
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

    def add_samples(
        self, samples: list[Sample], ids: Optional[list[int]] = None
    ) -> list[int]:
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

                from plaid import Dataset
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

                from plaid import Dataset
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
    def get_scalar_names(self, ids: Optional[list[int]] = None) -> list[str]:
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
    def get_field_names(
        self,
        ids: Optional[list[int]] = None,
        location: Optional[str] = None,
        zone_name: Optional[str] = None,
        base_name: Optional[str] = None,
        time: Optional[float] = None,
    ) -> list[str]:
        """Return union of fields names in all samples with id in ids.

        Args:
            ids (list[int], optional): Select fields depending on sample id. If None, take all samples. Defaults to None.
            location (str, optional): If provided, only field names from this location will be included. Defaults to None.
            zone_name (str, optional): If provided, only field names from this zone will be included. Defaults to None.
            base_name (str, optional): If provided, only field names containing this base name will be included. Defaults to None.
            time (float, optional): If provided, only field names from this time will be included. Defaults to None.

        Returns:
            list[str]: List of all fields names.
        """
        if ids is not None and len(set(ids)) != len(ids):  # pragma: no cover
            logger.warning("Provided ids are not unique")

        fields_names = []
        for sample in self.get_samples(ids, as_list=True):
            times = [time] if time else sample.features.get_all_time_values()
            for time in times:
                base_names = (
                    [base_name]
                    if base_name
                    else sample.features.get_base_names(time=time)
                )
                for base_name in base_names:
                    zone_names = (
                        [zone_name]
                        if zone_name
                        else sample.features.get_zone_names(
                            time=time, base_name=base_name
                        )
                    )
                    for zone_name in zone_names:
                        locations = [location] if location else CGNS_FIELD_LOCATIONS
                        for location in locations:
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
        self, ids: Optional[list[int]] = None
    ) -> list[FeatureIdentifier]:
        """Get all features identifiers from the dataset.

        Args:
            ids (list[int], optional): Sample id from which returning feature identifiers. If None, take all samples. Defaults to None.

        Returns:
            list[FeatureIdentifier]: A list of dictionaries containing the identifiers of all features in the dataset.
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
        self,
        feature_type: Literal["scalar", "nodes", "field"],
        ids: list[int] = None,
    ) -> list[FeatureIdentifier]:
        """Get all features identifiers from the dataset.

        Args:
            feature_type (str): Type of features to return
            ids (list[int], optional): Sample id from which returning feature identifiers. If None, take all samples. Defaults to None.

        Returns:
            list[FeatureIdentifier]: A list of dictionaries containing the identifiers of all features of a given type  in the dataset.
        """
        all_features_identifiers = self.get_all_features_identifiers(ids)
        return [
            feat_id
            for feat_id in all_features_identifiers
            if feat_id["type"] == feature_type
        ]

    # -------------------------------------------------------------------------#

    def add_tabular_scalars(
        self, tabular: np.ndarray, names: Optional[list[str]] = None
    ) -> None:
        """Add tabular scalar data to the summary.

        Args:
            tabular (np.ndarray): A 2D NumPy array containing tabular scalar data.
            names (list[str], optional): A list of column names for the tabular data. Defaults to None.

        Raises:
            ShapeError: Raised if the input tabular array does not have the correct shape (2D).
            ShapeError: Raised if the number of columns in the tabular data does not match the number of names provided.

        Note:
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
        scalar_names: Optional[list[str]] = None,
        sample_ids: Optional[list[int]] = None,
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
            logger.warning(
                "Provided scalar names are not unique, this will lead to duplicate columns in output array"
            )

        if sample_ids is None:
            sample_ids = self.get_sample_ids()
        elif len(set(sample_ids)) != len(sample_ids):
            logger.warning(
                "Provided sample ids are not unique, this will lead to duplicate rows in output array"
            )
        nb_samples = len(sample_ids)

        named_tabular = {}
        for s_name in scalar_names:
            tmp = self[sample_ids[0]].get_scalar(s_name)
            res = np.empty(nb_samples)
            if isinstance(tmp, np.ndarray) and tmp.size > 1:
                assert tmp.ndim < 3
                res = np.empty((nb_samples, tmp.size))
            res.fill(None)
            for i_, id in enumerate(sample_ids):
                val = self[id].get_scalar(s_name)
                if val is not None:
                    res[i_] = val.reshape((-1,)) if isinstance(val, np.ndarray) else val
            named_tabular[s_name] = res

        if as_nparray:
            named_tabular = np.concatenate(
                [v.reshape((nb_samples, -1)) for v in named_tabular.values()], axis=1
            )
        return named_tabular

    # -------------------------------------------------------------------------#
    def get_feature_from_string_identifier(
        self, feature_string_identifier: str
    ) -> dict[int, Feature]:
        """Get a list of features from the dataset based on the provided feature string identifier.

        Args:
            feature_string_identifier (str): A string identifier for the feature.

        Returns:
            dict[int, Feature]: A list of features matching the provided string identifier.
        """
        return {
            id: self[id].get_feature_from_string_identifier(feature_string_identifier)
            for id in self.get_sample_ids()
        }

    def get_feature_from_identifier(
        self, feature_identifier: FeatureIdentifier
    ) -> dict[int, Feature]:
        """Get a list of features from the dataset based on the provided feature identifier.

        Args:
            feature_identifier (FeatureIdentifier): A dictionary containing the feature identifier.

        Returns:
            dict[int, Feature]: A list of features matching the provided identifier.
        """
        return {
            id: self[id].get_feature_from_identifier(feature_identifier)
            for id in self.get_sample_ids()
        }

    def get_features_from_identifiers(
        self, feature_identifiers: list[FeatureIdentifier]
    ) -> dict[int, list[Feature]]:
        """Get a list of features from the dataset based on the provided feature identifiers.

        Args:
            feature_identifiers (FeatureIdentifier): A dictionary containing the feature identifier.

        Returns:
            dict[int, list[Feature]]: A list of features matching the provided identifier.
        """
        return {
            id: self[id].get_features_from_identifiers(feature_identifiers)
            for id in self.get_sample_ids()
        }

    def update_features_from_identifier(
        self,
        feature_identifiers: Union[FeatureIdentifier, list[FeatureIdentifier]],
        features: dict[int, Union[Feature, list[Feature]]],
        in_place: bool = False,
    ) -> Self:
        """Update one or several features of the dataset by their identifier(s).

        This method applies updates to scalars, fields, or nodes
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

    def extract_dataset_from_identifier(
        self,
        feature_identifiers: Union[FeatureIdentifier, list[FeatureIdentifier]],
    ) -> Self:
        """Extract features of the dataset by their identifier(s) and return a new dataset containing these features.

        This method applies updates to scalars, fields, or nodes
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
            extracted_sample = self[id].extract_sample_from_identifier(
                feature_identifiers
            )
            dataset.add_sample(sample=extracted_sample, id=id)
        return dataset

    @deprecated(
        "`Dataset.from_features_identifier(...)` is deprecated, use instead `Dataset.extract_dataset_from_identifier(...)`",
        version="0.1.8",
        removal="0.2",
    )
    def from_features_identifier(
        self,
        feature_identifiers: Union[FeatureIdentifier, list[FeatureIdentifier]],
    ) -> Self:
        """DEPRECATED: Use :meth:`Dataset.extract_dataset_from_identifier` instead."""
        return self.extract_dataset_from_identifier(
            feature_identifiers
        )  # pragma: no cover

    def get_tabular_from_homogeneous_identifiers(
        self,
        feature_identifiers: list[FeatureIdentifier],
    ) -> Array:
        """Extract features of the dataset by their identifier(s) and return an array containing these features.

        Features must have identic sizes to be casted in an array. The first dimension of the array is the number of samples in the dataset.
        This method applies updates to scalars, fields, or nodes using feature identifiers.

        Args:
            feature_identifiers (list of dict): Feature identifiers.

        Returns:
            Array: An containing the provided feature identifiers, size (nb_sample, nb_features, dim_features)

        Raises:
            AssertionError: If feature sizes are inconsistent.
        """
        features = self.get_features_from_identifiers(feature_identifiers)
        dim_features = check_features_size_homogeneity(feature_identifiers, features)

        tabular = np.stack(list(features.values()))
        if dim_features == 0:
            tabular = np.expand_dims(tabular, axis=-1)
        assert tabular.ndim == 3, (
            f"tabular must be constructed to have 3 dimensions: (nb_sample, nb_features, dim_features), but {tabular.ndim=} | {tabular.shape=}"
        )

        return tabular

    def get_tabular_from_stacked_identifiers(
        self,
        feature_identifiers: list[FeatureIdentifier],
    ) -> tuple[Array, Array]:
        """Extract features of the dataset by their identifier(s), stack them and return an array containing these features.

        After stacking, each sample has one feature of dimension dim_stacked_features

        Args:
            feature_identifiers (list of dict): Feature identifiers.

        Returns:
            Array: An array containing the provided feature identifiers, size (nb_sample, dim_stacked_features)
            Array: An array containing the cumulated feature dimensions, starts with 0, size (len(feature_identifiers)+1, )
        """
        features = self.get_features_from_identifiers(feature_identifiers)

        tabular = []
        for local_feats in features.values():
            tabular.append(np.hstack([np.atleast_1d(e) for e in local_feats]))
        tabular = np.stack(tabular)

        feat_dims = [0]
        feat_dims.extend([len(np.atleast_1d(e)) for e in local_feats])
        cumulated_feat_dims = np.cumsum(feat_dims)

        return tabular, cumulated_feat_dims

    def add_features_from_tabular(
        self,
        tabular: Array,
        feature_identifiers: list[FeatureIdentifier],
        restrict_to_features: bool = True,
    ) -> Self:
        """Add or update features in the dataset from tabular data using feature identifiers.

        This method takes tabular data and applies it to the dataset, either by updating existing features
        or adding new ones based on the provided feature identifiers. The method can either:
        1. Extract only the specified features and return a new dataset with just those features (if restrict_to_features=True)
        2. Update the specified features in the current dataset while keeping all other existing features (if restrict_to_features=False)

        Parameters:
            tabular (Array): of size (nb_sample, nb_features) or (nb_sample, nb_features, dim_feature) if dim_feature>1
            feature_identifiers (list of dict): One or more feature identifiers specifying which features to update/add.
            restrict_to_features (bool, optional): If True, only returns the features from feature identifiers, otherwise keep the other features as well. Defaults to True.

        Returns:
            Self: A new dataset with features updated/added from the tabular data. If restrict_to_features=True,
                  contains only the specified features. If restrict_to_features=False, contains all original
                  features plus the updated/added ones.

        Raises:
            AssertionError
                If the number of rows in `tabular` does not match the number of samples in the dataset,
                or if the number of feature identifiers does not match the number of columns in `tabular`.
        """
        for i_id, feat_id in enumerate(feature_identifiers):
            feature_identifiers[i_id] = FeatureIdentifier(feat_id)

        assert tabular.shape[0] == len(self)
        # assert tabular.shape[1] == len(feature_identifiers)

        features = {id: tabular[i] for i, id in enumerate(self.get_sample_ids())}

        if restrict_to_features:
            dataset = self.extract_dataset_from_identifier(feature_identifiers)
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

    @deprecated(
        "`Dataset.from_tabular(...)` is deprecated, use instead `Dataset.add_features_from_tabular(...)`",
        version="0.1.8",
        removal="0.2",
    )
    def from_tabular(
        self,
        tabular: Array,
        feature_identifiers: Union[FeatureIdentifier, list[FeatureIdentifier]],
        restrict_to_features: bool = True,
    ) -> Self:
        """DEPRECATED: Use :meth:`Dataset.add_features_from_tabular` instead."""
        return self.add_features_from_tabular(
            tabular, feature_identifiers, restrict_to_features
        )  # pragma: no cover

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

                from plaid import Dataset
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

                from plaid import Dataset
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

    def set_infos(self, infos: dict[str, dict[str, str]], warn: bool = True) -> None:
        """Set information to the :class:`Dataset <plaid.containers.dataset.Dataset>`, overwriting the existing one.

        Args:
            infos (dict[str,dict[str,str]]): Information to associate with this data set (Dataset).
            warn (bool, optional): If True, warns when replacing existing infos. Defaults to True.

        Raises:
            KeyError: Invalid category key format in provided infos.
            KeyError: Invalid info key format in provided infos.

        Example:
            .. code-block:: python

                from plaid import Dataset
                dataset = Dataset()
                infos = {"legal":{"owner":"CompX", "license":"li_X"}}
                dataset.set_infos(infos)
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}}
        """
        for cat_key in infos.keys():  # Format checking on "infos"
            if cat_key != "plaid":
                if cat_key not in AUTHORIZED_INFO_KEYS:
                    raise KeyError(
                        f"{cat_key=} not among authorized keys. Maybe you want to try among these keys {list(AUTHORIZED_INFO_KEYS.keys())}"
                    )
                for info_key in infos[cat_key].keys():
                    if info_key not in AUTHORIZED_INFO_KEYS[cat_key]:
                        raise KeyError(
                            f"{info_key=} not among authorized keys. Maybe you want to try among these keys {AUTHORIZED_INFO_KEYS[cat_key]}"
                        )

        # Check if there are any non-plaid infos being replaced
        has_user_infos = any(key != "plaid" for key in self._infos.keys())
        if has_user_infos and warn:
            logger.warning("infos not empty, replacing it anyway")
        self._infos = copy.deepcopy(infos)

        if "plaid" not in self._infos:
            self._infos["plaid"] = {}
        if "version" not in self._infos["plaid"]:
            self._infos["plaid"]["version"] = Version(plaid.__version__)

    def get_infos(self) -> dict[str, dict[str, str]]:
        """Get information from an instance of :class:`Dataset <plaid.containers.dataset.Dataset>`.

        Returns:
            dict[str,dict[str,str]]: Information associated with this data set (Dataset).

        Example:
            .. code-block:: python

                from plaid import Dataset
                dataset = Dataset()
                infos = {"legal":{"owner":"CompX", "license":"li_X"}}
                dataset.set_infos(infos)
                print(dataset.get_infos())
                >>> {'legal': {'owner': 'CompX', 'license': 'li_X'}}
        """
        return {
            k: {kk: str(vv) for kk, vv in v.items()} for k, v in self._infos.items()
        }

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
    def merge_dataset_by_features(cls, datasets_list: list[Self]) -> Self:
        """Merge features a list of datasets.

        Args:
            datasets_list (list[Dataset]): The list of datasets to be merged.

        Returns:
            Dataset: A new dataset containing all samples from the input datasets.
        """
        if len(datasets_list) == 1:
            return datasets_list[0]

        merged_dataset = datasets_list[0]
        for dataset in datasets_list[1:]:
            merged_dataset = merged_dataset.merge_features(dataset, in_place=False)
        return merged_dataset

    @deprecated_argument("directory_path", "path", version="0.1.8", removal="0.2.0")
    @deprecated(
        "`Dataset.save(...)` is deprecated, use instead `Dataset.save_to_file(...)`",
        version="0.1.10",
        removal="0.2.0",
    )
    def save(self, path: Union[str, Path]) -> None:
        """DEPRECATED: use :meth:`Dataset.save_to_file` instead."""
        self.save_to_file(path)

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Saves the data set to a TAR (Tape Archive) file.

        It creates a temporary intermediate directory to store temporary files during the loading process.

        Args:
            path (Union[str, Path]): The path to which the data set will be saved.

        Raises:
            ValueError: If the randomly generated temporary dir name is already used (extremely unlikely!).
        """
        path = Path(path)

        # First : creates a directory <tmp_dir> to save everything in an
        # arborescence on disk
        tmp_dir = path.parent / f"tmpsavedir_{generate_random_ASCII()}"
        if tmp_dir.is_dir():  # pragma: no cover
            raise ValueError(
                f"temporary intermediate directory <{tmp_dir}> already exits"
            )
        tmp_dir.mkdir(parents=True)

        self.save_to_dir(tmp_dir)

        # Then : tar dir in file <path>
        # TODO: avoid using subprocess by using lib tarfile
        ARGUMENTS = ["tar", "-cf", path, "-C", tmp_dir, "."]
        subprocess.call(ARGUMENTS)

        # Finally : removes directory <tmp_dir>
        shutil.rmtree(tmp_dir)

    def save_to_dir(self, path: Union[str, Path], verbose: bool = False) -> None:
        """Saves the dataset into a sub-directory `samples` and creates an 'infos.yaml' file to store additional information about the dataset.

        Args:
            path (Union[str, Path]): The path in which to save the files.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
        """
        path = Path(path)
        if not (path.is_dir()):
            path.mkdir(parents=True)

        self.path = path

        if verbose:  # pragma: no cover
            print(f"Saving database to: {path}")

        # Save infos
        assert "plaid" in self._infos, f"{self._infos.keys()=} should contain 'plaid'"
        assert "version" in self._infos["plaid"], (
            f"{self._infos['plaid'].keys()=} should contain 'version'"
        )
        plaid_version = Version(plaid.__version__)
        if self._infos["plaid"]["version"] != plaid_version:  # pragma: no cover
            logger.warning(
                f"Version mismatch: Dataset was loaded from version {self._infos['plaid']['version'] if self._infos['plaid']['version'] is not None else 'anterior to 0.1.10'}, and will be saved with version: {plaid_version}"
            )
        self._infos["plaid"]["version"] = str(plaid_version)
        infos_fname = path / "infos.yaml"
        with open(infos_fname, "w") as file:
            yaml.dump(self._infos, file, default_flow_style=False, sort_keys=False)

        # Save samples
        samples_dir = path / "samples"
        if not (samples_dir.is_dir()):
            samples_dir.mkdir(parents=True)

        for i_sample, sample in tqdm(self._samples.items(), disable=not (verbose)):
            sample_dname = samples_dir / f"sample_{i_sample:09d}"
            sample.save_to_dir(sample_dname)

    def summarize_features(self) -> str:
        """Show the name of each feature and the number of samples containing it.

        Returns:
            str: A summary of features across the dataset.

        Example:
            .. code-block:: bash

                Dataset Feature Summary:
                ==================================================
                Scalars (8 unique):
                - Pr: 30/32 samples (93.8%)
                - Q: 30/32 samples (93.8%)
                - Tr: 30/32 samples (93.8%)
                - angle_in: 32/32 samples (100.0%)
                - angle_out: 30/32 samples (93.8%)
                - eth_is: 30/32 samples (93.8%)
                - mach_out: 32/32 samples (100.0%)
                - power: 30/32 samples (93.8%)

                Fields (8 unique):
                - M_iso: 30/32 samples (93.8%)
                - mach: 30/32 samples (93.8%)
                - nut: 30/32 samples (93.8%)
                - ro: 30/32 samples (93.8%)
                - roe: 30/32 samples (93.8%)
                - rou: 30/32 samples (93.8%)
                - rov: 30/32 samples (93.8%)
                - sdf: 32/32 samples (100.0%)
        """
        summary = "Dataset Feature Summary:\n"
        summary += "=" * 50 + "\n"

        if len(self._samples) == 0:
            return summary + "No samples in dataset.\n"

        # Collect all feature names across all samples
        all_scalar_names = set()
        all_field_names = set()

        # Count occurrences of each feature
        scalar_counts = {}
        field_counts = {}

        for _, sample in self._samples.items():
            # Scalars
            scalar_names = sample.get_scalar_names()
            all_scalar_names.update(scalar_names)
            for name in scalar_names:
                scalar_counts[name] = scalar_counts.get(name, 0) + 1

            # Fields
            times = sample.features.get_all_time_values()
            for time in times:
                base_names = sample.features.get_base_names(time=time)
                for base_name in base_names:
                    zone_names = sample.features.get_zone_names(
                        base_name=base_name, time=time
                    )
                    for zone_name in zone_names:
                        field_names = sample.get_field_names(
                            zone_name=zone_name, base_name=base_name, time=time
                        )
                        all_field_names.update(field_names)
                        for name in field_names:
                            field_counts[name] = field_counts.get(name, 0) + 1

        total_samples = len(self._samples)

        # Scalars summary
        summary += f"Scalars ({len(all_scalar_names)} unique):\n"
        if all_scalar_names:
            for name in sorted(all_scalar_names):
                count = scalar_counts.get(name, 0)
                summary += f"  - {name}: {count}/{total_samples} samples ({count / total_samples * 100:.1f}%)\n"
        else:
            summary += "  None\n"
        summary += "\n"

        # Fields summary
        summary += f"Fields ({len(all_field_names)} unique):\n"
        if all_field_names:
            for name in sorted(all_field_names):
                count = field_counts.get(name, 0)
                summary += f"  - {name}: {count}/{total_samples} samples ({count / total_samples * 100:.1f}%)\n"
        else:
            summary += "  None\n"

        return summary

    def check_feature_completeness(self) -> str:
        """Detect and notify if some Samples don't contain all features.

        Returns:
            str: A report on feature completeness across the dataset.

        Example:
            .. code-block:: bash

                Dataset Feature Completeness Check:
                ========================================
                Complete samples: 30/32 (93.8%)
                Incomplete samples: 2/32 (6.2%)

                Samples with missing features:
                Sample 671: missing 13 features
                    - scalar:Tr
                    - scalar:angle_out
                    - scalar:power
                    - scalar:Pr
                    - scalar:Q
                    ... and 8 more
                Sample 672: missing 13 features
                    - scalar:Tr
                    - scalar:angle_out
                    - scalar:power
                    - scalar:Pr
                    - scalar:Q
                    ... and 8 more
        """
        report = "Dataset Feature Completeness Check:\n"
        report += "=" * 40 + "\n"

        if len(self._samples) == 0:
            return report + "No samples in dataset.\n"

        # Collect all possible features across all samples
        all_scalar_names = set()
        all_field_names = set()

        for sample in self._samples.values():
            all_scalar_names.update(sample.get_scalar_names())

            times = sample.features.get_all_time_values()
            for time in times:
                base_names = sample.features.get_base_names(time=time)
                for base_name in base_names:
                    zone_names = sample.features.get_zone_names(
                        base_name=base_name, time=time
                    )
                    for zone_name in zone_names:
                        all_field_names.update(
                            sample.get_field_names(
                                zone_name=zone_name, base_name=base_name, time=time
                            )
                        )

        total_samples = len(self._samples)
        incomplete_samples = []

        # Check each sample for missing features
        for sample_id, sample in self._samples.items():
            missing_features = []

            # Check scalars
            sample_scalars = set(sample.get_scalar_names())
            missing_scalars = all_scalar_names - sample_scalars
            if missing_scalars:
                missing_features.extend([f"scalar:{name}" for name in missing_scalars])

            # Check fields
            sample_fields = set()
            times = sample.features.get_all_time_values()
            for time in times:
                base_names = sample.features.get_base_names(time=time)
                for base_name in base_names:
                    zone_names = sample.features.get_zone_names(
                        base_name=base_name, time=time
                    )
                    for zone_name in zone_names:
                        sample_fields.update(
                            sample.get_field_names(
                                zone_name=zone_name, base_name=base_name, time=time
                            )
                        )

            missing_fields = all_field_names - sample_fields
            if missing_fields:
                missing_features.extend([f"field:{name}" for name in missing_fields])

            if missing_features:
                incomplete_samples.append((sample_id, missing_features))

        # Generate report
        complete_samples = total_samples - len(incomplete_samples)
        report += f"Complete samples: {complete_samples}/{total_samples} ({complete_samples / total_samples * 100:.1f}%)\n"
        report += f"Incomplete samples: {len(incomplete_samples)}/{total_samples} ({len(incomplete_samples) / total_samples * 100:.1f}%)\n\n"

        if incomplete_samples:
            report += "Samples with missing features:\n"
            for sample_id, missing_features in incomplete_samples:
                report += (
                    f"  Sample {sample_id}: missing {len(missing_features)} features\n"
                )
                for feature in missing_features[:5]:  # Show first 5 missing features
                    report += f"    - {feature}\n"
                if len(missing_features) > 5:
                    report += f"    ... and {len(missing_features) - 5} more\n"
        else:
            report += "All samples contain all features! ✓\n"

        return report

    @classmethod
    @deprecated(
        "`Dataset.from_list_of_samples(samples)` is deprecated, use instead `Dataset(samples=samples)`",
        version="0.1.8",
        removal="0.2.0",
    )
    def from_list_of_samples(
        cls, list_of_samples: list[Sample], ids: Optional[list[int]] = None
    ) -> Self:
        """DEPRECATED: use `Dataset(samples=..., sample_ids=...)` instead."""
        return cls(samples=list_of_samples, sample_ids=ids)

    @classmethod
    @deprecated_argument("fname", "path", version="0.1.8", removal="0.2.0")
    def load_from_file(
        cls, path: Union[str, Path], verbose: bool = False, processes_number: int = 0
    ) -> Self:
        """Load data from a specified TAR (Tape Archive) file.

        Args:
            path (Union[str, Path]): The path to the data file to be loaded.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.

        Returns:
            Self: The loaded dataset (Dataset).
        """
        path = Path(path)
        instance = cls()
        instance.load(path, verbose, processes_number)
        return instance

    @classmethod
    @deprecated_argument("fname", "path", version="0.1.8", removal="0.2.0")
    def load_from_dir(
        cls,
        path: Union[str, Path],
        ids: Optional[list[int]] = None,
        verbose: bool = False,
        processes_number: int = 0,
    ) -> Self:
        """Load data from a specified directory.

        Args:
            path (Union[str, Path]): The path from which to load files.
            ids (list, optional): The specific sample IDs to load from the dataset. Defaults to None.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.

        Returns:
            Self: The loaded dataset (Dataset).
        """
        path = Path(path)
        instance = cls()
        instance._load_from_dir_(
            path, ids=ids, verbose=verbose, processes_number=processes_number
        )
        return instance

    @deprecated_argument("fname", "path", version="0.1.8", removal="0.2.0")
    def load(
        self, path: Union[str, Path], verbose: bool = False, processes_number: int = 0
    ) -> None:
        """Load data from a specified file or directory.

        Note:
            If path is a file, it creates a temporary intermediate directory to extract the files from the archive during the loading process.

        Note:
            This method overwrites the content of the calling instance.

        Args:
            path (Union[str, Path]): The path to the data file to be loaded.
            verbose (bool, optional): Explicitly displays the operations performed. Defaults to False.
            processes_number (int, optional): Number of processes used to load files (-1 to use all available ressources, 0 to disable multiprocessing). Defaults to 0.

        Raises:
            ValueError: If a randomly generated temporary directory already exists,
            indicating a potential conflict during the loading process (extremely unlikely).
        """
        path = Path(path)

        if path.is_file():
            inputdir = path.parent / f"tmploaddir_{generate_random_ASCII()}"
            if inputdir.is_dir():  # pragma: no cover
                raise ValueError(
                    f"temporary intermediate directory <{inputdir}> already exits"
                )
            inputdir.mkdir(parents=True)

            # First : untar file <path> to a directory <inputdir>
            # TODO: avoid using subprocess by using a lib tarfile
            arguments = ["tar", "-xf", path, "-C", inputdir]
            subprocess.call(arguments)

            # Then : load data from directory <inputdir>
            self._load_from_dir_(
                inputdir, verbose=verbose, processes_number=processes_number
            )

            # Finally : removes directory <inputdir>
            shutil.rmtree(inputdir)
        else:
            self._load_from_dir_(
                path, verbose=verbose, processes_number=processes_number
            )

    # -------------------------------------------------------------------------#
    @deprecated_argument("save_dir", "path", version="0.1.8", removal="0.2.0")
    def add_to_dir(
        self,
        sample: Sample,
        path: Optional[Union[str, Path]] = None,
        verbose: bool = False,
    ) -> None:
        """Add a sample to the dataset and save it to the specified directory.

        Note:
            If `path` is None, will look for `self.path` which will be retrieved from last previous call to load or save.
            `path` given in argument will take precedence over `self.path` and overwrite it.

        Args:
            sample (Sample): The sample to add.
            path (Union[str, Path], optional): The directory in which to save the sample. Defaults to None.
            verbose (bool, optional): If True, will print additional information. Defaults to False.

        Raises:
            ValueError: If both self.path and path are None.
        """
        if path is not None:
            path = Path(path)
            self.path = path
        else:
            if not hasattr(self, "path") or self.path is None:
                raise ValueError(
                    "self.path and path are None, we don't know where to save, specify one of them before"
                )

        # --- sample is not only saved to dir, but also added to the dataset
        # self.add_sample(sample)
        # --- if dataset already contains other Samples, they will all be saved to path
        # self._save_to_dir_(self.path)

        if not self.path.is_dir():
            self.path.mkdir(parents=True)

        if verbose:
            print(f"Saving database to: {self.path}")

        samples_dir = self.path / "samples"
        if not samples_dir.is_dir():
            samples_dir.mkdir(parents=True)

        # find i_sample
        # if there are already samples in the instance, we should not take an already existing id
        # if there are already samples in the path, we should not take an already existing id
        sample_ids_in_path = [
            int(d.name.split("_")[-1])
            for d in samples_dir.glob("sample_*")
            if d.is_dir()
        ]
        i_sample = max(sample_ids_in_path) + 1 if len(sample_ids_in_path) > 0 else 0
        i_sample = max(len(self), i_sample)

        sample_dname = samples_dir / f"sample_{i_sample:09d}"
        sample.save_to_dir(sample_dname)

    @deprecated(
        "`Dataset._save_to_dir_(path)` is deprecated, use instead `Dataset.save_to_dir(path)`",
        version="0.1.10",
        removal="0.2.0",
    )
    def _save_to_dir_(self, path: Union[str, Path], verbose: bool = False) -> None:
        """DEPRECATED: use :meth:`Dataset.save_to_dir` instead."""
        self.save_to_dir(path, verbose=verbose)

    def _load_from_dir_(
        self,
        path: Union[str, Path],
        ids: Optional[list[int]] = None,
        verbose: bool = False,
        processes_number: int = 0,
    ) -> None:
        """DEPRECATED: use :meth:`Dataset.load` instead."""
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(
                f'"{path}" is not a directory or does not exist. Abort'
            )

        if processes_number < -1:
            raise ValueError("Number of processes cannot be < -1")

        self.path = path

        if verbose:  # pragma: no cover
            print(f"Reading database located at: {path}")

        # Load infos
        infos_fname = path / "infos.yaml"
        if infos_fname.is_file():
            with open(infos_fname, "r") as file:
                self._infos = yaml.safe_load(file)
        if (
            "plaid" not in self._infos or "version" not in self._infos["plaid"]
        ):  # pragma: no cover
            self._infos.setdefault("plaid", {}).setdefault("version", None)
        else:
            if not isinstance(self._infos["plaid"]["version"], Version):
                self._infos["plaid"]["version"] = Version(
                    self._infos["plaid"]["version"]
                )

        # Load samples
        sample_paths = sorted(
            [path for path in (path / "samples").glob("sample_*") if path.is_dir()]
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
                sample = Sample(path=sample_path)
                self.add_sample(sample, id)
        else:
            with Pool(processes_number) as p:
                for id, sample in list(
                    tqdm(
                        p.imap(_process_sample, sample_paths),
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
                    _process_sample,
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

        if len(self) == 0:  # pragma: no cover
            print("Warning: dataset contains no sample")

    @staticmethod
    def _load_number_of_samples_(_path: Union[str, Path]) -> int:
        """DEPRECATED: use :meth:`plaid.get_number_of_samples <plaid.containers.utils.get_number_of_samples>` instead."""
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
    def set_sample(
        self, id: int, sample: Sample, warning_overwrite: bool = True
    ) -> None:
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

                from plaid import Dataset
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

        Note:
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

                from plaid import Dataset
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

    def __str__(self) -> str:
        """Return a string representation of the dataset.

        Returns:
            str: A string representation of the overview of dataset content.

        Example:
            .. code-block:: python

                from plaid import Dataset
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

        # fields
        nb_fields = len(self.get_field_names())
        str_repr += f"{nb_fields} field{'' if nb_fields == 1 else 's'}, "

        if str_repr[-2:] == ", ":
            str_repr = str_repr[:-2]
        str_repr = str_repr + ")"
        return str_repr

    __repr__ = __str__
