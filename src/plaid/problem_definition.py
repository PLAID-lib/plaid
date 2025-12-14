"""Implementation of the `ProblemDefinition` class."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import csv
import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import yaml
from packaging.version import Version

import plaid
from plaid.constants import AUTHORIZED_SCORE_FUNCTIONS, AUTHORIZED_TASKS
from plaid.containers import FeatureIdentifier
from plaid.types import IndexType
from plaid.utils.deprecation import deprecated

# %% Globals

logger = logging.getLogger(__name__)

# %% Functions

# %% Classes


class ProblemDefinition(object):
    """Gathers all necessary informations to define a learning problem."""

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        directory_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """Initialize an empty :class:`ProblemDefinition <plaid.problem_definition.ProblemDefinition>`.

        Use :meth:`add_inputs <plaid.problem_definition.ProblemDefinition.add_inputs>` or :meth:`add_output_scalars_names <plaid.problem_definition.ProblemDefinition.add_output_scalars_names>` to feed the :class:`ProblemDefinition`

        Args:
            path (Union[str,Path], optional): The path from which to load PLAID problem definition files.
            directory_path (Union[str,Path], optional): Deprecated, use `path` instead.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition

                # 1. Create empty instance of ProblemDefinition
                problem_definition = ProblemDefinition()
                print(problem_definition)
                >>> ProblemDefinition()

                # 2. Load problem definition and create ProblemDefinition instance
                problem_definition = ProblemDefinition("path_to_plaid_prob_def")
                print(problem_definition)
                >>> ProblemDefinition(input_scalars_names=['s_1'], output_scalars_names=['s_2'], input_meshes_names=['mesh'], task='regression')
        """
        self._name: str = None
        self._version: Union[Version] = Version(plaid.__version__)
        self._task: str = None
        self._score_function: str = None
        self.in_features_identifiers: Sequence[Union[str, FeatureIdentifier]] = []
        self.out_features_identifiers: Sequence[Union[str, FeatureIdentifier]] = []
        self.constant_features_identifiers: list[str] = []
        self.in_scalars_names: list[str] = []
        self.out_scalars_names: list[str] = []
        self.in_timeseries_names: list[str] = []
        self.out_timeseries_names: list[str] = []
        self.in_fields_names: list[str] = []
        self.out_fields_names: list[str] = []
        self.in_meshes_names: list[str] = []
        self.out_meshes_names: list[str] = []
        self._split: Optional[dict[str, IndexType]] = None
        self._train_split: Optional[dict[str, dict[str, IndexType]]] = None
        self._test_split: Optional[dict[str, dict[str, IndexType]]] = None

        if directory_path is not None:
            if path is not None:
                raise ValueError(
                    "Arguments `path` and `directory_path` cannot be both set. Use only `path` as `directory_path` is deprecated."
                )
            else:
                path = directory_path
                logger.warning(
                    "DeprecationWarning: 'directory_path' is deprecated, use 'path' instead."
                )

        if path is not None:
            path = Path(path)
            self._load_from_dir_(path)

    # -------------------------------------------------------------------------#
    def get_name(self) -> str:
        """Get the name. None if not defined.

        Returns:
            str: The name, such as "regression_1".
        """
        return self._name

    def set_name(self, name: str) -> None:
        """Set the name.

        Args:
            name (str): The name, such as "regression_1".
        """
        if self._name is not None:
            raise ValueError(f"A name is already in self._name: (`{self._name}`)")
        else:
            self._name = name

    # -------------------------------------------------------------------------#
    def get_version(self) -> Version:
        """Get the version. None if not defined.

        Returns:
            Version: The version, such as "0.1.0".
        """
        return self._version

    # -------------------------------------------------------------------------#
    def get_task(self) -> str:
        """Get the authorized task. None if not defined.

        Returns:
            str: The authorized task, such as "regression" or "classification".
        """
        return self._task

    def set_task(self, task: str) -> None:
        """Set the authorized task.

        Args:
            task (str): The authorized task to be set, such as "regression" or "classification".
        """
        if self._task is not None:
            raise ValueError(f"A task is already in self._task: (`{self._task}`)")
        elif task in AUTHORIZED_TASKS:
            self._task = task
        else:
            raise TypeError(
                f"{task} not among authorized tasks. Maybe you want to try among: {AUTHORIZED_TASKS}"
            )

    # -------------------------------------------------------------------------#
    def get_score_function(self) -> str:
        """Get the authorized score function. None if not defined.

        Returns:
            str: The authorized score function, such as "RRMSE".
        """
        return self._score_function

    def set_score_function(self, score_function: str) -> None:
        """Set the authorized score function.

        Args:
            score_function (str): The authorized score function, such as "RRMSE".
        """
        if self._score_function is not None:
            raise ValueError(
                f"A score function is already in self._task: (`{self._score_function}`)"
            )
        elif score_function in AUTHORIZED_SCORE_FUNCTIONS:
            self._score_function = score_function
        else:
            raise TypeError(
                f"{score_function} not among authorized tasks. Maybe you want to try among: {AUTHORIZED_SCORE_FUNCTIONS}"
            )

    # -------------------------------------------------------------------------#

    def get_split(
        self, indices_name: Optional[str] = None
    ) -> Union[IndexType, dict[str, IndexType]]:
        """Get the split indices. This function returns the split indices, either for a specific split with the provided `indices_name` or all split indices if `indices_name` is not specified.

        Args:
            indices_name (str, optional): The name of the split for which indices are requested. Defaults to None.

        Raises:
            KeyError: If `indices_name` is specified but not found among split names.

        Returns:
            Union[IndexType,dict[str,IndexType]]: If `indices_name` is provided, it returns
            the indices for that split (IndexType). If `indices_name` is not provided, it
            returns a dictionary mapping split names (str) to their respective indices
            (IndexType).

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                split_indices = problem.get_split()
                print(split_indices)
                >>> {'train': [0, 1, 2, ...], 'test': [100, 101, ...]}

                test_indices = problem.get_split('test')
                print(test_indices)
                >>> [100, 101, ...]
        """
        if indices_name is None:
            return self._split
        else:
            assert indices_name in self._split, (
                indices_name + " not among split indices names"
            )
            return self._split[indices_name]

    def set_split(self, split: dict[str, IndexType]) -> None:
        """Set the split indices. This function allows you to set the split indices by providing a dictionary mapping split names (str) to their respective indices (IndexType).

        Args:
            split (dict[str,IndexType]):  A dictionary containing split names and their indices.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                new_split = {'train': [0, 1, 2], 'test': [3, 4]}
                problem.set_split(new_split)
        """
        if self._split is not None:  # pragma: no cover
            logger.warning("split already exists -> data will be replaced")
        self._split = split

    def get_train_split(
        self, indices_name: Optional[str] = None
    ) -> Union[dict[str, IndexType], dict[str, dict[str, IndexType]]]:
        """Get the train split indices for different subsets of the dataset.

        Args:
            indices_name (str, optional): The name of the specific train split subset
                for which indices are requested. Defaults to None.

        Returns:
            Union[dict[str, IndexType], dict[str, dict[str, IndexType]]]:
                If indices_name is provided:
                    - Returns a dictionary mapping split names to their indices for the specified subset.
                If indices_name is None:
                    - Returns the complete train split dictionary containing all subsets and their indices.

        Raises:
            AssertionError: If indices_name is provided but not found in the train split.
        """
        if indices_name is None:
            return self._train_split
        else:
            assert indices_name in self._train_split, (
                indices_name + " not among split indices names"
            )
            return self._train_split[indices_name]

    def set_train_split(self, split: dict[str, dict[str, Optional[IndexType]]]) -> None:
        """Set the train split dictionary containing subsets and their indices.

        Args:
            split (dict[str, dict[str, IndexType]]): Dictionary mapping train subset names
                to their split dictionaries. Each split dictionary maps split names (e.g., 'train', 'val')
                to their indices.

        Note:
            If a train split already exists, it will be replaced and a warning will be logged.
        """
        if self._train_split is not None:  # pragma: no cover
            logger.warning("split already exists -> data will be replaced")
        self._train_split = split

    def get_test_split(
        self, indices_name: Optional[str] = None
    ) -> Union[dict[str, IndexType], dict[str, dict[str, IndexType]]]:
        """Get the test split indices for different subsets of the dataset.

        Args:
            indices_name (str, optional): The name of the specific test split subset
                for which indices are requested. Defaults to None.

        Returns:
            Union[dict[str, IndexType], dict[str, dict[str, IndexType]]]:
                If indices_name is provided:
                    - Returns a dictionary mapping split names to their indices for the specified subset.
                If indices_name is None:
                    - Returns the complete test split dictionary containing all subsets and their indices.

        Raises:
            AssertionError: If indices_name is provided but not found in the test split.
        """
        if indices_name is None:
            return self._test_split
        else:
            assert indices_name in self._test_split, (
                indices_name + " not among split indices names"
            )
            return self._test_split[indices_name]

    def set_test_split(self, split: dict[str, dict[str, Optional[IndexType]]]) -> None:
        """Set the test split dictionary containing subsets and their indices.

        Args:
            split (dict[str, dict[str, IndexType]]): Dictionary mapping test subset names
                to their split dictionaries. Each split dictionary maps split names (e.g., 'test', 'test_ood')
                to their indices.

        Note:
            If a test split already exists, it will be replaced and a warning will be logged.
        """
        if self._test_split is not None:  # pragma: no cover
            logger.warning("split already exists -> data will be replaced")
        self._test_split = split

    # -------------------------------------------------------------------------#
    @staticmethod
    def _feature_sort_key(feat: Union[str, FeatureIdentifier]) -> tuple[str, str]:
        if isinstance(feat, str):
            # Strings first, sorted lexicographically
            return ("a_string", feat)
        else:
            assert isinstance(feat, FeatureIdentifier)
            # Then FeatureIdentifiers, sorted by their "type" field
            return ("b_feature", feat["type"])

    def get_in_features_identifiers(self) -> Sequence[Union[str, FeatureIdentifier]]:
        """Get the input features identifiers of the problem.

        Returns:
            Sequence[Union[str, FeatureIdentifier]]: A list of input feature identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                in_features_identifiers = problem.get_in_features_identifiers()
                print(in_features_identifiers)
                >>> ['omega', 'pressure']
        """
        return self.in_features_identifiers

    def add_in_features_identifiers(
        self, inputs: Sequence[Union[str, FeatureIdentifier]]
    ) -> None:
        """Add input features identifiers to the problem.

        Args:
            inputs (Sequence[Union[str, FeatureIdentifier]]): A list of input feature identifiers to add.

        Raises:
            ValueError: If some :code:`inputs` are redondant.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                in_features_identifiers = ['omega', 'pressure']
                problem.add_in_features_identifiers(in_features_identifiers)
        """
        if not (len(set(inputs)) == len(inputs)):
            raise ValueError("Some inputs have same identifiers")
        for input in inputs:
            self.add_in_feature_identifier(input)

    def add_in_feature_identifier(self, input: Union[str, FeatureIdentifier]) -> None:
        """Add an input feature identifier or identifier to the problem.

        Args:
            input (FeatureIdentifier):  The identifier or identifier of the input feature to add.

        Raises:
            ValueError: If the specified input feature is already in the list of inputs.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                input_identifier = 'pressure'
                problem.add_in_feature_identifier(input_identifier)
        """
        if input in self.in_features_identifiers:
            raise ValueError(f"{input} is already in self.in_features_identifiers")
        self.in_features_identifiers.append(input)
        self.in_features_identifiers.sort(key=self._feature_sort_key)

    def filter_in_features_identifiers(
        self, identifiers: Sequence[Union[str, FeatureIdentifier]]
    ) -> Sequence[Union[str, FeatureIdentifier]]:
        """Filter and get input features features corresponding to a sorted list of identifiers.

        Args:
            identifiers (Sequence[Union[str, FeatureIdentifier]]): A list of identifiers for which to retrieve corresponding input features.

        Returns:
            Sequence[Union[str, FeatureIdentifier]]: A sorted list of input feature identifiers or categories corresponding to the provided identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                features_identifiers = ['omega', 'pressure', 'temperature']
                input_features = problem.filter_in_features_identifiers(features_identifiers)
                print(input_features)
                >>> ['omega', 'pressure']
        """
        return sorted(set(identifiers).intersection(self.get_in_features_identifiers()))

    # -------------------------------------------------------------------------#
    def get_out_features_identifiers(self) -> Sequence[Union[str, FeatureIdentifier]]:
        """Get the output features identifiers of the problem.

        Returns:
            Sequence[Union[str, FeatureIdentifier]]: A list of output feature identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                outputs_identifiers = problem.get_out_features_identifiers()
                print(outputs_identifiers)
                >>> ['compression_rate', 'in_massflow', 'isentropic_efficiency']
        """
        return self.out_features_identifiers

    def add_out_features_identifiers(
        self, outputs: Sequence[Union[str, FeatureIdentifier]]
    ) -> None:
        """Add output features identifiers to the problem.

        Args:
            outputs (Sequence[Union[str, FeatureIdentifier]]): A list of output feature identifiers to add.

        Raises:
            ValueError: if some :code:`outputs` are redondant.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                out_features_identifiers = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                problem.add_out_features_identifiers(out_features_identifiers)
        """
        if not (len(set(outputs)) == len(outputs)):
            raise ValueError("Some outputs have same identifiers")
        for output in outputs:
            self.add_out_feature_identifier(output)

    def add_out_feature_identifier(self, output: Union[str, FeatureIdentifier]) -> None:
        """Add an output feature identifier or identifier to the problem.

        Args:
            output (FeatureIdentifier):  The identifier or identifier of the output feature to add.

        Raises:
            ValueError: If the specified output feature is already in the list of outputs.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                out_features_identifiers = 'pressure'
                problem.add_out_feature_identifier(out_features_identifiers)
        """
        if output in self.out_features_identifiers:
            raise ValueError(f"{output} is already in self.out_features_identifiers")
        self.out_features_identifiers.append(output)
        self.out_features_identifiers.sort(key=self._feature_sort_key)

    def filter_out_features_identifiers(
        self, identifiers: Sequence[Union[str, FeatureIdentifier]]
    ) -> Sequence[Union[str, FeatureIdentifier]]:
        """Filter and get output features corresponding to a sorted list of identifiers.

        Args:
            identifiers (Sequence[Union[str, FeatureIdentifier]]): A list of identifiers for which to retrieve corresponding output features.

        Returns:
            Sequence[Union[str, FeatureIdentifier]]: A sorted list of output feature identifiers or categories corresponding to the provided identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                features_identifiers = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                output_features = problem.filter_out_features_identifiers(features_identifiers)
                print(output_features)
                >>> ['in_massflow']
        """
        return sorted(
            set(identifiers).intersection(self.get_out_features_identifiers())
        )

    # -------------------------------------------------------------------------#
    def get_constant_features_identifiers(self) -> list[str]:
        """Get the constant features identifiers of the problem.

        Returns:
            list[str]: A list of constant feature identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                constant_features_identifiers = problem.get_constant_features_identifiers()
                print(constant_features_identifiers)
                >>> ['Global/P', 'Base_2_2/Zone/GridCoordinates']
        """
        return self.constant_features_identifiers

    def add_constant_features_identifiers(self, inputs: list[str]) -> None:
        """Add input features identifiers to the problem.

        Args:
            inputs (list[str]): A list of constant feature identifiers to add.

        Raises:
            ValueError: If some :code:`inputs` are redondant.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                constant_features_identifiers = ['Global/P', 'Base_2_2/Zone/GridCoordinates']
                problem.add_constant_features_identifiers(constant_features_identifiers)
        """
        if not (len(set(inputs)) == len(inputs)):
            raise ValueError("Some inputs have same identifiers")
        for input in inputs:
            self.add_constant_feature_identifier(input)

    def add_constant_feature_identifier(self, input: str) -> None:
        """Add an constant feature identifier to the problem.

        Args:
            input (str):  The identifier of the constant feature to add.

        Raises:
            ValueError: If the specified input feature is already in the list of constant features.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                constant_identifier = 'Global/P'
                problem.add_constant_feature_identifier(constant_identifier)
        """
        if input in self.constant_features_identifiers:
            raise ValueError(f"{input} is already in self.in_features_identifiers")
        self.constant_features_identifiers.append(input)
        self.constant_features_identifiers.sort(key=self._feature_sort_key)

    def filter_constant_features_identifiers(self, identifiers: list[str]) -> list[str]:
        """Filter and get input features features corresponding to a sorted list of identifiers.

        Args:
            identifiers (list[str]): A list of identifiers for which to retrieve corresponding constant features.

        Returns:
            list[str]: A sorted list of constant feature identifiers corresponding to the provided identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                features_identifiers = ['Global/P', 'Base_2_2/Zone/GridCoordinates']
                constant_features = problem.filter_constant_features_identifiers(features_identifiers)
                print(constant_features)
                >>> ['Global/P']
        """
        return sorted(
            set(identifiers).intersection(self.get_constant_features_identifiers())
        )

    # -------------------------------------------------------------------------#
    @deprecated(
        "use `get_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def get_input_scalars_names(self) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.get_in_features_identifiers` instead.

        Get the input scalars names of the problem.

        Returns:
            list[str]: A list of input feature names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_scalars_names = problem.get_input_scalars_names()
                print(input_scalars_names)
                >>> ['omega', 'pressure']
        """
        return self.in_scalars_names

    @deprecated(
        "use `add_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def add_input_scalars_names(self, inputs: list[str]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_in_features_identifiers` instead.

        Add input scalars names to the problem.

        Args:
            inputs (list[str]): A list of input feature names to add.

        Raises:
            ValueError: If some :code:`inputs` are redondant.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                input_scalars_names = ['omega', 'pressure']
                problem.add_input_scalars_names(input_scalars_names)
        """
        if not (len(set(inputs)) == len(inputs)):
            raise ValueError("Some inputs have same names")
        for input in inputs:
            self.add_input_scalar_name(input)

    @deprecated(
        "use `add_in_feature_identifier` instead", version="0.1.8", removal="0.2.0"
    )
    def add_input_scalar_name(self, input: str) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_in_feature_identifier` instead.

        Add an input scalar name to the problem.

        Args:
            input (str):  The name of the input feature to add.

        Raises:
            ValueError: If the specified input feature is already in the list of inputs.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                input_name = 'pressure'
                problem.add_input_scalar_name(input_name)
        """
        if input in self.in_scalars_names:
            raise ValueError(f"{input} is already in self.in_scalars_names")
        self.in_scalars_names.append(input)
        self.in_scalars_names.sort()

    @deprecated(
        "use `filter_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def filter_input_scalars_names(self, names: list[str]) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.filter_in_features_identifiers` instead.

        Filter and get input scalars features corresponding to a list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding input features.

        Returns:
            list[str]: A sorted list of input feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                scalars_names = ['omega', 'pressure', 'temperature']
                input_features = problem.filter_input_scalars_names(scalars_names)
                print(input_features)
                >>> ['omega', 'pressure']
        """
        return sorted(set(names).intersection(self.get_input_scalars_names()))

    # -------------------------------------------------------------------------#
    @deprecated(
        "use `get_out_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def get_output_scalars_names(self) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.get_out_features_identifiers` instead.

        Get the output scalars names of the problem.

        Returns:
            list[str]: A list of output feature names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                outputs_names = problem.get_output_scalars_names()
                print(outputs_names)
                >>> ['compression_rate', 'in_massflow', 'isentropic_efficiency']
        """
        return self.out_scalars_names

    @deprecated(
        "use `add_out_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def add_output_scalars_names(self, outputs: list[str]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_out_features_identifiers` instead.

        Add output scalars names to the problem.

        Args:
            outputs (list[str]): A list of output feature names to add.

        Raises:
            ValueError: if some :code:`outputs` are redondant.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                output_scalars_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                problem.add_output_scalars_names(output_scalars_names)
        """
        if not (len(set(outputs)) == len(outputs)):
            raise ValueError("Some outputs have same names")
        for output in outputs:
            self.add_output_scalar_name(output)

    @deprecated(
        "use `add_out_feature_identifier` instead", version="0.1.8", removal="0.2.0"
    )
    def add_output_scalar_name(self, output: str) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_out_feature_identifier` instead.

        Add an output scalar name to the problem.

        Args:
            output (str):  The name of the output feature to add.

        Raises:
            ValueError: If the specified output feature is already in the list of outputs.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                output_scalars_names = 'pressure'
                problem.add_output_scalar_name(output_scalars_names)
        """
        if output in self.out_scalars_names:
            raise ValueError(f"{output} is already in self.out_scalars_names")
        self.out_scalars_names.append(output)
        self.in_scalars_names.sort()

    def filter_output_scalars_names(self, names: list[str]) -> list[str]:
        """Filter and get output features corresponding to a list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding output features.

        Returns:
            list[str]: A sorted list of output feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                scalars_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                output_features = problem.filter_output_scalars_names(scalars_names)
                print(output_features)
                >>> ['in_massflow']
        """
        return sorted(set(names).intersection(self.get_output_scalars_names()))

    # -------------------------------------------------------------------------#
    @deprecated(
        "use `get_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def get_input_fields_names(self) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.get_in_features_identifiers` instead.

        Get the input fields names of the problem.

        Returns:
            list[str]: A list of input feature names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_fields_names = problem.get_input_fields_names()
                print(input_fields_names)
                >>> ['omega', 'pressure']
        """
        return self.in_fields_names

    @deprecated(
        "use `add_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def add_input_fields_names(self, inputs: list[str]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_in_features_identifiers` instead.

        Add input fields names to the problem.

        Args:
            inputs (list[str]): A list of input feature names to add.

        Raises:
            ValueError: If some :code:`inputs` are redondant.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                input_fields_names = ['omega', 'pressure']
                problem.add_input_fields_names(input_fields_names)
        """
        if not (len(set(inputs)) == len(inputs)):
            raise ValueError("Some inputs have same names")
        for input in inputs:
            self.add_input_field_name(input)

    @deprecated(
        "use `add_in_feature_identifier` instead", version="0.1.8", removal="0.2.0"
    )
    def add_input_field_name(self, input: str) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_in_feature_identifier` instead.

        Add an input field name to the problem.

        Args:
            input (str):  The name of the input feature to add.

        Raises:
            ValueError: If the specified input feature is already in the list of inputs.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                input_name = 'pressure'
                problem.add_input_field_name(input_name)
        """
        if input in self.in_fields_names:
            raise ValueError(f"{input} is already in self.in_fields_names")
        self.in_fields_names.append(input)
        self.in_fields_names.sort()

    def filter_input_fields_names(self, names: list[str]) -> list[str]:
        """Filter and get input fields features corresponding to a list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding input features.

        Returns:
            list[str]: A sorted list of input feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_fields_names = ['omega', 'pressure', 'temperature']
                input_features = problem.filter_input_fields_names(input_fields_names)
                print(input_features)
                >>> ['omega', 'pressure']
        """
        return sorted(set(names).intersection(self.get_input_fields_names()))

    # -------------------------------------------------------------------------#
    @deprecated(
        "use `get_out_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def get_output_fields_names(self) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.get_out_features_identifiers` instead.

        Get the output fields names of the problem.

        Returns:
            list[str]: A list of output feature names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                outputs_names = problem.get_output_fields_names()
                print(outputs_names)
                >>> ['compression_rate', 'in_massflow', 'isentropic_efficiency']
        """
        return self.out_fields_names

    @deprecated(
        "use `add_out_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def add_output_fields_names(self, outputs: list[str]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_out_features_identifiers` instead.

        Add output fields names to the problem.

        Args:
            outputs (list[str]): A list of output feature names to add.

        Raises:
            ValueError: if some :code:`outputs` are redondant.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                output_fields_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                problem.add_output_fields_names(output_fields_names)
        """
        if not (len(set(outputs)) == len(outputs)):
            raise ValueError("Some outputs have same names")
        for output in outputs:
            self.add_output_field_name(output)

    @deprecated(
        "use `add_out_feature_identifier` instead", version="0.1.8", removal="0.2.0"
    )
    def add_output_field_name(self, output: str) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_out_feature_identifier` instead.

        Add an output field name to the problem.

        Args:
            output (str):  The name of the output feature to add.

        Raises:
            ValueError: If the specified output feature is already in the list of outputs.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                output_fields_names = 'pressure'
                problem.add_output_field_name(output_fields_names)
        """
        if output in self.out_fields_names:
            raise ValueError(f"{output} is already in self.out_fields_names")
        self.out_fields_names.append(output)
        self.out_fields_names.sort()

    def filter_output_fields_names(self, names: list[str]) -> list[str]:
        """Filter and get output features corresponding to a list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding output features.

        Returns:
            list[str]: A sorted list of output feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                output_fields_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                output_features = problem.filter_output_fields_names(output_fields_names)
                print(output_features)
                >>> ['in_massflow']
        """
        return sorted(set(names).intersection(self.get_output_fields_names()))

    # -------------------------------------------------------------------------#
    @deprecated(
        "use `get_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def get_input_timeseries_names(self) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.get_in_features_identifiers` instead.

        Get the input timeseries names of the problem.

        Returns:
            list[str]: A list of input feature names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_timeseries_names = problem.get_input_timeseries_names()
                print(input_timeseries_names)
                >>> ['omega', 'pressure']
        """
        return self.in_timeseries_names

    @deprecated(
        "use `add_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def add_input_timeseries_names(self, inputs: list[str]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_in_features_identifiers` instead.

        Add input timeseries names to the problem.

        Args:
            inputs (list[str]): A list of input feature names to add.

        Raises:
            ValueError: If some :code:`inputs` are redondant.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                input_timeseries_names = ['omega', 'pressure']
                problem.add_input_timeseries_names(input_timeseries_names)
        """
        if not (len(set(inputs)) == len(inputs)):
            raise ValueError("Some inputs have same names")
        for input in inputs:
            self.add_input_timeseries_name(input)

    @deprecated(
        "use `add_in_feature_identifier` instead", version="0.1.8", removal="0.2.0"
    )
    def add_input_timeseries_name(self, input: str) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_in_feature_identifier` instead.

        Add an input timeseries name to the problem.

        Args:
            input (str):  The name of the input feature to add.

        Raises:
            ValueError: If the specified input feature is already in the list of inputs.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                input_name = 'pressure'
                problem.add_input_timeseries_name(input_name)
        """
        if input in self.in_timeseries_names:
            raise ValueError(f"{input} is already in self.in_timeseries_names")
        self.in_timeseries_names.append(input)
        self.in_timeseries_names.sort()

    def filter_input_timeseries_names(self, names: list[str]) -> list[str]:
        """Filter and get input timeseries features corresponding to a list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding input features.

        Returns:
            list[str]: A sorted list of input feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_timeseries_names = ['omega', 'pressure', 'temperature']
                input_features = problem.filter_input_timeseries_names(input_timeseries_names)
                print(input_features)
                >>> ['omega', 'pressure']
        """
        return sorted(set(names).intersection(self.get_input_timeseries_names()))

    # -------------------------------------------------------------------------#
    @deprecated(
        "use `get_out_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def get_output_timeseries_names(self) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.get_out_features_identifiers` instead.

        Get the output timeseries names of the problem.

        Returns:
            list[str]: A list of output feature names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                outputs_names = problem.get_output_timeseries_names()
                print(outputs_names)
                >>> ['compression_rate', 'in_massflow', 'isentropic_efficiency']
        """
        return self.out_timeseries_names

    @deprecated(
        "use `add_out_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def add_output_timeseries_names(self, outputs: list[str]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_out_features_identifiers` instead.

        Add output timeseries names to the problem.

        Args:
            outputs (list[str]): A list of output feature names to add.

        Raises:
            ValueError: if some :code:`outputs` are redondant.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                output_timeseries_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                problem.add_output_timeseries_names(output_timeseries_names)
        """
        if not (len(set(outputs)) == len(outputs)):
            raise ValueError("Some outputs have same names")
        for output in outputs:
            self.add_output_timeseries_name(output)

    @deprecated(
        "use `add_out_feature_identifier` instead", version="0.1.8", removal="0.2.0"
    )
    def add_output_timeseries_name(self, output: str) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_out_feature_identifier` instead.

        Add an output timeseries name to the problem.

        Args:
            output (str):  The name of the output feature to add.

        Raises:
            ValueError: If the specified output feature is already in the list of outputs.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                output_timeseries_names = 'pressure'
                problem.add_output_timeseries_name(output_timeseries_names)
        """
        if output in self.out_timeseries_names:
            raise ValueError(f"{output} is already in self.out_timeseries_names")
        self.out_timeseries_names.append(output)
        self.in_timeseries_names.sort()

    def filter_output_timeseries_names(self, names: list[str]) -> list[str]:
        """Filter and get output features corresponding to a list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding output features.

        Returns:
            list[str]: A sorted list of output feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                output_timeseries_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                output_features = problem.filter_output_timeseries_names(output_timeseries_names)
                print(output_features)
                >>> ['in_massflow']
        """
        return sorted(set(names).intersection(self.get_output_timeseries_names()))

    # -------------------------------------------------------------------------#
    @deprecated(
        "use `get_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def get_input_meshes_names(self) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.get_in_features_identifiers` instead.

        Get the input meshes names of the problem.

        Returns:
            list[str]: A list of input feature names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_meshes_names = problem.get_input_meshes_names()
                print(input_meshes_names)
                >>> ['omega', 'pressure']
        """
        return self.in_meshes_names

    @deprecated(
        "use `add_in_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def add_input_meshes_names(self, inputs: list[str]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_in_features_identifiers` instead.

        Add input meshes names to the problem.

        Args:
            inputs (list[str]): A list of input feature names to add.

        Raises:
            ValueError: If some :code:`inputs` are redondant.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                input_meshes_names = ['omega', 'pressure']
                problem.add_input_meshes_names(input_meshes_names)
        """
        if not (len(set(inputs)) == len(inputs)):
            raise ValueError("Some inputs have same names")
        for input in inputs:
            self.add_input_mesh_name(input)

    @deprecated(
        "use `add_in_feature_identifier` instead", version="0.1.8", removal="0.2.0"
    )
    def add_input_mesh_name(self, input: str) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_in_feature_identifier` instead.

        Add an input mesh name to the problem.

        Args:
            input (str):  The name of the input feature to add.

        Raises:
            ValueError: If the specified input feature is already in the list of inputs.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                input_name = 'pressure'
                problem.add_input_mesh_name(input_name)
        """
        if input in self.in_meshes_names:
            raise ValueError(f"{input} is already in self.in_meshes_names")
        self.in_meshes_names.append(input)
        self.in_meshes_names.sort()

    def filter_input_meshes_names(self, names: list[str]) -> list[str]:
        """Filter and get input meshes features corresponding to a list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding input features.

        Returns:
            list[str]: A sorted list of input feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_meshes_names = ['omega', 'pressure', 'temperature']
                input_features = problem.filter_input_meshes_names(input_meshes_names)
                print(input_features)
                >>> ['omega', 'pressure']
        """
        return sorted(set(names).intersection(self.get_input_meshes_names()))

    # -------------------------------------------------------------------------#
    @deprecated(
        "use `get_out_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def get_output_meshes_names(self) -> list[str]:
        """DEPRECATED: use :meth:`ProblemDefinition.get_out_features_identifiers` instead.

        Get the output meshes names of the problem.

        Returns:
            list[str]: A list of output feature names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                outputs_names = problem.get_output_meshes_names()
                print(outputs_names)
                >>> ['compression_rate', 'in_massflow', 'isentropic_efficiency']
        """
        return self.out_meshes_names

    @deprecated(
        "use `add_out_features_identifiers` instead", version="0.1.8", removal="0.2.0"
    )
    def add_output_meshes_names(self, outputs: list[str]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_out_features_identifiers` instead.

        Add output meshes names to the problem.

        Args:
            outputs (list[str]): A list of output feature names to add.

        Raises:
            ValueError: if some :code:`outputs` are redondant.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                output_meshes_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                problem.add_output_meshes_names(output_meshes_names)
        """
        if not (len(set(outputs)) == len(outputs)):
            raise ValueError("Some outputs have same names")
        for output in outputs:
            self.add_output_mesh_name(output)

    @deprecated(
        "use `add_out_feature_identifier` instead", version="0.1.8", removal="0.2.0"
    )
    def add_output_mesh_name(self, output: str) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.add_out_feature_identifier` instead.

        Add an output mesh name to the problem.

        Args:
            output (str):  The name of the output feature to add.

        Raises:
            ValueError: If the specified output feature is already in the list of outputs.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                output_meshes_names = 'pressure'
                problem.add_output_mesh_name(output_meshes_names)
        """
        if output in self.out_meshes_names:
            raise ValueError(f"{output} is already in self.out_meshes_names")
        self.out_meshes_names.append(output)
        self.in_meshes_names.sort()

    def filter_output_meshes_names(self, names: list[str]) -> list[str]:
        """Filter and get output features corresponding to a list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding output features.

        Returns:
            list[str]: A sorted list of output feature names or categories corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                output_meshes_names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                output_features = problem.filter_output_meshes_names(output_meshes_names)
                print(output_features)
                >>> ['in_massflow']
        """
        return sorted(set(names).intersection(self.get_output_meshes_names()))

    # -------------------------------------------------------------------------#
    def get_all_indices(self) -> list[int]:
        """Get all indices from splits.

        Returns:
            list[int]: list containing all unique indices.
        """
        all_indices = []
        for indices in self.get_split().values():
            all_indices += list(indices)
        return list(set(all_indices))

    # -------------------------------------------------------------------------#
    def _generate_problem_infos_dict(self) -> dict[str, Union[str, list]]:
        """Generate a dictionary containing all relevant problem definition data.

        Returns:
            dict[str, Union[str, list]]: A dictionary with keys for task, input/output features, scalars, fields, timeseries, and meshes.
        """
        data = {
            "task": self._task,
            "score_function": self._score_function,
            "constant_features": [],
            "input_features": [],
            "output_features": [],
        }
        for tup in self.in_features_identifiers:
            if isinstance(tup, FeatureIdentifier):
                data["input_features"].append(dict(**tup))
            else:
                data["input_features"].append(tup)
        for tup in self.out_features_identifiers:
            if isinstance(tup, FeatureIdentifier):
                data["output_features"].append(dict(**tup))
            else:
                data["output_features"].append(tup)
        for tup in self.constant_features_identifiers:
            data["constant_features"].append(tup)
        if self._train_split is not None:
            data["train_split"] = self._train_split
        if self._test_split is not None:
            data["test_split"] = self._test_split
        if self._name is not None:
            data["name"] = self._name
        if Version(plaid.__version__) < Version("0.2.0"):
            data.update(
                {
                    k: v
                    for k, v in {
                        "input_scalars": self.in_scalars_names,
                        "output_scalars": self.out_scalars_names,
                        "input_fields": self.in_fields_names,
                        "output_fields": self.out_fields_names,
                        "input_timeseries": self.in_timeseries_names,
                        "output_timeseries": self.out_timeseries_names,
                        "input_meshes": self.in_meshes_names,
                        "output_meshes": self.out_meshes_names,
                    }.items()
                    if v  # keeps only truthy (non-empty, non-None) lists
                }
            )

        # Handle version
        plaid_version = Version(plaid.__version__)
        if self._version != plaid_version:  # pragma: no cover
            logger.warning(
                f"Version mismatch: ProblemDefinition was loaded from version {self._version if self._version is not None else 'anterior to 0.1.10'}, and will be saved with version: {plaid_version}"
            )
            data["version"] = str(plaid_version)
        else:
            data["version"] = str(self._version)

        return data

        # Handle version
        plaid_version = Version(plaid.__version__)
        if self._version != plaid_version:  # pragma: no cover
            logger.warning(
                f"Version mismatch: ProblemDefinition was loaded from version {self._version if self._version is not None else 'anterior to 0.1.10'}, and will be saved with version: {plaid_version}"
            )
            data["version"] = str(plaid_version)
        else:
            data["version"] = str(self._version)

        # Save infos

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save problem information, inputs, outputs, and split to the specified file in YAML format.

        Args:
            path (Union[str,Path]): The filepath where the problem information will be saved.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                problem.save_to_file("/path/to/save_file")
        """
        problem_infos_dict = self._generate_problem_infos_dict()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")

        # Save infos
        with path.open("w") as file:
            yaml.dump(
                problem_infos_dict, file, default_flow_style=False, sort_keys=True
            )

    @deprecated(
        "`ProblemDefinition._save_to_dir_(...)` is deprecated. Use `ProblemDefinition.save_to_dir(...)` instead.",
        version="0.1.10",
        removal="0.2.0",
    )
    def _save_to_dir_(self, path: Union[str, Path]) -> None:
        """DEPRECATED: use :meth:`ProblemDefinition.save_to_dir` instead."""
        self.save_to_dir(path)

    def save_to_dir(self, path: Union[str, Path]) -> None:
        """Save problem information, inputs, outputs, and split to the specified directory in YAML and CSV formats.

        Args:
            path (Union[str,Path]): The directory where the problem information will be saved.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                problem.save_to_dir("/path/to/save_directory")
        """
        path = Path(path)

        if not (path.is_dir()):
            path.mkdir(parents=True)

        problem_infos_dict = self._generate_problem_infos_dict()

        # Save infos
        pbdef_fname = path / "problem_infos.yaml"
        with pbdef_fname.open("w") as file:
            yaml.dump(
                problem_infos_dict, file, default_flow_style=False, sort_keys=True
            )

        # Save split
        split_fname = path / "split.json"
        if self.get_split() is not None:
            with split_fname.open("w") as file:
                json.dump(self.get_split(), file)

        # # Save split
        # split_fname = path / "train_split.json"
        # if self.get_train_split() is not None:
        #     with split_fname.open("w") as file:
        #         json.dump(self.get_train_split(), file)

        # split_fname = path / "test_split.json"
        # if self.get_test_split() is not None:
        #     with split_fname.open("w") as file:
        #         json.dump(self.get_test_split(), file)

    @classmethod
    def load(cls, path: Union[str, Path]) -> Self:  # pragma: no cover
        """Load data from a specified directory.

        Args:
            path (Union[str,Path]): The path from which to load files.

        Returns:
            Self: The loaded dataset (Dataset).
        """
        instance = cls()
        instance._load_from_dir_(path)
        return instance

    def _initialize_from_problem_infos_dict(
        self, data: dict[str, Union[str, list]]
    ) -> None:
        if "version" not in data:
            self._version = None
        else:
            self._version = Version(data["version"])
        self._task = data["task"]
        self.in_features_identifiers = []
        if "input_features" in data:
            for tup in data["input_features"]:
                if isinstance(tup, dict):
                    self.in_features_identifiers.append(FeatureIdentifier(**tup))
                else:
                    self.in_features_identifiers.append(tup)
        self.out_features_identifiers = []
        if "output_features" in data:
            for tup in data["output_features"]:
                if isinstance(tup, dict):
                    self.out_features_identifiers.append(FeatureIdentifier(**tup))
                else:
                    self.out_features_identifiers.append(tup)
        self.constant_features_identifiers = []
        if "constant_features" in data:
            for tup in data["constant_features"]:
                self.constant_features_identifiers.append(tup)
        if "version" not in data or Version(data["version"]) < Version("0.2.0"):
            self.in_scalars_names = data.get("input_scalars", [])
            self.out_scalars_names = data.get("output_scalars", [])
            self.in_fields_names = data.get("input_fields", [])
            self.out_fields_names = data.get("output_fields", [])
            self.in_timeseries_names = data.get("input_timeseries", [])
            self.out_timeseries_names = data.get("output_timeseries", [])
            self.in_meshes_names = data.get("input_meshes", [])
            self.out_meshes_names = data.get("output_meshes", [])
        else:  # pragma: no cover
            old_keys = [
                "input_scalars",
                "input_fields",
                "input_timeseries",
                "input_meshes",
                "output_scalars",
                "output_fields",
                "output_timeseries",
                "output_meshes",
            ]
            for k in old_keys:
                if k in data:
                    logger.warning(
                        f"Key '{k}' is deprecated and will be ignored. You should convert your ProblemDefinition using FeatureIdentifiers to identify features instead of names."
                    )
        if "score_function" in data:
            self._score_function = data["score_function"]
        if "train_split" in data:
            self._train_split = data["train_split"]
        if "test_split" in data:
            self._test_split = data["test_split"]
        if "name" in data:
            self._name = data["name"]

    def _load_from_file_(self, path: Union[str, Path]) -> None:
        """Load problem information, inputs, outputs, and split from the specified file in YAML format.

        Args:
            path (Union[str,Path]): The filepath from which to load the problem information.

        Raises:
            FileNotFoundError: Triggered if the provided file does not exist.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                problem._load_from_file_("/path/to/load_file")
        """
        path = Path(path)

        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")

        if not path.exists():
            raise FileNotFoundError(f'File "{path}" does not exist. Abort')

        with path.open("r") as file:
            data = yaml.safe_load(file)

        self._initialize_from_problem_infos_dict(data)

    def _load_from_dir_(self, path: Union[str, Path]) -> None:
        """Load problem information, inputs, outputs, and split from the specified directory in YAML and CSV formats.

        Args:
            path (Union[str,Path]): The directory from which to load the problem information.

        Raises:
            FileNotFoundError: Triggered if the provided directory or file problem_infos.yaml does not exist
            FileExistsError: Triggered if the provided path is a file instead of a directory.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                problem._load_from_dir_("/path/to/load_directory")
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f'Directory "{path}" does not exist. Abort')

        if not path.is_dir():
            raise FileExistsError(f'"{path}" is not a directory. Abort')

        pbdef_fname = path / "problem_infos.yaml"
        data = {}  # To avoid crash if pbdef_fname does not exist
        if pbdef_fname.is_file():
            with pbdef_fname.open("r") as file:
                data = yaml.safe_load(file)
        else:
            raise FileNotFoundError(
                f"file with path `{pbdef_fname}` does not exist. Abort"
            )

        self._initialize_from_problem_infos_dict(data)

        # if it was saved with version <=0.1.7 it is a .csv else it is .json
        split = {}
        split_fname_csv = path / "split.csv"
        split_fname_json = path / "split.json"
        if split_fname_json.is_file():
            with split_fname_json.open("r") as file:
                split = json.load(file)
            if split_fname_csv.is_file():  # pragma: no cover
                logger.warning(
                    f"Both files with path `{split_fname_csv}` and `{split_fname_json}` exist. JSON file is the standard from 0.1.7 -> CSV file will be ignored"
                )
        elif split_fname_csv.is_file():  # pragma: no cover
            with split_fname_csv.open("r") as file:
                reader = csv.reader(file, delimiter=",")
                for row in reader:
                    split[row[0]] = [int(i) for i in row[1:]]
        else:  # pragma: no cover
            logger.warning(
                f"file with path `{split_fname_csv}` or `{split_fname_json}` does not exist. Splits will not be set"
            )
        self.set_split(split)

    def extract_problem_definition_from_identifiers(
        self, identifiers: Sequence[Union[str, FeatureIdentifier]]
    ) -> Self:
        """Create a new ProblemDefinition restricted to a subset of feature identifiers.

        Args:
            identifiers (Sequence[Union[str, FeatureIdentifier]]): List of identifiers to keep.

        Returns:
            ProblemDefinition: A new :class:`ProblemDefinition` instance.
        """
        new_problem_definition = ProblemDefinition()
        if self._task is not None:
            new_problem_definition.set_task(self.get_task())
        if self._name is not None:
            new_problem_definition.set_name(self.get_name())

        in_features = self.filter_in_features_identifiers(identifiers)
        if len(in_features) > 0:
            new_problem_definition.add_in_features_identifiers(in_features)

        out_features = self.filter_out_features_identifiers(identifiers)
        if len(out_features) > 0:
            new_problem_definition.add_out_features_identifiers(out_features)

        if self.get_split() is not None:
            new_problem_definition.set_split(self.get_split())

        return new_problem_definition

    # -------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """Return a string representation of the problem.

        Returns:
            str: A string representation of the overview of problem content.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                print(problem)
                >>> ProblemDefinition(input_scalars_names=['s_1'], output_scalars_names=['s_2'], input_meshes_names=['mesh'], task='regression', split_names=['train', 'val'])
        """
        str_repr = "ProblemDefinition("

        # ---# features
        if len(self.in_features_identifiers) > 0:
            in_features_identifiers = self.in_features_identifiers
            str_repr += f"{in_features_identifiers=}, "
        if len(self.out_features_identifiers) > 0:
            out_features_identifiers = self.out_features_identifiers
            str_repr += f"{out_features_identifiers=}, "

        # ---# scalars
        if len(self.in_scalars_names) > 0:
            input_scalars_names = self.in_scalars_names
            str_repr += f"{input_scalars_names=}, "
        if len(self.out_scalars_names) > 0:
            output_scalars_names = self.out_scalars_names
            str_repr += f"{output_scalars_names=}, "
        # ---# fields
        if len(self.in_fields_names) > 0:
            input_fields_names = self.in_fields_names
            str_repr += f"{input_fields_names=}, "
        if len(self.out_fields_names) > 0:
            output_fields_names = self.out_fields_names
            str_repr += f"{output_fields_names=}, "
        # ---# timeseries
        if len(self.in_timeseries_names) > 0:
            input_timeseries_names = self.in_timeseries_names
            str_repr += f"{input_timeseries_names=}, "
        if len(self.out_timeseries_names) > 0:
            output_timeseries_names = self.out_timeseries_names
            str_repr += f"{output_timeseries_names=}, "
        # ---# meshes
        if len(self.in_meshes_names) > 0:
            input_meshes_names = self.in_meshes_names
            str_repr += f"{input_meshes_names=}, "
        if len(self.out_meshes_names) > 0:
            output_meshes_names = self.out_meshes_names
            str_repr += f"{output_meshes_names=}, "
        # ---# task
        if self._task is not None:
            task = self._task
            str_repr += f"{task=}, "
        # ---# split
        if self._split is not None:
            split_names = list(self._split.keys())
            str_repr += f"{split_names=}, "

        if str_repr[-2:] == ", ":
            str_repr = str_repr[:-2]
        str_repr += ")"
        return str_repr
