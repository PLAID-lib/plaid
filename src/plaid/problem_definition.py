"""Implementation of the `ProblemDefinition` class."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

# %% Imports
# Standard library
import csv
import logging
import sys
from pathlib import Path
from typing import Optional, Union

# Typing support
if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

# Third party imports
import yaml

# Local imports
from plaid.constants import (
    AUTHORIZED_FEATURE_TYPES,
    AUTHORIZED_TASKS,
    CGNS_FIELD_LOCATIONS,
)
from plaid.types import IndexType
from plaid.types.feature_types import (
    FeatureIdentifier,
    FeatureIdentifierSequence,
)

# %% Globals

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="[%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s(%(lineno)d)]:%(message)s",
    level=logging.INFO,
)

# %% Functions

# %% Classes


class ProblemDefinition(object):
    """Gathers all necessary informations to define a learning problem.

    This class defines a machine learning problem by specifying:
    1. Input and output feature identifiers
    2. Training task (regression, classification, etc.)
    3. Train/test split configuration

    Features are identified by dictionaries containing:
    - type: Feature type (scalar, field, nodes, time_series)
    - name: Feature name (required for all except nodes)
    - base_name: Base name (required for field and nodes)
    - zone_name: Zone name (required for field and nodes)
    - location: Field location (required for field)
    - time: Timestamp (optional for field and nodes)
    """

    def __init__(self, directory_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize an empty :class:`ProblemDefinition <plaid.problem_definition.ProblemDefinition>`.

        Use :meth:`add_inputs <plaid.problem_definition.ProblemDefinition.add_inputs>` or :meth:`add_outputs <plaid.problem_definition.ProblemDefinition.add_outputs>` to feed the :class:`ProblemDefinition`

        Args:
            directory_path (Union[str, Path], optional): The path from which to load PLAID problem definition files.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition

                # 1. Create empty instance of ProblemDefinition
                problem_definition = ProblemDefinition()
                print(problem_definition)
                >>> ProblemDefinition()

                # 2. Load problem definition and create ProblemDefinition instance
                problem_definition = ProblemDefinition("path_to_plaid_prob_def")
                print(problem_definition)
                >>> ProblemDefinition(inputs=[{'type': 'scalar', 'name': 's_1'}], outputs=[{'type': 'scalar', 'name': 's_2'}], task='regression')
        """
        # Core attributes
        self._task: Optional[str] = None
        self.inputs: list[FeatureIdentifier] = []
        self.outputs: list[FeatureIdentifier] = []
        self._split: Optional[dict[str, IndexType]] = None

        if directory_path is not None:
            directory_path = Path(directory_path)
            self._load_from_dir_(directory_path)

    # -------------------------------------------------------------------------#
    def get_task(self) -> Optional[str]:
        """Get the authorized task.

        Returns:
            Optional[str]: The authorized task, such as "regression" or "classification".
                Returns None if no task is set.
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
    def get_split(
        self, split_name: Optional[str] = None
    ) -> Optional[Union[IndexType, dict[str, IndexType]]]:
        """Get the split indices. This function returns the split indices, either for a specific split with the provided `split_name` or all split indices if `split_name` is not specified.

        Args:
            split_name (Optional[str]): The name of the split for which indices are requested. Defaults to None.

        Raises:
            KeyError: If `split_name` is specified but not found among split names.

        Returns:
            Optional[Union[IndexType,dict[str,IndexType]]]: If `split_name` is provided, it returns
            the indices for that split (IndexType). If `split_name` is not provided, it
            returns a dictionary mapping split names (str) to their respective indices
            (IndexType). Returns None if no splits are defined.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                split_indices = problem.get_split()
                print(split_indices)
                >>> {'train': [0, 1, 2, ...], 'test': [100, 101, ...]}

                test_indices = problem.get_split('test')
                print(test_indices)
                >>> [100, 101, ...]
        """
        if self._split is None:
            return None

        if split_name is None:
            return self._split

        if split_name not in self._split:
            raise KeyError(
                f"Split name '{split_name}' not found in available splits: {list(self._split.keys())}"
            )

        return self._split[split_name]

    def set_split(self, split: dict[str, IndexType]) -> None:
        """Set the split indices. This function allows you to set the split indices by providing a dictionary mapping split names (str) to their respective indices (IndexType).

        Args:
            split (dict[str,IndexType]):  A dictionary containing split names and their indices.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                new_split = {'train': [0, 1, 2], 'test': [3, 4]}
                problem.set_split(new_split)
        """
        if self._split is not None:  # pragma: no cover
            logger.warning("split already exists -> data will be replaced")
        self._split = split

    def get_all_indices(self) -> list[int]:
        """Get all indices from splits.

        Returns:
            list[int]: list containing all unique indices.
        """
        split = self.get_split()
        if split is None:
            return []

        all_indices: list[int] = []
        for indices in split.values():
            all_indices.extend([int(i) for i in indices])
        return list(set(all_indices))

    # -------------------------------------------------------------------------#
    def add_input(self, identifier: FeatureIdentifier) -> None:
        """Add an input feature identifier to the problem.

        Args:
            identifier (FeatureIdentifier): Feature identifier to add.
                Must be a dict containing required and optional fields:
                - type: str - Feature type ("scalar", "field", "nodes", "time_series")
                - name: str - Feature name (required for all except "nodes")
                - base_name: str - Base name (required for "field" and "nodes")
                - zone_name: str - Zone name (required for "field" and "nodes")
                - location: str - Field location (required for "field")
                - time: float - Timestamp (optional for "field" and "nodes")

        Raises:
            ValueError: If:
                - Required fields are missing for the feature type
                - Feature with same identifiers already exists
                - Invalid field location specified
                - Invalid feature type specified

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()

                # Add a scalar
                problem.add_input({'type': 'scalar', 'name': 'pressure'})

                # Add a field
                problem.add_input({
                    'type': 'field',
                    'name': 'velocity',
                    'base_name': 'Base',
                    'zone_name': 'Zone1',
                    'location': 'Vertex',
                    'time': 0.0
                })
        """
        # Verify feature type
        feat_type = identifier.get("type")
        if not feat_type or feat_type not in AUTHORIZED_FEATURE_TYPES:
            raise ValueError(f"Invalid or missing feature type: {feat_type}")

        # Verify required fields based on type
        if feat_type != "nodes" and "name" not in identifier:
            raise ValueError(f"Name is required for {feat_type} features")

        if feat_type in ["field", "nodes"]:
            if "base_name" not in identifier:
                raise ValueError(f"base_name is required for {feat_type} features")
            if "zone_name" not in identifier:
                raise ValueError(f"zone_name is required for {feat_type} features")

        if feat_type == "field":
            if "location" not in identifier:
                raise ValueError("location is required for field features")
            if identifier["location"] not in CGNS_FIELD_LOCATIONS:
                raise ValueError(f"Invalid field location: {identifier['location']}")

        # Check for duplicates
        if self._has_duplicate_identifier(identifier, self.inputs):
            name = identifier.get("name", "<no name>")
            raise ValueError(f"Duplicate {feat_type} feature with name '{name}'")

        # Add and sort
        self.inputs.append(identifier)
        self.inputs.sort(key=lambda x: (x.get("type", ""), x.get("name", "")))

    def add_inputs(self, identifiers: list[FeatureIdentifier]) -> None:
        """Add multiple input feature identifiers to the problem.

        Args:
            identifiers (list[FeatureIdentifier]): A list of input feature identifiers to add.

        Raises:
            ValueError: If any features have duplicate names.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                inputs = [
                    {'type': 'scalar', 'name': 'pressure'},
                    {'type': 'scalar', 'name': 'temperature'}
                ]
                problem.add_inputs(inputs)
        """
        names = [f.get("name") for f in identifiers]
        if not (len(set(names)) == len(names)):
            raise ValueError("Some inputs have same names")
        for identifier in identifiers:
            self.add_input(identifier)

    def get_input_identifiers(self) -> FeatureIdentifierSequence:
        """Get all input feature identifiers of the problem.

        Returns:
            FeatureIdentifierSequence: A list of input feature identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                input_identifiers = problem.get_input_identifiers()
                print(input_identifiers)
                >>> [{'type': 'scalar', 'name': 'omega'},
                     {'type': 'scalar', 'name': 'pressure'}]
        """
        return self.inputs

    def filter_input_identifiers(self, names: list[str]) -> FeatureIdentifierSequence:
        """Filter and get input features corresponding to a sorted list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding input features.

        Returns:
            FeatureIdentifierSequence: A sorted list of input feature identifiers corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                names = ['omega', 'pressure', 'temperature']
                input_features = problem.filter_input_identifiers(names)
                print(input_features)
                >>> [{'type': 'scalar', 'name': 'omega'}, {'type': 'scalar', 'name': 'pressure'}]
        """
        return sorted(
            [f for f in self.inputs if f.get("name") in names],
            key=lambda x: (x.get("type", ""), x.get("name", "")),
        )

    # -------------------------------------------------------------------------#
    def add_output(self, identifier: FeatureIdentifier) -> None:
        """Add an output feature identifier to the problem.

        Args:
            identifier (FeatureIdentifier): The output feature identifier to add.

        Raises:
            ValueError: If a feature with the same identifier is already in the outputs.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                output = {'type': 'scalar', 'name': 'pressure'}
                problem.add_output(output)
        """
        # Verify no duplicate identifier exists
        if self._has_duplicate_identifier(identifier, self.outputs):
            identifier_str = str({k: v for k, v in identifier.items() if v is not None})
            raise ValueError(
                f"A feature with identifiers {identifier_str} already exists"
            )

        # Verify feature type and required fields
        feat_type = identifier["type"]
        if feat_type not in AUTHORIZED_FEATURE_TYPES:
            raise ValueError(f"Invalid feature type: {feat_type}")

        # For field type, verify location
        if feat_type == "field":
            location = identifier.get("location")
            if location not in CGNS_FIELD_LOCATIONS:
                raise ValueError(
                    f"Invalid field location {location}, must be one of {CGNS_FIELD_LOCATIONS}"
                )

        # Add and sort
        self.outputs.append(identifier)
        self.outputs.sort(key=lambda x: (x.get("type", ""), x.get("name", "")))

    def add_outputs(self, identifiers: list[FeatureIdentifier]) -> None:
        """Add output feature identifiers to the problem.

        Args:
            identifiers (list[FeatureIdentifier]): A list of output feature identifiers to add.

        Raises:
            ValueError: if any feature names are redundant.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                outputs = [
                    {'type': 'scalar', 'name': 'compression_rate'},
                    {'type': 'scalar', 'name': 'in_massflow'}
                ]
                problem.add_outputs(outputs)
        """
        names = [f.get("name") for f in identifiers]
        if not (len(set(names)) == len(names)):
            raise ValueError("Some outputs have same names")
        for identifier in identifiers:
            self.add_output(identifier)

    def get_output_identifiers(self) -> FeatureIdentifierSequence:
        """Get the output feature identifiers of the problem.

        Returns:
            list[FeatureIdentifier]: A list of output feature identifiers.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                outputs = problem.get_output_identifiers()
                print(outputs)
                >>> [{'type': 'scalar', 'name': 'compression_rate'}, {'type': 'scalar', 'name': 'in_massflow'}]
        """
        return self.outputs

    def filter_output_identifiers(self, names: list[str]) -> FeatureIdentifierSequence:
        """Filter and get output features corresponding to a sorted list of names.

        Args:
            names (list[str]): A list of names for which to retrieve corresponding output features.

        Returns:
            FeatureIdentifierSequence: A sorted list of output feature identifiers corresponding to the provided names.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                names = ['compression_rate', 'in_massflow', 'isentropic_efficiency']
                output_features = problem.filter_output_identifiers(names)
                print(output_features)
                >>> [{'type': 'scalar', 'name': 'in_massflow'}]
        """
        return sorted(
            [f for f in self.outputs if f.get("name") in names],
            key=lambda x: (x.get("type", ""), x.get("name", "")),
        )

    # -------------------------------------------------------------------------#
    @staticmethod
    def _has_duplicate_identifier(
        identifier: FeatureIdentifier, existing: list[FeatureIdentifier]
    ) -> bool:
        """Check if an identifier would be a duplicate in the given list.

        Args:
            identifier (FeatureIdentifier): The identifier to check
            existing (list[FeatureIdentifier]): List of existing identifiers

        Returns:
            bool: True if this would be a duplicate, False otherwise
        """
        feat_type = identifier.get("type")
        if not feat_type:
            return False

        if feat_type in ["scalar", "time_series"]:
            # For scalars and time series, only name needs to match
            name = identifier.get("name")
            if not name:
                return False
            return any(
                e.get("type") == feat_type and e.get("name") == name for e in existing
            )
        else:
            # For fields and nodes, check all relevant fields
            keys_to_check = [
                "base_name",
                "zone_name",
            ]  # Required for both field and nodes
            if feat_type == "field":
                keys_to_check.extend(
                    ["name", "location"]
                )  # Additional required for fields
            elif feat_type == "nodes":
                pass  # No additional required fields for nodes
            else:
                return False  # Unknown type

            # Check optional 'time' field if present
            if "time" in identifier:
                keys_to_check.append("time")

            return any(
                e.get("type") == feat_type
                and all(
                    identifier.get(k) is not None and e.get(k) == identifier.get(k)
                    for k in keys_to_check
                )
                for e in existing
            )

    # -------------------------------------------------------------------------#
    def _save_to_dir_(self, savedir: Path) -> None:
        """Save problem information, inputs, outputs, and split to the specified directory in YAML and CSV formats.

        Args:
            savedir (Path): The directory where the problem information will be saved.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                problem._save_to_dir_("/path/to/save_directory")
        """
        if not (savedir.is_dir()):  # pragma: no cover
            savedir.mkdir()

        # Save core problem definition
        data = {
            "task": self._task,
            "inputs": self.inputs,
            "outputs": self.outputs,
        }

        pbdef_fname = savedir / "problem_infos.yaml"
        with open(pbdef_fname, "w") as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)

        # Save split information
        split_fname = savedir / "split.csv"
        if self._split is not None:
            with open(split_fname, "w", newline="") as file:
                write = csv.writer(file)
                for name, indices in self._split.items():
                    write.writerow([name] + list(indices))

    # -------------------------------------------------------------------------#
    @classmethod
    def load(cls, save_dir: Union[str, Path]) -> Self:  # pragma: no cover
        """Load a problem definition from a specified directory.

        Args:
            save_dir (Union[str, Path]): The path from which to load files.

        Returns:
            Self: The loaded ProblemDefinition.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition.load("/path/to/plaid_prob_def")
                print(problem)
                >>> ProblemDefinition(inputs=[...], outputs=[...], task='regression')
        """
        instance = cls()
        instance._load_from_dir_(Path(save_dir))
        return instance

    def _load_from_dir_(self, save_dir: Path) -> None:
        """Load problem information, inputs, outputs, and split from the specified directory in YAML and CSV formats.

        Args:
            save_dir (Path): The directory from which to load the problem information.

        Raises:
            FileNotFoundError: Triggered if the provided directory does not exist.
            FileExistsError: Triggered if the provided path is a file instead of a directory.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                problem._load_from_dir_("/path/to/load_directory")
        """
        if not save_dir.exists():  # pragma: no cover
            raise FileNotFoundError(f'Directory "{save_dir}" does not exist. Abort')

        if not save_dir.is_dir():  # pragma: no cover
            raise FileExistsError(f'"{save_dir}" is not a directory. Abort')

        # Load core problem definition
        pbdef_fname = save_dir / "problem_infos.yaml"
        if pbdef_fname.is_file():
            with open(pbdef_fname, "r") as file:
                data = yaml.safe_load(file) or {}

            # Load task
            self._task = data.get("task")

            # Load feature identifiers
            for identifier in data.get("inputs", []):
                self.add_input(identifier)
            for identifier in data.get("outputs", []):
                self.add_output(identifier)
        else:  # pragma: no cover
            logger.warning(
                f"file with path `{pbdef_fname}` does not exist. Task, inputs, and outputs will not be set"
            )

        # Load split information
        split_fname = save_dir / "split.csv"
        split = {}
        if split_fname.is_file():
            with open(split_fname) as file:
                reader = csv.reader(file, delimiter=",")
                for row in reader:
                    split[row[0]] = [int(i) for i in row[1:]]
        else:  # pragma: no cover
            logger.warning(
                f"file with path `{split_fname}` does not exist. Splits will not be set"
            )
        self._split = split

    # -------------------------------------------------------------------------#
    def __repr__(self) -> str:
        """Return a string representation of the problem.

        Returns:
            str: A string representation of the overview of problem content.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                # [...]
                print(problem)
                >>> ProblemDefinition(inputs=[{'type': 'scalar', 'name': 's_1'}], outputs=[{'type': 'scalar', 'name': 's_2'}], task='regression', split_name=['train', 'val'])
        """
        parts = []

        # Add non-empty feature lists
        if self.inputs:
            parts.append(f"inputs={self.inputs}")
        if self.outputs:
            parts.append(f"outputs={self.outputs}")

        # Add task if set
        if self._task is not None:
            parts.append(f"task='{self._task}'")

        # Add split information if available
        if self._split is not None:
            parts.append(f"split_names={list(self._split.keys())}")

        return f"ProblemDefinition({', '.join(parts)})"
