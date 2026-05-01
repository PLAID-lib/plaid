"""Implementation of the `ProblemDefinition` class."""
# %% Imports

from os import name, path
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

from .constants import (
    AUTHORIZED_SCORE_FUNCTIONS,
    AUTHORIZED_TASKS,
    AUTHORIZED_TASKS_T,
    AUTHORIZED_SCORE_FUNCTIONS_T,
)

from .types import IndexType

# %% Globals

logger = logging.getLogger(__name__)

# %% Functions

# %% Classes

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator


def _normalize_list(v):
    return sorted(map(str, v))


class ProblemDefinition(BaseModel):
    """Defines the input and output features for a machine learning problem."""

    model_config = ConfigDict(
        revalidate_instances="always", validate_assignment=True, extra="forbid"
    )

    name: Optional[str] = Field(default=None)
    task: Optional[AUTHORIZED_TASKS_T] = Field(default=None)
    input_features: list[str] = Field(default_factory=list)
    output_features: list[str] = Field(default_factory=list)
    score_function: Optional[AUTHORIZED_SCORE_FUNCTIONS_T] = Field(default=None)
    train_split: Optional[dict[str, Sequence[int] | Literal["all"]]] = Field(
        default=None
    )
    test_split: Optional[dict[str, Sequence[int] | Literal["all"]]] = Field(
        default=None
    )

    # verifier que tab autotocopleate marche bien dans vscode

    @staticmethod
    def from_path(
        path: str | Path, name: str | None = None, **overrides
    ) -> "ProblemDefinition":

        from plaid.storage import load_problem_definitions_from_disk

        all_pb_def = load_problem_definitions_from_disk(path=Path(path))
        name = overrides.get("name", name)
        available = ", ".join(sorted(all_pb_def))
        if name is not None:
            if name not in all_pb_def:
                raise ValueError(
                    f"Problem definition '{name}' not found in {path}. "
                    f"Available definitions: {available}"
                )
            data2 = all_pb_def[name].model_dump()
            data2.update(overrides)
            data = data2
        else:
            if len(all_pb_def) > 1:
                raise RuntimeError(
                    f"Non name specified, but more than one Problem definition. Available definitions: {available}"
                )
            else:
                data2 = next(iter(all_pb_def.values())).model_dump()
                data2.update(overrides)
                data = data2

        # return data
        return ProblemDefinition(**data)

    @field_validator("input_features", mode="before")
    @classmethod
    def normalize_in_features_identifiers(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("duplicated values in input_features")
        return _normalize_list(v)

    @field_validator("train_split", "test_split", mode="after")
    @classmethod
    def check_split_has_only_one_obj(cls, v):
        if len(v) > 1:
            raise ValueError(
                "Splits only support one element (dict with only one object)"
            )
        return v

    @field_validator("output_features", mode="before")
    @classmethod
    def normalize_out_features_identifiers(cls, v):
        if len(set(v)) != len(v):
            raise ValueError("duplicated values in output_features")
        return _normalize_list(v)

    def __setattr__(self, name: str, value: Any) -> None:
        # to set the name, task, score_function only once and oly once
        if name in ["name", "task", "score_function"]:
            current_value = getattr(self, name, None)
            if (
                current_value is not None
                and value is not None
                and current_value != value
            ):
                raise AttributeError(f"'{name}' is already set and cannot be changed.")
        # warning if set
        if name in ["train_split", "test_split"]:
            current_value = getattr(self, name, None)
            if (
                current_value is not None
                and value is not None
                and current_value != value
            ):
                logger.warning("'%s' already exists -> data will be replaced", name)

        super().__setattr__(name, value)

    # def get_name(self) -> str | None:
    #    return self.name

    #     # -------------------------------------------------------------------------#

    def get_split(self, indices_name: Optional[str] = None) -> dict[str, IndexType]:
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
        if indices_name is not None:
            if indices_name == "train":
                return next(iter(self.train_split.values()))
            elif indices_name == "test":
                return next(iter(self.test_split.values()))
            else:
                raise ValueError(
                    f'indices_name can be None, "train" or "test". given "{indices_name}"'
                )
        res = {}

        if self.train_split != None:
            assert len(self.train_split) == 1
            train_split_ids = next(iter(self.train_split.values()))
            res["train"] = train_split_ids

        if self.test_split != None:
            assert len(self.test_split) == 1
            test_split_ids = next(iter(self.test_split.values()))
            res["test"] = test_split_ids

        return res


    def add_in_features_identifiers(self, inputs: Union[str, Sequence[str]]) -> None:
        """Add input features identifiers to the problem.

        Args:
            inputs (Sequence[str] or str ): A list of or a single input feature identifier to add.

        Raises:
            ValueError: If some :code:`inputs` are duplicated.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                in_features_identifiers = ['omega', 'pressure']
                problem.add_in_features_identifiers(in_features_identifiers)

                # or for a single feature

                problem.add_in_features_identifiers("angle")
        """

        if isinstance(inputs, str):
            input_feature = inputs
            if input_feature in self.input_features:
                raise ValueError(
                    f"{input_feature} is already in self.input_features"
                )

            self.input_features.append(input_feature)
            self.input_features.sort()
            return

        if not (len(set(inputs)) == len(inputs)):
            raise ValueError("Some input features share the same identifier")

        for input_feature in inputs:
            self.add_in_features_identifiers(input_feature)

    def add_out_features_identifiers(self, outputs: Union[str, Sequence[str]]) -> None:
        """Add output features identifiers to the problem.

        Args:
            outputs (Sequence[str] or str ): A list of or a single input feature identifier to add.

        Raises:
            ValueError: If some :code:`outputs` are duplicated.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition()
                out_features_identifiers = ['omega', 'pressure']
                problem.add_out_features_identifiers(out_features_identifiers)

                # or for a single feature

                problem.add_out_features_identifiers("angle")
        """

        if isinstance(outputs, str):
            output_feature = outputs
            if output_feature in self.output_features:
                raise ValueError(
                    f"{output_feature} is already in self.output_features"
                )

            self.output_features.append(output_feature)
            self.output_features.sort()
            return

        if not (len(set(outputs)) == len(outputs)):
            raise ValueError("Some output features share the same identifier")

        for output_feature in outputs:
            self.add_out_features_identifiers(output_feature)

    def _generate_problem_infos_dict(self) -> dict[str, Union[str, list]]:
        """Generate a dictionary containing all relevant problem definition data.

        Returns:
            dict[str, Union[str, list]]: A dictionary with keys for task, input/output features, scalars, fields, timeseries, and meshes.
        """
        data = {
            "task": self.task,
            "score_function": self.score_function,
            "input_features": [],
            "output_features": [],
        }
        for tup in self.input_features:
            data["input_features"].append(tup)

        for tup in self.output_features:
            data["output_features"].append(tup)


        if self.train_split is not None:
            data["train_split"] = self.train_split

        if self.test_split is not None:
            data["test_split"] = self.test_split
        if self.name is not None:
            data["name"] = self.name

        # Handle version
        return data


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

        for key, value in data.items():
            if key in type(self).model_fields.keys():
                setattr(self, key, value)
            else:
                logger.warning(f" Data ignored! : {key}: {value}")
