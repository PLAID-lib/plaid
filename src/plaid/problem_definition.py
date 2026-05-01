"""Implementation of the `ProblemDefinition` class."""
# %% Imports

import sys

if sys.version_info >= (3, 11):
    from typing import Self
else:  # pragma: no cover
    from typing import TypeVar

    Self = TypeVar("Self")

import logging
from pathlib import Path
from typing import Any, Literal, Optional, Sequence, Union, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from .constants import (
    AUTHORIZED_SCORE_FUNCTIONS_T,
    AUTHORIZED_TASKS_T,
)
from .types import IndexType

# %% Globals

logger = logging.getLogger(__name__)

# %% Functions

# %% Classes


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
        """Load a problem definition from a YAML file located at the specified path.

        The YAML file should contain one or more problem definitions, and the desired definition can be selected by its name.

        Args:
            path (str | Path): The file path to the YAML file containing problem definitions.
            name (str | None, optional): The name of the problem definition to load. If None, it will attempt to load the
            only problem definition available in the file. Defaults to None.
            **overrides: Additional keyword arguments to override specific fields in the loaded problem definition.

        Raises:
            ValueError: If the specified name is not found in the YAML file or if multiple problem definitions are present
            without a specified name.

        Returns:
            ProblemDefinition: The loaded problem definition.
        """
        from plaid.storage import load_problem_definitions_from_disk

        all_pb_def = load_problem_definitions_from_disk(path=Path(path))
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
        """Normalize input features identifiers by ensuring they are unique and sorted."""
        if len(set(v)) != len(v):
            raise ValueError("duplicated values in input_features")
        return _normalize_list(v)

    @field_validator("train_split", "test_split", mode="after")
    @classmethod
    def check_split_has_only_one_obj(cls, v):
        """Ensure that the split dictionaries contain only one key-value pair."""
        if len(v) > 1:
            raise ValueError(
                "Splits only support one element (dict with only one object)"
            )
        return v

    @field_validator("output_features", mode="before")
    @classmethod
    def normalize_out_features_identifiers(cls, v):
        """Normalize output features identifiers by ensuring they are unique and sorted."""
        if len(set(v)) != len(v):
            raise ValueError("duplicated values in output_features")
        return _normalize_list(v)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override the default attribute setting behavior to enforce immutability for certain fields and log warnings for others."""
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
    def get_train_split_name(self) -> str:
        """Return the name of the train split."""
        if self.train_split is None:
            raise ValueError("train_split is not defined.")
        return list(self.train_split.keys())[0]

    def get_train_split_indices(self) -> IndexType | Literal["all"]:
        """Return the indices associated with the train split.

        Raises:
            ValueError: If `train_split` is not defined.

        Returns:
            IndexType | Literal["all"]: The indices associated with the train split.
        """
        if self.train_split is None:
            raise ValueError("train_split is not defined.")
        return cast(IndexType | Literal["all"], next(iter(self.train_split.values())))

    def get_test_split_name(self) -> str:
        """Return the name of the test split."""
        if self.test_split is None:
            raise ValueError("test_split is not defined.")
        return list(self.test_split.keys())[0]

    def get_test_split_indices(self) -> IndexType | Literal["all"]:
        """Return the indices associated with the test split.

        Raises:
            ValueError: If `test_split` is not defined.

        Returns:
            IndexType | Literal["all"]: The indices associated with the test split.
        """
        if self.test_split is None:
            raise ValueError("test_split is not defined.")
        return cast(IndexType | Literal["all"], next(iter(self.test_split.values())))

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

    def _generate_problem_infos_dict(self) -> dict[str, Union[str, list[str|int]]]:
        """Generate a dictionary containing all relevant problem definition data.

        Returns:
            dict[str, Union[str, list]]: A dictionary with keys for task, input/output features, scalars, fields, timeseries, and meshes.
        """
        data = {
            "task": self.task,
            "score_function": self.score_function,
            "input_features": self.input_features.copy(),
            "output_features": self.output_features.copy(),
        }

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
