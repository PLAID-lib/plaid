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
from typing import Any, Literal, Sequence, Union

import yaml
from pydantic import BaseModel, field_validator

# %% Globals

logger = logging.getLogger(__name__)

_KEY_ORDER = [
    "input_features",
    "output_features",
    "train_split",
    "test_split",
]


def _normalize_list(v):
    return sorted(map(str, v))


class ProblemDefinition(
    BaseModel,
    revalidate_instances="always",
    str_strip_whitespace=True,
    validate_assignment=True,
    extra="forbid",
):
    """Defines the input and output features for a machine learning problem."""

    input_features: list[str]
    output_features: list[str]
    train_split: dict[str, Sequence[int] | Literal["all"]]
    test_split: dict[str, Sequence[int] | Literal["all"]]

    @classmethod
    def from_path(cls, path: str | Path) -> "ProblemDefinition":
        """Load and validate one problem definition from a YAML file.

        Args:
            path: Path to the problem-definition YAML file. If no suffix is
                provided, ``.yaml`` is appended.

        Returns:
            Validated problem definition instance.

        Raises:
            FileNotFoundError: If the resolved YAML file does not exist.
            IsADirectoryError: If ``path`` points to a directory.
        """
        path = Path(path)
        if path.is_dir():
            raise IsADirectoryError(
                f'Expected a YAML file path, got directory "{path}"'
            )
        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")
        if not path.exists():
            raise FileNotFoundError(f'File "{path}" does not exist. Abort')

        with path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file) or {}

        return cls.model_validate(data)

    @field_validator("input_features", mode="before")
    @classmethod
    def normalize_input_features(cls, v):
        """Normalize input features identifiers by ensuring they are unique and sorted."""
        if not v:
            raise ValueError("input_features must not be empty")
        if len(set(v)) != len(v):
            raise ValueError("duplicated values in input_features")
        return _normalize_list(v)

    @field_validator("output_features", mode="before")
    @classmethod
    def normalize_output_features(cls, v):
        """Normalize output features identifiers by ensuring they are unique and sorted."""
        if not v:
            raise ValueError("output_features must not be empty")
        if len(set(v)) != len(v):
            raise ValueError("duplicated values in output_features")
        return _normalize_list(v)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override attribute setting to log warnings when split fields are replaced."""
        if name in ["train_split", "test_split"]:
            current_value = getattr(self, name, None)
            if (
                current_value is not None
                and value is not None
                and current_value != value
            ):
                logger.warning("'%s' already exists -> data will be replaced", name)

        super().__setattr__(name, value)

    def add_input_features(self, inputs: Union[str, Sequence[str]]) -> None:
        """Add input features identifiers to the problem.

        Args:
            inputs (Sequence[str] or str ): A list of or a single input feature identifier to add.

        Raises:
            ValueError: If some :code:`inputs` are duplicated.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition(
                    input_features=["angle"],
                    output_features=["pressure"],
                    train_split={"train": "all"},
                    test_split={"test": "all"},
                )
                input_features = ['omega', 'pressure']
                problem.add_input_features(input_features)

                # or for a single feature

                problem.add_input_features("angle")
        """
        if isinstance(inputs, str):
            input_feature = inputs
            if input_feature in self.input_features:
                raise ValueError(f"{input_feature} is already in self.input_features")

            self.input_features.append(input_feature)
            self.input_features.sort()
            return

        if not (len(set(inputs)) == len(inputs)):
            raise ValueError("Some input features share the same identifier")

        for input_feature in inputs:
            self.add_input_features(input_feature)

    def add_output_features(self, outputs: Union[str, Sequence[str]]) -> None:
        """Add output features identifiers to the problem.

        Args:
            outputs (Sequence[str] or str ): A list of or a single input feature identifier to add.

        Raises:
            ValueError: If some :code:`outputs` are duplicated.

        Example:
            .. code-block:: python

                from plaid.problem_definition import ProblemDefinition
                problem = ProblemDefinition(
                    input_features=["angle"],
                    output_features=["pressure"],
                    train_split={"train": "all"},
                    test_split={"test": "all"},
                )
                output_features = ['omega', 'pressure']
                problem.add_output_features(output_features)

                # or for a single feature

                problem.add_output_features("angle")
        """
        if isinstance(outputs, str):
            output_feature = outputs
            if output_feature in self.output_features:
                raise ValueError(f"{output_feature} is already in self.output_features")

            self.output_features.append(output_feature)
            self.output_features.sort()
            return

        if not (len(set(outputs)) == len(outputs)):
            raise ValueError("Some output features share the same identifier")

        for output_feature in outputs:
            self.add_output_features(output_feature)

    def save_to_file(self, path: Union[str, Path]) -> None:
        """Save problem information, inputs, outputs, and split to the specified file in YAML format.

        Args:
            path (Union[str,Path]): The filepath where the problem information will be saved.

        Example:
            .. code-block:: python

                from plaid import ProblemDefinition
                problem = ProblemDefinition(
                    input_features=["angle"],
                    output_features=["pressure"],
                    train_split={"train": "all"},
                    test_split={"test": "all"},
                )
                problem.save_to_file("/path/to/save_file")
        """
        path = Path(path)
        if path.is_dir():
            raise IsADirectoryError(
                f'Expected a YAML file path, got directory "{path}"'
            )

        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")

        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump()
        ordered_data = {key: data[key] for key in _KEY_ORDER if key in data}

        # Save infos
        with path.open("w") as file:
            yaml.safe_dump(
                ordered_data,
                file,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )
