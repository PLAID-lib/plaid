"""Problem definition schema based on Pydantic."""

# -*- coding: utf-8 -*-
#
# This file is subject to the terms and conditions defined in
# file 'LICENSE.txt', which is part of this source code package.
#
#

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import yaml
from packaging.version import Version
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

import plaid
from plaid.constants import AUTHORIZED_SCORE_FUNCTIONS, AUTHORIZED_TASKS
from plaid.containers import FeatureIdentifier
from plaid.types import IndexType

logger = logging.getLogger(__name__)


def _feature_sort_key(feat: Union[str, FeatureIdentifier]) -> tuple[str, str]:
    if isinstance(feat, str):
        return ("a_string", feat)
    return ("b_feature", feat["type"])


class ProblemDefinition(BaseModel):
    """Canonical representation of a learning problem."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid",
    )

    name: Optional[str] = None
    version: Version = Field(default_factory=lambda: Version(plaid.__version__))
    task: Optional[str] = None
    score_function: Optional[str] = None
    input_features: list[Union[str, FeatureIdentifier]] = Field(default_factory=list)
    output_features: list[Union[str, FeatureIdentifier]] = Field(default_factory=list)
    constant_features: list[str] = Field(default_factory=list)
    input_scalars: list[str] = Field(default_factory=list)
    output_scalars: list[str] = Field(default_factory=list)
    input_fields: list[str] = Field(default_factory=list)
    output_fields: list[str] = Field(default_factory=list)
    input_timeseries: list[str] = Field(default_factory=list)
    output_timeseries: list[str] = Field(default_factory=list)
    input_meshes: list[str] = Field(default_factory=list)
    output_meshes: list[str] = Field(default_factory=list)
    split: Optional[dict[str, IndexType]] = None
    train_split: Optional[dict[str, dict[str, IndexType]]] = None
    test_split: Optional[dict[str, dict[str, IndexType]]] = None

    # Validators / serializers
    @field_validator("version", mode="before")
    @classmethod
    def _coerce_version(cls, value: Optional[Union[str, Version]]) -> Optional[Version]:
        if value is None:
            return Version(plaid.__version__)
        if isinstance(value, Version):
            return value
        return Version(value)

    @field_validator("task")
    @classmethod
    def _validate_task(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in AUTHORIZED_TASKS:
            raise ValueError(
                f"{value} not among authorized tasks. Maybe you want to try among: {AUTHORIZED_TASKS}"
            )
        return value

    @field_validator("score_function")
    @classmethod
    def _validate_score_function(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in AUTHORIZED_SCORE_FUNCTIONS:
            raise ValueError(
                f"{value} not among authorized tasks. Maybe you want to try among: {AUTHORIZED_SCORE_FUNCTIONS}"
            )
        return value

    @field_validator("input_features", "output_features", mode="before")
    @classmethod
    def _coerce_features(
        cls, value: Optional[Sequence[Union[str, FeatureIdentifier, dict]]]
    ) -> list[Union[str, FeatureIdentifier]]:
        if value is None:
            return []
        coerced: list[Union[str, FeatureIdentifier]] = []
        for item in value:
            if isinstance(item, dict):
                coerced.append(FeatureIdentifier(**item))
            else:
                coerced.append(item)
        return coerced

    @field_serializer("version")
    def _serialize_version(self, value: Optional[Version]) -> Optional[str]:
        return str(value) if value is not None else None

    @field_serializer("input_features", "output_features")
    def _serialize_features(
        self, value: list[Union[str, FeatureIdentifier]]
    ) -> list[Union[str, dict]]:
        serialized: list[Union[str, dict]] = []
        for item in value:
            if isinstance(item, FeatureIdentifier):
                serialized.append(dict(**item))
            else:
                serialized.append(item)
        return serialized

    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        """Validate and possibly load from file."""
        if isinstance(obj, (str, Path)):
            return cls.load(obj)
        return super().model_validate(obj, *args, **kwargs)

    def __init__(
        self,
        path: Optional[Union[str, Path]] = None,
        directory_path: Optional[Union[str, Path]] = None,
        **data,
    ):
        """Create a problem definition, optionally loading it from disk."""
        if path is not None and directory_path is not None:
            raise ValueError(
                "Arguments `path` and `directory_path` cannot be both set. Use only `path`."
            )
        load_path = directory_path or path
        if load_path is not None:
            loaded = self.load(load_path)
            super().__init__(**loaded.model_dump())
        else:
            super().__init__(**data)

    # Basic setters/getters -------------------------------------------------
    def get_name(self) -> Optional[str]:
        """Return the problem name."""
        return self.name

    def set_name(self, name: str) -> None:
        """Set the problem name once."""
        if self.name is not None:
            raise ValueError(f"A name is already set (`{self.name}`)")
        self.name = name

    def get_version(self) -> Version:
        """Return the stored version."""
        return self.version

    def get_task(self) -> Optional[str]:
        """Return the task type."""
        return self.task

    def set_task(self, task: str) -> None:
        """Set the task, enforcing allowed values and preventing overwrite."""
        if self.task is not None:
            raise ValueError(f"A task is already set (`{self.task}`)")
        if task not in AUTHORIZED_TASKS:
            raise TypeError(
                f"{task} not among authorized tasks. Maybe you want to try among: {AUTHORIZED_TASKS}"
            )
        self.task = task

    def get_score_function(self) -> Optional[str]:
        """Return the score function."""
        return self.score_function

    def set_score_function(self, score_function: str) -> None:
        """Set the score function, enforcing allowed values and preventing overwrite."""
        if self.score_function is not None:
            raise ValueError(
                f"A score function is already set (`{self.score_function}`)"
            )
        if score_function not in AUTHORIZED_SCORE_FUNCTIONS:
            raise TypeError(
                f"{score_function} not among authorized tasks. Maybe you want to try among: {AUTHORIZED_SCORE_FUNCTIONS}"
            )
        self.score_function = score_function

    # Feature helpers -------------------------------------------------------
    def get_in_features_identifiers(self) -> list[Union[str, FeatureIdentifier]]:
        """Return input feature identifiers."""
        return list(self.input_features)

    def add_in_features_identifiers(
        self, inputs: Sequence[Union[str, FeatureIdentifier]]
    ) -> None:
        """Add multiple input feature identifiers, rejecting duplicates."""
        if len(set(inputs)) != len(inputs):
            raise ValueError("Some inputs have same identifiers")
        for inp in inputs:
            self.add_in_feature_identifier(inp)

    def add_in_feature_identifier(self, input: Union[str, FeatureIdentifier]) -> None:
        """Add a single input feature identifier."""
        if input in self.input_features:
            raise ValueError(f"{input} is already in input_features")
        self.input_features.append(input)
        self.input_features.sort(key=_feature_sort_key)

    def filter_in_features_identifiers(
        self, identifiers: Sequence[Union[str, FeatureIdentifier]]
    ) -> list[Union[str, FeatureIdentifier]]:
        """Return registered input identifiers matching the provided list."""
        return sorted(
            set(identifiers).intersection(self.get_in_features_identifiers()),
            key=_feature_sort_key,
        )

    def get_out_features_identifiers(self) -> list[Union[str, FeatureIdentifier]]:
        """Return output feature identifiers."""
        return list(self.output_features)

    def add_out_features_identifiers(
        self, outputs: Sequence[Union[str, FeatureIdentifier]]
    ) -> None:
        """Add multiple output feature identifiers, rejecting duplicates."""
        if len(set(outputs)) != len(outputs):
            raise ValueError("Some outputs have same identifiers")
        for out in outputs:
            self.add_out_feature_identifier(out)

    def add_out_feature_identifier(self, output: Union[str, FeatureIdentifier]) -> None:
        """Add a single output feature identifier."""
        if output in self.output_features:
            raise ValueError(f"{output} is already in output_features")
        self.output_features.append(output)
        self.output_features.sort(key=_feature_sort_key)

    def filter_out_features_identifiers(
        self, identifiers: Sequence[Union[str, FeatureIdentifier]]
    ) -> list[Union[str, FeatureIdentifier]]:
        """Return registered output identifiers matching the provided list."""
        return sorted(
            set(identifiers).intersection(self.get_out_features_identifiers()),
            key=_feature_sort_key,
        )

    def get_constant_features_identifiers(self) -> list[str]:
        """Return constant feature identifiers."""
        return list(self.constant_features)

    def add_constant_features_identifiers(self, inputs: Sequence[str]) -> None:
        """Add multiple constant feature identifiers, rejecting duplicates."""
        if len(set(inputs)) != len(inputs):
            raise ValueError("Some inputs have same identifiers")
        for inp in inputs:
            self.add_constant_feature_identifier(inp)

    def add_constant_feature_identifier(self, input: str) -> None:
        """Add a single constant feature identifier."""
        if input in self.constant_features:
            raise ValueError(f"{input} is already in constant_features")
        self.constant_features.append(input)
        self.constant_features.sort()

    def filter_constant_features_identifiers(
        self, identifiers: Sequence[str]
    ) -> list[str]:
        """Return registered constant identifiers matching the provided list."""
        return sorted(
            set(identifiers).intersection(self.get_constant_features_identifiers())
        )

    # Legacy name-based helpers --------------------------------------------
    def get_input_scalars_names(self) -> list[str]:
        """Return input scalar names (legacy)."""
        return list(self.input_scalars)

    def add_input_scalars_names(self, inputs: Sequence[str]) -> None:
        """Add input scalar names (legacy)."""
        if len(set(inputs)) != len(inputs):
            raise ValueError("Some inputs have same names")
        for inp in inputs:
            self.add_input_scalar_name(inp)

    def add_input_scalar_name(self, input: str) -> None:
        """Add a single input scalar name (legacy)."""
        if input in self.input_scalars:
            raise ValueError(f"{input} is already in input_scalars")
        self.input_scalars.append(input)
        self.input_scalars.sort()

    def filter_input_scalars_names(self, names: Sequence[str]) -> list[str]:
        """Filter input scalar names (legacy)."""
        return sorted(set(names).intersection(self.get_input_scalars_names()))

    def get_output_scalars_names(self) -> list[str]:
        """Return output scalar names (legacy)."""
        return list(self.output_scalars)

    def add_output_scalars_names(self, outputs: Sequence[str]) -> None:
        """Add output scalar names (legacy)."""
        if len(set(outputs)) != len(outputs):
            raise ValueError("Some outputs have same names")
        for out in outputs:
            self.add_output_scalar_name(out)

    def add_output_scalar_name(self, output: str) -> None:
        """Add a single output scalar name (legacy)."""
        if output in self.output_scalars:
            raise ValueError(f"{output} is already in output_scalars")
        self.output_scalars.append(output)
        self.output_scalars.sort()

    def filter_output_scalars_names(self, names: Sequence[str]) -> list[str]:
        """Filter output scalar names (legacy)."""
        return sorted(set(names).intersection(self.get_output_scalars_names()))

    def get_input_fields_names(self) -> list[str]:
        """Return input field names (legacy)."""
        return list(self.input_fields)

    def add_input_fields_names(self, inputs: Sequence[str]) -> None:
        """Add input field names (legacy)."""
        if len(set(inputs)) != len(inputs):
            raise ValueError("Some inputs have same names")
        for inp in inputs:
            self.add_input_field_name(inp)

    def add_input_field_name(self, input: str) -> None:
        """Add a single input field name (legacy)."""
        if input in self.input_fields:
            raise ValueError(f"{input} is already in input_fields")
        self.input_fields.append(input)
        self.input_fields.sort()

    def filter_input_fields_names(self, names: Sequence[str]) -> list[str]:
        """Filter input field names (legacy)."""
        return sorted(set(names).intersection(self.get_input_fields_names()))

    def get_output_fields_names(self) -> list[str]:
        """Return output field names (legacy)."""
        return list(self.output_fields)

    def add_output_fields_names(self, outputs: Sequence[str]) -> None:
        """Add output field names (legacy)."""
        if len(set(outputs)) != len(outputs):
            raise ValueError("Some outputs have same names")
        for out in outputs:
            self.add_output_field_name(out)

    def add_output_field_name(self, output: str) -> None:
        """Add a single output field name (legacy)."""
        if output in self.output_fields:
            raise ValueError(f"{output} is already in output_fields")
        self.output_fields.append(output)
        self.output_fields.sort()

    def filter_output_fields_names(self, names: Sequence[str]) -> list[str]:
        """Filter output field names (legacy)."""
        return sorted(set(names).intersection(self.get_output_fields_names()))

    def get_input_timeseries_names(self) -> list[str]:
        """Return input timeseries names (legacy)."""
        return list(self.input_timeseries)

    def add_input_timeseries_names(self, inputs: Sequence[str]) -> None:
        """Add input timeseries names (legacy)."""
        if len(set(inputs)) != len(inputs):
            raise ValueError("Some inputs have same names")
        for inp in inputs:
            self.add_input_timeseries_name(inp)

    def add_input_timeseries_name(self, input: str) -> None:
        """Add a single input timeseries name (legacy)."""
        if input in self.input_timeseries:
            raise ValueError(f"{input} is already in input_timeseries")
        self.input_timeseries.append(input)
        self.input_timeseries.sort()

    def filter_input_timeseries_names(self, names: Sequence[str]) -> list[str]:
        """Filter input timeseries names (legacy)."""
        return sorted(set(names).intersection(self.get_input_timeseries_names()))

    def get_output_timeseries_names(self) -> list[str]:
        """Return output timeseries names (legacy)."""
        return list(self.output_timeseries)

    def add_output_timeseries_names(self, outputs: Sequence[str]) -> None:
        """Add output timeseries names (legacy)."""
        if len(set(outputs)) != len(outputs):
            raise ValueError("Some outputs have same names")
        for out in outputs:
            self.add_output_timeseries_name(out)

    def add_output_timeseries_name(self, output: str) -> None:
        """Add a single output timeseries name (legacy)."""
        if output in self.output_timeseries:
            raise ValueError(f"{output} is already in output_timeseries")
        self.output_timeseries.append(output)
        self.output_timeseries.sort()

    def filter_output_timeseries_names(self, names: Sequence[str]) -> list[str]:
        """Filter output timeseries names (legacy)."""
        return sorted(set(names).intersection(self.get_output_timeseries_names()))

    def get_input_meshes_names(self) -> list[str]:
        """Return input mesh names (legacy)."""
        return list(self.input_meshes)

    def add_input_meshes_names(self, inputs: Sequence[str]) -> None:
        """Add input mesh names (legacy)."""
        if len(set(inputs)) != len(inputs):
            raise ValueError("Some inputs have same names")
        for inp in inputs:
            self.add_input_mesh_name(inp)

    def add_input_mesh_name(self, input: str) -> None:
        """Add a single input mesh name (legacy)."""
        if input in self.input_meshes:
            raise ValueError(f"{input} is already in input_meshes")
        self.input_meshes.append(input)
        self.input_meshes.sort()

    def filter_input_meshes_names(self, names: Sequence[str]) -> list[str]:
        """Filter input mesh names (legacy)."""
        return sorted(set(names).intersection(self.get_input_meshes_names()))

    def get_output_meshes_names(self) -> list[str]:
        """Return output mesh names (legacy)."""
        return list(self.output_meshes)

    def add_output_meshes_names(self, outputs: Sequence[str]) -> None:
        """Add output mesh names (legacy)."""
        if len(set(outputs)) != len(outputs):
            raise ValueError("Some outputs have same names")
        for out in outputs:
            self.add_output_mesh_name(out)

    def add_output_mesh_name(self, output: str) -> None:
        """Add a single output mesh name (legacy)."""
        if output in self.output_meshes:
            raise ValueError(f"{output} is already in output_meshes")
        self.output_meshes.append(output)
        self.output_meshes.sort()

    def filter_output_meshes_names(self, names: Sequence[str]) -> list[str]:
        """Filter output mesh names (legacy)."""
        return sorted(set(names).intersection(self.get_output_meshes_names()))

    # Splits ----------------------------------------------------------------
    def get_split(
        self, indices_name: Optional[str] = None
    ) -> Union[IndexType, dict[str, IndexType], None]:
        """Return the full split or a named subset."""
        if self.split is None:
            return None
        if indices_name is None:
            return self.split
        if indices_name not in self.split:
            raise KeyError(indices_name + " not among split indices names")
        return self.split[indices_name]

    def set_split(self, split: dict[str, IndexType]) -> None:
        """Set the main split mapping."""
        if self.split is not None:
            logger.warning("split already exists -> data will be replaced")
        self.split = split

    def get_train_split(
        self, indices_name: Optional[str] = None
    ) -> Union[dict[str, IndexType], dict[str, dict[str, IndexType]], None]:
        """Return the train split dictionary or a named subset."""
        if self.train_split is None:
            return None
        if indices_name is None:
            return self.train_split
        if indices_name not in self.train_split:
            raise KeyError(indices_name + " not among split indices names")
        return self.train_split[indices_name]

    def set_train_split(self, split: dict[str, dict[str, IndexType]]) -> None:
        """Set the train split mapping."""
        if self.train_split is not None:
            logger.warning("train_split already exists -> data will be replaced")
        self.train_split = split

    def get_test_split(
        self, indices_name: Optional[str] = None
    ) -> Union[dict[str, IndexType], dict[str, dict[str, IndexType]], None]:
        """Return the test split dictionary or a named subset."""
        if self.test_split is None:
            return None
        if indices_name is None:
            return self.test_split
        if indices_name not in self.test_split:
            raise KeyError(indices_name + " not among split indices names")
        return self.test_split[indices_name]

    def set_test_split(self, split: dict[str, dict[str, IndexType]]) -> None:
        """Set the test split mapping."""
        if self.test_split is not None:
            logger.warning("test_split already exists -> data will be replaced")
        self.test_split = split

    def get_all_indices(self) -> list[int]:
        """Return the set of all indices present in the main split."""
        if self.split is None:
            return []
        all_indices: list[int] = []
        for indices in self.split.values():
            all_indices += list(indices)
        return list(set(all_indices))

    # Persistence -----------------------------------------------------------
    def save_to_file(self, path: Union[str, Path]) -> None:
        """Persist the problem definition to a single YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")
        with path.open("w") as file:
            yaml.dump(self.model_dump(exclude_none=True), file, sort_keys=True)

    def save_to_dir(self, path: Union[str, Path]) -> None:
        """Persist the problem definition to a directory (single YAML)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.save_to_file(path / "problem_infos.yaml")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ProblemDefinition":
        """Load a problem definition from a file or directory."""
        path = Path(path)
        if path.is_dir():
            return cls._load_from_dir(path)
        return cls._load_from_file(path)

    @classmethod
    def _load_from_file(cls, path: Union[str, Path]) -> "ProblemDefinition":
        """Load a problem definition from a YAML file."""
        path = Path(path)
        if path.suffix != ".yaml":
            path = path.with_suffix(".yaml")
        if not path.exists():
            raise FileNotFoundError(f'File "{path}" does not exist. Abort')
        with path.open("r") as file:
            data = yaml.safe_load(file) or {}
        return cls.model_validate(data)

    @classmethod
    def _load_from_dir(cls, path: Union[str, Path]) -> "ProblemDefinition":
        """Load a problem definition from a directory layout."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Directory "{path}" does not exist. Abort')
        if not path.is_dir():
            raise FileExistsError(f'"{path}" is not a directory. Abort')

        pbdef_fname = path / "problem_infos.yaml"
        if not pbdef_fname.is_file():
            raise FileNotFoundError(
                f"file with path `{pbdef_fname}` does not exist. Abort"
            )
        with pbdef_fname.open("r") as file:
            data = yaml.safe_load(file) or {}

        if "split" not in data:
            split_json = path / "split.json"
            if split_json.is_file():
                with split_json.open("r") as file:
                    data["split"] = json.load(file)
            else:
                split_csv = path / "split.csv"
                if split_csv.is_file():  # pragma: no cover
                    import csv as _csv

                    split: dict[str, list[int]] = {}
                    with split_csv.open("r") as file:
                        reader = _csv.reader(file, delimiter=",")
                        for row in reader:
                            split[row[0]] = [int(i) for i in row[1:]]
                    data["split"] = split

        return cls.model_validate(data)

    # Representation --------------------------------------------------------
    def __repr__(self) -> str:
        """Return a concise string representation of the problem definition."""
        pieces = []
        if self.input_features:
            pieces.append(f"input_features={self.input_features}")
        if self.output_features:
            pieces.append(f"output_features={self.output_features}")
        if self.constant_features:
            pieces.append(f"constant_features={self.constant_features}")
        if self.task:
            pieces.append(f"task='{self.task}'")
        if self.split:
            pieces.append(f"split_names={list(self.split.keys())}")
        if self.name:
            pieces.append(f"name='{self.name}'")
        joined = ", ".join(pieces)
        return f"ProblemDefinition({joined})"
