"""Tests for `plaid.storage.common.writer` validation paths."""

from pathlib import Path

import pytest

from plaid.problem_definition import ProblemDefinition
from plaid.storage.common.writer import save_problem_definitions_to_disk


def _make_pb_def() -> ProblemDefinition:
    return ProblemDefinition(
        input_features=["Global/in"],
        output_features=["Global/out"],
        train_split={"train": [0]},
        test_split={"test": [0]},
    )


def test_save_problem_definitions_to_disk_rejects_non_dict_non_pbdef(
    tmp_path: Path,
) -> None:
    """Passing a non-dict, non-ProblemDefinition value should raise TypeError."""
    with pytest.raises(TypeError, match=r"dict\[str, ProblemDefinition\]"):
        save_problem_definitions_to_disk(tmp_path, [("name", _make_pb_def())])  # type: ignore[arg-type]


def test_save_problem_definitions_to_disk_rejects_non_string_identifier(
    tmp_path: Path,
) -> None:
    """Non-string / empty identifiers should raise TypeError."""
    pb_def = _make_pb_def()
    with pytest.raises(TypeError, match="non-empty strings"):
        save_problem_definitions_to_disk(tmp_path, {123: pb_def})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match="non-empty strings"):
        save_problem_definitions_to_disk(tmp_path, {"": pb_def})


def test_save_problem_definitions_to_disk_rejects_non_pbdef_value(
    tmp_path: Path,
) -> None:
    """Non-ProblemDefinition values should raise TypeError."""
    with pytest.raises(TypeError, match="ProblemDefinition instances"):
        save_problem_definitions_to_disk(tmp_path, {"pb": "not a pb_def"})  # type: ignore[dict-item]


def test_save_problem_definitions_to_disk_rejects_bare_pbdef(tmp_path: Path) -> None:
    """Passing a bare ProblemDefinition (not wrapped in a dict) should raise."""
    with pytest.raises(TypeError, match="use the dictionary key as the problem"):
        save_problem_definitions_to_disk(tmp_path, _make_pb_def())  # type: ignore[arg-type]


def test_save_problem_definitions_to_disk_writes_each_definition(
    tmp_path: Path,
) -> None:
    """Happy path: each ProblemDefinition is delegated to its `save_to_file`."""
    pb_defs = {"pb_a": _make_pb_def(), "pb_b": _make_pb_def()}

    save_problem_definitions_to_disk(tmp_path, pb_defs)

    target_dir = tmp_path / "problem_definitions"
    assert target_dir.is_dir()
    # ProblemDefinition.save_to_file serialises each definition as a YAML file.
    assert (target_dir / "pb_a.yaml").is_file()
    assert (target_dir / "pb_b.yaml").is_file()
