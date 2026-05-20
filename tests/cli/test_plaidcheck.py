"""Tests for the plaidcheck CLI and checker helpers."""

import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from plaid.cli import plaidcheck
from plaid.cli.plaidcheck import (
    _check_numeric_content,
    _is_branch_without_data,
    _is_branch_without_data_in_mapping,
    check_dataset,
    main,
)


def _copy_reference_dataset(tmp_path: Path) -> Path:
    """Copy the small reference dataset used by container tests.

    Args:
        tmp_path: Temporary pytest directory.

    Returns:
        Path to the copied dataset root.
    """
    src = Path(__file__).resolve().parent.parent / "containers" / "dataset"
    dst = tmp_path / "dataset"
    shutil.copytree(src, dst)
    return dst


def test_check_dataset_valid_reference(tmp_path: Path) -> None:
    """Reference dataset should pass with no errors."""
    dataset_path = _copy_reference_dataset(tmp_path)

    report = check_dataset(dataset_path, max_samples=2)

    assert not report.has_errors()


def test_check_dataset_missing_infos(tmp_path: Path) -> None:
    """Missing infos.yaml should be reported as an error."""
    dataset_path = _copy_reference_dataset(tmp_path)
    (dataset_path / "infos.yaml").unlink()

    report = check_dataset(dataset_path)

    assert report.has_errors()
    assert any(msg.code == "MISSING_PATH" for msg in report.messages)


def test_check_dataset_num_samples_mismatch(tmp_path: Path) -> None:
    """Tampering with num_samples should raise split mismatch errors."""
    dataset_path = _copy_reference_dataset(tmp_path)
    infos_path = dataset_path / "infos.yaml"
    infos = yaml.safe_load(infos_path.read_text(encoding="utf-8"))
    infos["num_samples"]["train"] = 1
    infos_path.write_text(yaml.dump(infos, sort_keys=False), encoding="utf-8")

    report = check_dataset(dataset_path)

    assert any(msg.code == "SPLIT_COUNT_MISMATCH" for msg in report.messages)


def test_main_json_output_and_exit_code(tmp_path: Path, capsys) -> None:
    """CLI should output JSON and return expected status code."""
    dataset_path = _copy_reference_dataset(tmp_path)

    code = main([str(dataset_path), "--json", "--max-samples", "1"])
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert code == 0
    assert "counts" in payload
    assert "messages" in payload


def test_main_strict_fails_on_warning(tmp_path: Path) -> None:
    """In strict mode, warnings should make the command fail."""
    dataset_path = _copy_reference_dataset(tmp_path)
    infos_path = dataset_path / "infos.yaml"
    infos = yaml.safe_load(infos_path.read_text(encoding="utf-8"))
    infos["num_samples"]["train"] = 11
    infos_path.write_text(yaml.dump(infos, sort_keys=False), encoding="utf-8")

    code = main([str(dataset_path), "--strict"])

    assert code in {1, 2}


class _FakeSample:
    """Minimal sample-like object exposing `get_tree` for helper tests."""

    def get_tree(self) -> dict[str, str]:
        """Return a sentinel tree object used by monkeypatched CGU access."""
        return {"tree": "sentinel"}


def test_is_branch_without_data_true_for_none_with_children(monkeypatch) -> None:
    """Branch nodes with children and no data should be ignored by numeric checks."""

    def _fake_get_node_by_path(tree: Any, path: str) -> list[Any]:
        assert tree == {"tree": "sentinel"}
        assert path == "Global/Branch"
        return ["Branch", None, [["Child", 1.0, [], "DataArray_t"]], "UserDefinedData_t"]

    monkeypatch.setattr(plaidcheck.CGU, "getNodeByPath", _fake_get_node_by_path)

    assert _is_branch_without_data(_FakeSample(), "Global/Branch")


def test_is_branch_without_data_false_for_none_leaf(monkeypatch) -> None:
    """Leaf nodes with None data must still be reported as invalid values."""

    def _fake_get_node_by_path(tree: Any, path: str) -> list[Any]:
        assert tree == {"tree": "sentinel"}
        assert path == "Global/Leaf"
        return ["Leaf", None, [], "DataArray_t"]

    monkeypatch.setattr(plaidcheck.CGU, "getNodeByPath", _fake_get_node_by_path)

    assert not _is_branch_without_data(_FakeSample(), "Global/Leaf")
    assert _check_numeric_content(None) == "value is None"


def test_is_branch_without_data_in_mapping_true_for_branch_entry() -> None:
    """Dict-based branch entry with None data should be skipped in B-path checks."""
    feat_map = {
        "Global": None,
        "Global/ParamA": 1.0,
        "Global/ParamB": 2.0,
    }

    assert _is_branch_without_data_in_mapping("Global", None, feat_map)


def test_is_branch_without_data_in_mapping_false_for_leaf_none() -> None:
    """Dict-based leaf None entry should still be checked and reported."""
    feat_map = {
        "Global/Leaf": None,
        "Global/Other": 1.0,
    }

    assert not _is_branch_without_data_in_mapping("Global/Leaf", None, feat_map)
