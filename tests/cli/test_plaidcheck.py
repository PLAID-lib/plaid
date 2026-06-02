"""Tests for the plaidcheck CLI and checker helpers."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plaid.cli import plaidcheck
from plaid.cli.plaidcheck import (
    CheckReport,
    _check_numeric_content,
    _is_branch_without_data,
    _is_branch_without_data_in_mapping,
    check_dataset,
    main,
)
from plaid.infos import Infos

_REFERENCE_DATASETS = ("dataset_cgns", "dataset_hf")


def _copy_reference_dataset(tmp_path: Path, name: str = "dataset_cgns") -> Path:
    """Copy a reference dataset (CGNS or HF) used by container tests.

    Args:
        tmp_path: Temporary pytest directory.
        name: Reference dataset directory name under ``tests/containers``.

    Returns:
        Path to the copied dataset root.
    """
    src = Path(__file__).resolve().parent.parent / "containers" / name
    dst = tmp_path / "dataset"
    shutil.copytree(src, dst)
    return dst


@pytest.mark.parametrize("dataset_name", _REFERENCE_DATASETS)
def test_check_dataset_valid_reference(tmp_path: Path, dataset_name: str) -> None:
    """Reference dataset should pass with no errors."""
    dataset_path = _copy_reference_dataset(tmp_path, dataset_name)

    report = check_dataset(dataset_path)
    print(report)
    assert not report.has_errors()


@pytest.mark.parametrize("dataset_name", _REFERENCE_DATASETS)
def test_check_dataset_missing_infos(tmp_path: Path, dataset_name: str) -> None:
    """Missing infos.yaml should be reported as an error."""
    dataset_path = _copy_reference_dataset(tmp_path, dataset_name)
    (dataset_path / "infos.yaml").unlink()

    report = check_dataset(dataset_path)

    assert report.has_errors()
    assert any(msg.code == "MISSING_PATH" for msg in report.messages)


@pytest.mark.parametrize("dataset_name", _REFERENCE_DATASETS)
def test_check_dataset_num_samples_mismatch(tmp_path: Path, dataset_name: str) -> None:
    """Tampering with num_samples should raise split mismatch errors."""
    dataset_path = _copy_reference_dataset(tmp_path, dataset_name)
    infos_path = dataset_path / "infos.yaml"
    infos = Infos.from_path(infos_path)
    infos.num_samples["train"] = 1
    infos.save_to_file(infos_path)

    report = check_dataset(dataset_path)

    assert any(msg.code == "SPLIT_COUNT_MISMATCH" for msg in report.messages)


@pytest.mark.parametrize("dataset_name", _REFERENCE_DATASETS)
def test_main_json_output_and_exit_code(
    tmp_path: Path, capsys, dataset_name: str
) -> None:
    """CLI should output JSON and return expected status code."""
    dataset_path = _copy_reference_dataset(tmp_path, dataset_name)

    code = main(
        [
            str(dataset_path),
            "--json",
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(out)

    assert code == 0
    assert "counts" in payload
    assert "messages" in payload


@pytest.mark.parametrize("dataset_name", _REFERENCE_DATASETS)
def test_main_strict_fails_on_warning(tmp_path: Path, dataset_name: str) -> None:
    """In strict mode, warnings should make the command fail."""
    dataset_path = _copy_reference_dataset(tmp_path, dataset_name)
    infos_path = dataset_path / "infos.yaml"
    infos = Infos.from_path(infos_path)
    infos.num_samples["train"] = 11
    infos.save_to_file(infos_path)

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
        return [
            "Branch",
            None,
            [["Child", 1.0, [], "DataArray_t"]],
            "UserDefinedData_t",
        ]

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


def _make_minimal_layout(root: Path) -> Path:
    """Create the minimal expected dataset layout for checker entry checks.

    Args:
        root: Temporary path where the dataset directory is created.

    Returns:
        Path to the dataset root.
    """
    dataset = root / "dataset_min"
    dataset.mkdir()
    (dataset / "infos.yaml").write_text("storage_backend: zarr\n", encoding="utf-8")
    (dataset / "variable_schema.yaml").write_text("{}\n", encoding="utf-8")
    (dataset / "cgns_types.yaml").write_text("{}\n", encoding="utf-8")
    (dataset / "constants").mkdir()
    (dataset / "data").mkdir()
    return dataset


class _FakeSampleForCheck:
    """Sample-like object implementing methods used by `check_dataset`."""

    def __init__(
        self,
        global_value: Any = 1.0,
        field_value: Any = 1.0,
        global_names: list[str] | None = None,
        tree: Any = None,
        checksum: str = "same",
    ) -> None:
        self._global_value = global_value
        self._field_value = field_value
        self._global_names = ["G"] if global_names is None else global_names
        self._tree = tree
        self._checksum = checksum

    def get_zone_names(self, base: str, time: float) -> list[str]:  # noqa: ARG002
        """Return deterministic zone names for checker traversal.

        Args:
            base: Ignored.
            time: Ignored.

        Returns:
            A single zone name list.
        """
        return ["ZoneA"]

    def get_global_names(self) -> list[str]:
        """Return configured global names."""
        return self._global_names

    def get_feature_by_path(self, path: str) -> Any:  # noqa: ARG002
        """Return configured global value.

        Args:
            path: Ignored.

        Returns:
            Global value payload.
        """
        return self._global_value

    def get_tree(self):
        """Return no CGNS tree to disable branch skipping."""
        return self._tree

    def get_all_time_values(self) -> list[float]:
        """Return one time value."""
        return [0.0]

    def get_base_names(self, time: float) -> list[str]:  # noqa: ARG002
        """Return one base name.

        Args:
            time: Ignored.

        Returns:
            A single base name list.
        """
        return ["BaseA"]

    def get_field_names(
        self,
        location: str,  # noqa: ARG002
        zone: str,  # noqa: ARG002
        base: str,  # noqa: ARG002
        time: float,  # noqa: ARG002
    ) -> list[str]:
        """Return one field name.

        Args:
            location: Ignored.
            zone: Ignored.
            base: Ignored.
            time: Ignored.

        Returns:
            A single field name list.
        """
        return ["F"]

    def get_field(self, *args, **kwargs) -> Any:  # noqa: ARG002
        """Return configured field value.

        Returns:
            Field value payload.
        """
        return self._field_value


class _FakeDataset:
    """Dataset-like object exposing only `__len__`."""

    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        """Return dataset size."""
        return self._n


class _FakeConverter:
    """Converter-like object exposing `to_plaid`."""

    def __init__(
        self, samples: list[Any], fail_indices: set[int] | None = None
    ) -> None:
        self._samples = samples
        self._fail_indices = set() if fail_indices is None else fail_indices

    def to_plaid(self, dataset: _FakeDataset, idx: int) -> Any:  # noqa: ARG002
        """Return fake sample or raise conversion error.

        Args:
            dataset: Ignored.
            idx: Sample index.

        Returns:
            Fake sample instance.
        """
        if idx in self._fail_indices:
            raise RuntimeError("boom")
        return self._samples[idx]


def test_check_numeric_content_all_remaining_branches() -> None:
    """Numeric checker should report all remaining invalid content cases."""
    assert _check_numeric_content([]) == "value is empty"
    assert _check_numeric_content(np.array([1.0, np.nan])) == "contains NaN"
    assert _check_numeric_content(np.array([1.0, np.inf])) == "contains Inf"
    assert (
        _check_numeric_content(np.array([None, "x"], dtype=object))
        == "contains None in object array"
    )


def test_is_branch_without_data_false_variants(monkeypatch) -> None:
    """Branch helper should return False for missing tree/node/children layout."""

    class _SampleNoTree:
        def get_tree(self):
            return None

    class _SampleWithTree:
        def get_tree(self):
            return {"tree": 1}

    assert not _is_branch_without_data(_SampleNoTree(), "any")

    monkeypatch.setattr(plaidcheck.CGU, "getNodeByPath", lambda _tree, _path: None)
    assert not _is_branch_without_data(_SampleWithTree(), "any")

    monkeypatch.setattr(
        plaidcheck.CGU, "getNodeByPath", lambda _tree, _path: ["X", None]
    )
    assert not _is_branch_without_data(_SampleWithTree(), "any")


def test_is_branch_without_data_in_mapping_false_when_value_present() -> None:
    """Mapping helper should immediately reject non-None entries."""
    assert not _is_branch_without_data_in_mapping("Global", 1.0, {"Global/Child": 2.0})


def test_check_dataset_loader_failures_and_header_validations(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Checker should report infos/metadata/init failures and header errors."""
    dataset = _make_minimal_layout(tmp_path)

    monkeypatch.setattr(
        plaidcheck,
        "load_infos_from_disk",
        lambda path: (_ for _ in ()).throw(RuntimeError("infos")),  # noqa: ARG005
    )
    report_infos = check_dataset(dataset)
    assert any(msg.code == "INFOS_READ_ERROR" for msg in report_infos.messages)

    monkeypatch.setattr(
        plaidcheck,
        "load_infos_from_disk",
        lambda path: {"storage_backend": "zarr", "num_samples": {"train": 1}},  # noqa: ARG005
    )
    monkeypatch.setattr(
        plaidcheck,
        "load_metadata_from_disk",
        lambda path: (_ for _ in ()).throw(RuntimeError("meta")),  # noqa: ARG005
    )
    report_meta = check_dataset(dataset)
    assert any(msg.code == "METADATA_READ_ERROR" for msg in report_meta.messages)

    monkeypatch.setattr(
        plaidcheck,
        "load_metadata_from_disk",
        lambda path: ({"train": {}}, {}, {"train": {}}, None),  # noqa: ARG005
    )
    monkeypatch.setattr(
        plaidcheck,
        "init_from_disk",
        lambda path: (_ for _ in ()).throw(RuntimeError("init")),  # noqa: ARG005
    )
    report_init = check_dataset(dataset)
    assert any(msg.code == "DATASET_INIT_ERROR" for msg in report_init.messages)

    monkeypatch.setattr(
        plaidcheck,
        "init_from_disk",
        lambda path: ({"train": _FakeDataset(0)}, {"train": _FakeConverter([])}),  # noqa: ARG005
    )
    monkeypatch.setattr(
        plaidcheck,
        "load_infos_from_disk",
        lambda path: {"storage_backend": 12, "num_samples": "bad"},  # noqa: ARG005
    )
    report_header = check_dataset(dataset)
    assert any(msg.code == "BACKEND_MISSING" for msg in report_header.messages)
    assert any(msg.code == "NUM_SAMPLES_INVALID" for msg in report_header.messages)


def test_check_dataset_split_and_data_warnings_and_duplicates(
    tmp_path: Path, monkeypatch
) -> None:
    """Checker should report split errors, warnings and duplicated samples."""
    dataset = _make_minimal_layout(tmp_path)

    monkeypatch.setattr(
        plaidcheck,
        "load_infos_from_disk",
        lambda path: {"storage_backend": "zarr", "num_samples": {"train": 3}},  # noqa: ARG005
    )
    monkeypatch.setattr(
        plaidcheck, "load_metadata_from_disk", lambda _path: ({}, {"Var": {}}, {}, None)
    )

    monkeypatch.setattr(
        plaidcheck.CGU,
        "getNodeByPath",
        lambda tree, path: [  # noqa: ARG005
            "Branch",
            None,
            [["child", 1.0, [], "DataArray_t"]],
            "UserDefinedData_t",
        ],
    )

    samples = [
        _FakeSampleForCheck(global_value=1.0, field_value=1.0, checksum="dup"),
        _FakeSampleForCheck(
            global_value=np.array([np.nan]),
            field_value=np.array([np.nan]),
            tree={"branch": 1},
            checksum="unique",
        ),
        _FakeSampleForCheck(
            global_value=np.array([np.nan]),
            field_value=np.array([np.nan]),
            checksum="dup",
        ),
    ]
    converter = _FakeConverter(samples=samples)
    datasetdict = {"train": _FakeDataset(3)}
    monkeypatch.setattr(
        plaidcheck,
        "init_from_disk",
        lambda path: (datasetdict, {"train": converter}),  # noqa: ARG005
    )
    monkeypatch.setattr(plaidcheck, "compute_checksum", lambda sample: sample._checksum)

    report = check_dataset(dataset, splits=["train", "ghost"])

    assert any(msg.code == "UNKNOWN_SPLIT" for msg in report.messages)
    assert any(msg.code == "MISSING_CONSTANT_SCHEMA" for msg in report.messages)
    assert any(msg.code == "MISSING_CONSTANT_VALUES" for msg in report.messages)
    assert any(msg.code == "INVALID_DATA_VALUE A" for msg in report.messages)
    assert any(msg.code == "DUPLICATED_DATA" for msg in report.messages)


def test_check_dataset_sample_conversion_error(tmp_path: Path, monkeypatch) -> None:
    """Checker should emit conversion errors when converter fails on an index."""
    dataset = _make_minimal_layout(tmp_path)

    monkeypatch.setattr(
        plaidcheck,
        "load_infos_from_disk",
        lambda path: {"storage_backend": "zarr", "num_samples": {"train": 1}},  # noqa: ARG005
    )
    monkeypatch.setattr(
        plaidcheck,
        "load_metadata_from_disk",
        lambda path: ({"train": {}}, {"Var": {}}, {"train": {}}, None),  # noqa: ARG005
    )
    monkeypatch.setattr(
        plaidcheck,
        "init_from_disk",
        lambda path: (  # noqa: ARG005
            {"train": _FakeDataset(1)},
            {"train": _FakeConverter([_FakeSampleForCheck()], fail_indices={0})},
        ),
    )

    report = check_dataset(dataset, splits=["train"])

    assert any(msg.code == "SAMPLE_CONVERSION_ERROR" for msg in report.messages)


def test_check_dataset_problem_definition_validation_paths(
    tmp_path: Path, monkeypatch
) -> None:
    """Checker should cover problem-definition read/validation branches."""
    dataset = _make_minimal_layout(tmp_path)
    (dataset / "problem_definitions").mkdir()

    monkeypatch.setattr(
        plaidcheck,
        "load_infos_from_disk",
        lambda path: {"storage_backend": "zarr", "num_samples": {"train": 2}},  # noqa: ARG005
    )
    monkeypatch.setattr(
        plaidcheck,
        "load_metadata_from_disk",
        lambda path: (  # noqa: ARG005
            {"train": {"Known": 1.0}},
            {"KnownVar": {}},
            {"train": {}},
            None,
        ),
    )
    monkeypatch.setattr(
        plaidcheck,
        "init_from_disk",
        lambda path: (  # noqa: ARG005
            {"train": _FakeDataset(2)},
            {"train": _FakeConverter([_FakeSampleForCheck(), _FakeSampleForCheck()])},
        ),
    )

    monkeypatch.setattr(
        plaidcheck,
        "load_problem_definitions_from_disk",
        lambda path: (_ for _ in ()).throw(RuntimeError("pb")),  # noqa: ARG005
    )
    report_read = check_dataset(dataset)
    assert any(msg.code == "PB_DEF_READ_ERROR" for msg in report_read.messages)

    class _PBDef:
        def __init__(self, train_split, test_split):
            self.input_features = ["UnknownInput"]
            self.output_features = ["UnknownOutput"]
            self.train_split = train_split
            self.test_split = test_split

    pb_defs = {
        "pb_many": _PBDef(train_split={"train": [0], "other": [1]}, test_split=None),
        "pb_unknown_split": _PBDef(train_split={"ghost": [0]}, test_split=None),
        "pb_indices": _PBDef(
            train_split={"train": [0, 0, -1, 9]}, test_split={"train": "all"}
        ),
    }
    monkeypatch.setattr(
        plaidcheck,
        "load_problem_definitions_from_disk",
        lambda path: pb_defs,  # noqa: ARG005
    )
    report_pb = check_dataset(dataset)

    assert any(msg.code == "PB_DEF_UNKNOWN_INPUT" for msg in report_pb.messages)
    assert any(msg.code == "PB_DEF_UNKNOWN_OUTPUT" for msg in report_pb.messages)
    assert any(msg.code == "PB_DEF_SPLIT" for msg in report_pb.messages)
    assert any(msg.code == "PB_DEF_UNKNOWN_SPLIT" for msg in report_pb.messages)
    assert any(msg.code == "PB_DEF_DUPLICATE_INDICES" for msg in report_pb.messages)
    assert any(msg.code == "PB_DEF_OUT_OF_RANGE_INDICES" for msg in report_pb.messages)


def test_main_strict_returns_warning_exit_code(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    """Main should return exit code 2 when strict mode sees warnings only."""
    report = CheckReport(messages=[])
    report.add("warning", "W", "loc", "msg")

    monkeypatch.setattr(plaidcheck, "check_dataset", lambda path, splits=None: report)  # noqa: ARG005
    code = main([str(tmp_path), "--strict"])
    _ = capsys.readouterr().out

    assert code == 2
