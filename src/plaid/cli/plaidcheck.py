"""CLI tool to validate integrity of a PLAID dataset stored on disk."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import CGNS.PAT.cgnsutils as CGU
import numpy as np

from plaid.storage import init_from_disk
from plaid.storage.common.reader import (
    load_infos_from_disk,
    load_metadata_from_disk,
    load_problem_definitions_from_disk,
)


@dataclass
class CheckMessage:
    """One integrity check message.

    Args:
        severity: Message severity (`error`, `warning`, or `info`).
        code: Stable message code identifier.
        location: Path-like location string related to the issue.
        message: Human-readable message.
    """

    severity: str
    code: str
    location: str
    message: str


@dataclass
class CheckReport:
    """Container for check results and summary helpers.

    Args:
        messages: Integrity check messages collected during validation.
    """

    messages: list[CheckMessage]

    def add(self, severity: str, code: str, location: str, message: str) -> None:
        """Append a new message to the report.

        Args:
            severity: Message severity (`error`, `warning`, or `info`).
            code: Stable message code identifier.
            location: Path-like location string related to the issue.
            message: Human-readable message.
        """
        self.messages.append(
            CheckMessage(
                severity=severity,
                code=code,
                location=location,
                message=message,
            )
        )

    def counts(self) -> dict[str, int]:
        """Return counts by severity.

        Returns:
            Mapping from severity names to message counts.
        """
        return {
            "error": sum(msg.severity == "error" for msg in self.messages),
            "warning": sum(msg.severity == "warning" for msg in self.messages),
            "info": sum(msg.severity == "info" for msg in self.messages),
        }

    def has_errors(self) -> bool:
        """Return whether at least one error was reported.

        Returns:
            True when the report contains one or more error messages.
        """
        return any(msg.severity == "error" for msg in self.messages)

    def has_warnings(self) -> bool:
        """Return whether at least one warning was reported.

        Returns:
            True when the report contains one or more warning messages.
        """
        return any(msg.severity == "warning" for msg in self.messages)

    def to_json(self) -> str:
        """Serialize report to JSON string.

        Returns:
            JSON string containing severity counts and message details.
        """
        payload = {
            "counts": self.counts(),
            "messages": [asdict(msg) for msg in self.messages],
        }
        return json.dumps(payload, indent=2)


def _check_required_layout(path: Path, report: CheckReport) -> None:
    """Validate that the dataset directory has the required PLAID layout.

    Args:
        path: Dataset directory to inspect.
        report: Report updated with missing path errors.

    Returns:
        None.
    """
    required_paths = [
        "infos.yaml",
        "variable_schema.yaml",
        "cgns_types.yaml",
        "constants",
        "data",
    ]
    for rel in required_paths:
        p = path / rel
        if not p.exists():
            report.add("error", "MISSING_PATH", rel, f"Missing file/path path: {rel}")


def _check_numeric_content(value: Any) -> Optional[str]:
    """Inspect a feature value for invalid numeric or object content.

    Args:
        value: Feature value to validate.

    Returns:
        Description of the detected issue, or None when the value is valid.
    """
    if value is None:
        return "value is None"
    arr = np.asarray(value)
    if arr.size == 0:
        return "value is empty"
    if np.issubdtype(arr.dtype, np.floating):
        if np.isnan(arr).any():
            return "contains NaN"
        if np.isinf(arr).any():
            return "contains Inf"
    if arr.dtype == object:
        if any(v is None for v in arr.flat):
            return "contains None in object array"
    return None


def _is_branch_without_data(sample: Any, path: str) -> bool:
    """Return True when `path` points to a branch node with no direct value.

    Args:
        sample: Sample-like object exposing `get_tree()`.
        path: CGNS path to inspect.

    Returns:
        bool: True if node exists, node value is None, and node has children.
    """
    tree = sample.get_tree()
    if tree is None:
        return False
    node = CGU.getNodeByPath(tree, path)
    if node is None:
        return False
    if len(node) < 3:
        return False
    return node[1] is None and bool(node[2])


def _is_branch_without_data_in_mapping(
    feature_name: str,
    value: Any,
    feat_map: dict[str, Any],
) -> bool:
    """Return True when a dict-based feature entry represents a branch node.

    Args:
        feature_name: Feature path/key currently inspected.
        value: Feature value currently inspected.
        feat_map: Mapping of feature names to values for a given sample/time.

    Returns:
        bool: True when current entry is a branch with no direct data and
            child entries exist in `feat_map`.
    """
    if value is not None:
        return False
    prefix = f"{feature_name}/"
    return any(name.startswith(prefix) for name in feat_map)


def compute_checksum(sample):
    """Compute a SHA-256 checksum for a converted sample representation.

    Args:
        sample: Sample object or dictionary representation to checksum.

    Returns:
        Hexadecimal SHA-256 digest of the pickled sample.
    """
    import hashlib
    import pickle

    sha256 = hashlib.sha256()
    sha256.update(pickle.dumps(sample))
    return sha256.hexdigest()


def check_dataset(
    path: Path,
    splits: Optional[list[str]] = None,
    max_samples: Optional[int] = None,
) -> CheckReport:
    """Run integrity checks on a local PLAID dataset.

    Args:
        path: Dataset directory.
        splits: Optional selected split names.
        max_samples: Optional cap for per-sample checks.

    Returns:
        A populated :class:`CheckReport`.
    """
    report = CheckReport(messages=[])

    # First verify the dataset has the required on-disk files and folders.
    # Later checks rely on these paths being present and readable.
    _check_required_layout(path, report)
    if report.has_errors():
        return report

    # Load dataset descriptors and metadata before touching sample payloads.
    # Each loading step is isolated so the report points to the failing layer.
    try:
        infos = load_infos_from_disk(path)
    except Exception as exc:
        report.add("error", "INFOS_READ_ERROR", "infos.yaml", str(exc))
        return report

    try:
        flat_cst, variable_schema, constant_schema, _ = load_metadata_from_disk(path)
    except Exception as exc:
        report.add("error", "METADATA_READ_ERROR", str(path), str(exc))
        return report

    try:
        datasetdict, converterdict = init_from_disk(path)
    except Exception as exc:
        report.add("error", "DATASET_INIT_ERROR", str(path), str(exc))
        return report

    # Validate top-level dataset declarations from infos.yaml.
    declared_backend = infos.get("storage_backend")
    if not isinstance(declared_backend, str):
        report.add(
            "error",
            "BACKEND_MISSING",
            "infos.yaml",
            "Missing or invalid 'storage_backend' in infos.yaml",
        )

    num_samples = infos.get("num_samples", {})
    if not isinstance(num_samples, dict):
        report.add(
            "error", "NUM_SAMPLES_INVALID", "infos.yaml", "'num_samples' must be a dict"
        )
        num_samples = {}

    # Resolve the user-requested splits against the splits actually available.
    dataset_splits = set(datasetdict.keys())
    target_splits = set(splits) if splits else dataset_splits
    unknown_splits = target_splits - dataset_splits
    for split in sorted(unknown_splits):
        report.add("error", "UNKNOWN_SPLIT", split, "Split not found in dataset")
    target_splits = target_splits & dataset_splits

    checksum_report = {}
    for split in sorted(target_splits):
        dataset = datasetdict[split]
        converter = converterdict[split]

        # Check split-level consistency between metadata, schemas, and storage.
        expected_n = num_samples.get(split)
        actual_n = len(dataset)
        if isinstance(expected_n, int) and expected_n != actual_n:
            report.add(
                "error",
                "SPLIT_COUNT_MISMATCH",
                split,
                f"Expected {expected_n} samples from infos.yaml, found {actual_n}",
            )

        if split not in constant_schema:
            report.add(
                "error",
                "MISSING_CONSTANT_SCHEMA",
                split,
                "No constant schema for split",
            )
        if split not in flat_cst:
            report.add(
                "error",
                "MISSING_CONSTANT_VALUES",
                split,
                "No constant values for split",
            )

        ids = getattr(dataset, "ids", None)
        if ids is not None and isinstance(expected_n, int):
            id_list = [int(i) for i in ids]
            duplicates = len(id_list) - len(set(id_list))
            if duplicates > 0:
                report.add(
                    "warning",
                    "DUPLICATED_SAMPLE_IDS",
                    split,
                    f"Found {duplicates} duplicated sample id(s)",
                )

            expected_ids = set(range(expected_n))
            missing_ids = sorted(expected_ids - set(id_list))
            extra_ids = sorted(set(id_list) - expected_ids)
            if missing_ids:
                report.add(
                    "warning",
                    "MISSING_SAMPLE_IDS",
                    split,
                    f"Missing sample ids (first 10): {missing_ids[:10]}",
                )
            if extra_ids:
                report.add(
                    "warning",
                    "UNEXPECTED_SAMPLE_IDS",
                    split,
                    f"Unexpected sample ids (first 10): {extra_ids[:10]}",
                )

        # Deep-check a bounded number of samples to validate content and detect
        # duplicated scalar DOE signatures without necessarily scanning all data.
        n_to_check = actual_n if max_samples is None else min(max_samples, actual_n)
        doe_signatures: dict[tuple[Any, ...], int] = {}
        for idx in range(n_to_check):
            # CGNS-backed datasets expose sample trees directly through
            # `to_plaid`, so validate global scalar values from the tree API.
            if converter.backend == "cgns":
                try:
                    sample = converter.to_plaid(dataset, idx)
                except Exception as exc:
                    report.add(
                        "error",
                        "SAMPLE_CONVERSION_ERROR",
                        f"{split}[{idx}]",
                        str(exc),
                    )
                    continue

                # Track whole-sample checksums to detect duplicated data across
                # all checked splits after the per-split loop completes.
                sample_checksum = compute_checksum(sample)
                checksum_report[(idx, split)] = sample_checksum

                scalar_signature: list[Any] = []
                for global_name in sample.get_global_names():
                    value = sample.get_feature_by_path(global_name)
                    if _is_branch_without_data(sample, global_name):
                        continue
                    issue = _check_numeric_content(value)
                    if issue is not None:
                        report.add(
                            "warning",
                            "INVALID_DATA_VALUE A",
                            f"{split}[{idx}] global/{global_name}",
                            issue,
                        )
                    arr = np.asarray(value)
                    if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                        scalar_signature.append(
                            ("global", global_name, float(arr.ravel()[0]))
                        )

                sig_key = tuple(sorted(scalar_signature))
                if sig_key:
                    doe_signatures[sig_key] = doe_signatures.get(sig_key, 0) + 1
                continue

            # Non-CGNS backends are normalized to dictionaries so validation can
            # iterate over time keys and feature maps in a backend-neutral way.
            try:
                sample_dict = converter.to_dict(dataset, idx)
            except Exception as exc:
                report.add(
                    "error",
                    "SAMPLE_CONVERSION_ERROR",
                    f"{split}[{idx}]",
                    str(exc),
                )
                continue

            # Track whole-sample checksums for duplicate data detection.
            sample_checksum = compute_checksum(sample_dict)
            checksum_report[(idx, split)] = sample_checksum

            # Validate each materialized feature value while ignoring branch
            # entries that exist only to group child feature paths.
            for time_key, feat_map in sample_dict.items():
                for feature_name, value in feat_map.items():
                    if _is_branch_without_data_in_mapping(
                        feature_name, value, feat_map
                    ):
                        continue

                    issue = _check_numeric_content(value)

                    if issue is not None:
                        report.add(
                            "warning",
                            "INVALID_DATA_VALUE B",
                            f"{split}[{idx}]/{time_key} {feature_name}",
                            issue,
                        )

            # Build a scalar signature for duplicate DOE input detection within
            # this split. Only scalar numeric values are included.
            scalar_signature: list[Any] = []
            for time_key, feat_map in sample_dict.items():
                for feature_name, value in feat_map.items():
                    arr = np.asarray(value)
                    if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                        scalar_signature.append(
                            (str(time_key), feature_name, float(arr.ravel()[0]))
                        )

            sig_key = tuple(sorted(scalar_signature))
            if sig_key:
                doe_signatures[sig_key] = doe_signatures.get(sig_key, 0) + 1

        repeated = sum(1 for count in doe_signatures.values() if count > 1)
        if repeated > 0:
            report.add(
                "warning",
                "DUPLICATED_DOE_INPUTS",
                split,
                f"Detected {repeated} duplicated scalar signature(s) in checked samples",
            )

    # Compare checksums from every checked sample to flag identical sample data.
    if len(checksum_report) != len(np.unique(checksum_report.values())):
        k = list(checksum_report.keys())
        v = list(checksum_report.values())
        uni, cou = np.unique(v, return_counts=True)
        for u, c in zip(uni, cou):
            if c == 1:
                continue
            duplicated = k[v == u]

            report.add(
                "warning",
                "DUPLICATED_DATA",
                str(duplicated),
                "duplicated sample",
            )

    # If problem definitions are present, verify that their feature references,
    # split names, and sample indices are compatible with the dataset.
    pb_def_dir = path / "problem_definitions"
    if pb_def_dir.exists():
        try:
            pb_defs = load_problem_definitions_from_disk(path)
        except Exception as exc:
            report.add(
                "error",
                "PB_DEF_READ_ERROR",
                "problem_definitions",
                str(exc),
            )
            pb_defs = {}

        all_features = set(variable_schema.keys())
        for split_cst in flat_cst.values():
            all_features.update(split_cst.keys())

        for pb_name, pb_def in pb_defs.items():
            for feat in pb_def.input_features:
                if feat not in all_features:
                    report.add(
                        "error",
                        "PB_DEF_UNKNOWN_INPUT",
                        f"problem_definitions/{pb_name}",
                        f"Unknown input feature: {feat}",
                    )
                if "GlobalConvergenceHistory" not in feat and "Global" not in feat:
                    report.add(
                        "warning",
                        "DOE_INPUT_NOT_SCALAR",
                        f"problem_definitions/{pb_name}",
                        f"Input feature may not be scalar/global for DOE: {feat}",
                    )

            for feat in pb_def.output_features:
                if feat not in all_features:
                    report.add(
                        "error",
                        "PB_DEF_UNKNOWN_OUTPUT",
                        f"problem_definitions/{pb_name}",
                        f"Unknown output feature: {feat}",
                    )

            for split_dict_name in ["train_split", "test_split"]:
                split_dict = getattr(pb_def, split_dict_name)
                if split_dict is None:
                    continue
                split_name = next(iter(split_dict.keys()))
                split_ids = next(iter(split_dict.values()))
                if split_name not in dataset_splits:
                    report.add(
                        "error",
                        "PB_DEF_UNKNOWN_SPLIT",
                        f"problem_definitions/{pb_name}",
                        f"Unknown split in {split_dict_name}: {split_name}",
                    )
                    continue
                if split_ids == "all":
                    continue
                ids_list = list(split_ids)
                if len(ids_list) != len(set(ids_list)):
                    report.add(
                        "error",
                        "PB_DEF_DUPLICATE_INDICES",
                        f"problem_definitions/{pb_name}",
                        f"Duplicated indices in {split_dict_name}",
                    )
                split_len = len(datasetdict[split_name])
                bad = [i for i in ids_list if i < 0 or i >= split_len]
                if bad:
                    report.add(
                        "error",
                        "PB_DEF_OUT_OF_RANGE_INDICES",
                        f"problem_definitions/{pb_name}",
                        f"Out-of-range indices in {split_dict_name} (first 10): {bad[:10]}",
                    )

    # Emit an explicit success message when no errors or warnings were found.
    if not report.messages:
        report.add("info", "OK", str(path), "No issue detected")
    return report


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the dataset checker.

    Returns:
        Configured argument parser for the `plaid-check` command.
    """
    parser = argparse.ArgumentParser(description="Check integrity of a PLAID dataset.")
    parser.add_argument("path", type=Path, help="Path to local PLAID dataset")
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        help="Split to check (can be provided multiple times)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per split for deep checks",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print report in JSON format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failure",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point for `plaid-check`.

    Args:
        argv: Optional command-line args.

    Returns:
        Process exit code.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    report = check_dataset(
        path=args.path,
        splits=args.split,
        max_samples=args.max_samples,
    )

    if args.json:
        print(report.to_json())
    else:
        for msg in report.messages:
            print(f"[{msg.severity.upper()}] {msg.code} {msg.location}: {msg.message}")
        counts = report.counts()
        print(
            f"Summary: {counts['error']} error(s), "
            f"{counts['warning']} warning(s), {counts['info']} info message(s)"
        )

    if report.has_errors():
        return 1
    if args.strict and report.has_warnings():
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
