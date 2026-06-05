"""CLI tool to validate integrity of a PLAID dataset stored on disk."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import CGNS.PAT.cgnsutils as CGU
import numpy as np
from tqdm import tqdm

from plaid.constants import CGNS_FIELD_LOCATIONS
from plaid.infos import Infos
from plaid.storage import init_from_disk
from plaid.storage.common.reader import (
    load_metadata_from_disk,
    load_problem_definitions_from_disk,
)


def load_infos_from_disk(path: Path) -> Infos:
    """Load infos for checker diagnostics without persisted-field enforcement."""
    return Infos.from_path(path, require_persisted=False)


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


def _check_required_layout(
    path: Path, report: CheckReport, backend: Optional[str] = None
) -> None:
    """Validate that the dataset directory has the required PLAID layout.

    Args:
        path: Dataset directory to inspect.
        report: Report updated with missing path errors.
        backend: Storage backend identifier. The CGNS backend stores
            self-contained samples and intentionally omits the derived
            ``constants/``, ``variable_schema.yaml`` and ``cgns_types.yaml``
            metadata produced by other backends.

    Returns:
        None.
    """
    required_paths = ["infos.yaml", "data"]
    if backend != "cgns":
        required_paths += [
            "variable_schema.yaml",
            "cgns_types.yaml",
            "constants",
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


def _format_missing_split_message(split: object) -> str:
    """Return an actionable message for missing split declarations.

    Args:
        split: Split name/key reported by a low-level ``KeyError``.

    Returns:
        Human-readable explanation suitable for a checker error.
    """
    return (
        f"Split {split!r} exists in the stored dataset or metadata but is missing "
        "from infos.yaml > num_samples. Add this split to num_samples, or remove "
        "the corresponding split data/metadata from the dataset."
    )


def _check_num_samples_declares_splits(
    num_samples: dict[str, Any],
    split_names: set[str],
    report: CheckReport,
) -> None:
    """Validate that ``infos.yaml > num_samples`` declares known disk splits.

    Args:
        num_samples: Mapping loaded from ``infos.yaml``.
        split_names: Split names discovered from storage files/metadata.
        report: Report updated with missing declaration errors.

    Returns:
        None.
    """
    declared_splits = {str(split) for split in num_samples}
    for split in sorted(split_names - declared_splits):
        report.add(
            "error",
            "NUM_SAMPLES_MISSING_SPLIT",
            "infos.yaml",
            _format_missing_split_message(split),
        )


def _discover_split_names_from_disk(
    path: Path,
    backend: Optional[str],
    flat_cst: dict[str, Any],
    constant_schema: dict[str, Any],
) -> set[str]:
    """Discover split names from files/metadata without building converters.

    Args:
        path: Dataset root.
        backend: Declared storage backend, if valid.
        flat_cst: Flattened constants keyed by split for non-CGNS backends.
        constant_schema: Constant schema keyed by split for non-CGNS backends.

    Returns:
        Split names discovered from the on-disk dataset structure and metadata.
    """
    split_names: set[str] = set()
    data_path = path / "data"
    if data_path.exists():
        if backend in {"zarr", "cgns"}:
            split_names.update(p.name for p in data_path.iterdir() if p.is_dir())
        elif backend == "hf_datasets":
            split_names.update(flat_cst.keys())
            split_names.update(constant_schema.keys())
    if backend != "cgns":
        constants_path = path / "constants"
        if constants_path.exists():
            split_names.update(p.name for p in constants_path.iterdir() if p.is_dir())
        split_names.update(flat_cst.keys())
        split_names.update(constant_schema.keys())
    return {str(split) for split in split_names}


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


def _progress(
    iterable: Any, *, desc: str, show_progress: bool, total: int | None = None
):
    """Wrap an iterable in a tqdm progress bar when requested.

    Args:
        iterable: Iterable to wrap.
        desc: Progress bar description.
        show_progress: Whether the progress bar is enabled.
        total: Optional total length.

    Returns:
        Iterable, possibly wrapped by :class:`tqdm.tqdm`.
    """
    return tqdm(iterable, desc=desc, total=total, disable=not show_progress)


def _resolve_problem_split_indices(
    split_ids: Any,
    split_len: int,
) -> list[int]:
    """Resolve a problem-definition split declaration into concrete indices.

    Args:
        split_ids: Either the special all-samples marker or an iterable of indices.
        split_len: Number of samples available in the referenced split.

    Returns:
        Concrete sample indices to instantiate.
    """
    if split_ids == "all":
        return list(range(split_len))
    return list(split_ids)


def _check_problem_definition_sample_features(
    *,
    pb_name: str,
    split_dict_name: str,
    split_name: str,
    idx: int,
    dataset: Any,
    converter: Any,
    features: list[str],
    report: CheckReport,
) -> None:
    """Instantiate and validate one problem-definition sample view.

    The sample is instantiated with the exact feature subset requested by the
    problem definition, then each requested feature is read back to validate
    that the requested feature paths can actually be resolved.

    Numeric content (NaN, Inf, None, empty arrays, ...) is intentionally not
    re-checked here: the per-split loop in :func:`check_dataset` already walks
    every sample's globals and fields and reports such issues with the
    ``INVALID_DATA_VALUE A`` code. Re-checking them in this loop would only
    produce duplicate warnings under a different code/location.

    Args:
        pb_name: Problem-definition name.
        split_dict_name: ``train_split`` or ``test_split``.
        split_name: Dataset split name.
        idx: Sample index.
        dataset: Backend dataset object.
        converter: Storage converter exposing ``to_plaid``.
        features: Feature paths to request and validate.
        report: Report updated with detected errors/warnings.
    """
    location = f"problem_definitions/{pb_name}/{split_dict_name}/{split_name}[{idx}]"
    try:
        sample = converter.to_plaid(dataset, idx, features=features)
    except Exception as exc:
        report.add(
            "error",
            "PB_DEF_SAMPLE_CONVERSION_ERROR",
            location,
            str(exc),
        )
        return

    for feature in features:
        try:
            sample.get_feature_by_path(feature)
        except Exception as exc:
            report.add(
                "error",
                "PB_DEF_FEATURE_READ_ERROR",
                f"{location} {feature}",
                str(exc),
            )


def compute_checksum(sample: Any) -> str:
    """Compute a SHA-256 checksum for a converted sample representation.

    Args:
        sample: Sample object or dictionary representation to checksum.

    Returns:
        str: Hexadecimal SHA-256 digest of the pickled sample.
    """
    import hashlib
    import pickle

    sha256 = hashlib.sha256()
    sha256.update(pickle.dumps(sample))
    return sha256.hexdigest()


def check_dataset(
    path: Path,
    splits: Optional[list[str]] = None,
    show_progress: bool = True,
    problem_definitions: Optional[list[str]] = None,
) -> CheckReport:
    """Run integrity checks on a local PLAID dataset.

    Algorithm overview:
        1. Validate the required on-disk PLAID layout.
        2. Load infos, metadata, and split-specific dataset/converter objects.
        3. Validate top-level declarations from ``infos.yaml`` (backend, sample counts).
        4. Resolve requested splits and report unknown ones.
        5. For each checked split:
           - verify split-level schema/value consistency,
           - validate sample IDs,
           - convert each sample and validate values,
           - compute checksums for duplicate-data detection,
           - build scalar signatures to detect duplicated DOE-like inputs.
        6. Validate optional problem definitions against available features/splits/indices.
        7. Emit an ``OK`` info message when no issue is detected.

    Args:
        path: Dataset directory.
        splits: Optional selected split names.
        show_progress: Whether to display tqdm progress bars for expensive checks.
        problem_definitions: Optional selected problem-definition names. When
            omitted, all discovered problem definitions are checked.

    Returns:
        A populated :class:`CheckReport`.
    """
    report = CheckReport(messages=[])

    # Load infos first so we can branch on the declared backend.
    if not (path / "infos.yaml").exists():
        report.add(
            "error",
            "MISSING_PATH",
            "infos.yaml",
            "Missing file/path path: infos.yaml",
        )
        return report
    try:
        infos = load_infos_from_disk(path)
    except Exception as exc:
        report.add("error", "INFOS_READ_ERROR", "infos.yaml", str(exc))
        return report

    declared_backend_for_layout = infos.storage_backend
    if not isinstance(declared_backend_for_layout, str):
        declared_backend_for_layout = None

    # Verify the dataset has the required on-disk files and folders for the
    # detected backend. Later checks rely on these paths being present and
    # readable.
    _check_required_layout(path, report, backend=declared_backend_for_layout)
    if report.has_errors():
        return report

    # Validate top-level dataset declarations from infos.yaml before calling
    # init_from_disk(), because storage initialization indexes num_samples by
    # split and otherwise reports missing entries as opaque KeyError messages.
    declared_backend = infos.storage_backend
    if not isinstance(declared_backend, str):
        report.add(
            "error",
            "BACKEND_MISSING",
            "infos.yaml",
            "Missing or invalid 'storage_backend' in infos.yaml",
        )

    num_samples = infos.num_samples
    if not isinstance(num_samples, dict):
        report.add(
            "error", "NUM_SAMPLES_INVALID", "infos.yaml", "'num_samples' must be a dict"
        )
        num_samples = {}

    # Load metadata when the backend defines it. The CGNS backend stores
    # self-contained samples and intentionally writes no derived metadata.
    if declared_backend_for_layout == "cgns":
        flat_cst: dict = {}
        variable_schema: dict = {}
        constant_schema: dict = {}
    else:
        try:
            flat_cst, variable_schema, constant_schema, _ = load_metadata_from_disk(
                path
            )
        except Exception as exc:
            report.add("error", "METADATA_READ_ERROR", str(path), str(exc))
            return report

    discovered_splits = _discover_split_names_from_disk(
        path,
        declared_backend_for_layout,
        flat_cst,
        constant_schema,
    )
    _check_num_samples_declares_splits(num_samples, discovered_splits, report)
    if report.has_errors():
        return report

    try:
        datasetdict, converterdict = init_from_disk(path)
    except KeyError as exc:
        report.add(
            "error",
            "NUM_SAMPLES_MISSING_SPLIT",
            "infos.yaml",
            _format_missing_split_message(exc.args[0] if exc.args else str(exc)),
        )
        return report
    except Exception as exc:
        report.add("error", "DATASET_INIT_ERROR", str(path), str(exc))
        return report

    # Resolve the user-requested splits against the splits actually available.
    dataset_splits = set(datasetdict.keys())
    target_splits = set(splits) if splits else dataset_splits
    unknown_splits = target_splits - dataset_splits
    for split in sorted(unknown_splits):
        available = " and ".join(f'"{x}"' for x in dataset_splits)
        report.add(
            "error",
            "UNKNOWN_SPLIT",
            split,
            f"Split not found in dataset, available are {available}",
        )
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

        if declared_backend_for_layout != "cgns":
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

        # Deep-check to validate content and detect non valide data in fields (nan inf)
        for idx in _progress(
            range(actual_n),
            desc=f"Checking split {split}",
            show_progress=show_progress,
            total=actual_n,
        ):
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

            for global_name in sample.get_global_names():
                global_path = "Global/" + global_name
                value = sample.get_feature_by_path(global_path)

                if _is_branch_without_data(sample, global_path):
                    continue

                issue = _check_numeric_content(value)
                if issue is not None:
                    report.add(
                        "warning",
                        "INVALID_DATA_VALUE A",
                        f"{split}[{idx}] global/{global_name}",
                        issue,
                    )

            for time in sample.get_all_time_values():
                local_bases = sample.get_base_names(time=time)
                for base in local_bases:
                    zone_names = sample.get_zone_names(base=base, time=time)
                    for zone in zone_names:
                        for location in CGNS_FIELD_LOCATIONS:
                            field_names = sample.get_field_names(
                                location=location,
                                zone=zone,
                                base=base,
                                time=time,
                            )

                            for f_name in field_names:
                                field_value = sample.get_field(
                                    f_name,
                                    location=location,
                                    zone=zone,
                                    base=base,
                                    time=time,
                                )
                                issue = _check_numeric_content(field_value)
                                if issue is not None:
                                    report.add(
                                        "warning",
                                        "INVALID_DATA_VALUE A",
                                        f"{split}[{idx}][{time}] {base}/{zone}/{location}/{f_name}",
                                        issue,
                                    )

    # Compare checksums from every checked sample to flag identical sample data.
    checksum_values = list(checksum_report.values())
    if len(checksum_report) != len(np.unique(checksum_values)):
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
            return report

        all_features = set(variable_schema.keys())
        for split_cst in flat_cst.values():
            all_features.update(split_cst.keys())
        # The CGNS backend stores self-contained samples and writes no
        # derived feature schema, so we have no authoritative catalogue
        # to validate problem-definition feature paths against. Skip the
        # feature-name checks in that case (split / index checks below
        # still run).
        validate_pb_def_features = declared_backend_for_layout != "cgns"

        target_pb_names = (
            set(problem_definitions) if problem_definitions else set(pb_defs)
        )
        unknown_pb_names = target_pb_names - set(pb_defs)
        for pb_name in sorted(unknown_pb_names):
            available = " and ".join(f'"{x}"' for x in sorted(pb_defs))
            report.add(
                "error",
                "PB_DEF_UNKNOWN",
                f"problem_definitions/{pb_name}",
                f"Problem definition not found, available are {available}",
            )
        target_pb_names = target_pb_names & set(pb_defs)

        for pb_name, pb_def in pb_defs.items():
            if pb_name not in target_pb_names:
                continue
            if validate_pb_def_features:
                for feat in pb_def.input_features:
                    if feat not in all_features:
                        report.add(
                            "error",
                            "PB_DEF_UNKNOWN_INPUT",
                            f"problem_definitions/{pb_name}",
                            f"Unknown input feature: {feat}",
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
                # split_dict must have only one elements
                if len(split_dict) > 1:
                    report.add(
                        "error",
                        "PB_DEF_SPLIT",
                        f"problem_definitions/{pb_name}",
                        f"{split_dict_name} has more than 1 split: {list(split_dict.keys())}",
                    )
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
                split_len = len(datasetdict[split_name])
                ids_list = _resolve_problem_split_indices(split_ids, split_len)
                if len(ids_list) != len(set(ids_list)):
                    report.add(
                        "error",
                        "PB_DEF_DUPLICATE_INDICES",
                        f"problem_definitions/{pb_name}",
                        f"Duplicated indices in {split_dict_name}",
                    )
                bad = [i for i in ids_list if i < 0 or i >= split_len]
                if bad:
                    report.add(
                        "error",
                        "PB_DEF_OUT_OF_RANGE_INDICES",
                        f"problem_definitions/{pb_name}",
                        f"Out-of-range indices in {split_dict_name} (first 10): {bad[:10]}",
                    )
                    continue

                if split_dict_name == "train_split":
                    features = list(pb_def.input_features) + list(
                        pb_def.output_features
                    )
                else:
                    features = list(pb_def.input_features)

                for idx in _progress(
                    ids_list,
                    desc=f"Checking problem {pb_name} {split_dict_name}",
                    show_progress=show_progress,
                    total=len(ids_list),
                ):
                    _check_problem_definition_sample_features(
                        pb_name=pb_name,
                        split_dict_name=split_dict_name,
                        split_name=split_name,
                        idx=idx,
                        dataset=datasetdict[split_name],
                        converter=converterdict[split_name],
                        features=features,
                        report=report,
                    )

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
        "--json",
        action="store_true",
        help="Print report in JSON format",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failure",
    )
    parser.add_argument(
        "--problem-definition",
        action="append",
        default=None,
        help="Problem definition to check (can be provided multiple times)",
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
        show_progress=not args.json,
        problem_definitions=args.problem_definition,
    )

    if args.json:
        print(report.to_json())
    else:
        if not report.messages:
            print(f"[OK] {args.path}: No issue detected")
        else:
            for msg in report.messages:
                print(
                    f"[{msg.severity.upper()}] {msg.code} {msg.location}: {msg.message}"
                )
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
