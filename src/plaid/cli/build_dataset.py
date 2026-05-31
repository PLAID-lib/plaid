"""CLI to build a PLAID dataset from a raw CSV directory layout."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from plaid import Dataset, Sample


FileMap = dict[int, Path]
FieldMaps = dict[str, FileMap]


def _extract_sample_id(path: Path) -> int:
    """Extract integer sample id from a filename stem.

    Expected stems end with digits (e.g. ``scalars_00012``).
    """
    stem = path.stem
    i = len(stem) - 1
    while i >= 0 and stem[i].isdigit():
        i -= 1
    suffix = stem[i + 1 :]
    if suffix == "":# pragma: no cover 
        msg = (
            f"Could not extract sample id from filename '{path.name}'. "
            "Expected a numeric suffix."
        )
        raise ValueError(msg)
    return int(suffix)


def _read_scalar_row(path: Path) -> dict[str, str]:
    """Read a scalar CSV file as a single-row dict.

    The file must contain exactly one data row with headers.
    """
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if len(rows) != 1:# pragma: no cover 
        msg = (
            f"Scalar file '{path}' must contain exactly one data row, "
            f"got {len(rows)}."
        )
        raise ValueError(msg)
    return rows[0]


def _read_field_csv(path: Path) -> np.ndarray:
    """Read field values from CSV into a numpy array."""
    field = np.loadtxt(path, delimiter=",")
    field = np.asarray(field)
    if field.ndim == 0:  # pragma: no cover 
        field = field.reshape(1)
    return field


def _coerce_scalar(value: str) -> float:
    """Convert scalar text to float.

    Raises:
        ValueError: If the value cannot be parsed as a float.
    """
    txt = value.strip()
    if txt == "":  # pragma: no cover 
        raise ValueError("Empty scalar value is not allowed.")
    try:
        return float(txt)
    except ValueError:
        raise ValueError(f"Scalar value '{value}' is not numeric.") from None


def _discover_sample_files(directory: Path, file_glob: str) -> dict[int, Path]:
    """Map sample ids to files for one directory."""
    files = sorted(directory.glob(file_glob))
    if not files:  # pragma: no cover  
        msg = f"No files found in '{directory}' with pattern '{file_glob}'."
        raise FileNotFoundError(msg)

    mapped = {}
    for file in files:
        if not file.is_file():
            continue
        sid = _extract_sample_id(file)
        if sid in mapped:  # pragma: no cover 
            msg = (
                f"Duplicate sample id {sid} in '{directory}' for "
                f"'{mapped[sid]}' and '{file}'."
            )
            raise ValueError(msg)
        mapped[sid] = file

    if not mapped:  # pragma: no cover 
        msg = (
            f"No matching files found in '{directory}' with pattern "
            f"'{file_glob}'."
        )
        raise FileNotFoundError(msg)
    return mapped


def _validate_raw_layout(
    input_dir: Path,
    input_scalars_dir_name: str,
    output_scalars_dir_name: str,
    file_glob: str,
    field_dirs: list[str] | None,
) -> tuple[FileMap, FileMap, FieldMaps]:
    """Validate raw layout and return discovered file maps."""
    in_scalars_dir = input_dir / input_scalars_dir_name
    out_scalars_dir = input_dir / output_scalars_dir_name

    if not in_scalars_dir.is_dir():
        msg = f"Missing input scalars directory: '{in_scalars_dir}'."
        raise FileNotFoundError(msg)
    if not out_scalars_dir.is_dir():  # pragma: no cover 
        msg = f"Missing output scalars directory: '{out_scalars_dir}'."
        raise FileNotFoundError(msg)

    in_scalars = _discover_sample_files(in_scalars_dir, file_glob)
    out_scalars = _discover_sample_files(out_scalars_dir, file_glob)

    if set(in_scalars) != set(out_scalars):
        missing_in_output = sorted(set(in_scalars) - set(out_scalars))
        missing_in_input = sorted(set(out_scalars) - set(in_scalars))
        msg = (
            "Input/output scalar sample IDs mismatch. "
            "Missing in output: "
            f"{missing_in_output}; missing in input: {missing_in_input}."
        )
        raise ValueError(msg)

    if field_dirs is None:
        excluded = {input_scalars_dir_name, output_scalars_dir_name}
        field_dirs = [
            directory.name
            for directory in input_dir.iterdir()
            if directory.is_dir() and directory.name not in excluded
        ]
        field_dirs = sorted(field_dirs)

    fields_map: FieldMaps = {}
    for field_name in field_dirs:
        field_dir = input_dir / field_name
        if not field_dir.is_dir():  # pragma: no cover 
            msg = (
                f"Field directory '{field_name}' not found under "
                f"'{input_dir}'."
            )
            raise FileNotFoundError(msg)

        local_map = _discover_sample_files(field_dir, file_glob)
        if set(local_map) != set(in_scalars):
            missing_in_field = sorted(set(in_scalars) - set(local_map))
            extra_in_field = sorted(set(local_map) - set(in_scalars))
            msg = (
                f"Field directory '{field_name}' sample IDs mismatch "
                "with scalar IDs. "
                f"Missing in field: {missing_in_field}; "
                f"extra in field: {extra_in_field}."
            )
            raise ValueError(msg)
        fields_map[field_name] = local_map

    return in_scalars, out_scalars, fields_map


def _initialize_sample_geometry(
    sample: Sample,
    fields: dict[str, np.ndarray],
    field_location: str,
    base_name: str,
    zone_name: str,
) -> None:
    """Initialize minimal mesh required to store fields in a Sample."""
    if not fields:
        return

    if field_location != "Vertex":
        msg = (
            "Only 'Vertex' field location is currently supported by "
            "this CLI builder."
        )
        raise ValueError(msg)

    first_field_name, first_field = next(iter(fields.items()))
    n_nodes = first_field.shape[0] if first_field.ndim > 1 else first_field.size
    if n_nodes <= 0:  # pragma: no cover 
        raise ValueError("Field size must be positive.")

    for field_name, field_array in fields.items():
        local_n_nodes = (
            field_array.shape[0] if field_array.ndim > 1 else field_array.size
        )
        if local_n_nodes != n_nodes:
            msg = (
                "All fields must share the same number of support points. "
                f"Reference field '{first_field_name}' has {n_nodes}, "
                f"field '{field_name}' has {local_n_nodes}."
            )
            raise ValueError(msg)

    if sample.features is None:  # pragma: no cover 
        raise ValueError("Sample features are not initialized.")
    features = sample.features

    features.init_base(1, 1, base_name=base_name)
    zone_shape = np.array([[n_nodes, 0, 0]], dtype=np.int32)
    features.init_zone(
        zone_shape=zone_shape,
        zone_name=zone_name,
        base_name=base_name,
    )
    nodes = np.arange(n_nodes, dtype=np.float64).reshape(-1, 1)
    features.set_nodes(
        nodes=nodes,
        zone_name=zone_name,
        base_name=base_name,
    )


def build_dataset_from_raw(
    input_dir: Path,
    output_dir: Path,
    input_scalars_dir_name: str = "input_scalars",
    output_scalars_dir_name: str = "output_scalars",
    field_dirs: list[str] | None = None,
    field_location: str = "Vertex",
    base_name: str = "Base_1_1",
    zone_name: str = "Zone",
    file_glob: str = "scalars_*.csv",
    overwrite: bool = False,
) -> Dataset:
    """Build and save a PLAID dataset from raw directory data."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.is_dir():  # pragma: no cover 
        msg = f"Input directory does not exist: '{input_dir}'."
        raise FileNotFoundError(msg)

    in_scalars, out_scalars, fields_map = _validate_raw_layout(
        input_dir=input_dir,
        input_scalars_dir_name=input_scalars_dir_name,
        output_scalars_dir_name=output_scalars_dir_name,
        file_glob=file_glob,
        field_dirs=field_dirs,
    )

    if output_dir.exists():
        if not overwrite:
            msg = (
                f"Output directory already exists: '{output_dir}'. "
                "Use --overwrite to replace it."
            )
            raise FileExistsError(msg)
        shutil.rmtree(output_dir)

    dataset = Dataset()

    for sid in sorted(in_scalars.keys()):
        sample = Sample(path=None)
        if sample.features is None:  # pragma: no cover 
            raise ValueError("Sample features are not initialized.")
        features = sample.features

        in_row = _read_scalar_row(in_scalars[sid])
        out_row = _read_scalar_row(out_scalars[sid])

        duplicate_keys = sorted(set(in_row).intersection(set(out_row)))
        if duplicate_keys:
            msg = (
                f"Sample {sid}: duplicated scalar names between "
                f"input/output scalar files: {duplicate_keys}."
            )
            raise ValueError(msg)

        for key, value in in_row.items():
            sample.add_scalar(key, _coerce_scalar(value))
        for key, value in out_row.items():
            sample.add_scalar(key, _coerce_scalar(value))

        sample_fields = {
            field_name: _read_field_csv(field_file_map[sid])
            for field_name, field_file_map in fields_map.items()
        }

        _initialize_sample_geometry(
            sample=sample,
            fields=sample_fields,
            field_location=field_location,
            base_name=base_name,
            zone_name=zone_name,
        )

        for field_name, field_array in sample_fields.items():
            features.add_field(
                name=field_name,
                field=field_array,
                location=field_location,
                base_name=base_name,
                zone_name=zone_name,
            )

        dataset.add_sample(sample=sample, id=sid)

    dataset.save_to_dir(output_dir)
    return dataset


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for dataset builder CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a PLAID dataset from raw CSV directories "
            "in a single command."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the raw input directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path where the PLAID dataset will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--input-scalars-dir",
        default="input_scalars",
        help="Subdirectory name that contains input scalar CSV files.",
    )
    parser.add_argument(
        "--output-scalars-dir",
        default="output_scalars",
        help="Subdirectory name that contains output scalar CSV files.",
    )
    parser.add_argument(
        "--field-dirs",
        nargs="*",
        default=None,
        help=(
            "Field directory names under --input-dir. If omitted, "
            "all non-scalar subdirs are used."
        ),
    )
    parser.add_argument(
        "--field-location",
        default="Vertex",
        choices=["Vertex"],
        help="Field location in PLAID samples (currently only 'Vertex').",
    )
    parser.add_argument(
        "--base-name",
        default="Base_1_1",
        help="Base name used when initializing sample features.",
    )
    parser.add_argument(
        "--zone-name",
        default="Zone",
        help="Zone name used when initializing sample features.",
    )
    parser.add_argument(
        "--file-glob",
        default="scalars_*.csv",
        help="Glob pattern used to discover CSV files in each subdirectory.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a completion message with the generated output path.",
    )
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()

    dataset = build_dataset_from_raw(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        input_scalars_dir_name=args.input_scalars_dir,
        output_scalars_dir_name=args.output_scalars_dir,
        field_dirs=args.field_dirs,
        field_location=args.field_location,
        base_name=args.base_name,
        zone_name=args.zone_name,
        file_glob=args.file_glob,
        overwrite=args.overwrite,
    )

    if args.verbose:
        out = Path(args.output_dir).resolve()
        print(f"Built PLAID dataset with {len(dataset)} samples at '{out}'.")


if __name__ == "__main__":  # pragma: no cover 
    main()
