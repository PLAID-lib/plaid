from pathlib import Path

import pytest

from plaid.cli.build_dataset import build_dataset_from_raw
from plaid.cli.build_dataset import build_parser
from plaid.containers.dataset import Dataset


def _write_csv(path: Path, header: str, row: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")


def _write_field(path: Path, values: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(v) for v in values) + "\n", encoding="utf-8")


def test_build_dataset_from_raw_happy_path(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    _write_csv(
        raw / "input_scalars" / "scalars_00000.csv",
        "in_a,in_b",
        "1.0,2.0",
    )
    _write_csv(
        raw / "output_scalars" / "scalars_00000.csv",
        "out_y",
        "3.5",
    )
    _write_field(raw / "field_1" / "scalars_00000.csv", [10.0, 20.0, 30.0])

    dataset = build_dataset_from_raw(input_dir=raw, output_dir=out)

    assert len(dataset) == 1
    assert dataset.get_sample_ids() == [0]
    assert dataset.get_scalar_names() == ["in_a", "in_b", "out_y"]
    assert dataset.get_field_names() == ["field_1"]
    assert out.is_dir()

    loaded = Dataset(out)
    assert len(loaded) == 1
    assert loaded.get_scalar_names() == ["in_a", "in_b", "out_y"]


def test_build_dataset_from_raw_mismatched_ids_raises(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    _write_csv(raw / "input_scalars" / "scalars_00000.csv", "in_a", "1.0")
    _write_csv(raw / "output_scalars" / "scalars_00001.csv", "out_y", "2.0")

    with pytest.raises(ValueError, match="mismatch"):
        build_dataset_from_raw(input_dir=raw, output_dir=out)


def test_build_dataset_from_raw_missing_input_scalars(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    _write_csv(
        raw / "output_scalars" / "scalars_00000.csv",
        "out_y",
        "2.0",
    )

    with pytest.raises(FileNotFoundError, match="input scalars"):
        build_dataset_from_raw(input_dir=raw, output_dir=out)


def test_build_dataset_from_raw_non_numeric_scalar(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    _write_csv(
        raw / "input_scalars" / "scalars_00000.csv",
        "in_a",
        "abc",
    )
    _write_csv(
        raw / "output_scalars" / "scalars_00000.csv",
        "out_y",
        "2.0",
    )

    with pytest.raises(ValueError, match="not numeric"):
        build_dataset_from_raw(input_dir=raw, output_dir=out)


def test_build_dataset_from_raw_duplicate_scalar_names(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    _write_csv(
        raw / "input_scalars" / "scalars_00000.csv",
        "x",
        "1.0",
    )
    _write_csv(
        raw / "output_scalars" / "scalars_00000.csv",
        "x",
        "2.0",
    )

    with pytest.raises(ValueError, match="duplicated scalar names"):
        build_dataset_from_raw(input_dir=raw, output_dir=out)


def test_build_dataset_from_raw_field_ids_mismatch(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    _write_csv(
        raw / "input_scalars" / "scalars_00000.csv",
        "in_a",
        "1.0",
    )
    _write_csv(
        raw / "output_scalars" / "scalars_00000.csv",
        "out_y",
        "2.0",
    )
    _write_field(raw / "field_1" / "scalars_00001.csv", [1.0, 2.0])

    with pytest.raises(ValueError, match="Field directory"):
        build_dataset_from_raw(input_dir=raw, output_dir=out)


def test_build_dataset_from_raw_invalid_location(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    _write_csv(
        raw / "input_scalars" / "scalars_00000.csv",
        "in_a",
        "1.0",
    )
    _write_csv(
        raw / "output_scalars" / "scalars_00000.csv",
        "out_y",
        "2.0",
    )
    _write_field(raw / "field_1" / "scalars_00000.csv", [1.0, 2.0])

    with pytest.raises(ValueError, match="Only 'Vertex'"):
        build_dataset_from_raw(
            input_dir=raw,
            output_dir=out,
            field_location="CellCenter",
        )


def test_build_dataset_from_raw_overwrite(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    out = tmp_path / "out"

    _write_csv(raw / "input_scalars" / "scalars_00000.csv", "in_a", "1.0")
    _write_csv(raw / "output_scalars" / "scalars_00000.csv", "out_y", "2.0")

    out.mkdir(parents=True, exist_ok=True)
    (out / "obsolete.txt").write_text("old", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_dataset_from_raw(input_dir=raw, output_dir=out, overwrite=False)

    build_dataset_from_raw(input_dir=raw, output_dir=out, overwrite=True)
    assert out.is_dir()
    assert not (out / "obsolete.txt").exists()


def test_build_parser_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args([
        "--input-dir",
        "raw",
        "--output-dir",
        "out",
    ])
    assert args.input_scalars_dir == "input_scalars"
    assert args.output_scalars_dir == "output_scalars"
    assert args.field_location == "Vertex"
