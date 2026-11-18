"""Round-trip and validation tests for dataset operations."""

import runpy
import sys

import numpy as np
import pytest

from plaid.cli import tools
from plaid.containers.sample import Sample
from plaid.infos import Infos
from plaid.problem_definition import ProblemDefinition
from plaid.storage import init_from_disk, save_to_disk


def _sample(index):
    sample = Sample()
    sample.add_global("value", float(index))
    sample.add_global("vector", np.array([index, index + 1], dtype=np.float32))
    return sample


def _source(path, backend="cgns", owner="owner", splits=None, definitions=False):
    save_to_disk(
        path,
        sample_constructor=_sample,
        ids={"train": [0, 1, 2, 3], "test": [4, 5]} if splits is None else splits,
        backend=backend,
        infos=Infos(owner=owner, license="BSD-3-Clause"),
        pb_defs={
            "regression": ProblemDefinition(
                input_features=["vector"],
                output_features=["value"],
                train_split={"train": "all"},
                test_split={"test": "all"},
            )
        }
        if definitions
        else None,
    )
    return path


@pytest.fixture
def source(tmp_path):
    return _source(tmp_path / "source")


def _values(path):
    datasets, converters = init_from_disk(path)
    return {
        split: [
            float(converters[split].to_plaid(dataset, i).get_global("value"))
            for i in range(len(dataset))
        ]
        for split, dataset in datasets.items()
    }


@pytest.mark.parametrize("source_backend", ["cgns", "hf_datasets", "zarr"])
@pytest.mark.parametrize("output_backend", [None, "cgns", "hf_datasets", "zarr"])
def test_subset_backends(tmp_path, source_backend, output_backend):
    source = _source(tmp_path / "source", source_backend)
    output = tmp_path / "nested" / "output"
    tools.subset(
        source, output, select=["train:3,0:2", "test:1"], backend=output_backend
    )
    assert _values(output) == {"train": [3, 0, 1], "test": [5]}
    info = Infos.from_path(output / "infos.yaml")
    assert info.storage_backend == (output_backend or source_backend)
    assert info.num_samples == {"train": 3, "test": 1}
    assert _values(source) == {"train": [0, 1, 2, 3], "test": [4, 5]}


def test_subset_all_and_warning(tmp_path):
    source = _source(tmp_path / "source", definitions=True)
    with pytest.warns(UserWarning, match="Problem definitions.*omitted"):
        tools.subset(source, tmp_path / "output")
    assert _values(tmp_path / "output") == _values(source)
    assert not (tmp_path / "output" / "problem_definitions").exists()


@pytest.mark.parametrize("backend", [None, "cgns", "hf_datasets", "zarr"])
def test_merge_mixed_backends(source, tmp_path, backend):
    second = _source(tmp_path / "second", "zarr", splits={"train": [6, 7], "OOD": [8]})
    third = _source(tmp_path / "third", "hf_datasets", splits={"train": [9]})
    output = tmp_path / "merged"
    tools.merge([source, second, third], output, backend=backend)
    assert _values(output) == {
        "train": [0, 1, 2, 3, 6, 7, 9],
        "test": [4, 5],
        "OOD": [8],
    }
    assert Infos.from_path(output / "infos.yaml").storage_backend == (backend or "cgns")


def test_merge_metadata(source, tmp_path):
    second = _source(tmp_path / "second", owner="other")
    with pytest.raises(ValueError, match="Conflicting"):
        tools.merge([source, second], tmp_path / "output")
    assert not (tmp_path / "output").exists()
    tools.merge([source, second], tmp_path / "output", metadata="first")
    assert Infos.from_path(tmp_path / "output" / "infos.yaml").owner == "owner"
    assert _values(tmp_path / "output")["train"] == [0, 1, 2, 3] * 2


@pytest.mark.parametrize("backend", [None, "cgns", "hf_datasets", "zarr"])
def test_partition_reproducible(source, tmp_path, backend):
    for name in ("a", "b"):
        tools.partition(
            source,
            tmp_path / name,
            ratios=["train=0.5", "val=0.25", "test=0.25"],
            backend=backend,
        )
    values = _values(tmp_path / "a")
    assert values == _values(tmp_path / "b")
    assert {name: len(rows) for name, rows in values.items()} == {
        "train": 3,
        "val": 2,
        "test": 1,
    }
    assert sorted(sum(values.values(), [])) == list(range(6))
    assert Infos.from_path(tmp_path / "a" / "infos.yaml").storage_backend == (
        backend or "cgns"
    )


def test_partition_counts_selected(source, tmp_path):
    tools.partition(
        source,
        tmp_path / "output",
        counts=["a=2", "b=1"],
        select=["train:3,0", "test:1"],
        shuffle=False,
    )
    assert _values(tmp_path / "output") == {"a": [3, 0], "b": [5]}


@pytest.mark.parametrize(
    "expression,expected",
    [
        (":", [0, 1, 2, 3]),
        ("::2", [0, 2]),
        ("1:3:", [1, 2]),
        ("0,2:4", [0, 2, 3]),
    ],
)
def test_indices(expression, expected):
    assert tools._indices(expression, 4) == expected


@pytest.mark.parametrize(
    "selector",
    [
        "missing",
        "train:4",
        "train:-1",
        "train:0:5",
        "train:3:1",
        "train:0:2:0",
        "train:0:2:-1",
        "train:0:1:2:3",
        "train:",
        "train:no",
        "train:0,0",
        "train:0:0",
        "../train",
        "train:0,",
    ],
)
def test_invalid_selection(source, tmp_path, selector):
    with pytest.raises(ValueError):
        tools.subset(source, tmp_path / "output", select=[selector])
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("selection", [[], ["train", "train:0"]])
def test_empty_or_repeated_selection(source, tmp_path, selection):
    with pytest.raises(ValueError):
        tools.subset(source, tmp_path / "output", select=selection)


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b", "a:b"])
def test_split_name(name):
    with pytest.raises(ValueError, match="Invalid split"):
        tools._split_name(name)


@pytest.mark.parametrize(
    "spec,total,ratios",
    [
        ([], 6, True),
        (["a"], 6, True),
        (["a=1", "a=1"], 6, False),
        (["a=nan"], 6, True),
        (["a=inf"], 6, True),
        (["a=-1"], 6, False),
        (["a=0"], 6, False),
        (["a=0.3", "b=0.3"], 6, True),
        (["a=2"], 6, False),
        (["a=0.5", "b=0.5"], 1, True),
        (["a=1.5"], 6, False),
        (["a=no"], 6, True),
    ],
)
def test_invalid_sizes(spec, total, ratios):
    with pytest.raises(ValueError):
        tools._sizes(spec, total, ratios)


@pytest.mark.parametrize(
    "options",
    [{}, {"counts": ["a=6"], "ratios": ["a=1"]}, {"counts": ["a=6"], "seed": -1}],
)
def test_invalid_partition(source, tmp_path, options):
    with pytest.raises(ValueError):
        tools.partition(source, tmp_path / "output", **options)


def test_merge_invalid(source, tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        tools.merge([source], tmp_path / "output")
    with pytest.raises(ValueError, match="metadata"):
        tools.merge([source, source], tmp_path / "output", metadata="invalid")


def test_output_protection(source, tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        tools.subset(source, output)
    with pytest.raises(FileExistsError):
        tools.subset(source, source)
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        tools.subset(source, link)
    with pytest.raises(ValueError, match="backend"):
        tools.subset(source, tmp_path / "unknown", backend="unknown")
    loaded = tools._load([source])
    with pytest.raises(ValueError, match="nonempty"):
        tools._write(loaded, {"train": []}, tmp_path / "empty", None)


def test_failed_write_cleanup(source, tmp_path):
    datasets, converters, info = tools._load([source])[0]
    # A real invalid source index fails during sample conversion, not a mock.
    with pytest.raises(AssertionError):
        tools._write(
            [(datasets, converters, info)],
            {"train": [(0, "train", 999)]},
            tmp_path / "output",
            "cgns",
        )
    assert not (tmp_path / "output").exists()
    assert not list(tmp_path.glob(".plaid-tools-*"))


def test_output_created_during_write(source, tmp_path):
    output = tmp_path / "output"
    datasets, converters, info = tools._load([source])[0]

    class CreatingConverter:
        def to_plaid(self, dataset, index):
            output.mkdir(exist_ok=True)
            (output / "marker").write_text("keep")
            return converters["train"].to_plaid(dataset, index)

    with pytest.raises(FileExistsError):
        tools._write(
            [(datasets, {"train": CreatingConverter()}, info)],
            {"train": [(0, "train", 0)]},
            output,
            "cgns",
        )
    assert (output / "marker").read_text() == "keep"
    assert not list(tmp_path.glob(".plaid-tools-*"))


@pytest.mark.parametrize(
    "command,extra",
    [
        ("subset", ["--select", "train:0:2"]),
        ("merge", []),
        ("partition", ["--counts", "a=3", "b=3", "--no-shuffle"]),
    ],
)
def test_cli(source, tmp_path, capsys, command, extra):
    output = tmp_path / "output"
    inputs = [str(source)] * (2 if command == "merge" else 1)
    assert tools.main([command, *inputs, str(output), *extra, "--backend", "zarr"]) == 0
    assert "Saved dataset" in capsys.readouterr().out
    values = _values(output)
    assert (
        sum(map(len, values.values()))
        == {"subset": 2, "merge": 12, "partition": 6}[command]
    )


@pytest.mark.parametrize(
    "arguments",
    [[], ["subset"], ["partition", "a", "b"], ["subset", "a", "b", "--backend", "bad"]],
)
def test_cli_parse_errors(arguments):
    with pytest.raises(SystemExit) as exc:
        tools.main(arguments)
    assert exc.value.code == 2


def test_cli_operation_error(source, tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        tools.main(
            ["subset", str(source), str(tmp_path / "out"), "--select", "missing"]
        )
    assert exc.value.code == 2
    assert "Unknown split" in capsys.readouterr().err


def test_module_entrypoint(source, tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["plaid-tools", "subset", str(source), str(tmp_path / "output")]
    )
    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        with pytest.raises(SystemExit) as exc:
            runpy.run_module("plaid.cli.tools", run_name="__main__")
    assert exc.value.code == 0
    assert _values(tmp_path / "output") == _values(source)
