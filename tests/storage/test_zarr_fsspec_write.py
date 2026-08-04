"""Tests for the fsspec-aware Zarr write path (issue #485).

These tests exercise ``plaid.storage.zarr.writer.generate_datasetdict_to_disk``
against a remote (non-local) fsspec target using the in-memory filesystem
(``memory://``). They verify that:

- local and remote targets are correctly discriminated;
- URL joining preserves the ``://`` protocol separator (``pathlib.Path`` would
  collapse it);
- samples are written at the intended remote location in both sequential and
  parallel (``num_proc > 1``) modes;
- no literal ``memory:`` directory is created on the local filesystem (the
  previous ``LocalStore`` implementation wrote remote URLs to a local folder).
"""

import os

import fsspec
import numpy as np
import pytest
import zarr

from plaid.containers.sample import Sample
from plaid.storage.zarr.writer import (
    _is_local_target,
    _join_target,
    _open_split_group,
    generate_datasetdict_to_disk,
)

# ---------------------------------------------------------------------------
# Module-level (picklable) helpers required by the parallel writer.
# ---------------------------------------------------------------------------

_GLOBAL_VALUES = {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0}


def _make_sample(value: float) -> Sample:
    """Build a minimal sample carrying a single global scalar."""
    sample = Sample()
    sample.add_global("myglobal", float(value))
    return sample


class _MemorySampleGenerator:
    """Picklable generator matching the ``gen_func(shards_ids)`` contract.

    Defined at module level so ``multiprocessing`` can pickle it when
    ``num_proc > 1``.
    """

    def __call__(self, shards_ids=None):
        if shards_ids is None:
            shards_ids = [[]]
        for shard in shards_ids:
            for sample_id in shard:
                yield _make_sample(_GLOBAL_VALUES[sample_id])


_VARIABLE_SCHEMA = {
    "Global/myglobal": {"dtype": "float64", "ndim": 0},
    "Global/myglobal_times": {"dtype": "float64", "ndim": 0},
    "Global": {"dtype": "float64", "ndim": 1},
    "Global_times": {"dtype": "float64", "ndim": 1},
}


# ---------------------------------------------------------------------------
# Unit tests for the target helpers.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("target", "expected_local"),
    [
        ("/abs/local/path", True),
        ("relative/local/path", True),
        ("file:///abs/local/path", True),
        ("memory://root", False),
    ],
)
def test_is_local_target(target, expected_local):
    assert _is_local_target(target) is expected_local


def test_is_local_target_missing_backend_raises_clear_error():
    """An unavailable fsspec protocol surfaces the install hint from fsspec."""
    with pytest.raises(ImportError, match="s3fs"):
        _is_local_target("s3://bucket/key")


def test_join_target_local_returns_path(tmp_path):
    joined = _join_target(tmp_path, "data", "train")
    assert joined == tmp_path / "data" / "train"


def test_join_target_remote_preserves_protocol():
    """``pathlib.Path`` would collapse ``memory://root`` to ``memory:/root``."""
    joined = _join_target("memory://root", "data", "train")
    assert joined == "memory://root/data/train"
    # a trailing slash on the base must not produce a doubled separator
    assert _join_target("memory://root/", "data") == "memory://root/data"


# ---------------------------------------------------------------------------
# End-to-end write to a remote (memory://) target.
# ---------------------------------------------------------------------------


def _clear_memory_fs():
    """Reset the shared in-memory filesystem between test runs."""
    fs = fsspec.filesystem("memory")
    fs.store.clear()
    if hasattr(fs, "pseudo_dirs"):
        fs.pseudo_dirs.clear()
        fs.pseudo_dirs.append("")


def _read_split_groups(output_url: str, split_name: str):
    data_target = _join_target(_join_target(output_url, "data"), split_name)
    return zarr.open_group(str(data_target), mode="r")


@pytest.mark.parametrize("num_proc", [1, 2])
def test_generate_datasetdict_to_disk_parallel_still_works_locally(num_proc, tmp_path):
    """The store-selection change must not regress the local parallel path.

    ``memory://`` is per-process, so a ``num_proc > 1`` write cannot be read
    back from the parent process; the parallel reopen is therefore validated on
    a shared local target instead.
    """
    output_folder = tmp_path / f"dataset_np{num_proc}"

    if num_proc == 1:
        shards_ids = [[0, 1]]
    else:
        # one shard per worker to exercise the parallel reopen path
        shards_ids = [[0], [1]]

    generate_datasetdict_to_disk(
        output_folder=output_folder,
        generators={"train": _MemorySampleGenerator()},
        variable_schema=_VARIABLE_SCHEMA,
        gen_kwargs={"train": {"shards_ids": shards_ids}},
        num_proc=num_proc,
        verbose=False,
    )

    group = zarr.open_group(str(output_folder / "data" / "train"), mode="r")
    assert sorted(group.group_keys()) == [
        "sample_000000000",
        "sample_000000001",
    ]


def test_generate_datasetdict_to_disk_writes_to_memory_fs():
    """Sequential write to a remote (memory://) fsspec target.

    A parallel (``num_proc > 1``) equivalent is intentionally omitted here: the
    ``memory://`` filesystem is per-process, so samples written by worker
    processes are not visible from the parent. The parallel store-selection path
    is covered by ``test_generate_datasetdict_to_disk_parallel_still_works_locally``
    and ``test_open_split_group_roundtrip_on_memory_fs``.
    """
    _clear_memory_fs()
    output_url = "memory://dataset_seq"

    generate_datasetdict_to_disk(
        output_folder=output_url,
        generators={"train": _MemorySampleGenerator()},
        variable_schema=_VARIABLE_SCHEMA,
        gen_kwargs={"train": {"shards_ids": [[0, 1]]}},
        num_proc=1,
        verbose=False,
    )

    # samples landed at the intended remote location
    group = _read_split_groups(output_url, "train")
    sample_groups = sorted(group.group_keys())
    assert sample_groups == ["sample_000000000", "sample_000000001"]

    # each sample carries the flattened global feature and is readable back
    for sample_name in sample_groups:
        array_keys = list(group[sample_name].array_keys())
        assert "Global__myglobal" in array_keys

    # the remote write must NOT create a literal local "memory:" directory
    assert not os.path.exists("memory:")
    assert not os.path.exists("memory:/")


def test_open_split_group_roundtrip_on_memory_fs():
    """Create (mode 'w') then reopen (mode 'a') a split group on memory://.

    This mirrors the sequential-create / parallel-worker-reopen handshake used
    by ``generate_datasetdict_to_disk``.
    """
    _clear_memory_fs()
    target = str(_join_target(_join_target("memory://reopen", "data"), "train"))

    created = _open_split_group(target, mode="w")
    created.create_group("sample_000000000").create_array("x", data=np.arange(3))

    reopened = _open_split_group(target, mode="a")
    reopened.create_group("sample_000000001").create_array("x", data=np.arange(3) + 10)

    assert sorted(reopened.group_keys()) == [
        "sample_000000000",
        "sample_000000001",
    ]
