"""Tests for dataset discovery and indexing in :class:`PlaidDatasetService`.

The service builds on ``plaid.storage.init_from_disk``. To keep these tests
lightweight and free from real CGNS/arrow fixtures, we monkey-patch that
function to return small in-memory stand-ins for ``dataset_dict`` and
``converter_dict``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plaid.viewer.config import ViewerConfig
from plaid.viewer.models import SampleRef
from plaid.viewer.services import PlaidDatasetService


class _FakeDataset(list):
    """Minimal list-like stand-in for ``datasets.Dataset``."""


class _FakeConverter:
    def __init__(self, samples_by_index: dict[int, object]) -> None:
        self._samples = samples_by_index

    def to_plaid(self, dataset, index: int):  # noqa: ARG002 - interface match
        return self._samples[index]


def _make_dataset_dir(root: Path, name: str) -> Path:
    base = root / name
    (base / "data").mkdir(parents=True, exist_ok=True)
    return base


def _install_fake_init_from_disk(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, tuple[dict, dict]],
) -> None:
    """Patch ``plaid.storage.init_from_disk`` to return per-directory fixtures."""

    def _fake(path: str):
        base_name = Path(path).name
        return payload[base_name]

    import plaid.storage as storage  # noqa: PLC0415

    monkeypatch.setattr(storage, "init_from_disk", _fake)


def test_list_datasets_returns_all_subdirectories_with_data(tmp_path: Path) -> None:
    _make_dataset_dir(tmp_path, "ds_a")
    _make_dataset_dir(tmp_path, "ds_b")
    (tmp_path / "not_a_dataset").mkdir()  # missing data/ subfolder
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    ids = {d.dataset_id for d in service.list_datasets()}
    assert ids == {"ds_a", "ds_b"}


def test_list_samples_uses_converter_to_plaid_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset_dir(tmp_path, "ds")
    dataset_dict = {
        "train": _FakeDataset(range(2)),
        "test": _FakeDataset(range(1)),
    }
    converter_dict = {
        "train": _FakeConverter({0: object(), 1: object()}),
        "test": _FakeConverter({0: object()}),
    }
    _install_fake_init_from_disk(monkeypatch, {"ds": (dataset_dict, converter_dict)})

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    refs = service.list_samples("ds")
    assert len(refs) == 3
    assert {(r.split, r.sample_id) for r in refs} == {
        ("train", "0"),
        ("train", "1"),
        ("test", "0"),
    }


def test_load_sample_calls_converter_to_plaid_with_integer_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset_dir(tmp_path, "ds")
    target = object()
    dataset_dict = {"train": _FakeDataset(range(3))}
    converter_dict = {"train": _FakeConverter({2: target})}
    _install_fake_init_from_disk(monkeypatch, {"ds": (dataset_dict, converter_dict)})

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    ref = SampleRef(backend_id="disk", dataset_id="ds", split="train", sample_id="2")
    assert service.load_sample(ref) is target


def test_get_dataset_reports_split_counts_from_dataset_dict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset_dir(tmp_path, "ds")
    dataset_dict = {
        "train": _FakeDataset(range(3)),
        "test": _FakeDataset(range(2)),
    }
    converter_dict = {
        "train": _FakeConverter({}),
        "test": _FakeConverter({}),
    }
    _install_fake_init_from_disk(monkeypatch, {"ds": (dataset_dict, converter_dict)})

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    detail = service.get_dataset("ds")
    assert detail.splits == {"train": 3, "test": 2}


def test_describe_non_visual_bases_lists_zoneless_bases_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bases that carry no ``Zone_t`` child are reported with their arrays."""
    import types

    import numpy as np
    from CGNS.PAT import cgnskeywords as CK

    _make_dataset_dir(tmp_path, "ds")

    # Build a minimal CGNS tree with one visual base (has a zone) and one
    # non-visual base (only DataArrays under a UserDefinedData_t node).
    pressure = np.array([1.5], dtype=np.float32)
    visual_base = ["Geom", None, [["Zone1", None, [], CK.Zone_ts]], CK.CGNSBase_ts]
    aux_base = [
        "Constants",
        None,
        [
            [
                "UD",
                None,
                [["Pressure", pressure, [], CK.DataArray_ts]],
                "UserDefinedData_t",
            ],
        ],
        CK.CGNSBase_ts,
    ]
    tree = ["CGNSTree", None, [visual_base, aux_base], "CGNSTree_t"]

    features = types.SimpleNamespace(data={0.0: tree})
    sample = types.SimpleNamespace(features=features)

    dataset_dict = {"train": _FakeDataset(range(1))}
    converter_dict = {"train": _FakeConverter({0: sample})}
    _install_fake_init_from_disk(monkeypatch, {"ds": (dataset_dict, converter_dict)})

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    ref = SampleRef(backend_id="disk", dataset_id="ds", split="train", sample_id="0")
    summary = service.describe_non_visual_bases(ref)

    assert list(summary.keys()) == ["Constants"]
    entries = summary["Constants"]
    assert len(entries) == 1
    entry = entries[0]
    assert entry["name"] == "Pressure"
    assert entry["shape"] == [1]
    assert "float32" in entry["dtype"]
    assert "1.5" in entry["preview"]


def test_load_sample_rejects_non_integer_sample_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset_dir(tmp_path, "ds")
    dataset_dict = {"train": _FakeDataset(range(1))}
    converter_dict = {"train": _FakeConverter({0: object()})}
    _install_fake_init_from_disk(monkeypatch, {"ds": (dataset_dict, converter_dict)})

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    ref = SampleRef(
        backend_id="disk", dataset_id="ds", split="train", sample_id="not-an-int"
    )
    with pytest.raises(ValueError):
        service.load_sample(ref)


def test_set_datasets_root_rejects_outside_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    service = PlaidDatasetService(
        ViewerConfig(datasets_root=sandbox, browse_roots=(sandbox,))
    )
    with pytest.raises(Exception):
        service.set_datasets_root(outside)


def test_set_datasets_root_updates_config(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    sub = sandbox / "sub"
    sub.mkdir()
    service = PlaidDatasetService(
        ViewerConfig(datasets_root=sandbox, browse_roots=(sandbox,))
    )
    resolved = service.set_datasets_root(sub)
    assert resolved == sub.resolve()
    assert service.datasets_root == sub.resolve()


def test_list_subdirs_returns_entries(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "a").mkdir()
    (sandbox / "b").mkdir()
    (sandbox / "b" / "data").mkdir()
    (sandbox / "b" / "problem_definitions").mkdir()
    service = PlaidDatasetService(
        ViewerConfig(datasets_root=sandbox, browse_roots=(sandbox,))
    )
    listing = service.list_subdirs(sandbox)
    names = {e["name"] for e in listing["entries"]}
    assert names == {"a", "b"}
    plaid_entry = next(e for e in listing["entries"] if e["name"] == "b")
    assert plaid_entry["is_plaid_candidate"] is True


def test_list_subdirs_rejects_outside_sandbox(tmp_path: Path) -> None:
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    service = PlaidDatasetService(
        ViewerConfig(datasets_root=sandbox, browse_roots=(sandbox,))
    )
    with pytest.raises(Exception):
        service.list_subdirs(outside)


# ---------------------------------------------------------------------------
# Hugging Face Hub streaming
# ---------------------------------------------------------------------------


def _install_fake_init_streaming_from_hub(
    monkeypatch: pytest.MonkeyPatch,
    payload: dict[str, tuple[dict, dict]],
) -> None:
    """Patch ``plaid.storage.init_streaming_from_hub`` to return fixtures."""

    def _fake(repo_id: str):
        return payload[repo_id]

    import plaid.storage as storage  # noqa: PLC0415

    monkeypatch.setattr(storage, "init_streaming_from_hub", _fake, raising=False)


def test_add_hub_dataset_rejects_invalid_repo_id(tmp_path: Path) -> None:
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    with pytest.raises(ValueError):
        service.add_hub_dataset("")
    with pytest.raises(ValueError):
        service.add_hub_dataset("missing-slash")


def test_add_hub_dataset_is_listed_alongside_local(tmp_path: Path) -> None:
    _make_dataset_dir(tmp_path, "local_ds")
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset("PLAID-lib/VKI-LS59")
    entries = service.list_datasets()
    ids = {d.dataset_id: d.backend_id for d in entries}
    assert ids == {"local_ds": "disk", "PLAID-lib/VKI-LS59": "hub"}
    # Idempotent add
    service.add_hub_dataset("PLAID-lib/VKI-LS59")
    assert len(service.list_datasets()) == 2


def test_list_samples_streams_from_hub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "PLAID-lib/VKI-LS59"
    dataset_dict = {
        "train": _FakeDataset(range(2)),
    }
    converter_dict = {
        "train": _FakeConverter({0: object(), 1: object()}),
    }
    _install_fake_init_streaming_from_hub(
        monkeypatch, {repo_id: (dataset_dict, converter_dict)}
    )
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)
    refs = service.list_samples(repo_id)
    assert {(r.backend_id, r.split, r.sample_id) for r in refs} == {
        ("hub", "train", "0"),
        ("hub", "train", "1"),
    }


def test_remove_hub_dataset_clears_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "org/ds"
    _install_fake_init_streaming_from_hub(
        monkeypatch,
        {
            repo_id: (
                {"train": _FakeDataset(range(1))},
                {"train": _FakeConverter({0: object()})},
            )
        },
    )
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)
    service.list_samples(repo_id)  # populates cache
    assert repo_id in service._store_cache  # noqa: SLF001
    service.remove_hub_dataset(repo_id)
    assert repo_id not in service._store_cache  # noqa: SLF001
    assert repo_id not in [d.dataset_id for d in service.list_datasets()]


# ---------------------------------------------------------------------------
# Streaming cursor behaviour (IterableDataset without __len__)
# ---------------------------------------------------------------------------


class _FakeIterableDataset:
    """Stand-in for ``datasets.IterableDataset`` - no ``__len__``."""

    def __init__(self, records: list[object]) -> None:
        self._records = records

    def __iter__(self):
        return iter(self._records)


class _FakeStreamingConverter:
    """Converter exposing ``sample_to_plaid`` (streaming API)."""

    def __init__(self, mapping: dict[int, object]) -> None:
        # Maps the raw record itself to a PLAID sample, using id() lookup
        # so we can assert the correct record was forwarded.
        self._mapping = mapping

    def sample_to_plaid(self, record):
        return self._mapping[record]

    # Intentionally no ``to_plaid`` method: streaming paths must not use it.


def _install_fake_streaming_dataset(
    monkeypatch: pytest.MonkeyPatch, repo_id: str
) -> tuple[list[object], dict[int, object]]:
    """Register a 3-record streaming dataset and return (records, mapping)."""
    records = [object(), object(), object()]
    mapping = {rec: object() for rec in records}
    dataset_dict = {"train": _FakeIterableDataset(records)}
    converter_dict = {"train": _FakeStreamingConverter(mapping)}
    _install_fake_init_streaming_from_hub(
        monkeypatch, {repo_id: (dataset_dict, converter_dict)}
    )
    return records, mapping


def test_streaming_dataset_is_detected_as_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "org/stream"
    _install_fake_streaming_dataset(monkeypatch, repo_id)
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)
    assert service.is_streaming(repo_id) is True
    # Splits without __len__ report a ``None`` count in the detail view.
    detail = service.get_dataset(repo_id)
    assert detail.splits == {"train": None}


def test_list_samples_emits_single_cursor_ref_for_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "org/stream"
    _install_fake_streaming_dataset(monkeypatch, repo_id)
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)
    refs = service.list_samples(repo_id)
    # Streaming splits surface a single synthetic reference using the
    # sentinel sample id, regardless of how many records the stream holds.
    assert len(refs) == 1
    assert refs[0].backend_id == "hub"
    assert refs[0].sample_id == "cursor"
    assert refs[0].split == "train"


def test_advance_stream_cursor_walks_records_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "org/stream"
    records, mapping = _install_fake_streaming_dataset(monkeypatch, repo_id)
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)

    # No sample fetched yet.
    assert service.stream_cursor_position(repo_id, "train") == -1

    ref0 = service.advance_stream_cursor(repo_id, "train")
    assert service.stream_cursor_position(repo_id, "train") == 0
    # ``load_sample`` must materialise the record that the cursor just
    # consumed, going through ``converter.sample_to_plaid``.
    sample0 = service.load_sample(ref0)
    assert sample0 is mapping[records[0]]

    # Advancing again moves forward and does not re-consume the first
    # record.
    ref1 = service.advance_stream_cursor(repo_id, "train")
    assert service.stream_cursor_position(repo_id, "train") == 1
    assert service.load_sample(ref1) is mapping[records[1]]


def test_advance_stream_cursor_raises_when_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "org/stream"
    _install_fake_streaming_dataset(monkeypatch, repo_id)
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)
    # Three records in the fake stream; the fourth advance must stop.
    for _ in range(3):
        service.advance_stream_cursor(repo_id, "train")
    with pytest.raises(StopIteration):
        service.advance_stream_cursor(repo_id, "train")


def test_reset_stream_cursor_rewinds_to_first_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_id = "org/stream"
    records, mapping = _install_fake_streaming_dataset(monkeypatch, repo_id)
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)
    service.advance_stream_cursor(repo_id, "train")
    service.advance_stream_cursor(repo_id, "train")
    assert service.stream_cursor_position(repo_id, "train") == 1

    service.reset_stream_cursor(repo_id, "train")
    assert service.stream_cursor_position(repo_id, "train") == -1
    ref = service.advance_stream_cursor(repo_id, "train")
    assert service.load_sample(ref) is mapping[records[0]]


# ---------------------------------------------------------------------------
# Feature filtering
# ---------------------------------------------------------------------------


def _install_fake_metadata(
    monkeypatch: pytest.MonkeyPatch,
    *,
    variable_schema: dict[str, object],
    constant_schema: dict[str, dict[str, object]],
) -> None:
    """Patch ``load_metadata_from_disk`` / ``load_metadata_from_hub``."""
    from plaid.storage.common import reader as reader_mod  # noqa: PLC0415

    def _fake(*_args, **_kwargs):
        return ({}, variable_schema, constant_schema, {})

    monkeypatch.setattr(reader_mod, "load_metadata_from_disk", _fake, raising=False)
    monkeypatch.setattr(reader_mod, "load_metadata_from_hub", _fake, raising=False)


class _FeatureAwareConverter:
    """Converter recording the feature list handed to ``to_plaid``."""

    def __init__(
        self,
        samples_by_index: dict[int, object],
        *,
        constant_features: set[str] | None = None,
        variable_features: set[str] | None = None,
    ) -> None:
        self._samples = samples_by_index
        self.constant_features = constant_features or set()
        self.variable_features = variable_features or set()
        self.last_features: list[str] | None = None

    def to_plaid(self, dataset, index: int, features=None):  # noqa: ARG002
        self.last_features = list(features) if features is not None else None
        return self._samples[index]


def test_list_available_features_only_exposes_field_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset_dir(tmp_path, "ds")
    variable = {
        "Base_2_2/Zone/VertexFields/pressure": None,
        "Base_2_2/Zone/GridCoordinates/CoordinateX": None,
    }
    constant = {
        "train": {
            "Base_2_2/Zone/VertexFields/sdf": None,
            "Base_2_2/Zone/VertexFields/sdf_times": None,
            "Base_2_2/Zone/VertexFields/GridLocation": None,
            "Global/angle_in": None,
        }
    }
    _install_fake_metadata(
        monkeypatch, variable_schema=variable, constant_schema=constant
    )
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    fields = service.list_available_features("ds")
    assert "Base_2_2/Zone/VertexFields/pressure" in fields
    assert "Base_2_2/Zone/VertexFields/sdf" in fields
    # Coordinates, time bookkeeping, GridLocation metadata and scalars
    # must not appear in the user-facing feature list.
    assert "Base_2_2/Zone/GridCoordinates/CoordinateX" not in fields
    assert "Base_2_2/Zone/VertexFields/sdf_times" not in fields
    assert "Base_2_2/Zone/VertexFields/GridLocation" not in fields
    assert "Global/angle_in" not in fields


def test_set_features_rejects_unknown_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset_dir(tmp_path, "ds")
    _install_fake_metadata(
        monkeypatch,
        variable_schema={"Base/Zone/VertexFields/pressure": None},
        constant_schema={"train": {}},
    )
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    with pytest.raises(ValueError):
        service.set_features("ds", ["not/a/feature"])


def test_load_sample_forwards_selected_features_on_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disk path: ``to_plaid`` receives the filtered feature list."""
    _make_dataset_dir(tmp_path, "ds")
    variable = {"Base/Zone/VertexFields/pressure": None}
    constant = {
        "train": {
            "Base": None,
            "Base/Zone": None,
            "Base/Zone/VertexFields": None,
        }
    }
    _install_fake_metadata(
        monkeypatch, variable_schema=variable, constant_schema=constant
    )
    target = object()
    dataset_dict = {"train": _FakeDataset(range(1))}
    converter = _FeatureAwareConverter(
        {0: target},
        constant_features=set(constant["train"].keys()),
        variable_features=set(variable.keys()),
    )
    converter_dict = {"train": converter}
    _install_fake_init_from_disk(monkeypatch, {"ds": (dataset_dict, converter_dict)})

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.set_features("ds", ["Base/Zone/VertexFields/pressure"])
    ref = SampleRef(backend_id="disk", dataset_id="ds", split="train", sample_id="0")
    assert service.load_sample(ref) is target
    # The user-selected field is forwarded, but the split's constant
    # features (mesh supports + globals) are always appended so the
    # rendered sample keeps its scalars/globals on top of the
    # user-selected variable fields.
    assert converter.last_features is not None
    assert "Base/Zone/VertexFields/pressure" in converter.last_features
    for path in constant["train"]:
        assert path in converter.last_features


def test_load_sample_without_filter_does_not_forward_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset_dir(tmp_path, "ds")
    target = object()
    dataset_dict = {"train": _FakeDataset(range(1))}
    converter = _FeatureAwareConverter({0: target})
    converter_dict = {"train": converter}
    _install_fake_init_from_disk(monkeypatch, {"ds": (dataset_dict, converter_dict)})

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    ref = SampleRef(backend_id="disk", dataset_id="ds", split="train", sample_id="0")
    assert service.load_sample(ref) is target
    assert converter.last_features is None


def test_streaming_open_expands_features_via_cgns_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming path: ``init_streaming_from_hub`` receives the expanded list.

    The expansion is delegated to
    ``plaid.utils.cgns_helper.update_features_for_CGNS_compatibility``;
    we patch that helper to a deterministic stub and assert the service
    hands the stub's output through.
    """
    repo_id = "org/stream_filter"
    variable = {"Base/Zone/VertexFields/pressure": None}
    constant = {"train": {"Base": None, "Base/Zone": None}}
    _install_fake_metadata(
        monkeypatch, variable_schema=variable, constant_schema=constant
    )

    captured: dict[str, object] = {}

    def _fake_init_streaming_from_hub(_repo, features=None):
        captured["features"] = features
        return (
            {"train": _FakeDataset(range(1))},
            {"train": _FakeConverter({0: object()})},
        )

    import plaid.storage as storage  # noqa: PLC0415

    monkeypatch.setattr(
        storage,
        "init_streaming_from_hub",
        _fake_init_streaming_from_hub,
        raising=False,
    )

    from plaid.utils import cgns_helper  # noqa: PLC0415

    def _fake_expand(features, _constant, _variable):
        # Deterministic: append a sentinel so we can verify that the
        # service actually routes through the helper instead of
        # forwarding the raw user selection.
        return sorted(set(features) | {"__expanded__"})

    monkeypatch.setattr(
        cgns_helper,
        "update_features_for_CGNS_compatibility",
        _fake_expand,
    )

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)
    service.set_features(repo_id, ["Base/Zone/VertexFields/pressure"])
    service.list_samples(repo_id)  # triggers ``_open``
    assert captured["features"] == [
        "Base/Zone/VertexFields/pressure",
        "__expanded__",
    ]


def test_set_features_invalidates_store_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changing the feature selection must force a reload of the dataset."""
    _make_dataset_dir(tmp_path, "ds")
    variable = {"Base/Zone/VertexFields/pressure": None}
    _install_fake_metadata(
        monkeypatch,
        variable_schema=variable,
        constant_schema={"train": {}},
    )
    dataset_dict = {"train": _FakeDataset(range(1))}
    converter = _FeatureAwareConverter(
        {0: object()},
        variable_features=set(variable.keys()),
    )
    _install_fake_init_from_disk(
        monkeypatch, {"ds": (dataset_dict, {"train": converter})}
    )
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.list_samples("ds")  # populates cache
    assert "ds" in service._store_cache  # noqa: SLF001
    service.set_features("ds", ["Base/Zone/VertexFields/pressure"])
    assert "ds" not in service._store_cache  # noqa: SLF001


def test_load_sample_auto_advances_cursor_on_first_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling ``load_sample`` with a cursor ref before any advance acts
    like "give me the first sample".
    """
    from plaid.viewer.models import SampleRef

    repo_id = "org/stream"
    records, mapping = _install_fake_streaming_dataset(monkeypatch, repo_id)
    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    service.add_hub_dataset(repo_id)
    ref = SampleRef(
        backend_id="hub", dataset_id=repo_id, split="train", sample_id="cursor"
    )
    sample = service.load_sample(ref)
    assert sample is mapping[records[0]]
    assert service.stream_cursor_position(repo_id, "train") == 0


class _KeyErrorOnFilteredConverter:
    """Converter whose filtered ``to_plaid`` path raises like PLAID does.

    Mirrors the real failure mode: the converter declares
    ``constant_features`` containing a path that its backing store
    cannot materialise, so passing ``features=sorted(constant_features)``
    triggers ``KeyError("Missing features in …")`` deep inside PLAID.
    The service must degrade gracefully and fall back to an unfiltered
    load instead of letting the error surface to the user.
    """

    def __init__(
        self,
        samples_by_index: dict[int, object],
        *,
        constant_features: set[str],
        variable_features: set[str] | None = None,
    ) -> None:
        self._samples = samples_by_index
        self.constant_features = constant_features
        self.variable_features = variable_features or set()
        self.unfiltered_calls = 0

    def to_plaid(self, dataset, index: int, features=None):  # noqa: ARG002
        if features is not None:
            raise KeyError("Missing features in dataset/converter: ['bogus']")
        self.unfiltered_calls += 1
        return self._samples[index]


def test_load_sample_falls_back_when_empty_filter_triggers_missing_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing the selection on a split whose ``constant_features`` trip
    the CGNS expander should not raise ``Missing features``.

    Reproduces the viewer bug where, on a split that shares none of the
    user-selected fields, the "geometry-only" fallback in
    :meth:`PlaidDatasetService.load_sample` used to hand the split's
    ``constant_features`` straight to ``Converter.to_plaid`` and crash
    with ``KeyError("Missing features in …")``. The service must now
    degrade to an unfiltered load so the user still sees the mesh.
    """
    _make_dataset_dir(tmp_path, "ds")
    variable = {"Base/Zone/VertexFields/pressure": None}
    constant = {"train": {"Base": None, "Base/Zone": None}}
    _install_fake_metadata(
        monkeypatch, variable_schema=variable, constant_schema=constant
    )
    target = object()
    dataset_dict = {"train": _FakeDataset(range(1))}
    converter = _KeyErrorOnFilteredConverter(
        {0: target},
        constant_features=set(constant["train"].keys()),
        variable_features=set(),  # split has no variable features at all
    )
    converter_dict = {"train": converter}
    _install_fake_init_from_disk(monkeypatch, {"ds": (dataset_dict, converter_dict)})

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    # Emulate the UI: the user selected a field that exists elsewhere in
    # the dataset metadata but not in this split.
    service.set_features("ds", ["Base/Zone/VertexFields/pressure"])
    ref = SampleRef(backend_id="disk", dataset_id="ds", split="train", sample_id="0")

    assert service.load_sample(ref) is target
    assert converter.unfiltered_calls == 1


def test_load_sample_does_not_reinject_deselected_constant_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user-visible *field* declared as a split constant must not be
    silently re-added to the request when the user deselects it.

    In PLAID, ``constant_features`` can hold genuine field paths (a
    field whose values happen to be constant across the split's
    samples, e.g. a signed-distance field precomputed offline). Those
    fields appear in the UI feature list and are toggleable. An
    earlier fix for the "Missing features" crash blindly re-injected
    every split constant on top of the user's selection, which
    defeated the filter: deselecting ``sdf`` still loaded ``sdf``.

    The service must only re-inject CGNS bookkeeping paths
    (coordinates, connectivities, ...), not user-visible fields.
    """
    _make_dataset_dir(tmp_path, "ds")
    variable = {"Base_2_2/Zone/VertexFields/pressure": None}
    constant = {
        "train": {
            # User-visible field -> must drop when deselected.
            "Base_2_2/Zone/VertexFields/sdf": None,
            # Time-series bookkeeping for ``sdf`` -> must drop with it.
            "Base_2_2/Zone/VertexFields/sdf_times": None,
            # CGNS bookkeeping -> must always be kept.
            "Base_2_2": None,
            "Base_2_2/Zone": None,
            "Base_2_2/Zone/GridCoordinates/CoordinateX": None,
        }
    }
    _install_fake_metadata(
        monkeypatch, variable_schema=variable, constant_schema=constant
    )
    target = object()
    dataset_dict = {"train": _FakeDataset(range(1))}
    converter = _FeatureAwareConverter(
        {0: target},
        constant_features=set(constant["train"].keys()),
        variable_features=set(variable.keys()),
    )
    _install_fake_init_from_disk(
        monkeypatch, {"ds": (dataset_dict, {"train": converter})}
    )

    service = PlaidDatasetService(ViewerConfig(datasets_root=tmp_path))
    # User clears the selection -> load only the geometry.
    service.set_features("ds", [])
    ref = SampleRef(backend_id="disk", dataset_id="ds", split="train", sample_id="0")
    assert service.load_sample(ref) is target
    # Bookkeeping paths are preserved so the renderer can draw the mesh...
    assert converter.last_features is not None
    assert "Base_2_2/Zone/GridCoordinates/CoordinateX" in converter.last_features
    # ... but the deselected user-visible field must NOT be re-injected,
    # and its ``_times`` bookkeeping path must follow the same fate.
    assert "Base_2_2/Zone/VertexFields/sdf" not in converter.last_features
    assert "Base_2_2/Zone/VertexFields/sdf_times" not in converter.last_features
    assert "Base_2_2/Zone/VertexFields/pressure" not in converter.last_features
