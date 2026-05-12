"""Tests for :mod:`plaid.storage.in_memory`."""

from typing import Any, cast

import pytest

from plaid.containers.sample import Sample
from plaid.storage.in_memory import InMemoryBackend, _find_first_missing


def _new_sample() -> Sample:
    """Build a minimal valid sample for in-memory storage tests."""
    return Sample(path=None)


def test_find_first_missing():
    """Helper returns the first non-negative missing integer key."""
    assert _find_first_missing([]) == 0
    assert _find_first_missing([0, 1, 3]) == 2


def test_backend_basic_len_and_getitem():
    """Backend supports len and indexing by int/slice/sequence."""
    backend = InMemoryBackend()
    assert backend.name == "in_memory"
    assert len(backend) == 0

    samples = [_new_sample(), _new_sample(), _new_sample()]
    ids = backend.add_sample(samples)
    assert ids == [0, 1, 2]
    assert len(backend) == 3

    assert backend[0] is samples[0]
    assert backend[1:3] == [samples[1], samples[2]]
    assert backend[[2, 0]] == [samples[2], samples[0]]


def test_add_sample_single_and_validation_errors():
    """Single add_sample path validates sample_id and sample types."""
    backend = InMemoryBackend()
    sample = _new_sample()

    new_id = backend.add_sample(sample)
    assert new_id == 0
    assert backend[0] is sample

    with pytest.raises(TypeError, match="sample_id must be an int"):
        backend.add_sample(sample, sample_id=[1])

    with pytest.raises(TypeError, match="sample must be a Sample"):
        backend.add_sample(123)  # type: ignore[arg-type]


def test_add_sample_sequence_and_validation_errors():
    """Sequence add_sample path checks IDs shape/type/uniqueness."""
    backend = InMemoryBackend()
    samples = [_new_sample(), _new_sample()]

    assert backend.add_sample(samples, sample_id=[10, 11]) == [10, 11]
    assert backend[10] is samples[0]
    assert backend[11] is samples[1]

    with pytest.raises(TypeError, match="sample_id must be a sequence"):
        backend.add_sample(samples, sample_id=1)

    with pytest.raises(ValueError, match="sample_ids must be unique"):
        backend.add_sample(samples, sample_id=[0, 0])

    with pytest.raises(ValueError, match="length of the list of samples"):
        backend.add_sample(samples, sample_id=[0])


def test_set_sample_single_iterable_and_validation_errors():
    """set_sample supports single and iterable paths with validations."""
    backend = InMemoryBackend()

    s0 = _new_sample()
    s1 = _new_sample()
    s2 = _new_sample()

    assert backend.set_sample(s0, sample_id=None) == 0
    assert backend.set_sample(s1, sample_id=2) == 2
    assert backend.set_sample(s2, sample_id=None) == 1

    # Overwrite existing id
    replacement = _new_sample()
    assert backend.set_sample(replacement, sample_id=2) == 2
    assert backend[2] is replacement

    # Iterable path with explicit ids
    s3 = _new_sample()
    s4 = _new_sample()
    ids = backend.set_sample([s3, s4], sample_id=[5, 6])
    assert ids == [5, 6]
    assert backend[5] is s3
    assert backend[6] is s4

    # Iterable path with inferred ids
    s5 = _new_sample()
    s6 = _new_sample()
    ids = backend.set_sample([s5, s6], sample_id=None)
    assert len(ids) == 2
    assert all(isinstance(i, int) for i in ids)

    with pytest.raises(TypeError, match="sample should be of type Sample"):
        backend.set_sample(3.14, sample_id=None)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="sample_id should be of type"):
        backend.set_sample(_new_sample(), sample_id="abc")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="sample_id should be positive"):
        backend.set_sample(_new_sample(), sample_id=-1)


def test_merge_dataset_and_unsupported_operations():
    """merge_dataset behavior and unsupported hub/disk methods."""
    backend = InMemoryBackend()
    assert backend.merge_dataset(None) is None

    source = InMemoryBackend()
    src_samples = [_new_sample(), _new_sample()]
    source.add_sample(src_samples)

    merged_ids = backend.merge_dataset(source)
    assert merged_ids == [0, 1]
    assert backend[0] is src_samples[0]
    assert backend[1] is src_samples[1]

    with pytest.raises(NotImplementedError):
        InMemoryBackend.init_from_disk("/tmp/dummy")

    with pytest.raises(NotImplementedError):
        backend.download_from_hub("repo/id", "/tmp/dummy")

    with pytest.raises(NotImplementedError):
        backend.download_from_hub(
            "repo/id",
            "/tmp/dummy",
            split_ids={"train": [0]},
            features=["Base/Zone/Field"],
            overwrite=True,
        )

    with pytest.raises(NotImplementedError):
        backend.init_datasetdict_streaming_from_hub("repo/id")

    with pytest.raises(NotImplementedError):
        backend.init_datasetdict_streaming_from_hub(
            "repo/id", split_ids={"train": [0]}, features=["Base/Zone/Field"]
        )

    with pytest.raises(NotImplementedError):
        backend.generate_to_disk("/tmp/out", generators={})

    with pytest.raises(NotImplementedError):
        backend.push_local_to_hub("repo/id", "/tmp/out")

    with pytest.raises(NotImplementedError):
        backend.configure_dataset_card("repo/id", cast(dict[str, Any], {"a": 1}))
