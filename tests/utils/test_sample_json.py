"""Tests for language-neutral Sample JSON serialization helpers."""

import json

import numpy as np
import pytest

from plaid.containers.sample import Sample
from plaid.utils.cgns_helper import compare_cgns_trees
from plaid.utils.sample_json import (
    sample_from_json,
    sample_from_json_payload,
    sample_to_json,
    sample_to_json_payload,
)


def _assert_no_numpy_objects(value):
    """Assert recursively that a JSON payload contains no NumPy objects."""
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_numpy_objects(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_numpy_objects(item)
    else:
        assert not isinstance(value, (np.ndarray, np.generic))


def _assert_same_sample_content(reference: Sample, candidate: Sample) -> None:
    """Assert that two Samples carry the same timestamped CGNS trees."""
    assert sorted(reference.data.keys()) == sorted(candidate.data.keys())

    for time in reference.data:
        assert compare_cgns_trees(
            reference.data[time],
            candidate.data[time],
        )


def test_empty_sample_json_payload_roundtrip(sample: Sample):
    """An empty Sample can be serialized and reconstructed from payload."""
    payload = sample_to_json_payload(sample)
    decoded = sample_from_json_payload(payload)

    assert payload["format"] == "plaid-sample-json"
    assert payload["version"] == 1
    assert payload["trees"] == []
    _assert_same_sample_content(sample, decoded)


def test_real_sample_json_roundtrip(sample_with_tree):
    """A Sample with a real CGNS tree survives JSON-string roundtrip."""
    text = sample_to_json(sample_with_tree)
    decoded = sample_from_json(text)

    _assert_same_sample_content(sample_with_tree, decoded)


def test_sample_json_payload_roundtrip_with_multiple_timestamps(sample_with_tree, tree):
    """All timestamps are serialized and restored for full Sample deserialization."""
    sample_with_tree.add_tree(tree, time=1.0)

    payload = sample_to_json_payload(sample_with_tree)
    decoded = sample_from_json_payload(payload)

    assert len(payload["trees"]) == 2
    _assert_same_sample_content(sample_with_tree, decoded)


def test_sample_json_payload_is_json_compatible(sample_with_tree):
    """Serialized sample payload contains only JSON-compatible scalar/container types."""
    payload = sample_to_json_payload(sample_with_tree)

    _assert_no_numpy_objects(payload)
    json.dumps(payload)


def test_sample_json_rejects_invalid_payloads():
    """Invalid Sample payload metadata and malformed trees raise explicit errors."""
    with pytest.raises(ValueError, match="Unsupported Sample JSON format"):
        sample_from_json_payload({"format": "other", "version": 1, "trees": []})

    with pytest.raises(ValueError, match="Unsupported Sample JSON version"):
        sample_from_json_payload(
            {"format": "plaid-sample-json", "version": 999, "trees": []}
        )

    with pytest.raises(ValueError, match="must contain a list in 'trees'"):
        sample_from_json_payload(
            {"format": "plaid-sample-json", "version": 1, "trees": {}}
        )

    with pytest.raises(ValueError, match="must be a dictionary"):
        sample_from_json_payload(
            {
                "format": "plaid-sample-json",
                "version": 1,
                "trees": ["not-a-dict"],
            }
        )

    with pytest.raises(ValueError, match="must contain 'time' and 'tree'"):
        sample_from_json_payload(
            {
                "format": "plaid-sample-json",
                "version": 1,
                "trees": [{"time": 0.0}],
            }
        )
