"""JSON serialization helpers for :class:`plaid.containers.sample.Sample`.

This module provides a language-neutral payload for full ``Sample`` objects.
Each timestamped CGNS tree is encoded with :mod:`plaid.utils.cgns_json` and
wrapped in a versioned top-level schema.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from ..containers.sample import Sample
from .cgns_json import cgns_tree_from_json_payload, cgns_tree_to_json_payload

FORMAT_NAME = "plaid-sample-json"
FORMAT_VERSION = 1


def sample_to_json_payload(sample: Sample) -> dict[str, Any]:
    """Convert a full Sample to a JSON-compatible payload.

    Args:
        sample: Sample instance to serialize.

    Returns:
        A JSON-compatible dictionary containing format metadata and all
        timestamped CGNS trees from the sample.
    """
    trees_payload = []
    for time, tree in sample.data.items():
        trees_payload.append(
            {
                "time": _encode_time(time),
                "tree": cgns_tree_to_json_payload(tree),
            }
        )

    return {
        "format": FORMAT_NAME,
        "version": FORMAT_VERSION,
        "trees": trees_payload,
    }


def sample_from_json_payload(payload: dict[str, Any]) -> Sample:
    """Rebuild a full Sample from a JSON-compatible payload.

    Args:
        payload: Payload produced by :func:`sample_to_json_payload`.

    Returns:
        A reconstructed :class:`Sample`.

    Raises:
        ValueError: If payload format or version is unsupported.
    """
    if payload.get("format") != FORMAT_NAME:
        raise ValueError(f"Unsupported Sample JSON format: {payload.get('format')!r}")
    if payload.get("version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported Sample JSON version: {payload.get('version')!r}")

    trees = payload.get("trees")
    if not isinstance(trees, list):
        raise ValueError("Sample JSON payload must contain a list in 'trees'")

    from ..containers.sample import Sample

    sample = Sample(path=None)
    for entry in trees:
        if not isinstance(entry, dict):
            raise ValueError("Each Sample JSON tree entry must be a dictionary")
        if "time" not in entry or "tree" not in entry:
            raise ValueError(
                "Each Sample JSON tree entry must contain 'time' and 'tree'"
            )

        time_value = _decode_time(entry["time"])
        sample.data[time_value] = cgns_tree_from_json_payload(entry["tree"])

    return sample


def sample_to_json(sample: "Sample", **json_kwargs: Any) -> str:
    """Convert a full Sample to a JSON string.

    Args:
        sample: Sample instance to serialize.
        **json_kwargs: Extra keyword arguments forwarded to :func:`json.dumps`.

    Returns:
        A JSON string containing the encoded sample.
    """
    return json.dumps(sample_to_json_payload(sample), **json_kwargs)


def sample_from_json(text: str) -> Sample:
    """Rebuild a Sample from a JSON string.

    Args:
        text: JSON string produced by :func:`sample_to_json`.

    Returns:
        A reconstructed :class:`Sample`.
    """
    payload = json.loads(text)
    return sample_from_json_payload(payload)


def _encode_time(value: Any) -> float | int:
    """Convert time keys to JSON scalar values.

    Args:
        value: Time key from ``sample.data``.

    Returns:
        A Python ``float`` or ``int``.
    """
    if isinstance(value, np.generic):
        value = value.item()

    if isinstance(value, (int, float)):
        return value

    raise TypeError(f"Unsupported time key type for Sample JSON: {type(value)!r}")


def _decode_time(value: Any) -> float:
    """Decode a serialized time value to float.

    Args:
        value: Encoded JSON scalar time.

    Returns:
        Time as float.
    """
    if not isinstance(value, (int, float)):
        raise ValueError("Sample JSON time entries must be numeric")
    return float(value)
