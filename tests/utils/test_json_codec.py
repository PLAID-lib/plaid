"""Tests for the generic JSON value codec helpers."""

import numpy as np
import pytest

from plaid.utils.json_codec import (
    _decode_array,
    decode_json_value,
    decode_leaf_value,
    encode_json_value,
    encode_leaf_value,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        (np.int32(7), 7),
        (np.float64(1.25), 1.25),
        ((np.int64(1), np.int64(2)), [1, 2]),
    ],
)
def test_encode_leaf_value_normalizes_numpy_scalars_and_tuples(value, expected):
    """NumPy scalar values and tuples are converted to JSON-compatible data."""
    assert encode_leaf_value(value) == expected


def test_encode_decode_leaf_value_roundtrips_bytes():
    """Bytes values are encoded as base64 objects and decoded back to bytes."""
    value = b"CGNS bytes"

    encoded = encode_leaf_value(value)

    assert encoded["kind"] == "bytes"
    assert decode_leaf_value(encoded) == value


def test_decode_leaf_value_decodes_nested_lists():
    """List payloads are decoded recursively."""
    encoded_bytes = encode_leaf_value(b"nested bytes")

    assert decode_leaf_value([encoded_bytes, [1, encoded_bytes]]) == [
        b"nested bytes",
        [1, b"nested bytes"],
    ]


def test_encode_leaf_value_rejects_unsupported_values():
    """Unsupported value types raise a TypeError with a clear message."""
    with pytest.raises(TypeError, match="Unsupported value type for JSON serialization"):
        encode_leaf_value({"not": "a supported value"})


def test_decode_leaf_value_leaves_unknown_dict_kind_unchanged():
    """Unknown dictionary payloads are passed through unchanged."""
    value = {"kind": "custom", "data": [1, 2, 3]}

    assert decode_leaf_value(value) is value


def test_decode_array_rejects_unknown_encoding():
    """Only JSON and base64 ndarray encodings are supported."""
    with pytest.raises(ValueError, match="Unsupported ndarray encoding"):
        _decode_array(
            {
                "encoding": "unsupported",
                "dtype": "<f8",
                "shape": [0],
                "data": "",
            }
        )


def test_encode_decode_array_roundtrips_numeric_dtypes():
    """Numeric NumPy arrays survive the base64 encode/decode roundtrip."""
    array = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    encoded = encode_leaf_value(array)
    decoded = decode_leaf_value(encoded)

    assert encoded["kind"] == "ndarray"
    assert np.array_equal(decoded, array)
    assert decoded.dtype == array.dtype


def test_encode_decode_array_roundtrips_unicode_dtype():
    """Unicode arrays are encoded with the JSON encoding and restored."""
    array = np.array(["alpha", "beta"], dtype="<U5")

    encoded = encode_leaf_value(array)
    decoded = decode_leaf_value(encoded)

    assert encoded["encoding"] == "json"
    assert np.array_equal(decoded, array)
    assert decoded.dtype == array.dtype


def test_encode_array_rejects_object_dtype():
    """Object dtype arrays are intentionally not part of the portable schema."""
    array = np.array([{"not": "portable"}], dtype=object)

    with pytest.raises(TypeError, match="Object dtype arrays"):
        encode_leaf_value(array)


def test_encode_json_value_roundtrips_nested_structures():
    """Nested dicts and lists with arrays survive the JSON value roundtrip."""
    value = {
        "scalar": 3,
        "text": "hello",
        "array": np.array([1, 2, 3], dtype=np.int64),
        "nested": [np.float64(1.5), {"bytes": b"data"}],
    }

    encoded = encode_json_value(value)
    decoded = decode_json_value(encoded)

    assert decoded["scalar"] == 3
    assert decoded["text"] == "hello"
    assert np.array_equal(decoded["array"], value["array"])
    assert decoded["nested"][0] == 1.5
    assert decoded["nested"][1]["bytes"] == b"data"